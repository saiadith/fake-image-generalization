"""CPU-side study on frozen features: all experiments, tables and figures.

Experiments
  E1 shortcut test      : train on 'raw' (real=JPEG, fake=PNG) vs 'jpeg95' (format equalised),
                          test on jpeg95. Exposes detectors that learned file format, not generation.
  E2 leave-one-generator-out (LOGO): train on 7 generators, test on the held-out one.
  E3 robustness         : LOGO models tested on jpeg75/jpeg50/resize50/blur views.
  E4 method             : CLIP+DINO+spectrum probe with perturbation-augmented training,
                          temperature scaling, and an OOD-based abstention rule.
Metrics: AUROC, AP, accuracy at the source-domain threshold (exposes threshold drift),
ECE, risk-coverage; bootstrap 95% CIs on headline numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ data
class Feats:
    def __init__(self, root):
        self.root = Path(root)
        self.meta = pd.read_csv(self.root / "meta.csv")
        self._c = {}

    def get(self, kind, view):
        key = (kind, view)
        if key not in self._c:
            if kind == "clip+dino":
                x = np.hstack([self.get("clip", view), self.get("dino", view)])
            elif kind == "all":
                x = np.hstack([self.get("clip", view), self.get("dino", view), self.get("spec", view)])
            else:
                x = np.load(self.root / f"{kind}_{view}.npy").astype(np.float32)
                extra = self.root / f"{kind}_{view}_train.npy"  # degraded views of the TRAIN split (augmentation)
                if extra.exists():
                    have = np.load(self.root / f"have_{view}_train.npy")
                    x[have] = np.load(extra).astype(np.float32)[have]
                if kind in ("clip", "dino"):
                    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            self._c[key] = x
        return self._c[key]


# ------------------------------------------------------------------ metrics
def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    conf = np.where(p >= 0.5, p, 1 - p); pred = (p >= 0.5).astype(int)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def metrics(p, y, thr=0.5):
    if len(np.unique(y)) < 2:
        return {"auroc": np.nan, "ap": np.nan, "acc": float(((p >= thr) == y).mean()), "ece": ece(p, y)}
    return {"auroc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "acc": float(((p >= thr) == y).mean()), "ece": ece(p, y)}


def bootstrap_ci(p, y, fn=roc_auc_score, n=500):
    vals = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(fn(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ------------------------------------------------------------------ models
class Probe:
    """Standardise -> logistic regression -> temperature scaling (fit on held-out val)."""

    def __init__(self, C=1.0):
        self.sc, self.lr, self.T = StandardScaler(), LogisticRegression(C=C, max_iter=3000), 1.0

    def fit(self, X, y, Xval=None, yval=None):
        self.lr.fit(self.sc.fit_transform(X), y)
        if Xval is not None:
            z = self.logit(Xval)
            grid = np.exp(np.linspace(np.log(0.05), np.log(20), 120))
            nll = [np.mean(np.logaddexp(0, -np.where(yval == 1, 1, -1) * z / t)) for t in grid]
            self.T = float(grid[int(np.argmin(nll))])
        return self

    def logit(self, X):
        return self.lr.decision_function(self.sc.transform(X))

    def proba(self, X):
        return 1 / (1 + np.exp(-self.logit(X) / self.T))


class Abstainer:
    """OOD score = Mahalanobis distance to the nearest training class-generator centroid
    in standardised feature space (shared diagonal covariance). Unfamiliar generators score
    high -> abstain. Threshold set to the 95th percentile of in-distribution validation scores."""

    def fit(self, X, groups):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.mu = np.stack([Z[groups == g].mean(0) for g in np.unique(groups)])
        self.var = Z.var(0) + 1e-3
        return self

    def score(self, X):
        Z = self.sc.transform(X)
        d = ((Z[:, None, :] - self.mu[None]) ** 2 / self.var).sum(-1)
        return np.sqrt(d.min(1) / Z.shape[1])


def risk_coverage(p, y, score):
    """Sort by OOD score ascending (most familiar first); error rate as coverage grows."""
    order = np.argsort(score)
    err = ((p[order] >= 0.5) != y[order]).astype(float)
    cov = np.arange(1, len(y) + 1) / len(y)
    return cov, np.cumsum(err) / np.arange(1, len(y) + 1)


# ------------------------------------------------------------------ experiments
def split_val(idx, frac=0.15):
    idx = RNG.permutation(idx)
    k = int(len(idx) * frac)
    return idx[k:], idx[:k]


def run(feat_root, out_dir="results"):
    F = Feats(feat_root)
    m = F.meta
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    gens = sorted(m.generator.unique())
    tr_all = np.where(m.split == "train")[0]
    te_all = np.where(m.split == "test")[0]
    y = m.label.values
    R = {"generators": gens, "n": {"train": int(len(tr_all)), "test": int(len(te_all))}}
    R["ext_by_label"] = m.groupby(["label", "ext"]).size().unstack(fill_value=0).to_dict()

    # E1 shortcut test (in-distribution, all generators)
    E1 = {}
    for kind in ["clip", "dino", "spec"]:
        for train_view in ["raw", "jpeg95"]:
            tr, va = split_val(tr_all)
            pr = Probe().fit(F.get(kind, train_view)[tr], y[tr], F.get(kind, train_view)[va], y[va])
            for test_view in ["raw", "jpeg95"]:
                E1[f"{kind}|train={train_view}|test={test_view}"] = metrics(pr.proba(F.get(kind, test_view)[te_all]), y[te_all])
    R["E1_shortcut"] = E1

    # E2/E3/E4 leave-one-generator-out, all trained on format-equalised jpeg95
    kinds = ["clip", "dino", "clip+dino", "spec", "all"]
    rows, rc = [], {}
    for g in gens:
        tr = tr_all[m.generator.values[tr_all] != g]
        tr, va = split_val(tr)
        te = te_all[m.generator.values[te_all] == g]
        te_in = te_all[m.generator.values[te_all] != g]
        for kind in kinds + ["ours"]:
            if kind == "ours":
                # perturbation-augmented training: every training image contributes its clean,
                # format-varied and (if extracted) degraded views
                fk = "all"
                views = ["jpeg95", "raw"] + [v for v in ["jpeg75", "jpeg50", "resize50", "blur"]
                                             if (F.root / f"clip_{v}_train.npy").exists()]
                Xtr = np.vstack([F.get(fk, v)[tr] for v in views])
                ytr = np.concatenate([y[tr]] * len(views))
                R.setdefault("ours_train_views", views)
            else:
                fk = kind
                Xtr, ytr = F.get(fk, "jpeg95")[tr], y[tr]
            pr = Probe(C=0.5).fit(Xtr, ytr, F.get(fk, "jpeg95")[va], y[va])
            for view in ["jpeg95", "jpeg75", "jpeg50", "resize50", "blur"]:
                p = pr.proba(F.get(fk, view)[te])
                row = {"heldout": g, "model": kind, "view": view, **metrics(p, y[te])}
                if view == "jpeg95":
                    row["in_dist_auroc"] = metrics(pr.proba(F.get(fk, "jpeg95")[te_in]), y[te_in])["auroc"]
                rows.append(row)
            if kind == "ours":
                ab = Abstainer().fit(F.get(fk, "jpeg95")[tr], (m.generator.values[tr] + "_" + y[tr].astype(str)))
                X_mix = np.vstack([F.get(fk, "jpeg95")[te_in], F.get(fk, "jpeg95")[te]])
                y_mix = np.concatenate([y[te_in], y[te]])
                s = ab.score(X_mix)
                is_new = np.r_[np.zeros(len(te_in)), np.ones(len(te))]
                thr = np.percentile(ab.score(F.get(fk, "jpeg95")[va]), 95)
                p_mix = pr.proba(X_mix)
                kept = s <= thr
                rc[g] = {"ood_auroc_new_vs_seen": float(roc_auc_score(is_new, s)),
                         "abstain_rate_seen": float((~kept[is_new == 0]).mean()),
                         "abstain_rate_new": float((~kept[is_new == 1]).mean()),
                         "acc_all": float(((p_mix >= .5) == y_mix).mean()),
                         "acc_kept": float(((p_mix[kept] >= .5) == y_mix[kept]).mean()) if kept.any() else None}
        print("done heldout", g, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "logo.csv", index=False)
    R["E2_logo_mean"] = (df[df.view == "jpeg95"].groupby("model")[["auroc", "acc", "ece", "in_dist_auroc"]]
                         .mean().round(4).to_dict())
    R["E3_robustness_auroc"] = df.pivot_table(index="model", columns="view", values="auroc").round(4).to_dict()
    R["E4_abstention"] = rc
    (out / "summary.json").write_text(json.dumps(R, indent=2, default=float))
    make_figures(df, R, out)
    return R


def make_figures(df, R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df.view == "jpeg95"].pivot_table(index="heldout", columns="model", values="auroc")
    fig, ax = plt.subplots(figsize=(8, 4))
    d.plot.bar(ax=ax, width=0.8)
    ax.set(ylabel="AUROC on held-out generator", ylim=(0.4, 1.0), title="Leave-one-generator-out")
    ax.axhline(0.5, color="k", lw=0.6, ls="--"); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "logo_auroc.png", dpi=150)
    r = df.pivot_table(index="model", columns="view", values="auroc")[["jpeg95", "jpeg75", "jpeg50", "resize50", "blur"]]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(r.values, vmin=0.3, vmax=1, cmap="viridis")
    ax.set_xticks(range(r.shape[1]), r.columns, rotation=30, ha="right"); ax.set_yticks(range(r.shape[0]), r.index)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            ax.text(j, i, f"{r.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im); ax.set_title("Robustness (mean LOGO AUROC)")
    fig.tight_layout(); fig.savefig(out / "robustness.png", dpi=150)


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results"), indent=1, default=float)[:4000])
