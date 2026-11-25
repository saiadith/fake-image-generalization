"""Matched compression history: does the JPEG-50 collapse come from a leftover format shortcut?

Baseline : reals = JPEG (original) -> q95,  fakes = PNG -> q95             (reals compressed twice, fakes once)
Matched  : reals = same,                    fakes = JPEG (real-like q) -> q95  (both compressed twice)

Both are tested on the same JPEG-50 / clean views, on unseen GenImage generators (LOGO) and on Synthbuster.
If matched recall at JPEG-50 rises a lot, the collapse was mostly the shortcut; if not, it's real information loss.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from evaluate import Feats, Probe, metrics, split_val  # noqa: E402

F = Feats("feats_raw/feats"); m = F.meta; y = m.label.values
E = Feats("feats_ext"); me = E.meta
have_m = np.load("feats_raw/feats/have_matched.npy")


def train_X(fk, kind, rows):
    if kind == "baseline":
        return F.get(fk, "jpeg95")[rows]
    X = F.get(fk, "jpeg95")[rows].copy()
    M = F.get(fk, "matched")[rows]
    fake = (y[rows] == 1) & have_m[rows]
    X[fake] = M[fake]
    return X


tr_all = np.where(m.split == "train")[0]; te_all = np.where(m.split == "test")[0]
rows = []
for fk in ["clip", "all"]:
    for kind in ["baseline", "matched"]:
        # LOGO on GenImage
        for g in sorted(m.generator.unique()):
            tr = tr_all[m.generator.values[tr_all] != g]; tr, va = split_val(tr)
            te = te_all[m.generator.values[te_all] == g]
            pr = Probe(C=0.5).fit(train_X(fk, kind, tr), y[tr], train_X(fk, kind, va), y[va])
            for view in ["jpeg95", "jpeg50"]:
                p = pr.proba(F.get(fk, view)[te])
                rows.append({"features": fk, "train": kind, "test": f"genimage-logo/{view}", "heldout": g, **metrics(p, y[te]),
                             "recall": float((p[y[te] == 1] >= .5).mean()), "specificity": float((p[y[te] == 0] < .5).mean())})
        # all 7 generators -> Synthbuster (fakes only) + GenImage test reals for specificity
        tr, va = split_val(tr_all)
        pr = Probe(C=0.5).fit(train_X(fk, kind, tr), y[tr], train_X(fk, kind, va), y[va])
        sb = me.generator.str.startswith("sb_").values
        reals = te_all[y[te_all] == 0]
        for view in ["jpeg95", "jpeg50"]:
            rows.append({"features": fk, "train": kind, "test": f"synthbuster/{view}",
                         "recall": float((pr.proba(E.get(fk, view)[sb]) >= .5).mean()),
                         "specificity": float((pr.proba(F.get(fk, view)[reals]) < .5).mean())})

df = pd.DataFrame(rows); df.to_csv("results/matched.csv", index=False)
t = df.groupby(["features", "train", "test"])[["recall", "specificity", "auroc"]].mean().round(3)
pd.set_option("display.width", 200); print(t)
json.dump(json.loads(t.reset_index().to_json(orient="records")), open("results/matched_summary.json", "w"), indent=1)
