"""Leave-one-perturbation-out: does degradation augmentation transfer to an UNSEEN degradation?
Generators are also held out (LOGO), so each test is unseen-generator x unseen-degradation."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from evaluate import Feats, Probe, metrics, split_val

F = Feats(sys.argv[1] if len(sys.argv) > 1 else "feats_raw/feats"); m = F.meta; y = m.label.values
tr_all = np.where(m.split == "train")[0]; te_all = np.where(m.split == "test")[0]
DEG = ["jpeg75", "jpeg50", "resize50", "blur"]
rows = []
for held in DEG + [None]:
    train_views = ["jpeg95", "raw"] + [v for v in DEG if v != held and not (held and held.startswith("jpeg") and v.startswith("jpeg"))]
    for g in sorted(m.generator.unique()):
        tr = tr_all[m.generator.values[tr_all] != g]; tr, va = split_val(tr)
        te = te_all[m.generator.values[te_all] == g]
        pr = Probe(C=0.5).fit(np.vstack([F.get("all", v)[tr] for v in train_views]), np.concatenate([y[tr]] * len(train_views)),
                              F.get("all", "jpeg95")[va], y[va])
        for tv in DEG:
            rows.append({"held_out_deg": held or "none", "train_views": "+".join(train_views), "test_view": tv,
                         "heldout_gen": g, **metrics(pr.proba(F.get("all", tv)[te]), y[te])})
df = pd.DataFrame(rows); df.to_csv("results/lopo.csv", index=False)
t = df.pivot_table(index="held_out_deg", columns="test_view", values=["auroc", "acc"]).round(3)
print(t); json.dump(json.loads(t.to_json()), open("results/lopo_summary.json", "w"), indent=1)
