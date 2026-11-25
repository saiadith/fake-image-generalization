"""Out-of-distribution generators: train on all 7 GenImage generators (format-equalised), then measure
fake recall at the trained threshold on Synthbuster (DALL-E 2/3, Firefly, Midjourney v5, SD 1.3/1.4/2/XL, GLIDE)
and an unverified community Nano Banana 2.0 set. Real counterparts (RAISE) are not available on Kaggle, so we
report recall (the documented failure mode), plus specificity on held-out GenImage real images for reference."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from evaluate import Feats, Probe, split_val

F = Feats("feats_raw/feats"); m = F.meta; y = m.label.values
E = Feats("feats_ext"); me = E.meta
tr_all = np.where(m.split == "train")[0]; te_real = np.where((m.split == "test") & (m.label == 0))[0]
tr, va = split_val(tr_all)
DEG = ["jpeg75", "jpeg50", "resize50", "blur"]
models = {"clip": ("clip", ["jpeg95"]), "clip+dino+spec": ("all", ["jpeg95"]),
          "ours (deg-augmented)": ("all", ["jpeg95", "raw"] + DEG)}
rows = []
for name, (fk, views) in models.items():
    pr = Probe(C=0.5).fit(np.vstack([F.get(fk, v)[tr] for v in views]), np.concatenate([y[tr]] * len(views)),
                          F.get(fk, "jpeg95")[va], y[va])
    for view in ["jpeg95", "jpeg50"]:
        spec = float((pr.proba(F.get(fk, view)[te_real]) < .5).mean())
        p = pr.proba(E.get(fk, view))
        for g in sorted(me.generator.unique()):
            mask = me.generator.values == g
            rows.append({"model": name, "view": view, "generator": g, "recall": float((p[mask] >= .5).mean()),
                         "mean_p_fake": float(p[mask].mean()), "real_specificity_genimage": spec})
df = pd.DataFrame(rows); df.to_csv("results/external.csv", index=False)
t = df.pivot_table(index="generator", columns=["view", "model"], values="recall").round(3)
pd.set_option("display.width", 200); print(t)
print(df.groupby(["view", "model"])[["recall", "real_specificity_genimage"]].mean().round(3))
json.dump(json.loads(t.to_json()), open("results/external_summary.json", "w"), indent=1)
