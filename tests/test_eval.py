import numpy as np
import pandas as pd
import pytest

from evaluate import Abstainer, Probe, ece, metrics, run


def test_ece_perfect_and_bad():
    y = np.array([0, 1] * 50)
    assert ece(y.astype(float) * 0.999 + 0.0005, y) < 0.01
    assert ece(np.full(100, 0.99), y) > 0.4  # always 99% "fake", half wrong


def test_metrics_threshold_drift_visible():
    y = np.array([0] * 50 + [1] * 50)
    p = np.r_[np.linspace(0.6, 0.7, 50), np.linspace(0.71, 0.9, 50)]  # perfect ranking, bad threshold
    m = metrics(p, y)
    assert m["auroc"] == 1.0 and m["acc"] == 0.5


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5)); y = (X[:, 0] + rng.normal(scale=1.5, size=600) > 0).astype(int)
    pr = Probe(C=1e4).fit(X[:400], y[:400], X[400:500], y[400:500])
    assert pr.T > 0.0
    assert ece(pr.proba(X[500:]), y[500:]) < 0.15


def test_abstainer_scores_far_points_higher():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)); g = np.array(["a"] * 100 + ["b"] * 100); X[100:] += 3
    ab = Abstainer().fit(X, g)
    assert ab.score(np.full((1, 4), 20.0))[0] > ab.score(X[:5]).max()


@pytest.fixture
def fake_feats(tmp_path):
    rng = np.random.default_rng(1)
    gens = ["g1", "g2", "g3"]
    rows = [dict(path="x", generator=g, split=s, label=l, ext=".png" if l else ".jpeg")
            for g in gens for s, n in (("train", 60), ("test", 30)) for l in (0, 1) for _ in range(n)]
    m = pd.DataFrame(rows); m.to_csv(tmp_path / "meta.csv", index=False)
    d = rng.normal(size=768)
    for kind, dim in (("clip", 768), ("dino", 768), ("spec", 64)):
        for v in ["raw", "jpeg95", "jpeg75", "jpeg50", "resize50", "blur"]:
            X = rng.normal(size=(len(m), dim)).astype(np.float32)
            X[m.label.values == 1] += 0.25 * d[:dim]
            if v == "raw":
                X[:, 0] += m.label.values * 5  # planted format shortcut
            np.save(tmp_path / f"{kind}_{v}.npy", X)
    return tmp_path


def test_pipeline_detects_planted_shortcut(fake_feats, tmp_path):
    R = run(fake_feats, tmp_path / "out")
    e1 = R["E1_shortcut"]
    assert e1["spec|train=raw|test=raw"]["acc"] > e1["spec|train=raw|test=jpeg95"]["acc"] + 0.1
    assert set(R["E2_logo_mean"]["auroc"]) >= {"clip", "dino", "spec", "ours"}
    assert (tmp_path / "out" / "logo_auroc.png").exists()
