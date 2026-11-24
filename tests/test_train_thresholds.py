import numpy as np

from suspackage.train.thresholds import choose_thresholds, decide, threshold_at_fpr

POL = {"lenient": {"block_fpr": 0.005, "warn_fpr": 0.03},
       "balanced": {"block_fpr": 0.01, "warn_fpr": 0.05},
       "strict": {"block_fpr": 0.03, "warn_fpr": 0.10}}


def _scores(seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(3000), np.ones(3000)].astype(int)
    p = np.r_[rng.beta(2, 8, 3000), rng.beta(6, 3, 3000)]
    return y, p


def test_thresholds_are_monotonic_across_policies():
    y, p = _scores()
    t = choose_thresholds(y, p, POL)
    assert t["lenient"]["block"] >= t["balanced"]["block"] >= t["strict"]["block"]
    assert t["lenient"]["warn"] >= t["balanced"]["warn"] >= t["strict"]["warn"]
    for v in t.values():
        assert v["warn"] <= v["block"]


def test_block_threshold_respects_fpr_target():
    y, p = _scores()
    for name, cfg in POL.items():
        t = choose_thresholds(y, p, {name: cfg})[name]
        assert t["oof_block_fpr"] <= cfg["block_fpr"] + 1e-9
    assert threshold_at_fpr(y, p, 0.0) <= 1.0


def test_decide_maps_scores_to_actions():
    thr = {"warn": 0.4, "block": 0.8}
    assert [decide(s, thr) for s in (0.1, 0.4, 0.79, 0.8, 1.0)] == ["allow", "warn", "warn", "block", "block"]
