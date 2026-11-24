import numpy as np

from suspackage.train import data as D
from suspackage.train import splits as S


def _ctx(usable):
    feats = D.feature_columns(usable)
    return usable.label.to_numpy(), D.build_split_groups(usable, feats), usable.source.to_numpy()


def test_holdout_has_no_group_leak(usable):
    y, g, _ = _ctx(usable)
    dev, hold = S.make_holdout_split(y, g, 0.2, seed=1)
    assert not set(g[dev]) & set(g[hold])
    assert len(dev) + len(hold) == len(y) and 0.1 < len(hold) / len(y) < 0.35


def test_cv_folds_have_no_group_leak_and_cover_all_rows(usable):
    y, g, _ = _ctx(usable)
    folds = S.make_grouped_folds(y, g, 5, seed=1, cap=3)
    seen = np.concatenate([va for _, va in folds])
    assert sorted(seen) == list(range(len(y)))
    for tr, va in folds:
        assert not set(g[tr]) & set(g[va])
        assert np.bincount(g[tr]).max() <= 3


def test_cap_per_group_is_seeded_and_capped(usable):
    y, g, _ = _ctx(usable)
    idx = np.arange(len(y))
    a, b = S.limit_rows_per_group(idx, g, 2, 7), S.limit_rows_per_group(idx, g, 2, 7)
    assert (a == b).all() and np.bincount(g[a]).max() <= 2
    assert (S.limit_rows_per_group(idx, g, None, 7) == idx).all()


def test_cross_split_keeps_datadog_out_of_training(usable):
    y, g, src = _ctx(usable)
    sp = S.make_cross_dataset_split(src, y, g, seed=3)
    assert not (src[sp.train] == "datadog").any()
    assert (src[sp.test] == "datadog").sum() == (src == "datadog").sum()
    assert not set(g[sp.train]) & set(g[sp.test])
    assert (y[sp.test] == 0).any() and (y[sp.train] == 1).any()
