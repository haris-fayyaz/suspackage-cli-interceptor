import numpy as np

from suspackage.train import data as D


def test_feature_selection_excludes_metadata_and_x_columns(usable):
    cols = D.feature_columns(usable)
    assert not set(cols) & set(D.META_COLS)
    assert not any(c.startswith("x_") for c in cols)
    assert len(cols) == 32
    assert D.input_columns(usable)[-1] == "x_py_kb"
    assert "x_nearest_popular" not in D.input_columns(usable)


def test_usable_rows_drops_failures(raw_features, usable):
    assert (raw_features.ok == 0).sum() == 5
    assert (usable.ok == 1).all() and len(usable) == len(raw_features) - 5


def test_groups_merge_families_and_identical_vectors(usable):
    feats = D.feature_columns(usable)
    g = D.build_split_groups(usable, feats)
    # every family sits in one group
    assert usable.groupby("family").apply(lambda d: len(set(g[d.index]))).max() == 1
    # the 30 campaign rows share one vector across 30 different families, so they share one group
    camp = usable.index[usable.name.str.startswith("camp")]
    assert len(set(g[camp])) == 1
    # identical vectors never straddle two groups
    h = D.feature_vector_hash(usable, feats)
    for v in np.unique(h):
        assert len(set(g[h == v])) == 1


def test_group_stats_shape(usable):
    s = D.group_stats(D.build_split_groups(usable, D.feature_columns(usable)))
    assert s["max_size"] >= 30 and len(s["top20_sizes"]) <= 20


def test_data_version_reads_dvc_pointer(tmp_path):
    p = tmp_path / "features.dvc"
    p.write_text("outs:\n- md5: cc0edd43212870d6f6d6a360b27da8b5.dir\n  size: 1\n")
    assert D.data_version(p) == "cc0edd43212870d6f6d6a360b27da8b5.dir"
    assert D.data_version(tmp_path / "nope.dvc").startswith("unknown")
