import numpy as np
import pytest

from suspackage.train import data as D
from suspackage.train.models import DEFAULT_BUILDER, MODEL_NAMES, FeatureBuilder, build_pipeline, feature_names


def test_builder_adds_densities_and_drops_x_py_kb(usable):
    cols = D.input_columns(usable)
    fb = FeatureBuilder(cols).fit(usable[cols])
    assert "x_py_kb" not in fb.feature_names_
    assert "d_c_subprocess" in fb.feature_names_
    out = fb.transform(usable[cols])
    i = fb.feature_names_.index("d_c_subprocess")
    expected = usable["c_subprocess"] / (usable["x_py_kb"] + 1)
    assert np.allclose(out[:, i], expected)


def test_builder_ablation_columns_are_removed(usable):
    cols = D.input_columns(usable)
    fb = FeatureBuilder(cols, ["meta_"], ["sp_present", "is_typosquat"], True).fit(usable[cols])
    names = fb.feature_names_
    assert not any(n.startswith("meta_") for n in names)
    assert "sp_present" not in names and "is_typosquat" not in names
    raw = FeatureBuilder(cols, use_density=False).fit(usable[cols])
    assert not any(n.startswith("d_") for n in raw.feature_names_)


def test_builder_reports_missing_columns(usable):
    cols = D.input_columns(usable)
    fb = FeatureBuilder(cols).fit(usable[cols])
    with pytest.raises(KeyError):
        fb.transform(usable[cols].drop(columns=["c_http"]))


@pytest.mark.parametrize("kind", MODEL_NAMES)
def test_every_pipeline_fits_and_scores(usable, kind):
    cols = D.input_columns(usable)
    X, y = usable[cols].astype(float), usable.label.to_numpy()
    params = {"n_estimators": 20} if kind in ("random_forest", "xgboost") else {}
    pipe = build_pipeline(kind, cols, 0, params, DEFAULT_BUILDER).fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (len(y),) and ((p >= 0) & (p <= 1)).all()
    assert "x_py_kb" not in feature_names(pipe)
    if kind != "dummy_baseline":
        assert p[y == 1].mean() > p[y == 0].mean()
