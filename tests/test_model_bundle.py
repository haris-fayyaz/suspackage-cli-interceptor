import logging

import numpy as np
import pytest

from suspackage.model_bundle import ModelBundle
from suspackage.train import data as D
from suspackage.train.models import DEFAULT_BUILDER, build_pipeline

THR = {"balanced": {"warn": 0.3, "block": 0.7}}


@pytest.fixture(scope="module")
def bundle(usable):
    cols = D.input_columns(usable)
    pipe = build_pipeline("random_forest", cols, 0, {"n_estimators": 20}, DEFAULT_BUILDER)
    pipe.fit(usable[cols].astype(float), usable.label.to_numpy())
    return ModelBundle(pipeline=pipe, feature_columns=cols, thresholds=THR, data_hash="abc")


def test_round_trip_gives_identical_scores(bundle, usable, tmp_path):
    bundle.save(tmp_path / "m.joblib")
    loaded = ModelBundle.load(tmp_path / "m.joblib")
    assert loaded.feature_columns == bundle.feature_columns and loaded.data_hash == "abc"
    row = usable.iloc[0].to_dict()
    assert loaded.predict_proba(row) == bundle.predict_proba(row)
    assert np.allclose(loaded.predict_frame(usable.iloc[:50]), bundle.predict_frame(usable.iloc[:50]))


def test_missing_features_default_to_zero_with_warning(bundle, usable, caplog):
    row = usable.iloc[0].to_dict()
    row.pop("c_http")
    with caplog.at_level(logging.WARNING):
        score = bundle.predict_proba(row)
    assert "c_http" in caplog.text and 0.0 <= score <= 1.0
    assert score == bundle.predict_proba({**row, "c_http": 0.0})


def test_extra_keys_are_ignored(bundle, usable):
    row = usable.iloc[1].to_dict()
    assert bundle.predict_proba({**row, "surprise": 99, "x_nearest_popular": "numpy"}) == bundle.predict_proba(row)


def test_decide_and_policy_errors(bundle):
    assert bundle.decide(0.1) == "allow" and bundle.decide(0.5) == "warn" and bundle.decide(0.9) == "block"
    with pytest.raises(ValueError):
        bundle.decide(0.5, "nope")


def test_schema_mismatch_is_rejected(bundle, tmp_path):
    bundle.schema_version = 99
    bundle.save(tmp_path / "bad.joblib")
    bundle.schema_version = 1
    with pytest.raises(ValueError):
        ModelBundle.load(tmp_path / "bad.joblib")
