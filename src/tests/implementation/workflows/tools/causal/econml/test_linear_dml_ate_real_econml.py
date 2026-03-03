from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch
from sklearn.compose import ColumnTransformer

from python.implementation.workflows.tools.causal.causal_command import (
    ATECommand,
    ATEInputsModel,
    ATESuccess,
    CommandFailure,
    FitCommand,
    FitInputs,
    FitSuccess,
)
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.econml.dml.linear_dml import LinearDMLCausalModel

from tests.implementation.workflows.conftest import InMemoryDataRepo, InMemoryModelsRepo


# -----------------------------------------------------------------------------
# Synthetic DGPs (raw, stable)
# -----------------------------------------------------------------------------
class CausalDataGenerator:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def binary_t_continuous_y(self, n: int = 700) -> pd.DataFrame:
        rng = self.rng
        x1 = rng.normal(size=n)
        w1 = rng.normal(size=n)

        logits = 0.35 * x1 + 0.45 * w1
        p = 1.0 / (1.0 + np.exp(-logits))
        t = rng.binomial(1, p, size=n).astype(float)

        # true ATE approx +2.0
        y = 1.0 + 2.0 * t + 0.4 * x1 + 0.6 * w1 + rng.normal(scale=1.0, size=n)
        return pd.DataFrame({"y": y, "t": t, "x1": x1, "w1": w1})

    def categorical_t_continuous_y(self, n: int = 900) -> pd.DataFrame:
        rng = self.rng
        x1 = rng.normal(size=n)
        w1 = rng.normal(size=n)

        # 3-level treatment 0/1/2, depends on X/W
        logits1 = 0.2 * x1 + 0.2 * w1
        logits2 = -0.1 * x1 + 0.3 * w1
        scores = np.vstack([np.zeros(n), logits1, logits2]).T
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        t = np.array([rng.choice([0.0, 1.0, 2.0], p=p) for p in probs], dtype=float)

        # effects: 0->1 ~ +1, 0->2 ~ +2
        y = (
            0.5
            + 1.0 * (t == 1.0)
            + 2.0 * (t == 2.0)
            + 0.4 * x1
            + 0.6 * w1
            + rng.normal(scale=1.0, size=n)
        )
        return pd.DataFrame({"y": y, "t": t, "x1": x1, "w1": w1})


# -----------------------------------------------------------------------------
# Specs + preprocessors
# -----------------------------------------------------------------------------
def spec_binary_t_cont_y() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "Y": {"kind": "continuous", "column": "y"},
            "T": {"kind": "binary", "column": "t", "treated_values": [1.0], "control_values": [0.0]},
            "W": ["w1"],
            "X": ["x1"],
            "Z": [],
        }
    )


def spec_categorical_t_cont_y() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "Y": {"kind": "continuous", "column": "y"},
            "T": {"kind": "categorical", "column": "t", "levels": [0.0, 1.0, 2.0], "baseline": 0.0},
            "W": ["w1"],
            "X": ["x1"],
            "Z": [],
        }
    )


def preprocessors_x_and_xw(*, x_dim: int, w_dim: int) -> tuple[ColumnTransformer, ColumnTransformer]:
    pre_X = ColumnTransformer([("x", "passthrough", list(range(x_dim)))], remainder="drop", sparse_threshold=1.0)
    pre_XW = ColumnTransformer([("xw", "passthrough", list(range(x_dim + w_dim)))], remainder="drop", sparse_threshold=1.0)
    return pre_X, pre_XW


# -----------------------------------------------------------------------------
# SUT fixture: real EconML, but shrink heavy default models
# -----------------------------------------------------------------------------
@pytest.fixture()
def sut(tmp_path, monkeypatch: MonkeyPatch):
    import python.implementation.workflows.tools.causal.econml.dml.linear_dml as mod

    from sklearn.ensemble import (
        ExtraTreesClassifier as _ETC,
        ExtraTreesRegressor as _ETR,
        RandomForestClassifier as _RFC,
        RandomForestRegressor as _RFR,
        HistGradientBoostingClassifier as _HGBC,
        HistGradientBoostingRegressor as _HGBR,
    )

    def ETC_small(*args, **kwargs):
        kwargs["n_estimators"] = 25
        kwargs["min_samples_leaf"] = 5
        kwargs["n_jobs"] = 1
        return _ETC(*args, **kwargs)

    def ETR_small(*args, **kwargs):
        kwargs["n_estimators"] = 25
        kwargs["min_samples_leaf"] = 5
        kwargs["n_jobs"] = 1
        return _ETR(*args, **kwargs)

    def RFC_small(*args, **kwargs):
        kwargs["n_estimators"] = 25
        kwargs["min_samples_leaf"] = 5
        kwargs["n_jobs"] = 1
        return _RFC(*args, **kwargs)

    def RFR_small(*args, **kwargs):
        kwargs["n_estimators"] = 25
        kwargs["min_samples_leaf"] = 5
        kwargs["n_jobs"] = 1
        return _RFR(*args, **kwargs)

    def HGBC_small(*args, **kwargs):
        kwargs["max_iter"] = 60
        kwargs["learning_rate"] = 0.05
        return _HGBC(*args, **kwargs)

    def HGBR_small(*args, **kwargs):
        kwargs["max_iter"] = 60
        kwargs["learning_rate"] = 0.05
        return _HGBR(*args, **kwargs)

    monkeypatch.setattr(mod, "ExtraTreesClassifier", ETC_small)
    monkeypatch.setattr(mod, "ExtraTreesRegressor", ETR_small)
    monkeypatch.setattr(mod, "RandomForestClassifier", RFC_small)
    monkeypatch.setattr(mod, "RandomForestRegressor", RFR_small)
    monkeypatch.setattr(mod, "HistGradientBoostingClassifier", HGBC_small)
    monkeypatch.setattr(mod, "HistGradientBoostingRegressor", HGBR_small)

    data_repo = InMemoryDataRepo(base_dir=tmp_path)
    models_repo = InMemoryModelsRepo()
    model = LinearDMLCausalModel(data_repo=data_repo, models_repo=models_repo)

    user_id = uuid4()
    conv_id = uuid4()
    return model, data_repo, models_repo, user_id, conv_id


def _fit_first(*, model, data_repo, user_id, conv_id, df: pd.DataFrame, spec: CausalSpec) -> tuple[UUID, UUID]:
    dataset_id = uuid4()
    fit_id = uuid4()
    data_repo.save_csv_data(user_id, conv_id, dataset_id, df)

    pre_X, pre_XW = preprocessors_x_and_xw(x_dim=1, w_dim=1)

    fit_cmd = FitCommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=fit_id,
        protocol_specs=spec,
        inputs=FitInputs(pre_X=pre_X, pre_XW=pre_XW),
    )
    fit_res = model.execute(user_id=user_id, conversation_id=conv_id, command=fit_cmd)
    assert isinstance(fit_res, FitSuccess), f"FIT must succeed first. got={type(fit_res)} err={getattr(fit_res, 'error', None)}"
    return dataset_id, fit_id


# -----------------------------------------------------------------------------
# ATE tests
# -----------------------------------------------------------------------------
def test_ate_binary_treatment_returns_one_contrast_and_finite(sut):
    model, data_repo, _models_repo, user_id, conv_id = sut
    df = CausalDataGenerator(seed=0).binary_t_continuous_y(n=700)
    spec = spec_binary_t_cont_y()

    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec)

    ate_cmd = ATECommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=uuid4(),
        protocol_specs=spec,
        fitted_model_id=fit_id,
        inputs=ATEInputsModel(alpha=0.1),
    )

    res = model.execute(user_id=user_id, conversation_id=conv_id, command=ate_cmd)
    assert isinstance(res, ATESuccess), f"Expected ATESuccess, got {type(res)}: {getattr(res, 'error', None)}"
    assert res.fitted_model_id == fit_id
    assert len(res.ate) == 1

    item = res.ate[0]
    assert "for_treatment" in item and "ate" in item
    ate_val = np.asarray(item["ate"], dtype=float)
    assert np.isfinite(ate_val).all()
    assert float(np.mean(ate_val)) > 0.2


def test_ate_categorical_baseline_vs_all_returns_expected_contrasts(sut):
    model, data_repo, _models_repo, user_id, conv_id = sut
    df = CausalDataGenerator(seed=2).categorical_t_continuous_y(n=900)
    spec = spec_categorical_t_cont_y()

    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec)

    ate_cmd = ATECommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=uuid4(),
        protocol_specs=spec,
        fitted_model_id=fit_id,
        inputs=ATEInputsModel(alpha=0.1),
    )

    res = model.execute(user_id=user_id, conversation_id=conv_id, command=ate_cmd)
    assert isinstance(res, ATESuccess), f"Expected ATESuccess, got {type(res)}: {getattr(res, 'error', None)}"

    # baseline 0 vs {1,2} => 2 contrasts expected
    assert len(res.ate) == 2

    means = []
    for item in res.ate:
        m = float(np.mean(np.asarray(item["ate"], dtype=float)))
        assert np.isfinite(m)
        means.append(m)

    assert abs(means[0] - means[1]) > 1e-4


def test_ate_model_not_found_returns_failure(sut):
    model, data_repo, _models_repo, user_id, conv_id = sut
    df = CausalDataGenerator(seed=4).binary_t_continuous_y(n=300)
    spec = spec_binary_t_cont_y()

    dataset_id = uuid4()
    data_repo.save_csv_data(user_id, conv_id, dataset_id, df)

    ate_cmd = ATECommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=uuid4(),
        protocol_specs=spec,
        fitted_model_id=uuid4(),  # missing
        inputs=ATEInputsModel(alpha=0.1),
    )

    res = model.execute(user_id=user_id, conversation_id=conv_id, command=ate_cmd)
    assert isinstance(res, CommandFailure)


def test_ate_is_serializable(sut):
    model, data_repo, _models_repo, user_id, conv_id = sut
    df = CausalDataGenerator(seed=6).binary_t_continuous_y(n=450)
    spec = spec_binary_t_cont_y()

    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec)

    ate_cmd = ATECommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=uuid4(),
        protocol_specs=spec,
        fitted_model_id=fit_id,
        inputs=ATEInputsModel(alpha=0.1),
    )

    res = model.execute(user_id=user_id, conversation_id=conv_id, command=ate_cmd)
    assert isinstance(res, ATESuccess)
    asdict(res)