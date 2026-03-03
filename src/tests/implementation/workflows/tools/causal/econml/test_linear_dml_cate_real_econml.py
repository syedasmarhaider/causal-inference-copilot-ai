from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from python.implementation.workflows.tools.causal.causal_command import (
    CATECommand,
    CATEInputs,
    CATESuccess,
    CommandFailure,
    FitCommand,
    FitInputs,
    FitSuccess,
)
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.econml.dml.linear_dml import LinearDMLCausalModel

from tests.implementation.workflows.conftest import InMemoryDataRepo, InMemoryModelsRepo


# -----------------------------------------------------------------------------
# sklearn version-safe OneHotEncoder
# -----------------------------------------------------------------------------
def _make_ohe(*, dense: bool, categories: list[list[str]] | None = None) -> OneHotEncoder:
    kwargs = dict(handle_unknown="ignore")
    if categories is not None:
        kwargs["categories"] = categories
    try:
        kwargs["sparse_output"] = not dense  # sklearn >= 1.2
        return OneHotEncoder(**kwargs)
    except TypeError:
        kwargs["sparse"] = not dense  # sklearn < 1.2
        return OneHotEncoder(**kwargs)


# -----------------------------------------------------------------------------
# DGPs
# -----------------------------------------------------------------------------
class CateDGP:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def binary_t_heterogeneous_effect(self, n: int = 900) -> pd.DataFrame:
        """
        True CATE:
          tau(x) = 1.0  if x_num <= 0
                 = 2.0  if x_num > 0
        """
        rng = self.rng
        x_num = rng.normal(size=n)
        w_num = rng.normal(size=n)

        logits = 0.4 * x_num + 0.4 * w_num
        p = 1.0 / (1.0 + np.exp(-logits))
        t = rng.binomial(1, p, size=n).astype(float)

        tau = 1.0 + 1.0 * (x_num > 0).astype(float)
        y = 0.5 + tau * t + 0.3 * x_num + 0.6 * w_num + rng.normal(scale=1.0, size=n)

        return pd.DataFrame({"y": y, "t": t, "x_num": x_num, "w_num": w_num})

    def binary_t_string_category_effect(self, n: int = 1200) -> pd.DataFrame:
        """
        True CATE depends on x_cat:
          A -> 0.8
          B -> 1.6
          C -> 2.4
        """
        rng = self.rng
        x_num = rng.normal(size=n)
        w_num = rng.normal(size=n)
        x_cat = rng.choice(["A", "B", "C"], size=n, p=[0.34, 0.33, 0.33]).astype(object)

        tau = np.where(x_cat == "A", 0.8, np.where(x_cat == "B", 1.6, 2.4))

        logits = 0.25 * x_num + 0.35 * w_num + np.where(x_cat == "C", 0.15, 0.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        t = rng.binomial(1, p, size=n).astype(float)

        y = 0.1 + tau * t + 0.3 * x_num + 0.7 * w_num + rng.normal(scale=1.0, size=n)
        return pd.DataFrame({"y": y, "t": t, "x_num": x_num, "x_cat": x_cat, "w_num": w_num})

    def categorical_t_continuous_y(self, n: int = 1400) -> pd.DataFrame:
        """
        Treatment levels: 0,1,2 (baseline=0).
        Effects: 0->1 ~ +1.0, 0->2 ~ +2.0
        """
        rng = self.rng
        x_num = rng.normal(size=n)
        w_num = rng.normal(size=n)

        # multinomial-ish via softmax scores
        s1 = 0.25 * x_num + 0.20 * w_num
        s2 = -0.10 * x_num + 0.30 * w_num
        scores = np.vstack([np.zeros(n), s1, s2]).T
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        t = np.array([rng.choice([0.0, 1.0, 2.0], p=p) for p in probs], dtype=float)

        y = (
            0.5
            + 1.0 * (t == 1.0).astype(float)
            + 2.0 * (t == 2.0).astype(float)
            + 0.4 * x_num
            + 0.6 * w_num
            + rng.normal(scale=1.0, size=n)
        )
        return pd.DataFrame({"y": y, "t": t, "x_num": x_num, "w_num": w_num})


# -----------------------------------------------------------------------------
# Specs
# -----------------------------------------------------------------------------
def spec_binary_xnum_wnum() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "Y": {"kind": "continuous", "column": "y"},
            "T": {"kind": "binary", "column": "t", "treated_values": [1.0], "control_values": [0.0]},
            "X": ["x_num"],
            "W": ["w_num"],
            "Z": [],
        }
    )


def spec_binary_xnum_xcat_wnum() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "Y": {"kind": "continuous", "column": "y"},
            "T": {"kind": "binary", "column": "t", "treated_values": [1.0], "control_values": [0.0]},
            "X": ["x_num", "x_cat"],
            "W": ["w_num"],
            "Z": [],
        }
    )


def spec_categorical_t_xnum_wnum() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "Y": {"kind": "continuous", "column": "y"},
            "T": {"kind": "categorical", "column": "t", "levels": [0.0, 1.0, 2.0], "baseline": 0.0},
            "X": ["x_num"],
            "W": ["w_num"],
            "Z": [],
        }
    )


# -----------------------------------------------------------------------------
# Preprocessors
# NOTE: pre_XW must be built for RAW concatenated [X, W] layout.
# -----------------------------------------------------------------------------
def pre_numeric_x_w(*, x_dim: int, w_dim: int) -> tuple[ColumnTransformer, ColumnTransformer]:
    # pre_X sees raw X only
    pre_X = ColumnTransformer([("x", "passthrough", list(range(x_dim)))], remainder="drop", sparse_threshold=1.0)
    # pre_XW sees raw [X, W]
    pre_XW = ColumnTransformer([("xw", "passthrough", list(range(x_dim + w_dim)))], remainder="drop", sparse_threshold=1.0)
    return pre_X, pre_XW


def pre_xnum_xcat_w(*, x_cat_levels: list[str]) -> tuple[ColumnTransformer, ColumnTransformer]:
    """
    raw X: [x_num(0), x_cat(1)]
    raw XW: [x_num(0), x_cat(1), w_num(2)]
    """
    pre_X = ColumnTransformer(
        transformers=[
            ("x_num", StandardScaler(), [0]),
            ("x_cat", _make_ohe(dense=True, categories=[x_cat_levels]), [1]),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    pre_XW = ColumnTransformer(
        transformers=[
            ("x_num", StandardScaler(), [0]),
            ("x_cat", _make_ohe(dense=False, categories=[x_cat_levels]), [1]),
            ("w_num", StandardScaler(), [2]),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # force sparse
    )
    return pre_X, pre_XW


# -----------------------------------------------------------------------------
# SUT fixture (real EconML; shrink heavy defaults)
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


def _fit_first(*, model, data_repo, user_id, conv_id, df: pd.DataFrame, spec: CausalSpec, pre_X, pre_XW) -> tuple[UUID, UUID]:
    dataset_id = uuid4()
    fit_id = uuid4()
    data_repo.save_csv_data(user_id, conv_id, dataset_id, df)

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
# CATE tests
# -----------------------------------------------------------------------------
def test_cate_binary_heterogeneous_effect_separates_groups(sut):
    """
    Strong sanity check: predicted CATE should be larger for x_num > 0 than x_num <= 0.
    """
    model, data_repo, _models_repo, user_id, conv_id = sut

    df = CateDGP(seed=0).binary_t_heterogeneous_effect(n=900)
    spec = spec_binary_xnum_wnum()
    pre_X, pre_XW = pre_numeric_x_w(x_dim=1, w_dim=1)

    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec, pre_X=pre_X, pre_XW=pre_XW)

    x_rows = df[spec.X].copy()
    cate_cmd = CATECommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=uuid4(),
        protocol_specs=spec,
        fitted_model_id=fit_id,
        inputs=CATEInputs(x_rows=x_rows, alpha=0.1),
    )

    res = model.execute(user_id=user_id, conversation_id=conv_id, command=cate_cmd)
    assert isinstance(res, CATESuccess), f"Expected CATESuccess, got {type(res)}: {getattr(res, 'error', None)}"
    assert len(res.effects) == 1

    cate = np.asarray(res.effects[0]["cate"], dtype=float).reshape(-1)
    assert cate.shape[0] == x_rows.shape[0]
    assert np.isfinite(cate).all()

    x = x_rows["x_num"].to_numpy()
    hi = float(np.mean(cate[x > 0]))
    lo = float(np.mean(cate[x <= 0]))

    # True gap ~ 1.0. Keep loose; we just want monotonic separation.
    assert hi > lo + 0.1


def test_cate_binary_with_string_category_runs_and_orders_group_means(sut):
    """
    End-to-end: string x_cat in X + OHE in pre_X, and CATE query runs on raw x_rows.
    """
    model, data_repo, _models_repo, user_id, conv_id = sut

    df = CateDGP(seed=1).binary_t_string_category_effect(n=1200)
    spec = spec_binary_xnum_xcat_wnum()
    pre_X, pre_XW = pre_xnum_xcat_w(x_cat_levels=["A", "B", "C"])

    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec, pre_X=pre_X, pre_XW=pre_XW)

    x_rows = df[spec.X].copy()
    cate_cmd = CATECommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=uuid4(),
        protocol_specs=spec,
        fitted_model_id=fit_id,
        inputs=CATEInputs(x_rows=x_rows, alpha=0.1),
    )
    res = model.execute(user_id=user_id, conversation_id=conv_id, command=cate_cmd)
    assert isinstance(res, CATESuccess), f"Expected CATESuccess, got {type(res)}: {getattr(res, 'error', None)}"

    cate = np.asarray(res.effects[0]["cate"], dtype=float).reshape(-1)
    assert np.isfinite(cate).all()

    # group means should follow A < B < C in expectation (loose)
    mA = float(np.mean(cate[x_rows["x_cat"] == "A"]))
    mB = float(np.mean(cate[x_rows["x_cat"] == "B"]))
    mC = float(np.mean(cate[x_rows["x_cat"] == "C"]))
    assert mB > mA - 0.2
    assert mC > mB - 0.2


def test_cate_categorical_treatment_produces_multiple_contrasts(sut):
    model, data_repo, _models_repo, user_id, conv_id = sut

    df = CateDGP(seed=2).categorical_t_continuous_y(n=1400)
    spec = spec_categorical_t_xnum_wnum()
    pre_X, pre_XW = pre_numeric_x_w(x_dim=1, w_dim=1)

    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec, pre_X=pre_X, pre_XW=pre_XW)

    x_rows = df[spec.X].copy()
    cate_cmd = CATECommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=uuid4(),
        protocol_specs=spec,
        fitted_model_id=fit_id,
        inputs=CATEInputs(x_rows=x_rows, alpha=0.1),
    )
    res = model.execute(user_id=user_id, conversation_id=conv_id, command=cate_cmd)
    assert isinstance(res, CATESuccess), f"Expected CATESuccess, got {type(res)}: {getattr(res, 'error', None)}"

    # baseline 0 vs {1,2} => 2 contrasts expected
    assert len(res.effects) == 2
    for item in res.effects:
        assert "for_treatment" in item and "cate" in item
        c = np.asarray(item["cate"], dtype=float).reshape(-1)
        assert c.shape[0] == x_rows.shape[0]
        assert np.isfinite(c).all()


def test_cate_model_not_found_returns_failure(sut):
    model, data_repo, _models_repo, user_id, conv_id = sut
    df = CateDGP(seed=3).binary_t_heterogeneous_effect(n=300)
    spec = spec_binary_xnum_wnum()
    pre_X, pre_XW = pre_numeric_x_w(x_dim=1, w_dim=1)

    dataset_id = uuid4()
    data_repo.save_csv_data(user_id, conv_id, dataset_id, df)

    cate_cmd = CATECommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=uuid4(),
        protocol_specs=spec,
        fitted_model_id=uuid4(),  # missing
        inputs=CATEInputs(x_rows=df[spec.X].copy(), alpha=0.1),
    )

    res = model.execute(user_id=user_id, conversation_id=conv_id, command=cate_cmd)
    assert isinstance(res, CommandFailure)
    assert res.error.code == "MODEL_NOT_FOUND"


def test_cate_binary_multiple_treated_values_is_rejected(sut):
    """
    Your adapter explicitly rejects binary CATE when treated/control lists have size != 1.
    """
    model, data_repo, _models_repo, user_id, conv_id = sut
    df = CateDGP(seed=4).binary_t_heterogeneous_effect(n=400)

    # invalid spec: treated_values has 2 items
    spec = CausalSpec.model_validate(
        {
            "Y": {"kind": "continuous", "column": "y"},
            "T": {"kind": "binary", "column": "t", "treated_values": [1.0, 2.0], "control_values": [0.0]},
            "X": ["x_num"],
            "W": ["w_num"],
            "Z": [],
        }
    )
    pre_X, pre_XW = pre_numeric_x_w(x_dim=1, w_dim=1)
    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec, pre_X=pre_X, pre_XW=pre_XW)

    cate_cmd = CATECommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=uuid4(),
        protocol_specs=spec,
        fitted_model_id=fit_id,
        inputs=CATEInputs(x_rows=df[spec.X].copy(), alpha=0.1),
    )

    res = model.execute(user_id=user_id, conversation_id=conv_id, command=cate_cmd)
    assert isinstance(res, CommandFailure)
    assert res.error.code == "OPTIONS_INVALID"
    assert "exactly one control_value" in res.error.message


def test_cate_rejects_x_rows_with_wrong_columns_or_order(sut):
    """
    Enforces your x_rows schema check.
    (This assumes you fixed the bug in _cate: df -> X_query for the checker call.)
    """
    model, data_repo, _models_repo, user_id, conv_id = sut

    df = CateDGP(seed=5).binary_t_heterogeneous_effect(n=450)
    spec = spec_binary_xnum_wnum()
    pre_X, pre_XW = pre_numeric_x_w(x_dim=1, w_dim=1)
    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec, pre_X=pre_X, pre_XW=pre_XW)

    # wrong: missing x_num
    bad_x = pd.DataFrame({"wrong": df["x_num"].to_numpy()})
    cate_cmd = CATECommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=uuid4(),
        protocol_specs=spec,
        fitted_model_id=fit_id,
        inputs=CATEInputs(x_rows=bad_x, alpha=0.1),
    )
    res = model.execute(user_id=user_id, conversation_id=conv_id, command=cate_cmd)
    assert isinstance(res, CommandFailure)
    # Depending on your checker implementation it may be OPTIONS_INVALID or ESTIMATOR_ERROR.
    assert res.error.code in ("OPTIONS_INVALID", "ESTIMATOR_ERROR")


def test_cate_result_is_dataclass_serializable(sut):
    """
    Transport-level check: dataclass asdict should not crash.
    (Not JSON-serializable necessarily, but good baseline.)
    """
    model, data_repo, _models_repo, user_id, conv_id = sut

    df = CateDGP(seed=7).binary_t_heterogeneous_effect(n=500)
    spec = spec_binary_xnum_wnum()
    pre_X, pre_XW = pre_numeric_x_w(x_dim=1, w_dim=1)
    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec, pre_X=pre_X, pre_XW=pre_XW)

    cate_cmd = CATECommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=uuid4(),
        protocol_specs=spec,
        fitted_model_id=fit_id,
        inputs=CATEInputs(x_rows=df[spec.X].copy(), alpha=0.1),
    )
    res = model.execute(user_id=user_id, conversation_id=conv_id, command=cate_cmd)
    assert isinstance(res, CATESuccess)
    asdict(res)