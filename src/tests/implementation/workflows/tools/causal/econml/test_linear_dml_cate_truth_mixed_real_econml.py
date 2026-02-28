from __future__ import annotations

import os
import warnings
from dataclasses import asdict
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

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
# Make compute deterministic-ish and avoid oversubscription in CI
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _limit_threads():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


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


def _to_epoch_seconds_2d(X):
    s = pd.to_datetime(pd.Series(np.asarray(X).reshape(-1)), utc=True, errors="raise")
    dt64 = s.to_numpy(dtype="datetime64[ns]")
    out = (dt64.astype("int64") / 1e9).reshape(-1, 1)
    return out


# -----------------------------------------------------------------------------
# DGPs with explicit ground-truth CATE tau(x)
# -----------------------------------------------------------------------------
class CateTruthDGP:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def tau_linear_in_xnum(self, x_num: np.ndarray) -> np.ndarray:
        # strong, smooth heterogeneity that LinearDML can learn with a linear featurizer
        return 1.0 + 1.2 * x_num

    def tau_by_category(self, x_cat: np.ndarray) -> np.ndarray:
        return np.where(x_cat == "A", 0.8, np.where(x_cat == "B", 1.6, 2.4)).astype(float)

    def tau_mixed(self, x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
        # additive heterogeneity: numeric + category shift
        return (1.0 + 1.0 * x_num + 0.8 * (x_cat == "C").astype(float)).astype(float)

    def make_linear_xnum(self, n: int = 1400, noise: float = 0.35) -> tuple[pd.DataFrame, np.ndarray]:
        rng = self.rng
        x_num = rng.normal(size=n)
        w_num = rng.normal(size=n)

        tau = self.tau_linear_in_xnum(x_num)

        logits = 0.35 * x_num + 0.35 * w_num
        p = 1.0 / (1.0 + np.exp(-logits))
        t = rng.binomial(1, p, size=n).astype(float)

        y = 0.2 + tau * t + 0.3 * x_num + 0.6 * w_num + rng.normal(scale=noise, size=n)

        df = pd.DataFrame({"y": y, "t": t, "x_num": x_num, "w_num": w_num})
        return df, tau

    def make_string_category_x(self, n: int = 1800, noise: float = 0.35) -> tuple[pd.DataFrame, np.ndarray]:
        rng = self.rng
        x_num = rng.normal(size=n)
        w_num = rng.normal(size=n)
        x_cat = rng.choice(["A", "B", "C"], size=n, p=[0.34, 0.33, 0.33]).astype(object)

        tau = self.tau_by_category(x_cat)

        logits = 0.25 * x_num + 0.35 * w_num + 0.2 * (x_cat == "C").astype(float)
        p = 1.0 / (1.0 + np.exp(-logits))
        t = rng.binomial(1, p, size=n).astype(float)

        y = 0.1 + tau * t + 0.25 * x_num + 0.7 * w_num + rng.normal(scale=noise, size=n)

        df = pd.DataFrame({"y": y, "t": t, "x_num": x_num, "x_cat": x_cat, "w_num": w_num})
        return df, tau

    def make_mixed_x(self, n: int = 2200, noise: float = 0.35) -> tuple[pd.DataFrame, np.ndarray]:
        rng = self.rng
        x_num = rng.normal(size=n)
        x_cat = rng.choice(["A", "B", "C"], size=n, p=[0.34, 0.33, 0.33]).astype(object)
        x_ord = rng.choice(["low", "med", "high"], size=n, p=[0.33, 0.34, 0.33]).astype(object)

        base = pd.Timestamp("2025-01-01", tz="UTC")
        x_dt = (base + pd.to_timedelta(rng.integers(0, 60 * 60 * 24 * 30, size=n), unit="s")).astype(object)

        w_num = rng.normal(size=n)

        tau = self.tau_mixed(x_num, x_cat)

        logits = 0.25 * x_num + 0.35 * w_num + 0.15 * (x_cat == "C").astype(float) + 0.10 * (x_ord == "high").astype(float)
        p = 1.0 / (1.0 + np.exp(-logits))
        t = rng.binomial(1, p, size=n).astype(float)

        y = (
            0.3
            + tau * t
            + 0.25 * x_num
            + 0.65 * w_num
            + 0.10 * (x_ord == "high").astype(float)
            + rng.normal(scale=noise, size=n)
        )

        df = pd.DataFrame({"y": y, "t": t, "x_num": x_num, "x_cat": x_cat, "x_ord": x_ord, "x_dt": x_dt, "w_num": w_num})
        return df, tau


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


def spec_binary_mixed_x_wnum() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "Y": {"kind": "continuous", "column": "y"},
            "T": {"kind": "binary", "column": "t", "treated_values": [1.0], "control_values": [0.0]},
            "X": ["x_num", "x_cat", "x_ord", "x_dt"],
            "W": ["w_num"],
            "Z": [],
        }
    )


# -----------------------------------------------------------------------------
# Preprocessors
# IMPORTANT: pre_XW must be built for RAW concatenated [X, W] layout.
# -----------------------------------------------------------------------------
def pre_numeric_x_w(*, x_dim: int, w_dim: int) -> tuple[ColumnTransformer, ColumnTransformer]:
    pre_X = ColumnTransformer([("x", "passthrough", list(range(x_dim)))], remainder="drop", sparse_threshold=1.0)
    pre_XW = ColumnTransformer([("xw", "passthrough", list(range(x_dim + w_dim)))], remainder="drop", sparse_threshold=1.0)
    return pre_X, pre_XW


def pre_xnum_xcat_w() -> tuple[ColumnTransformer, ColumnTransformer]:
    # raw X: [x_num(0), x_cat(1)]
    # raw XW: [x_num(0), x_cat(1), w_num(2)]
    pre_X = ColumnTransformer(
        transformers=[
            ("x_num", StandardScaler(), [0]),
            ("x_cat", _make_ohe(dense=True, categories=[["A", "B", "C"]]), [1]),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    pre_XW = ColumnTransformer(
        transformers=[
            ("x_num", StandardScaler(), [0]),
            ("x_cat", _make_ohe(dense=False, categories=[["A", "B", "C"]]), [1]),
            ("w_num", StandardScaler(), [2]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre_X, pre_XW


def pre_mixed_x_w() -> tuple[ColumnTransformer, ColumnTransformer]:
    """
    raw X: [x_num(0), x_cat(1), x_ord(2), x_dt(3)]
    raw XW: [x_num(0), x_cat(1), x_ord(2), x_dt(3), w_num(4)]
    """
    pre_X = ColumnTransformer(
        transformers=[
            ("x_num", StandardScaler(), [0]),
            ("x_cat", _make_ohe(dense=True, categories=[["A", "B", "C"]]), [1]),
            ("x_ord", OrdinalEncoder(categories=[["low", "med", "high"]]), [2]),
            ("x_dt", Pipeline([("epoch", FunctionTransformer(_to_epoch_seconds_2d, validate=False)), ("sc", StandardScaler())]), [3]),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    pre_XW = ColumnTransformer(
        transformers=[
            ("x_num", StandardScaler(), [0]),
            ("x_cat", _make_ohe(dense=False, categories=[["A", "B", "C"]]), [1]),
            ("x_ord", OrdinalEncoder(categories=[["low", "med", "high"]]), [2]),
            ("x_dt", FunctionTransformer(_to_epoch_seconds_2d, validate=False), [3]),
            ("w_num", StandardScaler(), [4]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre_X, pre_XW


# -----------------------------------------------------------------------------
# SUT fixture (real EconML; shrink heavy defaults; force random_state stability)
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

    def _rs(kwargs):
        if kwargs.get("random_state", None) is None:
            kwargs["random_state"] = 0
        return kwargs

    def ETC_small(*args, **kwargs):
        kwargs = _rs(kwargs)
        kwargs["n_estimators"] = 25
        kwargs["min_samples_leaf"] = 5
        kwargs["n_jobs"] = 1
        return _ETC(*args, **kwargs)

    def ETR_small(*args, **kwargs):
        kwargs = _rs(kwargs)
        kwargs["n_estimators"] = 25
        kwargs["min_samples_leaf"] = 5
        kwargs["n_jobs"] = 1
        return _ETR(*args, **kwargs)

    def RFC_small(*args, **kwargs):
        kwargs = _rs(kwargs)
        kwargs["n_estimators"] = 25
        kwargs["min_samples_leaf"] = 5
        kwargs["n_jobs"] = 1
        return _RFC(*args, **kwargs)

    def RFR_small(*args, **kwargs):
        kwargs = _rs(kwargs)
        kwargs["n_estimators"] = 25
        kwargs["min_samples_leaf"] = 5
        kwargs["n_jobs"] = 1
        return _RFR(*args, **kwargs)

    def HGBC_small(*args, **kwargs):
        kwargs = _rs(kwargs)
        kwargs["max_iter"] = 60
        kwargs["learning_rate"] = 0.05
        return _HGBC(*args, **kwargs)

    def HGBR_small(*args, **kwargs):
        kwargs = _rs(kwargs)
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

    # guard against "warnings-as-errors" CI configs
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        fit_res = model.execute(user_id=user_id, conversation_id=conv_id, command=fit_cmd)

    assert isinstance(fit_res, FitSuccess), f"FIT must succeed first. got={type(fit_res)} err={getattr(fit_res, 'error', None)}"
    return dataset_id, fit_id


def _run_cate(*, model, user_id, conv_id, dataset_id, fit_id, spec: CausalSpec, x_rows: pd.DataFrame) -> CATESuccess:
    cate_cmd = CATECommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=uuid4(),
        protocol_specs=spec,
        fitted_model_id=fit_id,
        inputs=CATEInputs(x_rows=x_rows, alpha=0.1),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        res = model.execute(user_id=user_id, conversation_id=conv_id, command=cate_cmd)
    assert isinstance(res, CATESuccess), f"Expected CATESuccess, got {type(res)}: {getattr(res, 'error', None)}"
    return res


# -----------------------------------------------------------------------------
# Truth-check tests
# -----------------------------------------------------------------------------
def test_cate_tracks_truth_when_tau_is_linear_in_xnum(sut):
    """
    Truth check: tau(x)=1+1.2*x_num.
    We require positive correlation between predicted CATE and x_num and with true tau.
    """
    model, data_repo, _models_repo, user_id, conv_id = sut

    df, tau = CateTruthDGP(seed=0).make_linear_xnum(n=1400, noise=0.30)
    spec = spec_binary_xnum_wnum()
    pre_X, pre_XW = pre_numeric_x_w(x_dim=1, w_dim=1)

    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec, pre_X=pre_X, pre_XW=pre_XW)

    x_rows = df[spec.X].copy()
    res = _run_cate(model=model, user_id=user_id, conv_id=conv_id, dataset_id=dataset_id, fit_id=fit_id, spec=spec, x_rows=x_rows)

    cate = np.asarray(res.effects[0]["cate"], dtype=float).reshape(-1)
    assert np.isfinite(cate).all()

    x = x_rows["x_num"].to_numpy()
    corr_x = float(np.corrcoef(cate, x)[0, 1])
    corr_tau = float(np.corrcoef(cate, tau)[0, 1])

    assert corr_x > 0.15
    assert corr_tau > 0.15


def test_cate_orders_group_means_when_tau_depends_on_string_category(sut):
    """
    Truth check: tau(A)<tau(B)<tau(C).
    We verify estimated group mean ordering is consistent (loose margins).
    """
    model, data_repo, _models_repo, user_id, conv_id = sut

    df, tau = CateTruthDGP(seed=1).make_string_category_x(n=1800, noise=0.30)
    spec = spec_binary_xnum_xcat_wnum()
    pre_X, pre_XW = pre_xnum_xcat_w()

    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec, pre_X=pre_X, pre_XW=pre_XW)
    x_rows = df[spec.X].copy()

    res = _run_cate(model=model, user_id=user_id, conv_id=conv_id, dataset_id=dataset_id, fit_id=fit_id, spec=spec, x_rows=x_rows)
    cate = np.asarray(res.effects[0]["cate"], dtype=float).reshape(-1)
    assert np.isfinite(cate).all()

    mA = float(np.mean(cate[x_rows["x_cat"] == "A"]))
    mB = float(np.mean(cate[x_rows["x_cat"] == "B"]))
    mC = float(np.mean(cate[x_rows["x_cat"] == "C"]))

    # true: 0.8 < 1.6 < 2.4
    assert mB > mA - 0.15
    assert mC > mB - 0.15


def test_cate_tracks_truth_on_mixed_x_numeric_cat_ordinal_datetime_encodings(sut):
    """
    Mixed data smoke + truth check.
    True tau depends on x_num and x_cat; other X cols exist and are encoded.
    We require positive correlation with true tau.
    """
    model, data_repo, _models_repo, user_id, conv_id = sut

    df, tau = CateTruthDGP(seed=2).make_mixed_x(n=2200, noise=0.30)
    spec = spec_binary_mixed_x_wnum()
    pre_X, pre_XW = pre_mixed_x_w()

    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec, pre_X=pre_X, pre_XW=pre_XW)
    x_rows = df[spec.X].copy()

    res = _run_cate(model=model, user_id=user_id, conv_id=conv_id, dataset_id=dataset_id, fit_id=fit_id, spec=spec, x_rows=x_rows)
    cate = np.asarray(res.effects[0]["cate"], dtype=float).reshape(-1)
    assert np.isfinite(cate).all()

    corr_tau = float(np.corrcoef(cate, tau)[0, 1])
    assert corr_tau > 0.10


def test_cate_query_with_unseen_category_is_finite_when_handle_unknown_ignore(sut):
    """
    Query-time robustness: pass x_cat='D' never seen in training.
    With OneHotEncoder(handle_unknown='ignore'), this must not crash and must be finite.
    """
    model, data_repo, _models_repo, user_id, conv_id = sut

    df, _tau = CateTruthDGP(seed=3).make_string_category_x(n=1200, noise=0.35)
    spec = spec_binary_xnum_xcat_wnum()
    pre_X, pre_XW = pre_xnum_xcat_w()

    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec, pre_X=pre_X, pre_XW=pre_XW)

    x_rows = df[spec.X].copy()
    x_rows.loc[:50, "x_cat"] = "D"  # unseen

    res = _run_cate(model=model, user_id=user_id, conv_id=conv_id, dataset_id=dataset_id, fit_id=fit_id, spec=spec, x_rows=x_rows)
    cate = np.asarray(res.effects[0]["cate"], dtype=float).reshape(-1)
    assert np.isfinite(cate).all()


def test_cate_rejects_wrong_x_rows_schema_or_order(sut):
    """
    Edge case: x_rows wrong columns must fail.
    (Exact failure code depends on your validator; we accept OPTIONS_INVALID or ESTIMATOR_ERROR.)
    """
    model, data_repo, _models_repo, user_id, conv_id = sut

    df, _tau = CateTruthDGP(seed=4).make_linear_xnum(n=600, noise=0.35)
    spec = spec_binary_xnum_wnum()
    pre_X, pre_XW = pre_numeric_x_w(x_dim=1, w_dim=1)

    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec, pre_X=pre_X, pre_XW=pre_XW)

    bad_x = pd.DataFrame({"wrong": df["x_num"].to_numpy()})
    cate_cmd = CATECommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=uuid4(),
        protocol_specs=spec,
        fitted_model_id=fit_id,
        inputs=CATEInputs(x_rows=bad_x, alpha=0.1),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        res = model.execute(user_id=user_id, conversation_id=conv_id, command=cate_cmd)

    assert isinstance(res, CommandFailure)
    assert res.error.code in ("OPTIONS_INVALID", "ESTIMATOR_ERROR")


def test_cate_result_is_dataclass_serializable(sut):
    model, data_repo, _models_repo, user_id, conv_id = sut

    df, _tau = CateTruthDGP(seed=5).make_linear_xnum(n=700, noise=0.35)
    spec = spec_binary_xnum_wnum()
    pre_X, pre_XW = pre_numeric_x_w(x_dim=1, w_dim=1)

    dataset_id, fit_id = _fit_first(model=model, data_repo=data_repo, user_id=user_id, conv_id=conv_id, df=df, spec=spec, pre_X=pre_X, pre_XW=pre_XW)
    res = _run_cate(model=model, user_id=user_id, conv_id=conv_id, dataset_id=dataset_id, fit_id=fit_id, spec=spec, x_rows=df[spec.X].copy())

    asdict(res)