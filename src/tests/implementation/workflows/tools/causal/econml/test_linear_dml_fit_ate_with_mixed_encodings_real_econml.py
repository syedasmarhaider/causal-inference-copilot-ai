from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)

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
    # X is (n, 1) array-like
    s = pd.to_datetime(pd.Series(np.asarray(X).reshape(-1)), utc=True, errors="raise")
    dt64 = s.to_numpy(dtype="datetime64[ns]")
    out = (dt64.astype("int64") / 1e9).reshape(-1, 1)
    return out


def _to_token_lists(X):
    vals = np.asarray(X).reshape(-1)
    return [[str(v)] for v in vals]


# -----------------------------------------------------------------------------
# DGP 1: mixed string categorical + ordinal + datetime + hashing
# -----------------------------------------------------------------------------
class MixedEncodingDGP:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def make(self, n: int = 1000) -> pd.DataFrame:
        rng = self.rng

        # X (4 cols)
        x_num = rng.normal(size=n)
        x_cat = rng.choice(["A", "B", "C"], size=n, p=[0.34, 0.33, 0.33]).astype(object)
        x_ord = rng.choice(["low", "med", "high"], size=n, p=[0.33, 0.34, 0.33]).astype(object)
        base = pd.Timestamp("2025-01-01", tz="UTC")
        x_dt = (base + pd.to_timedelta(rng.integers(0, 60 * 60 * 24 * 30, size=n), unit="s")).astype(object)

        # W (3 cols)
        w_num = rng.normal(size=n)
        w_bin = rng.choice(["no", "yes"], size=n, p=[0.55, 0.45]).astype(object)
        w_tok = rng.choice([f"tok{i}" for i in range(15)], size=n).astype(object)

        # treatment propensity depends on X/W incl categories
        x_cat_eff = np.where(x_cat == "A", -0.2, np.where(x_cat == "B", 0.0, 0.2))
        x_ord_eff = np.where(x_ord == "low", -0.15, np.where(x_ord == "med", 0.0, 0.15))
        w_bin_eff = np.where(w_bin == "yes", 0.15, -0.05)

        logits = 0.35 * x_num + 0.35 * w_num + x_cat_eff + x_ord_eff + w_bin_eff
        p = 1.0 / (1.0 + np.exp(-logits))
        t = rng.binomial(1, p, size=n).astype(float)

        # outcome: true ATE ~ +1.6
        y = (
            0.5
            + 1.6 * t
            + 0.4 * x_num
            + 0.5 * w_num
            + 0.2 * (x_cat == "C").astype(float)
            + 0.15 * (x_ord == "high").astype(float)
            + rng.normal(scale=1.0, size=n)
        )

        return pd.DataFrame(
            {
                "y": y,
                "t": t,
                "x_num": x_num,
                "x_cat": x_cat,
                "x_ord": x_ord,
                "x_dt": x_dt,
                "w_num": w_num,
                "w_bin": w_bin,
                "w_tok": w_tok,
            }
        )


def spec_binary_with_mixed_encodings() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "Y": {"kind": "continuous", "column": "y"},
            "T": {"kind": "binary", "column": "t", "treated_values": [1.0], "control_values": [0.0]},
            "X": ["x_num", "x_cat", "x_ord", "x_dt"],
            "W": ["w_num", "w_bin", "w_tok"],
            "Z": [],
        }
    )


def preprocessors_mixed_encodings() -> tuple[ColumnTransformer, ColumnTransformer]:
    """
    IMPORTANT: EconML passes nuisance inputs as raw concatenated [X, W] with shape (n, d_x + d_w).
    Here d_x=4, d_w=3 => 7 columns with indices 0..6.
    """

    X_CAT_LEVELS = ["A", "B", "C"]
    X_ORD_LEVELS = ["low", "med", "high"]
    W_BIN_LEVELS = ["no", "yes"]

    # raw X: [x_num(0), x_cat(1), x_ord(2), x_dt(3)]
    pre_X = ColumnTransformer(
        transformers=[
            ("x_num", StandardScaler(), [0]),
            ("x_cat", _make_ohe(dense=True, categories=[X_CAT_LEVELS]), [1]),
            ("x_ord", Pipeline([("ord", OrdinalEncoder(categories=[X_ORD_LEVELS])), ("sc", StandardScaler())]), [2]),
            ("x_dt", Pipeline([("epoch", FunctionTransformer(_to_epoch_seconds_2d, validate=False)), ("sc", StandardScaler())]), [3]),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    # raw XW: [x_num(0), x_cat(1), x_ord(2), x_dt(3), w_num(4), w_bin(5), w_tok(6)]
    w_tok_hash = Pipeline(
        steps=[
            ("to_lists", FunctionTransformer(_to_token_lists, validate=False)),
            ("hash", FeatureHasher(n_features=8, input_type="string")),
        ]
    )

    pre_XW = ColumnTransformer(
        transformers=[
            ("x_num", StandardScaler(), [0]),
            ("x_cat", _make_ohe(dense=False, categories=[X_CAT_LEVELS]), [1]),
            ("x_ord", OrdinalEncoder(categories=[X_ORD_LEVELS]), [2]),
            ("x_dt", FunctionTransformer(_to_epoch_seconds_2d, validate=False), [3]),
            ("w_num", StandardScaler(), [4]),
            ("w_bin", _make_ohe(dense=False, categories=[W_BIN_LEVELS]), [5]),
            ("w_tok", w_tok_hash, [6]),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # force sparse output
    )

    return pre_X, pre_XW


# -----------------------------------------------------------------------------
# DGP 2: numeric-only but “fancy” (polynomial featurizer + binning in nuisance)
# -----------------------------------------------------------------------------
class PolyBinningDGP:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def make(self, n: int = 900) -> pd.DataFrame:
        rng = self.rng
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        w1 = rng.normal(size=n)
        w2 = rng.normal(size=n)

        logits = 0.25 * x1 - 0.20 * x2 + 0.35 * w1 - 0.1 * w2
        p = 1.0 / (1.0 + np.exp(-logits))
        t = rng.binomial(1, p, size=n).astype(float)

        y = 0.2 + 1.2 * t + 0.3 * x1 - 0.2 * x2 + 0.5 * w1 + 0.2 * (w2 > 0).astype(float) + rng.normal(scale=1.0, size=n)
        return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2, "w1": w1, "w2": w2})


def spec_binary_poly_binning() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "Y": {"kind": "continuous", "column": "y"},
            "T": {"kind": "binary", "column": "t", "treated_values": [1.0], "control_values": [0.0]},
            "X": ["x1", "x2"],
            "W": ["w1", "w2"],
            "Z": [],
        }
    )


def preprocessors_poly_binning() -> tuple[ColumnTransformer, ColumnTransformer]:
    """
    pre_X: polynomial features over raw X (2 cols)
    pre_XW: nuisance encoding over raw concatenated [X, W] (4 cols)
    """

    # raw X: [x1(0), x2(1)]
    pre_X = Pipeline(
        steps=[
            ("sc", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),  # 2 -> 5
        ]
    )

    # raw XW: [x1(0), x2(1), w1(2), w2(3)]
    pre_XW = ColumnTransformer(
        transformers=[
            ("x_poly", Pipeline([("sc", StandardScaler()), ("poly", PolynomialFeatures(degree=2, include_bias=False))]), [0, 1]),
            ("w1", StandardScaler(), [2]),
            ("w2_bin", KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile"), [3]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
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


def _fit_first(*, model, data_repo, user_id, conv_id, df: pd.DataFrame, spec: CausalSpec, pre_X, pre_XW):
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
    assert isinstance(fit_res, FitSuccess), f"FIT must succeed. got={type(fit_res)} err={getattr(fit_res, 'error', None)}"
    return dataset_id, fit_id


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_fit_and_ate_succeed_with_mixed_encodings_strings_ordinal_datetime_hashing(sut):
    model, data_repo, _models_repo, user_id, conv_id = sut

    df = MixedEncodingDGP(seed=0).make(n=1000)
    spec = spec_binary_with_mixed_encodings()
    pre_X, pre_XW = preprocessors_mixed_encodings()

    dataset_id, fit_id = _fit_first(
        model=model,
        data_repo=data_repo,
        user_id=user_id,
        conv_id=conv_id,
        df=df,
        spec=spec,
        pre_X=pre_X,
        pre_XW=pre_XW,
    )

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
    assert len(res.ate) == 1
    ate_val = float(np.mean(np.asarray(res.ate[0]["ate"], dtype=float)))
    assert np.isfinite(ate_val)
    assert ate_val > 0.1

    asdict(res)


def test_fit_and_ate_succeed_with_polynomial_featurizer_and_binning_sparse_w(sut):
    model, data_repo, _models_repo, user_id, conv_id = sut

    df = PolyBinningDGP(seed=1).make(n=900)
    spec = spec_binary_poly_binning()
    pre_X, pre_XW = preprocessors_poly_binning()

    dataset_id, fit_id = _fit_first(
        model=model,
        data_repo=data_repo,
        user_id=user_id,
        conv_id=conv_id,
        df=df,
        spec=spec,
        pre_X=pre_X,
        pre_XW=pre_XW,
    )

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
    assert len(res.ate) == 1
    ate_val = float(np.mean(np.asarray(res.ate[0]["ate"], dtype=float)))
    assert np.isfinite(ate_val)
    assert ate_val > 0.05

    asdict(res)


def test_fit_fails_if_pre_xw_is_built_for_featurized_x_instead_of_raw_xw(sut):
    """
    Guardrail: nuisance pre_XW must match raw XW layout (d_x + d_w),
    not the featurized X dimension.
    """
    model, data_repo, _models_repo, user_id, conv_id = sut

    df = MixedEncodingDGP(seed=2).make(n=500)
    spec = spec_binary_with_mixed_encodings()
    pre_X, _pre_XW = preprocessors_mixed_encodings()

    # WRONG: assumes nuisance sees 9 columns; actual raw XW is 7 cols.
    pre_XW_wrong = ColumnTransformer(
        transformers=[("bad", "passthrough", list(range(9)))],
        remainder="drop",
        sparse_threshold=1.0,
    )

    dataset_id = uuid4()
    fit_id = uuid4()
    data_repo.save_csv_data(user_id, conv_id, dataset_id, df)

    fit_cmd = FitCommand(
        model_name="econml_linear",
        dataset_id=dataset_id,
        run_id=fit_id,
        protocol_specs=spec,
        inputs=FitInputs(pre_X=pre_X, pre_XW=pre_XW_wrong),
    )

    res = model.execute(user_id=user_id, conversation_id=conv_id, command=fit_cmd)
    assert isinstance(res, CommandFailure)
    assert res.error.code == "ESTIMATOR_ERROR"