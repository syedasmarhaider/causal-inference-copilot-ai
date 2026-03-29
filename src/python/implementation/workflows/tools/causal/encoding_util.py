from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Union, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from python.implementation.workflows.tools.causal.encoding_plan import (
    EncodingPresetSpec,
    TransformPlan,
)


class EncodingUtil:
    NAME: ClassVar[str] = "ENCODING"
    def compile(self,plan: TransformPlan, *, effect_modifiers_order: Sequence[str], covariates_order: Sequence[str], dense_output: bool = True) -> CompiledTransformers:
        return compile_plan_to_transformers(
            plan=plan,
            effect_modifiers=effect_modifiers_order,
            covariates=covariates_order,
            dense_output=dense_output,
            require_full_coverage=True,
        )  



# =============================================================================
# Output contract (SRP: plan -> transformers)
# =============================================================================
@dataclass(frozen=True)
class CompiledTransformers:
    """
    pre_X  : transformer that expects effect-modifier matrix (n, dx)
             ordered as effect_modifiers_order
    pre_XW : transformer that expects concatenated
             [effect_modifiers|covariates] matrix (n, dx+dw)
             ordered as (effect_modifiers_order + covariates_order)
    """
    pre_X: ColumnTransformer
    pre_XW: ColumnTransformer

CTTransformer = Union[BaseEstimator, Literal["passthrough"], Literal["drop"]]

# =============================================================================
# Helpers: sklearn-compat OneHotEncoder (version differences) + sparse/dense control
# =============================================================================
def _make_one_hot_encoder(
    *,
    handle_unknown: Literal["ignore", "error"],
    drop_first: bool,
    max_categories: int | None,
    dense_output: bool,
) -> OneHotEncoder:
    drop = "first" if drop_first else None
    kwargs: dict[str, Any] = {"handle_unknown": handle_unknown, "drop": drop}
    if max_categories is not None:
        kwargs["max_categories"] = int(max_categories)

    # sklearn>=1.2 uses sparse_output; older uses sparse
    try:
        return OneHotEncoder(sparse_output=bool(not dense_output), **kwargs)  # type: ignore[arg-type]
    except TypeError:
        return OneHotEncoder(sparse=bool(not dense_output), **kwargs)  # type: ignore[arg-type]


# =============================================================================
# Custom transformers
# =============================================================================
class RaiseIfMissing(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        s = pd.Series(np.asarray(X, dtype=object).reshape(-1))
        if s.isna().any():
            raise ValueError("Missing values found but missing='error'.")
        return self

    def transform(self, X) -> np.ndarray:
        s = pd.Series(np.asarray(X, dtype=object).reshape(-1))
        if s.isna().any():
            raise ValueError("Missing values found but missing='error'.")
        return np.asarray(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None or len(input_features) == 0:
            return np.asarray(["feature"], dtype=object)
        return np.asarray(input_features, dtype=object)


class Log1pSafeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, *, allow_negative: bool):
        self.allow_negative = bool(allow_negative)

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype="float64")
        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError(f"num_log1p: expected shape (n,1), got {arr.shape}.")
        vals = arr[:, 0]
        if np.any(vals <= -1.0):
            raise ValueError("num_log1p: values <= -1 found (log1p undefined).")
        if (not self.allow_negative) and np.any(vals < 0.0):
            raise ValueError("num_log1p: negative values found but allow_negative=False.")
        return np.log1p(arr)

    def get_feature_names_out(self, input_features=None):
        if input_features is None or len(input_features) == 0:
            return np.asarray(["log1p"], dtype=object)
        return np.asarray([str(input_features[0])], dtype=object)


class MinMaxEpsScaler(BaseEstimator, TransformerMixin):
    """
    Min-max scaling with epsilon-clamped denominator to avoid divide-by-zero explosions
    for constant columns. Works on dense arrays (n,k).
    """
    def __init__(self, *, eps: float = 1e-12):
        self.eps = float(eps)
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype="float64")
        if arr.ndim != 2:
            raise ValueError(f"num_minmax: expected 2D array, got shape {arr.shape}.")
        self.data_min_ = np.nanmin(arr, axis=0)
        self.data_max_ = np.nanmax(arr, axis=0)
        return self

    def transform(self, X) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise ValueError("num_minmax: transformer is not fitted.")
        arr = np.asarray(X, dtype="float64")
        denom = self.data_max_ - self.data_min_
        denom = np.where(np.abs(denom) < self.eps, 1.0, denom)
        return (arr - self.data_min_) / denom

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(input_features, dtype=object)


class BinaryMapTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        mapping: dict[str, float],
        *,
        allow_unknown: bool,
        unknown_value: float | None,
        missing: Literal["as_unknown", "impute_token", "error"],
        missing_token: str | None,
    ):
        self.mapping = dict(mapping)
        self.allow_unknown = bool(allow_unknown)
        self.unknown_value = unknown_value
        self.missing = missing
        self.missing_token = missing_token

    def fit(self, X, y=None):
        if not self.mapping:
            raise ValueError("map_binary: mapping must be non-empty.")
        if self.missing == "impute_token":
            if self.missing_token is None:
                raise ValueError("map_binary: missing_token required when missing='impute_token'.")
            if self.missing_token not in self.mapping:
                raise ValueError("map_binary: missing_token must exist in mapping.")
        if (not self.allow_unknown) and (self.unknown_value is not None):
            raise ValueError("map_binary: unknown_value must be null when allow_unknown=False.")
        return self

    def transform(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=object)
        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError(f"map_binary: expected shape (n,1), got {arr.shape}.")

        s = pd.Series(arr[:, 0])
        is_na = s.isna()

        if self.missing == "error" and bool(is_na.any()):
            raise ValueError("map_binary: missing values found but missing='error'.")

        if self.missing == "impute_token":
            s = s.fillna(self.missing_token)

        out = s.map(lambda v: self.mapping.get(str(v), np.nan))
        unknown_mask = (~s.isna()) & out.isna()

        if bool(unknown_mask.any()):
            if not self.allow_unknown:
                sample = s.loc[unknown_mask].astype(str).head(25).tolist()
                raise ValueError(f"map_binary: unknown categories found (allow_unknown=False). sample={sample}")
            fill = float(self.unknown_value) if self.unknown_value is not None else np.nan
            out.loc[unknown_mask] = fill

        if self.missing == "as_unknown" and bool(is_na.any()):
            fill = float(self.unknown_value) if self.unknown_value is not None else np.nan
            out.loc[is_na] = fill

        return out.astype("float64").to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None or len(input_features) == 0:
            return np.asarray(["map_binary"], dtype=object)
        return np.asarray([str(input_features[0])], dtype=object)


class OrdinalMapTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        order: list[str],
        *,
        start: int,
        allow_unknown: bool,
        unknown_value: int | None,
        missing: Literal["as_unknown", "impute_token", "error"],
        missing_token: str | None,
        token_position: Literal["prepend", "append"] | None,
    ):
        self.order = list(order)
        self.start = int(start)
        self.allow_unknown = bool(allow_unknown)
        self.unknown_value = unknown_value
        self.missing = missing
        self.missing_token = missing_token
        self.token_position = token_position

    def fit(self, X, y=None):
        if len(self.order) != len(set(self.order)):
            raise ValueError("map_ordinal: 'order' must not contain duplicates.")
        if self.missing == "impute_token":
            if self.missing_token is None or self.token_position is None:
                raise ValueError("map_ordinal: missing_token and token_position required when missing='impute_token'.")
        if (not self.allow_unknown) and (self.unknown_value is not None):
            raise ValueError("map_ordinal: unknown_value must be null when allow_unknown=False.")
        return self

    def transform(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=object)
        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError(f"map_ordinal: expected shape (n,1), got {arr.shape}.")

        s = pd.Series(arr[:, 0])
        is_na = s.isna()

        if self.missing == "error" and bool(is_na.any()):
            raise ValueError("map_ordinal: missing values found but missing='error'.")

        effective_order = list(self.order)
        if self.missing == "impute_token":
            tok = self.missing_token
            if tok is None:
                raise ValueError("map_ordinal: missing_token required when missing='impute_token'.")
            if tok not in effective_order:
                if self.token_position == "prepend":
                    effective_order = [tok] + effective_order
                else:
                    effective_order = effective_order + [tok]
            s = s.fillna(tok)

        mapping = {cat: float(self.start + i) for i, cat in enumerate(effective_order)}
        out = s.map(lambda v: mapping.get(str(v), np.nan))
        unknown_mask = (~s.isna()) & out.isna()

        if bool(unknown_mask.any()):
            if not self.allow_unknown:
                sample = s.loc[unknown_mask].astype(str).head(25).tolist()
                raise ValueError(f"map_ordinal: unknown categories found (allow_unknown=False). sample={sample}")
            fill = float(self.unknown_value) if self.unknown_value is not None else np.nan
            out.loc[unknown_mask] = fill

        if self.missing == "as_unknown" and bool(is_na.any()):
            fill = float(self.unknown_value) if self.unknown_value is not None else np.nan
            out.loc[is_na] = fill

        return out.astype("float64").to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None or len(input_features) == 0:
            return np.asarray(["map_ordinal"], dtype=object)
        return np.asarray([str(input_features[0])], dtype=object)


class DateTimeToEpochSecondsTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        errors: Literal["coerce", "raise"],
        unit: Literal["s", "ms", "us", "ns"],
        add_missing_indicator: bool,
    ):
        self.errors = errors
        self.unit = unit
        self.add_missing_indicator = bool(add_missing_indicator)

    def fit(self, X, y=None):
        if self.unit not in ("s", "ms", "us", "ns"):
            raise ValueError(f"datetime_epoch_seconds: invalid unit {self.unit!r}")
        return self

    def transform(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=object)
        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError(f"datetime_epoch_seconds: expected shape (n,1), got {arr.shape}.")

        s = pd.Series(arr[:, 0])
        dt = pd.to_datetime(s, errors="coerce")

        invalid_mask = (~s.isna()) & dt.isna()
        if self.errors == "raise" and bool(invalid_mask.any()):
            sample = s.loc[invalid_mask].astype(str).head(25).tolist()
            raise ValueError(f"datetime_epoch_seconds: unparseable values found. sample={sample}")

        # timezone-aware -> UTC naive
        if getattr(dt.dtype, "tz", None) is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)

        ns = dt.values.view("int64").astype("float64")
        out = pd.Series(ns).where(~dt.isna(), np.nan).to_numpy()
        denom = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0}[self.unit]
        out = (out / denom).reshape(-1, 1)

        if not self.add_missing_indicator:
            return out

        ind = pd.isna(dt).astype("int8").to_numpy().reshape(-1, 1)
        return np.concatenate([out, ind], axis=1)

    def get_feature_names_out(self, input_features=None):
        base = "datetime"
        if input_features is not None and len(input_features) > 0:
            base = str(input_features[0])
        if not self.add_missing_indicator:
            return np.asarray([base], dtype=object)
        return np.asarray([base, f"{base}_missing"], dtype=object)


# =============================================================================
# SRP: compile plan -> (pre_X, pre_XW). effect_modifiers/covariates must be provided and non-empty.
# =============================================================================
def compile_plan_to_transformers(
    plan: TransformPlan,
    *,
    effect_modifiers: Sequence[str],
    covariates: Sequence[str],
    dense_output: bool = True,
    require_full_coverage: bool = True,
) -> CompiledTransformers:
    """
    Requires effect_modifiers and covariates (non-empty). No inference. No None.

    - pre_X  expects effect_modifier array columns ordered exactly as effect_modifiers
    - pre_XW expects concatenated [effect_modifiers|covariates] array columns ordered exactly as (effect_modifiers + covariates)

    If require_full_coverage=True:
      - every col in effect_modifiers/covariates must have a plan entry
      - and every plan entry must belong to effect_modifiers/covariates (no surprises)

    NOTE (DML best practice):
      - Do NOT fit these transformers globally before cross-fitting. Put them inside the nuisance model pipelines
        so each fold fits its own preprocessing to avoid leakage.
    """
    if len(effect_modifiers) == 0 and len(covariates) == 0:
        raise ValueError("effect_modifiers or covariates must be provided (not None).")

    effect_modifiers_l = list(effect_modifiers)
    covariates_l = list(covariates)
    xw_cols = effect_modifiers_l + covariates_l

    # Index maps (EconML passes numpy arrays, not DataFrames)
    x_index = {c: i for i, c in enumerate(effect_modifiers_l)}
    xw_index = {c: i for i, c in enumerate(xw_cols)}

    # Plan sanity
    plan_cols = [cp.column for cp in plan.columns]
    if len(plan_cols) != len(set(plan_cols)):
        dup = sorted({c for c in plan_cols if plan_cols.count(c) > 1})
        raise ValueError(f"Duplicate column plans are not allowed: {dup}")

    plan_by_col = {cp.column: cp for cp in plan.columns}

    if require_full_coverage:
        missing_plans = [c for c in xw_cols if c not in plan_by_col]
        if missing_plans:
            raise ValueError(f"Missing encoding plan for columns: {missing_plans}")

        extra_plans = [c for c in plan_by_col if c not in set(xw_cols)]
        if extra_plans:
            raise ValueError(
                "Plan contains columns not present in effect_modifiers_order/covariates_order: "
                f"{extra_plans}"
            )

    def _compile(enc: EncodingPresetSpec) -> CTTransformer:
        preset = enc.preset

        if preset == "drop":
            return "drop"
        if preset == "passthrough":
            return "passthrough"

        if preset == "cat_onehot":
            steps: list[tuple[str, BaseEstimator]] = []
            if enc.missing == "error":
                steps.append(("check_missing", RaiseIfMissing()))
                steps.append((
                    "ohe",
                    _make_one_hot_encoder(
                        handle_unknown=enc.handle_unknown,
                        drop_first=enc.drop_first,
                        max_categories=enc.max_categories,
                        dense_output=dense_output,
                    ),
                ))
                return Pipeline(steps)

            # Ensure missing always becomes a known category
            token = "__NA__" if enc.missing == "dummy_na" else enc.missing_token
            steps.append(("impute", SimpleImputer(strategy="constant", fill_value=token)))
            steps.append((
                "ohe",
                _make_one_hot_encoder(
                    handle_unknown=enc.handle_unknown,
                    drop_first=enc.drop_first,
                    max_categories=enc.max_categories,
                    dense_output=dense_output,
                ),
            ))
            return Pipeline(steps)

        if preset == "num_standard":
            return Pipeline([
                ("impute", SimpleImputer(strategy=enc.impute, add_indicator=enc.add_missing_indicator)),
                ("scale", StandardScaler()),
            ])

        if preset == "num_minmax":
            return Pipeline([
                ("impute", SimpleImputer(strategy=enc.impute, add_indicator=enc.add_missing_indicator)),
                ("scale", MinMaxEpsScaler(eps=enc.eps)),
            ])

        if preset == "num_log1p":
            steps2: list[tuple[str, BaseEstimator]] = [
                ("impute", SimpleImputer(strategy=enc.impute, add_indicator=enc.add_missing_indicator)),
                ("log1p", Log1pSafeTransformer(allow_negative=enc.allow_negative)),
            ]
            if enc.then_scale == "standard":
                steps2.append(("scale", StandardScaler()))
            elif enc.then_scale == "minmax":
                # default eps for log1p branch; keep consistent behavior
                steps2.append(("scale", MinMaxEpsScaler(eps=1e-12)))
            return Pipeline(steps2)

        if preset == "datetime_epoch_seconds":
            return DateTimeToEpochSecondsTransformer(
                errors=enc.errors,
                unit=enc.unit,
                add_missing_indicator=enc.add_missing_indicator,
            )

        if preset == "map_binary":
            return BinaryMapTransformer(
                mapping=cast(dict[str, float], enc.mapping),
                allow_unknown=enc.allow_unknown,
                unknown_value=enc.unknown_value,
                missing=enc.missing,
                missing_token=enc.missing_token,
            )

        if preset == "map_ordinal":
            return OrdinalMapTransformer(
                order=cast(list[str], enc.order),
                start=enc.start,
                allow_unknown=enc.allow_unknown,
                unknown_value=enc.unknown_value,
                missing=enc.missing,
                missing_token=enc.missing_token,
                token_position=enc.token_position,
            )

        raise ValueError(f"Unsupported preset: {preset!r}")

    # -------------------------------------------------------------------------
    # Compile with deterministic ordering: iterate by effect_modifiers / (effect_modifiers + covariates)
    # -------------------------------------------------------------------------
    x_trs: list[tuple[str, CTTransformer, list[int]]] = []
    xw_trs: list[tuple[str, CTTransformer, list[int]]] = []

    # pre_X: only effect_modifier columns, in effect_modifiers order
    for col in effect_modifiers_l:
        cp = plan_by_col.get(col)
        if cp is None:
            if require_full_coverage:
                raise ValueError(f"Missing encoding plan for effect_modifier column: {col!r}")
            continue
        if cp.role != "effect_modifier":
            raise ValueError(
                f"Column {col!r} is in effect_modifiers order but plan role is "
                f"{cp.role!r} (expected 'effect_modifier')."
            )
        x_trs.append((col, _compile(cp.encoding), [x_index[col]]))

    # pre_XW: effect_modifier then covariate, in (effect_modifiers + covariates)
    for col in xw_cols:
        cp = plan_by_col.get(col)
        if cp is None:
            if require_full_coverage:
                raise ValueError(f"Missing encoding plan for column: {col!r}")
            continue
        if cp.role not in ("effect_modifier", "covariate"):
            raise ValueError(f"Unknown role {cp.role!r} for column {cp.column!r}")
        xw_trs.append((col, _compile(cp.encoding), [xw_index[col]]))

    # Enforce “effect_modifier or covariate always”
    if not any(cp.role == "effect_modifier" for cp in plan.columns) and not any(cp.role == "covariate" for cp in plan.columns):
        raise ValueError(
            "At least one column must have role 'effect_modifier' or 'covariate' in the plan."
        )
    
    # Ensure we have at least one non-dropped transformer in each view
    def _has_non_drop(trs: list[tuple[str, CTTransformer, list[int]]]) -> bool:
        return any(t[1] != "drop" for t in trs)

    if not x_trs:
        raise ValueError("No effect_modifier transformers compiled (empty effect_modifiers_order?).")
    if not xw_trs:
        raise ValueError("No effect_modifier/covariate transformers compiled (empty effect_modifiers+covariates?).")
    if not _has_non_drop(x_trs):
        raise ValueError("All effect_modifier columns are dropped. At least one effect_modifier column must be kept.")
    if not _has_non_drop(xw_trs):
        raise ValueError("All effect_modifier/covariate columns are dropped. At least one column must be kept.")

    # If you requested dense output, force dense aggregation.
    # If not, allow sparse when beneficial (especially for one-hot).
    sparse_threshold = 0.0 if dense_output else 1.0

    pre_X = ColumnTransformer(
        transformers=x_trs,
        remainder="drop",
        sparse_threshold=sparse_threshold,
        verbose_feature_names_out=True,
    )

    pre_XW = ColumnTransformer(
        transformers=xw_trs,
        remainder="drop",
        sparse_threshold=sparse_threshold,
        verbose_feature_names_out=True,
    )

    return CompiledTransformers(
        pre_X=pre_X,
        pre_XW=pre_XW,
    )  
