from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, conint

from python.implementation.workflows.utils.validation import ValidationIssueModel


# TODO: move to tool
# =============================================================================
# Public encoding identifiers
# =============================================================================
EncodingType = Literal[
    "drop",
    "one_hot",
    "binary_map",       
    "binary_map_idx",  
    "ordinal_map",     
    "ordinal_map_idx",
    "to_numeric",
    "fillna",
    "log1p",
    "standardize",
    "minmax",
    "datetime_to_epoch_seconds",
]


# =============================================================================
# LLM-facing descriptions (stable + explicit)
# =============================================================================
DESCRIPTIONS: Dict[EncodingType, str] = {
    "drop": "Drop the column entirely. Params: none.",
    "one_hot": (
        "Categorical -> dummy columns. Example: 'Sex' => Sex__M, Sex__F (+ optional NaN indicator). "
        "Params (optional): dummy_na(bool), drop_first(bool), max_categories(int)."
    ),
    "binary_map": (
        "Map categories using explicit mapping dict. Example: {'Never':0,'Former/Current':1}. "
        "Params (required): mapping(dict[str, int|float]). Optional: allow_unknown(bool). "
        "WARNING: mapping keys must match dataset category strings EXACTLY."
    ),
    "binary_map_idx": (
        "Map categories to 0/1 using category INDICES from profiling. "
        "Params (required): pos(list[int]), neg(list[int]). Optional: drop(list[int]). "
        "Recommended (robust vs label paraphrasing)."
    ),
    "ordinal_map": (
        "Ordered categories -> integers. Example order ['I','II','III','IV'] -> 0..3. "
        "Params (required): order(list[str]). Optional: start(int), allow_unknown(bool). "
        "WARNING: order values must match dataset category strings EXACTLY."
    ),
    "ordinal_map_idx": (
        "Ordinal map using category INDICES from profiling. Params (required): order(list[int]). "
        "Optional: start(int), drop(list[int]). Recommended."
    ),
    "to_numeric": "Coerce to numeric; invalid values can become NaN. Params (optional): errors('coerce'|'raise').",
    "fillna": "Fill missing values with a constant sentinel. Params (required): value(any JSON scalar).",
    "log1p": "Numeric -> log(1+x). Params (optional): allow_negative(bool).",
    "standardize": "Z-score: (x-mean)/std. Params (optional): ddof(int), eps(float).",
    "minmax": "Scale into [0,1]. Params (optional): eps(float).",
    "datetime_to_epoch_seconds": (
        "Datetime -> epoch seconds (UTC). Params (optional): errors('coerce'|'raise'), unit('s'|'ms'|'us'|'ns')."
    ),
}


# =============================================================================
# Catalog model (simple)
# =============================================================================
class SupportedEncodingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    encodings: List[EncodingType] = Field(default_factory=list)


def get_supported_encodings_model() -> SupportedEncodingsModel:
    return SupportedEncodingsModel(
        encodings=[
            "drop",
            "one_hot",
            "binary_map",
            "binary_map_idx",
            "ordinal_map",
            "ordinal_map_idx",
            "to_numeric",
            "fillna",
            "log1p",
            "standardize",
            "minmax",
            "datetime_to_epoch_seconds",
        ]
    )


def get_encoding_models_with_description() -> str:
    allowed = get_supported_encodings_model().encodings
    return "\n".join(f"- {enc}: {DESCRIPTIONS[enc]}" for enc in allowed)


# =============================================================================
# Pydantic specs: LLM outputs THIS (per column)
# =============================================================================
class _BaseParams(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _BaseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- drop
class DropSpec(_BaseSpec):
    encoding: Literal["drop"]


# ---- one_hot
class OneHotParams(_BaseParams):
    dummy_na: bool = True
    drop_first: bool = False
    max_categories: Optional[int] = None


class OneHotSpec(_BaseSpec):
    encoding: Literal["one_hot"]
    params: Optional[OneHotParams] = None


# ---- binary_map (dict)
class BinaryMapParams(_BaseParams):
    mapping: Dict[str, Union[int, float]] = Field(..., min_length=1)
    allow_unknown: bool = False


class BinaryMapSpec(_BaseSpec):
    encoding: Literal["binary_map"]
    params: BinaryMapParams


# ---- binary_map_idx (recommended)
CatIdx = conint(ge=0)


class BinaryMapIdxParams(_BaseParams):
    pos: List[int] = Field(..., min_length=1)
    neg: List[int] = Field(..., min_length=1)
    drop: List[int] = Field(default_factory=list)


class BinaryMapIdxSpec(_BaseSpec):
    encoding: Literal["binary_map_idx"]
    params: BinaryMapIdxParams


# ---- ordinal_map (label order)
class OrdinalMapParams(_BaseParams):
    order: List[str] = Field(..., min_length=1)
    start: int = 0
    allow_unknown: bool = False


class OrdinalMapSpec(_BaseSpec):
    encoding: Literal["ordinal_map"]
    params: OrdinalMapParams


# ---- ordinal_map_idx (recommended)
class OrdinalMapIdxParams(_BaseParams):
    order: List[int] = Field(..., min_length=1)
    start: int = 0
    drop: List[int] = Field(default_factory=list)


class OrdinalMapIdxSpec(_BaseSpec):
    encoding: Literal["ordinal_map_idx"]
    params: OrdinalMapIdxParams


# ---- to_numeric
class ToNumericParams(_BaseParams):
    errors: Literal["coerce", "raise"] = "coerce"


class ToNumericSpec(_BaseSpec):
    encoding: Literal["to_numeric"]
    params: Optional[ToNumericParams] = None


# ---- fillna
class FillNaParams(_BaseParams):
    # Keep JSON-safe types. If you truly need object values, widen this.
    value: Union[int, float, str, bool]


class FillNaSpec(_BaseSpec):
    encoding: Literal["fillna"]
    params: FillNaParams


# ---- log1p
class Log1pParams(_BaseParams):
    allow_negative: bool = False


class Log1pSpec(_BaseSpec):
    encoding: Literal["log1p"]
    params: Optional[Log1pParams] = None


# ---- standardize
class StandardizeParams(_BaseParams):
    ddof: int = 0
    eps: float = 1e-12


class StandardizeSpec(_BaseSpec):
    encoding: Literal["standardize"]
    params: Optional[StandardizeParams] = None


# ---- minmax
class MinMaxParams(_BaseParams):
    eps: float = 1e-12


class MinMaxSpec(_BaseSpec):
    encoding: Literal["minmax"]
    params: Optional[MinMaxParams] = None


# ---- datetime_to_epoch_seconds
class DateTimeToEpochParams(_BaseParams):
    errors: Literal["coerce", "raise"] = "coerce"
    unit: Literal["s", "ms", "us", "ns"] = "s"


class DateTimeToEpochSpec(_BaseSpec):
    encoding: Literal["datetime_to_epoch_seconds"]
    params: Optional[DateTimeToEpochParams] = None


EncodingSpec = Union[
    DropSpec,
    OneHotSpec,
    BinaryMapSpec,
    BinaryMapIdxSpec,
    OrdinalMapSpec,
    OrdinalMapIdxSpec,
    ToNumericSpec,
    FillNaSpec,
    Log1pSpec,
    StandardizeSpec,
    MinMaxSpec,
    DateTimeToEpochSpec,
]


# =============================================================================
# Feature-map (optional but very useful downstream)
# =============================================================================
class FeatureMapModel(BaseModel):
    """
    produced_columns:
      raw column -> columns representing it in transformed df.

    dropped:
      raw columns removed.
    """
    model_config = ConfigDict(extra="forbid")

    produced_columns: Dict[str, List[str]] = Field(default_factory=dict)
    dropped: List[str] = Field(default_factory=list)


# =============================================================================
# Issues helper
# =============================================================================
def _issue(
    *,
    severity: Literal["WARN", "FAIL"],
    message: str,
    evidence: Optional[Dict[str, Any]] = None,
    fix_hint: Optional[str] = None,
) -> ValidationIssueModel:
    return ValidationIssueModel(
        severity=severity,
        message=message,
        evidence=evidence or {},
        fix_hint=fix_hint,
    )


# =============================================================================
# Deterministic naming helpers (no LLM involvement)
# =============================================================================
def _derived_name(column: str, suffix: str) -> str:
    return f"{column}__{suffix}"


# =============================================================================
# Public entrypoint: apply a single spec to a single column
# =============================================================================
def apply_encoding(
    *,
    df: pd.DataFrame,
    column: str,
    spec: EncodingSpec,
    # For *_idx encodings, this MUST be supplied from profiling:
    # categories_in_order[0] corresponds to index 0, etc.
    categories_in_order: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, FeatureMapModel, List[ValidationIssueModel]]:
    """
    Apply one encoding spec for one column.

    Returns:
      (new_df, feature_map_for_this_column, issues)

    Guarantees:
      - never mutates input df
      - deterministic output column names for derived numeric transforms
      - strict validation of required params via Pydantic
    """
    if column not in df.columns:
        return df.copy(), FeatureMapModel(), [
            _issue(
                severity="FAIL",
                message=f"Column '{column}' not found.",
                evidence={"column": column, "encoding": cast(Any, spec).encoding},
                fix_hint="Fix transform spec column name (case/spelling).",
            )
        ]

    if isinstance(spec, DropSpec):
        out = df.drop(columns=[column]).copy()
        return out, FeatureMapModel(dropped=[column]), []

    if isinstance(spec, OneHotSpec):
        return _encode_one_hot(df=df, column=column, params=spec.params)

    if isinstance(spec, BinaryMapSpec):
        return _encode_binary_map(df=df, column=column, params=spec.params, categories_in_order=categories_in_order)

    if isinstance(spec, BinaryMapIdxSpec):
        return _encode_binary_map_idx(df=df, column=column, params=spec.params, categories_in_order=categories_in_order)

    if isinstance(spec, OrdinalMapSpec):
        return _encode_ordinal_map(df=df, column=column, params=spec.params, categories_in_order=categories_in_order)

    if isinstance(spec, OrdinalMapIdxSpec):
        return _encode_ordinal_map_idx(df=df, column=column, params=spec.params, categories_in_order=categories_in_order)

    if isinstance(spec, ToNumericSpec):
        return _encode_to_numeric(df=df, column=column, params=spec.params)

    if isinstance(spec, FillNaSpec):
        return _encode_fillna(df=df, column=column, params=spec.params)

    if isinstance(spec, Log1pSpec):
        return _encode_log1p(df=df, column=column, params=spec.params)

    if isinstance(spec, StandardizeSpec):
        return _encode_standardize(df=df, column=column, params=spec.params)

    if isinstance(spec, MinMaxSpec):
        return _encode_minmax(df=df, column=column, params=spec.params)

    if isinstance(spec, DateTimeToEpochSpec):
        return _encode_datetime_to_epoch_seconds(df=df, column=column, params=spec.params)

    # Unreachable if EncodingSpec union stays in sync
    return df.copy(), FeatureMapModel(), [
        _issue(
            severity="FAIL",
            message="Unsupported encoding spec type.",
            evidence={"column": column, "spec_type": type(spec).__name__},
        )
    ]


# =============================================================================
# Encoding implementations
# =============================================================================
def _encode_one_hot(
    *,
    df: pd.DataFrame,
    column: str,
    params: Optional[OneHotParams],
) -> Tuple[pd.DataFrame, FeatureMapModel, List[ValidationIssueModel]]:
    p = params or OneHotParams()
    nunique = int(df[column].nunique(dropna=True))

    if p.max_categories is not None and nunique > p.max_categories:
        return df.copy(), FeatureMapModel(), [
            _issue(
                severity="FAIL",
                message=f"one_hot cardinality too high for '{column}': {nunique} categories.",
                evidence={"column": column, "nunique": nunique, "max_categories": p.max_categories},
                fix_hint="Reduce categories (group rare) or use *_map_idx.",
            )
        ]

    try:
        dummies = pd.get_dummies(
            df[column],
            prefix=column,
            prefix_sep="__",
            dummy_na=p.dummy_na,
            drop_first=p.drop_first,
        )
    except Exception as e:  # noqa: BLE001
        return df.copy(), FeatureMapModel(), [
            _issue(
                severity="FAIL",
                message=f"one_hot failed for '{column}': {e}",
                evidence={"column": column},
            )
        ]

    out = df.copy()
    out = pd.concat([out.drop(columns=[column]), dummies], axis=1)

    produced =  list(dummies.columns)
    return out, FeatureMapModel(produced_columns={column: produced}), []


def _require_categories(
    *,
    column: str,
    categories_in_order: Optional[Sequence[str]],
    encoding: str,
) -> Tuple[Optional[List[str]], List[ValidationIssueModel]]:
    if categories_in_order is None:
        return None, [
            _issue(
                severity="FAIL",
                message=f"'{encoding}' requires categories_in_order for '{column}', but none was provided.",
                evidence={"column": column, "encoding": encoding},
                fix_hint="Pass profiling categories list into apply_encoding(..., categories_in_order=...).",
            )
        ]
    return list(categories_in_order), []


def _encode_binary_map(
    *,
    df: pd.DataFrame,
    column: str,
    params: BinaryMapParams,
    categories_in_order: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, FeatureMapModel, List[ValidationIssueModel]]:
    """
    Dict-based mapping. Safer when you also pass categories_in_order and validate keys.
    Still less robust than *_idx, but supported.
    """
    issues: List[ValidationIssueModel] = []
    out = df.copy()

    mapping = params.mapping

    # Optional safety validation if categories are provided:
    if categories_in_order is not None:
        cats = set(categories_in_order)
        bad_keys = sorted([k for k in mapping.keys() if k not in cats])
        if bad_keys:
            return df.copy(), FeatureMapModel(), [
                _issue(
                    severity="FAIL",
                    message=f"binary_map mapping contains keys not present in dataset categories for '{column}'.",
                    evidence={"column": column, "bad_keys": bad_keys, "n_categories": len(cats)},
                    fix_hint="Use exact category strings from profiling or switch to binary_map_idx.",
                )
            ]

    mapped = out[column].map(mapping)

    unknown_mask = out[column].notna() & mapped.isna()
    if bool(unknown_mask.any()):
        unknown_values = sorted(set(out[column][unknown_mask].astype(str).tolist()))
        if params.allow_unknown:
            issues.append(
                _issue(
                    severity="WARN",
                    message=f"binary_map saw unknown values in '{column}' which were set to NaN.",
                    evidence={"column": column, "unknown_values": unknown_values},
                    fix_hint="Extend mapping or use binary_map_idx.",
                )
            )
        else:
            return df.copy(), FeatureMapModel(), [
                _issue(
                    severity="FAIL",
                    message=f"binary_map has unmapped values in '{column}'.",
                    evidence={"column": column, "unknown_values": unknown_values},
                    fix_hint="Extend mapping, set allow_unknown=True, or use binary_map_idx.",
                )
            ]

    out[column] = mapped
    return out, FeatureMapModel(produced_columns={column: [column]}), issues


def _encode_binary_map_idx(
    *,
    df: pd.DataFrame,
    column: str,
    params: BinaryMapIdxParams,
    categories_in_order: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, FeatureMapModel, List[ValidationIssueModel]]:
    cats, issues = _require_categories(column=column, categories_in_order=categories_in_order, encoding="binary_map_idx")
    if cats is None:
        return df.copy(), FeatureMapModel(), issues

    n = len(cats)
    pos = set(map(int, params.pos))
    neg = set(map(int, params.neg))
    drp = set(map(int, params.drop))

    bad = [i for i in sorted(pos | neg | drp) if i < 0 or i >= n]
    if bad:
        return df.copy(), FeatureMapModel(), [
            _issue(
                severity="FAIL",
                message=f"binary_map_idx has invalid category indices for '{column}': {bad}",
                evidence={"column": column, "n_categories": n, "bad_indices": bad},
                fix_hint="Indices must be within [0, n_categories-1] from profiling.",
            )
        ]

    inter = (pos & neg) | (pos & drp) | (neg & drp)
    if inter:
        return df.copy(), FeatureMapModel(), [
            _issue(
                severity="FAIL",
                message=f"binary_map_idx sets overlap for '{column}'.",
                evidence={"column": column, "overlap": sorted(inter)},
                fix_hint="pos/neg/drop must be disjoint.",
            )
        ]

    mapping: Dict[str, float] = {}
    for i in pos:
        mapping[cats[i]] = 1.0
    for i in neg:
        mapping[cats[i]] = 0.0
    for i in drp:
        mapping[cats[i]] = np.nan

    out = df.copy()
    out[column] = out[column].map(mapping)
    return out, FeatureMapModel(produced_columns={column: [column]}), []


def _encode_ordinal_map(
    *,
    df: pd.DataFrame,
    column: str,
    params: OrdinalMapParams,
    categories_in_order: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, FeatureMapModel, List[ValidationIssueModel]]:
    issues: List[ValidationIssueModel] = []
    out = df.copy()

    order = params.order
    mapping: Dict[str, float] = {v: float(params.start + i) for i, v in enumerate(order)}

    # Optional safety validation if categories are provided:
    if categories_in_order is not None:
        cats = set(categories_in_order)
        bad_vals = sorted([v for v in order if v not in cats])
        if bad_vals:
            return df.copy(), FeatureMapModel(), [
                _issue(
                    severity="FAIL",
                    message=f"ordinal_map order contains values not present in dataset categories for '{column}'.",
                    evidence={"column": column, "bad_values": bad_vals, "n_categories": len(cats)},
                    fix_hint="Use exact category strings from profiling or switch to ordinal_map_idx.",
                )
            ]

    mapped = out[column].map(mapping)

    unknown_mask = out[column].notna() & mapped.isna()
    if bool(unknown_mask.any()):
        unknown_values = sorted(set(out[column][unknown_mask].astype(str).tolist()))
        if params.allow_unknown:
            issues.append(
                _issue(
                    severity="WARN",
                    message=f"ordinal_map saw unknown values in '{column}' which were set to NaN.",
                    evidence={"column": column, "unknown_values": unknown_values},
                    fix_hint="Extend order list or use ordinal_map_idx.",
                )
            )
        else:
            return df.copy(), FeatureMapModel(), [
                _issue(
                    severity="FAIL",
                    message=f"ordinal_map has values not present in params['order'] for '{column}'.",
                    evidence={"column": column, "unknown_values": unknown_values},
                    fix_hint="Extend order list, set allow_unknown=True, or use ordinal_map_idx.",
                )
            ]

    out[column] = mapped
    return out, FeatureMapModel(produced_columns={column: [column]}), issues


def _encode_ordinal_map_idx(
    *,
    df: pd.DataFrame,
    column: str,
    params: OrdinalMapIdxParams,
    categories_in_order: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, FeatureMapModel, List[ValidationIssueModel]]:
    cats, issues = _require_categories(column=column, categories_in_order=categories_in_order, encoding="ordinal_map_idx")
    if cats is None:
        return df.copy(), FeatureMapModel(), issues

    n = len(cats)
    order = list(map(int, params.order))
    drp = set(map(int, params.drop))

    bad = [i for i in sorted(set(order) | drp) if i < 0 or i >= n]
    if bad:
        return df.copy(), FeatureMapModel(), [
            _issue(
                severity="FAIL",
                message=f"ordinal_map_idx has invalid category indices for '{column}': {bad}",
                evidence={"column": column, "n_categories": n, "bad_indices": bad},
                fix_hint="Indices must be within [0, n_categories-1] from profiling.",
            )
        ]

    if set(order) & drp:
        return df.copy(), FeatureMapModel(), [
            _issue(
                severity="FAIL",
                message=f"ordinal_map_idx: 'order' and 'drop' overlap for '{column}'.",
                evidence={"column": column, "overlap": sorted(set(order) & drp)},
                fix_hint="order and drop must be disjoint.",
            )
        ]

    mapping: Dict[str, float] = {}
    for j, idx in enumerate(order):
        mapping[cats[idx]] = float(params.start + j)
    for idx in drp:
        mapping[cats[idx]] = np.nan

    out = df.copy()
    out[column] = out[column].map(mapping)
    return out, FeatureMapModel(produced_columns={column: [column]}), []


def _encode_to_numeric(
    *,
    df: pd.DataFrame,
    column: str,
    params: Optional[ToNumericParams],
) -> Tuple[pd.DataFrame, FeatureMapModel, List[ValidationIssueModel]]:
    p = params or ToNumericParams()
    out = df.copy()

    try:
        out[column] = pd.to_numeric(out[column], errors=p.errors)
    except Exception as e:  # noqa: BLE001
        return df.copy(), FeatureMapModel(), [
            _issue(
                severity="FAIL",
                message=f"to_numeric failed for '{column}': {e}",
                evidence={"column": column, "errors": p.errors},
            )
        ]

    return out, FeatureMapModel(produced_columns={column: [column]}), []


def _encode_fillna(
    *,
    df: pd.DataFrame,
    column: str,
    params: FillNaParams,
) -> Tuple[pd.DataFrame, FeatureMapModel, List[ValidationIssueModel]]:
    out = df.copy()
    try:
        out[column] = out[column].fillna(params.value)
    except Exception as e:  # noqa: BLE001
        return df.copy(), FeatureMapModel(), [
            _issue(
                severity="FAIL",
                message=f"fillna failed for '{column}': {e}",
                evidence={"column": column, "value": params.value},
            )
        ]

    return out, FeatureMapModel(produced_columns={column: [column]}), []


def _encode_log1p(
    *,
    df: pd.DataFrame,
    column: str,
    params: Optional[Log1pParams],
) -> Tuple[pd.DataFrame, FeatureMapModel, List[ValidationIssueModel]]:
    p = params or Log1pParams()

    s = pd.to_numeric(df[column], errors="coerce").astype(float)
    if not p.allow_negative and bool((s < 0).any()):
        return df.copy(), FeatureMapModel(), [
            _issue(
                severity="FAIL",
                message=f"log1p requires non-negative values in '{column}'.",
                evidence={"column": column, "min": float(np.nanmin(s.to_numpy()))},
                fix_hint="Shift/clip negatives or set allow_negative=True.",
            )
        ]

    out = df.copy()
    new_col = _derived_name(column, "log1p")
    out[new_col] = np.log1p(s.to_numpy(dtype=float))
    out = out.drop(columns=[column])

    return out, FeatureMapModel(produced_columns={column: [new_col]}), []


def _encode_standardize(
    *,
    df: pd.DataFrame,
    column: str,
    params: Optional[StandardizeParams],
) -> Tuple[pd.DataFrame, FeatureMapModel, List[ValidationIssueModel]]:
    p = params or StandardizeParams()
    issues: List[ValidationIssueModel] = []

    s = pd.to_numeric(df[column], errors="coerce").astype(float)
    x = s.to_numpy(dtype=float)

    mu = float(np.nanmean(x))
    sigma = float(np.nanstd(x, ddof=p.ddof))

    if not np.isfinite(mu) or not np.isfinite(sigma):
        return df.copy(), FeatureMapModel(), [
            _issue(
                severity="FAIL",
                message=f"standardize: non-finite mean/std for '{column}'.",
                evidence={"column": column, "mean": mu, "std": sigma},
            )
        ]

    if sigma <= p.eps:
        issues.append(
            _issue(
                severity="WARN",
                message=f"standardize: near-zero variance in '{column}', output will be ~0.",
                evidence={"column": column, "std": sigma},
                fix_hint="Consider dropping constant columns.",
            )
        )
        sigma = 1.0

    out = df.copy()
    new_col = _derived_name(column, "z")
    out[new_col] = (x - mu) / sigma
    out = out.drop(columns=[column])

    return out, FeatureMapModel(produced_columns={column: [new_col]}), issues


def _encode_minmax(
    *,
    df: pd.DataFrame,
    column: str,
    params: Optional[MinMaxParams],
) -> Tuple[pd.DataFrame, FeatureMapModel, List[ValidationIssueModel]]:
    p = params or MinMaxParams()
    issues: List[ValidationIssueModel] = []

    s = pd.to_numeric(df[column], errors="coerce").astype(float)
    x = s.to_numpy(dtype=float)

    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    rng = mx - mn

    if not np.isfinite(mn) or not np.isfinite(mx):
        return df.copy(), FeatureMapModel(), [
            _issue(
                severity="FAIL",
                message=f"minmax: non-finite min/max for '{column}'.",
                evidence={"column": column, "min": mn, "max": mx},
            )
        ]

    if rng <= p.eps:
        issues.append(
            _issue(
                severity="WARN",
                message=f"minmax: near-zero range in '{column}', output will be ~0.",
                evidence={"column": column, "min": mn, "max": mx},
                fix_hint="Consider dropping constant columns.",
            )
        )
        rng = 1.0

    out = df.copy()
    new_col = _derived_name(column, "mm")
    out[new_col] = (x - mn) / rng
    out = out.drop(columns=[column])

    return out, FeatureMapModel(produced_columns={column: [new_col]}), issues


def _encode_datetime_to_epoch_seconds(
    *,
    df: pd.DataFrame,
    column: str,
    params: Optional[DateTimeToEpochParams],
) -> Tuple[pd.DataFrame, FeatureMapModel, List[ValidationIssueModel]]:
    p = params or DateTimeToEpochParams()
    out = df.copy()

    new_col = _derived_name(column, f"epoch_{p.unit}")

    try:
        dt = pd.to_datetime(out[column], errors=p.errors, utc=True)

        # int64 nanoseconds since epoch; NaT is min int64
        ns = dt.view("int64").astype("float64")
        ns[ns <= -9e18] = np.nan

        if p.unit == "s":
            out[new_col] = ns / 1e9
        elif p.unit == "ms":
            out[new_col] = ns / 1e6
        elif p.unit == "us":
            out[new_col] = ns / 1e3
        else:
            out[new_col] = ns

        out = out.drop(columns=[column])

    except Exception as e:  # noqa: BLE001
        return df.copy(), FeatureMapModel(), [
            _issue(
                severity="FAIL",
                message=f"datetime_to_epoch_seconds failed for '{column}': {e}",
                evidence={"column": column, "errors": p.errors, "unit": p.unit},
            )
        ]

    return out, FeatureMapModel(produced_columns={column: [new_col]}), []