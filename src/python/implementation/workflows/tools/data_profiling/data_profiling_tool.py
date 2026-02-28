from __future__ import annotations

import json
import math
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Errors (structured + parseable)
# =============================================================================


class ColumnProfileErrorDetailsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: Optional[str] = None
    reason: str
    hint: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None


class DatasetProfilingError(RuntimeError):
    def __init__(self, details: ColumnProfileErrorDetailsModel):
        self.details = details
        msg = details.reason
        if details.column:
            msg = f"Column '{details.column}': {msg}"
        if details.hint:
            msg = f"{msg} Hint: {details.hint}"
        super().__init__(msg)


# =============================================================================
# Pydantic output schema (discriminated union)
# =============================================================================

InferredKind = Literal["NUMERIC", "DATETIME", "BOOLEAN", "CATEGORICAL", "OTHER"]


class NumericSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    quantiles: Optional[Dict[str, float]] = None  # {"0.05": 1.2, ...}


class DatetimeSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min: Optional[str] = None  # isoformat-ish
    max: Optional[str] = None


class BooleanSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counts: Dict[str, int]  # keys are stringified values


class CategoryCountModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    count: int


class CategoricalSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_categories: List[CategoryCountModel]
    other_count: int


class OtherSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distinct_values_sample: List[str]


class ColumnProfileCommonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str
    dtype: Optional[str] = None
    n_rows: int
    n_missing: int
    missing_rate: float
    distinct_count: Optional[int] = None
    note: Optional[str] = None  # only used in non-strict mode fallbacks


class NumericColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["NUMERIC"]
    summary: NumericSummaryModel


class DatetimeColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["DATETIME"]
    summary: DatetimeSummaryModel


class BooleanColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["BOOLEAN"]
    summary: BooleanSummaryModel


class CategoricalColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["CATEGORICAL"]
    summary: CategoricalSummaryModel


class OtherColumnProfileModel(ColumnProfileCommonModel):
    inferred_kind: Literal["OTHER"]
    summary: OtherSummaryModel


ColumnProfileModel = Union[
    NumericColumnProfileModel,
    DatetimeColumnProfileModel,
    BooleanColumnProfileModel,
    CategoricalColumnProfileModel,
    OtherColumnProfileModel,
]

# Discriminator annotation (pydantic v2)
DiscriminatedColumnProfile = Annotated[ColumnProfileModel, Field(discriminator="inferred_kind")]


class DatasetSummaryModel(BaseModel):
    """
    Deterministic order: profiles follow df.columns order.
    """
    model_config = ConfigDict(extra="forbid")

    n_rows: int
    profiles: List[DiscriminatedColumnProfile] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]


# =============================================================================
# Public API (STATE-style profiling tool; no constructor; pure funcs)
# =============================================================================

class DatasetProfilingStateTool:
    """
    "State tool" API:
      - extract_dataset_summary(df, ...) -> DatasetSummaryModel
      - dataset_summary_to_json(summary, ...) -> strict JSON (no NaN/Inf)
      - dataset_summary_from_json(payload) -> DatasetSummaryModel
    """
    

    def get_tool_name(self) -> str:
        return "DATA_PROFILING_TOOL"
    


    def extract_dataset_summary(
        self,
        df: pd.DataFrame,
        *,
        max_categories: int = 50,
        sample_distinct: int = 50,
        compute_quantiles: bool = True,
        strict: bool = True,
    ) -> DatasetSummaryModel:
        _validate_params(max_categories=max_categories, sample_distinct=sample_distinct)

        cols = _get_columns(df, strict=strict)
        n_rows = _safe_n_rows(df, strict=strict)
        dtypes = getattr(df, "dtypes", None)

        profiles: List[DiscriminatedColumnProfile] = []

        for col_key in cols:
            name = str(col_key).strip()
            if not name:
                if strict:
                    raise DatasetProfilingError(
                        ColumnProfileErrorDetailsModel(
                            column=None,
                            reason="Dataset contains an empty column name.",
                            hint="Rename the column to a non-empty string.",
                            evidence={"raw_column": repr(col_key)},
                        )
                    )
                continue

            try:
                s = _get_series(df, col_key, name, strict=strict)
                dtype_str = _dtype_to_str(dtypes, col_key)
                kind = _infer_kind(s, dtype_str)

                n_missing, missing_rate = _missingness(s, n_rows=n_rows)
                distinct = _distinct_count(s)

                base = ColumnProfileCommonModel(
                    name=name,
                    dtype=dtype_str,
                    n_rows=n_rows,
                    n_missing=n_missing,
                    missing_rate=missing_rate,
                    distinct_count=distinct,
                )

                if kind == "NUMERIC":
                    profiles.append(
                        NumericColumnProfileModel(
                            **base.model_dump(),
                            inferred_kind="NUMERIC",
                            summary=_numeric_summary(s, compute_quantiles=compute_quantiles),
                        )
                    )
                elif kind == "DATETIME":
                    profiles.append(
                        DatetimeColumnProfileModel(
                            **base.model_dump(),
                            inferred_kind="DATETIME",
                            summary=_datetime_summary(s),
                        )
                    )
                elif kind == "BOOLEAN":
                    profiles.append(
                        BooleanColumnProfileModel(
                            **base.model_dump(),
                            inferred_kind="BOOLEAN",
                            summary=_boolean_summary(s),
                        )
                    )
                elif kind == "CATEGORICAL":
                    profiles.append(
                        CategoricalColumnProfileModel(
                            **base.model_dump(),
                            inferred_kind="CATEGORICAL",
                            summary=_categorical_summary(s, max_categories=max_categories),
                        )
                    )
                else:
                    profiles.append(
                        OtherColumnProfileModel(
                            **base.model_dump(),
                            inferred_kind="OTHER",
                            summary=_other_summary(s, sample_distinct=sample_distinct),
                        )
                    )

            except DatasetProfilingError:
                if strict:
                    raise
                # Non-strict fallback: still a VALID profile
                profiles.append(
                    OtherColumnProfileModel(
                        name=name,
                        dtype=_dtype_to_str(dtypes, col_key),
                        n_rows=n_rows,
                        n_missing=0,
                        missing_rate=0.0,
                        distinct_count=None,
                        inferred_kind="OTHER",
                        summary=OtherSummaryModel(distinct_values_sample=[]),
                        note="Profiling failed for this column in non-strict mode.",
                    )
                )

        if strict and not profiles:
            raise DatasetProfilingError(
                ColumnProfileErrorDetailsModel(
                    column=None,
                    reason="No columns could be profiled.",
                    hint="Verify the dataset is tabular and contains at least one named column.",
                )
            )

        return DatasetSummaryModel(n_rows=n_rows, profiles=profiles)

    def dataset_summary_to_json(self,
        summary: DatasetSummaryModel,
        *,
        indent: int | None = None,
    ) -> str:
        return summary.model_dump_json(indent=indent)

    def dataset_summary_from_json(self, payload: str) -> DatasetSummaryModel:
        data = json.loads(payload)
        return DatasetSummaryModel.model_validate(data)


# =============================================================================
# Internals (logic preserved from your reference)
# =============================================================================

def _validate_params(*, max_categories: int, sample_distinct: int) -> None:
    if max_categories <= 0:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="max_categories must be > 0.",
                evidence={"max_categories": max_categories},
            )
        )
    if sample_distinct <= 0:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="sample_distinct must be > 0.",
                evidence={"sample_distinct": sample_distinct},
            )
        )


def _get_columns(df: Any, *, strict: bool) -> List[Any]:
    raw_cols = getattr(df, "columns", None)
    if raw_cols is None:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Dataset object has no 'columns' attribute; not a DataFrame-like table.",
                hint="Use a pandas DataFrame (recommended) or provide an object exposing .columns and df[col].",
                evidence={"df_type": type(df).__name__},
            )
        )
    try:
        cols = list(raw_cols)
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Could not iterate dataset columns.",
                hint="Ensure df.columns is iterable.",
                evidence={"df_type": type(df).__name__, "error": repr(e)},
            )
        )
    if strict and not cols:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Dataset has zero columns.",
                hint="Provide at least one column.",
            )
        )
    return cols


def _safe_n_rows(df: Any, *, strict: bool) -> int:
    try:
        shape = getattr(df, "shape", None)
        if shape is not None and len(shape) >= 1:
            n = int(shape[0])
            if n < 0:
                raise ValueError("negative row count")
            return n
    except Exception as e:
        if strict:
            raise DatasetProfilingError(
                ColumnProfileErrorDetailsModel(
                    column=None,
                    reason="Could not read df.shape[0].",
                    hint="Ensure df.shape is valid (pandas DataFrame recommended).",
                    evidence={"df_type": type(df).__name__, "error": repr(e)},
                )
            )
    try:
        n = int(len(df))
        if n < 0:
            raise ValueError("negative row count")
        return n
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Could not determine dataset row count.",
                hint="Ensure df implements __len__ or provides df.shape.",
                evidence={"df_type": type(df).__name__, "error": repr(e)},
            )
        )


def _get_series(df: Any, col_key: Any, col_name: str, *, strict: bool) -> Any:
    try:
        return df[col_key]
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=col_name,
                reason="Could not access column via df[col].",
                hint="Verify the column exists and df supports __getitem__ (pandas DataFrame recommended).",
                evidence={"col_key": repr(col_key), "df_type": type(df).__name__, "error": repr(e)},
            )
        )


def _dtype_to_str(dtypes: Any, col_key: Any) -> Optional[str]:
    try:
        if dtypes is None:
            return None
        return str(dtypes[col_key])
    except Exception:
        return None


def _infer_kind(series: Any, dtype_str: Optional[str]) -> InferredKind:
    ds = (dtype_str or "").lower()
    if "datetime" in ds or "date" in ds or "timestamp" in ds:
        return "DATETIME"
    if "bool" in ds:
        return "BOOLEAN"
    if any(x in ds for x in ("int", "float", "double", "numeric", "decimal")):
        return "NUMERIC"
    if any(x in ds for x in ("object", "string", "category")):
        return "CATEGORICAL"
    return "OTHER"


def _missingness(series: Any, *, n_rows: int) -> Tuple[int, float]:
    try:
        if hasattr(series, "isna"):
            n_missing = int(series.isna().sum())
        elif hasattr(series, "isnull"):
            n_missing = int(series.isnull().sum())
        else:
            vals = list(series)
            n_missing = sum(1 for x in vals if x is None)
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Could not compute missingness.",
                hint="Ensure the column supports isna()/isnull() or is iterable.",
                evidence={"error": repr(e)},
            )
        )

    rate = (n_missing / n_rows) if n_rows > 0 else 0.0
    return n_missing, float(rate)


def _distinct_count(series: Any) -> Optional[int]:
    try:
        if hasattr(series, "nunique"):
            return int(series.nunique(dropna=True))
    except Exception:
        return None
    return None


def _as_float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        fv = float(v)
        if not math.isfinite(fv):
            return None
        return fv
    except Exception:
        return None


def _as_datetime_str(v: Any) -> Optional[str]:
    try:
        if v is None:
            return None
        if hasattr(v, "isoformat"):
            return str(v.isoformat())
        return str(v)
    except Exception:
        return None


def _numeric_summary(series: Any, *, compute_quantiles: bool) -> NumericSummaryModel:
    try:
        s = series.dropna() if hasattr(series, "dropna") else series

        if hasattr(s, "astype"):
            try:
                s = s.astype(float)
            except Exception:
                # Preserve your old behavior: don't crash; emit empty numeric stats.
                return NumericSummaryModel()

        mn = _as_float_or_none(getattr(s, "min", lambda: None)())
        mx = _as_float_or_none(getattr(s, "max", lambda: None)())
        mean = _as_float_or_none(getattr(s, "mean", lambda: None)())
        std = _as_float_or_none(getattr(s, "std", lambda: None)())

        quantiles: Optional[Dict[str, float]] = None
        if compute_quantiles and hasattr(s, "quantile"):
            qs = [0.05, 0.25, 0.5, 0.75, 0.95]
            qvals = s.quantile(qs)
            items = list(qvals.items()) if hasattr(qvals, "items") else []

            out: Dict[str, float] = {}
            for k, v in items:
                fv = _as_float_or_none(v)
                if fv is not None:
                    out[str(k)] = fv
            quantiles = out or None

        return NumericSummaryModel(min=mn, max=mx, mean=mean, std=std, quantiles=quantiles)
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Numeric summary failed.",
                hint="Verify the column is numeric or coercible to float.",
                evidence={"error": repr(e)},
            )
        )


def _datetime_summary(series: Any) -> DatetimeSummaryModel:
    try:
        s = series.dropna() if hasattr(series, "dropna") else series
        mn = getattr(s, "min", lambda: None)()
        mx = getattr(s, "max", lambda: None)()
        return DatetimeSummaryModel(min=_as_datetime_str(mn), max=_as_datetime_str(mx))
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Datetime summary failed.",
                hint="Verify the column is datetime-like.",
                evidence={"error": repr(e)},
            )
        )


def _boolean_summary(series: Any) -> BooleanSummaryModel:
    try:
        if hasattr(series, "value_counts"):
            vc = series.value_counts(dropna=True)
            items = list(vc.items()) if hasattr(vc, "items") else []
            counts: Dict[str, int] = {}
            for k, v in items:
                counts[str(k)] = int(v)
            return BooleanSummaryModel(counts=counts)

        vals = list(series)
        counts = {
            "True": sum(1 for x in vals if x is True),
            "False": sum(1 for x in vals if x is False),
        }
        return BooleanSummaryModel(counts=counts)
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Boolean summary failed.",
                hint="Verify the column is boolean-like or has value_counts().",
                evidence={"error": repr(e)},
            )
        )


def _categorical_summary(series: Any, *, max_categories: int) -> CategoricalSummaryModel:
    try:
        if hasattr(series, "value_counts"):
            vc = series.value_counts(dropna=True)
            items = list(vc.items()) if hasattr(vc, "items") else []
        else:
            vals = [x for x in list(series) if x is not None]
            freq: Dict[str, int] = {}
            for x in vals:
                sx = str(x)
                freq[sx] = freq.get(sx, 0) + 1
            items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)

        top_items = items[:max_categories]
        other_count = sum(int(v) for _, v in items[max_categories:]) if len(items) > max_categories else 0

        top = [CategoryCountModel(value=str(k), count=int(v)) for k, v in top_items]
        return CategoricalSummaryModel(top_categories=top, other_count=int(other_count))
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Categorical summary failed.",
                hint="Verify the column is categorical/string-like or iterable.",
                evidence={"error": repr(e)},
            )
        )


def _other_summary(series: Any, *, sample_distinct: int) -> OtherSummaryModel:
    try:
        s = series.dropna() if hasattr(series, "dropna") else [x for x in list(series) if x is not None]

        if hasattr(s, "unique"):
            uniq: List[Any] = list(s.unique()) # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownArgumentType]
            distinct = [str(x) for x in uniq[:sample_distinct]]
        else:
            seen: List[str] = []
            for x in list(s):
                sx = str(x)
                if sx not in seen:
                    seen.append(sx)
                if len(seen) >= sample_distinct:
                    break
            distinct = seen

        return OtherSummaryModel(distinct_values_sample=distinct)
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetailsModel(
                column=None,
                reason="Other-type summary failed.",
                hint="Verify the column is iterable or provides .unique().",
                evidence={"error": repr(e)},
            )
        )