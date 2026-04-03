from __future__ import annotations

import json
import math
from typing import  Any, ClassVar, Literal

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)
from pydantic import BaseModel, ConfigDict

from python.domain.workflows.tool import Tool
from python.implementation.workflows.tools.common.model.data_summary import (
    BooleanColumnProfileModel,
    BooleanSummaryModel,
    CategoricalColumnProfileModel,
    CategoricalSummaryModel,
    CategoryCountModel,
    ColumnProfileCommonModel,
    DatasetSummaryModel,
    DatetimeColumnProfileModel,
    DatetimeSummaryModel,
    DiscriminatedColumnProfile,
    NumericColumnProfileModel,
    NumericSummaryModel,
    OtherColumnProfileModel,
    OtherSummaryModel,
)
InferredKind = Literal["NUMERIC", "DATETIME", "BOOLEAN", "CATEGORICAL", "OTHER"]

# =============================================================================
# Errors (structured + parseable)
# =============================================================================


class ColumnProfileErrorDetailsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: str | None = None
    reason: str
    hint: str | None = None
    evidence: dict[str, Any] | None = None


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
# Public output contract
# =============================================================================
class DatasetProfilingTool(Tool):
    NAME: ClassVar[str] = "DATA_PROFILING"
    def get_tool_name(self) -> str:
        return self.NAME
    
    def get_tool_info(self) -> str:
        return (
            "Tool for profiling tabular datasets. Analyzes each column to determine data types, missingness, distinct counts, and provides summaries such as numeric stats, category frequencies, and sample distinct values. "
            "Designed to handle a variety of DataFrame-like inputs with robust error handling and informative reporting."
        )
    
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

        profiles: list[DiscriminatedColumnProfile] = []

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
                kind = _infer_kind(s)

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


def _get_columns(df: Any, *, strict: bool) -> list[Any]:
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
        ) from e
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
            ) from e
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
        ) from e


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
        ) from e


def _dtype_to_str(dtypes: Any, col_key: Any) -> str | None:
    try:
        if dtypes is None:
            return None
        return str(dtypes[col_key])
    except Exception:
        return None


def _missingness(series: Any, *, n_rows: int) -> tuple[int, float]:
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
        ) from e

    rate = (n_missing / n_rows) if n_rows > 0 else 0.0
    return n_missing, float(rate)


def _distinct_count(series: Any) -> int | None:
    try:
        if hasattr(series, "nunique"):
            return int(series.nunique(dropna=True))
    except Exception:
        return None
    return None


def _as_float_or_none(v: Any) -> float | None:
    try:
        if v is None:
            return None
        fv = float(v)
        if not math.isfinite(fv):
            return None
        return fv
    except Exception:
        return None


def _as_datetime_str(v: Any) -> str | None:
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

        quantiles: dict[str, float] | None = None
        if compute_quantiles and hasattr(s, "quantile"):
            qs = [0.05, 0.25, 0.5, 0.75, 0.95]
            qvals = s.quantile(qs)
            items = list(qvals.items()) if hasattr(qvals, "items") else []

            out: dict[str, float] = {}
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
        ) from e


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
        ) from e


def _boolean_summary(series: Any) -> BooleanSummaryModel:
    try:
        if hasattr(series, "value_counts"):
            vc = series.value_counts(dropna=True)
            items = list(vc.items()) if hasattr(vc, "items") else []
            counts: dict[str, int] = {}
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
        ) from e


def _categorical_summary(series: Any, *, max_categories: int) -> CategoricalSummaryModel:
    try:
        if hasattr(series, "value_counts"):
            vc = series.value_counts(dropna=True)
            items = list(vc.items()) if hasattr(vc, "items") else []
        else:
            vals = [x for x in list(series) if x is not None]
            freq: dict[str, int] = {}
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
        ) from e


def _other_summary(series: Any, *, sample_distinct: int) -> OtherSummaryModel:
    try:
        s = series.dropna() if hasattr(series, "dropna") else [x for x in list(series) if x is not None]

        if hasattr(s, "unique"):
            try:
                uniq: list[Any] = list(s.unique()) # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownArgumentType]
                distinct = [str(x) for x in uniq[:sample_distinct]]
            except Exception:
                seen: list[str] = []
                for x in list(s):
                    sx = str(x)
                    if sx not in seen:
                        seen.append(sx)
                    if len(seen) >= sample_distinct:
                        break
                distinct = seen
        else:
            seen: list[str] = []
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
        ) from e
        
def _infer_kind(
    series: pd.Series,
    *,
    numeric_coerce_threshold: float = 0.90,
    datetime_coerce_threshold: float = 0.90,
    sample_size: int = 2000,
) -> InferredKind:
    # 1) Fast path: real dtypes
    if is_datetime64_any_dtype(series):
        return "DATETIME"
    if is_bool_dtype(series):
        return "BOOLEAN"
    if is_numeric_dtype(series):
        return "NUMERIC"

    # 2) Sample non-missing for heuristics (deterministic)
    s = series.dropna()
    if s.empty:
        return "OTHER"

    if len(s) > sample_size:
        s = s.sample(n=sample_size, random_state=0)

    # 3) Detect complex objects -> OTHER
    if s.map(lambda x: isinstance(x, (dict, list, set, tuple))).any():
        return "OTHER"

    # 4) Boolean-like strings
    low = s.astype(str).str.strip().str.lower()
    bool_vocab = {"true", "false", "0", "1", "yes", "no", "y", "n"}
    if float(low.isin(bool_vocab).mean()) >= 0.95:
        return "BOOLEAN"

    # 5) Numeric-like strings
    num = pd.to_numeric(s, errors="coerce")
    if float(num.notna().mean()) >= numeric_coerce_threshold:
        return "NUMERIC"

    # 6) Datetime-like strings
    dt = pd.to_datetime(s, errors="coerce", format="mixed")
    if float(dt.notna().mean()) >= datetime_coerce_threshold:
        return "DATETIME"

    return "CATEGORICAL"
