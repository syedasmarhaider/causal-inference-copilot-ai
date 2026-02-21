from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, NotRequired
from typing import Any, Dict, Optional

from python.implementation.workflows.utils.utils import json_sanitize


# TODO: later convert this to tool
@dataclass(frozen=True)
class ColumnProfileErrorDetails:
    column: Optional[str]
    reason: str
    hint: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None


class DatasetProfilingError(RuntimeError):
    def __init__(self, details: ColumnProfileErrorDetails):
        self.details = details
        msg = details.reason
        if details.column:
            msg = f"Column '{details.column}': {msg}"
        if details.hint:
            msg = f"{msg} Hint: {details.hint}"
        super().__init__(msg)


# =============================================================================
# Typed output schema (discriminated union, NO invalid overrides)
# =============================================================================

InferredKind = Literal["NUMERIC", "DATETIME", "BOOLEAN", "CATEGORICAL", "OTHER"]


class NumericSummary(TypedDict):
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    quantiles: Optional[Dict[str, float]]  # e.g. {"0.05": 1.2, "0.5": 3.4}


class DatetimeSummary(TypedDict):
    min: Optional[str]  # isoformat-ish
    max: Optional[str]


class BooleanSummary(TypedDict):
    counts: Dict[str, int]  # keys are stringified values


class CategoricalSummary(TypedDict):
    top_categories: List[Dict[str, int | str]]  # [{"value": "...", "count": 123}, ...]
    other_count: int


class OtherSummary(TypedDict):
    distinct_values_sample: List[str]


class ColumnProfileCommon(TypedDict):
    # Name is stored here => deterministic list, no duplicated "columns" list needed.
    name: str
    dtype: Optional[str]
    n_rows: int
    n_missing: int
    missing_rate: float
    distinct_count: Optional[int]
    note: NotRequired[str]  # only used in non-strict mode on failures


class NumericColumnProfile(ColumnProfileCommon):
    inferred_kind: Literal["NUMERIC"]
    summary: NumericSummary


class DatetimeColumnProfile(ColumnProfileCommon):
    inferred_kind: Literal["DATETIME"]
    summary: DatetimeSummary


class BooleanColumnProfile(ColumnProfileCommon):
    inferred_kind: Literal["BOOLEAN"]
    summary: BooleanSummary


class CategoricalColumnProfile(ColumnProfileCommon):
    inferred_kind: Literal["CATEGORICAL"]
    summary: CategoricalSummary


class OtherColumnProfile(ColumnProfileCommon):
    inferred_kind: Literal["OTHER"]
    summary: OtherSummary


ColumnProfile = (
    NumericColumnProfile
    | DatetimeColumnProfile
    | BooleanColumnProfile
    | CategoricalColumnProfile
    | OtherColumnProfile
)


class DatasetSummary(TypedDict):
    n_rows: int
    profiles: List[ColumnProfile]  # deterministic order = df.columns order


# =============================================================================
# Public API
# =============================================================================

class DatasetHelpers:
    @staticmethod
    def extract_dataset_summary(
        df: Any,
        *,
        max_categories: int = 30,
        sample_distinct: int = 50,
        compute_quantiles: bool = True,
        strict: bool = True,
    ) -> DatasetSummary:
        _validate_params(max_categories=max_categories, sample_distinct=sample_distinct)

        cols = _get_columns(df, strict=strict)
        n_rows = _safe_n_rows(df, strict=strict)
        dtypes = getattr(df, "dtypes", None)

        profiles: List[ColumnProfile] = []

        for col_key in cols:
            name = str(col_key).strip()
            if not name:
                if strict:
                    raise DatasetProfilingError(
                        ColumnProfileErrorDetails(
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

                base: ColumnProfileCommon = {
                    "name": name,
                    "dtype": dtype_str,
                    "n_rows": n_rows,
                    "n_missing": n_missing,
                    "missing_rate": missing_rate,
                    "distinct_count": distinct,
                }

                if kind == "NUMERIC":
                    numeric_prof: NumericColumnProfile = {
                        **base,
                        "inferred_kind": "NUMERIC",
                        "summary": _numeric_summary(s, compute_quantiles=compute_quantiles),
                    }
                    profiles.append(numeric_prof)

                elif kind == "DATETIME":
                    datetime_prof: DatetimeColumnProfile = {
                        **base,
                        "inferred_kind": "DATETIME",
                        "summary": _datetime_summary(s),
                    }
                    profiles.append(datetime_prof)

                elif kind == "BOOLEAN":
                    boolean_prof: BooleanColumnProfile = {
                        **base,
                        "inferred_kind": "BOOLEAN",
                        "summary": _boolean_summary(s),
                    }
                    profiles.append(boolean_prof)

                elif kind == "CATEGORICAL":
                    categorical_prof: CategoricalColumnProfile = {
                        **base,
                        "inferred_kind": "CATEGORICAL",
                        "summary": _categorical_summary(s, max_categories=max_categories),
                    }
                    profiles.append(categorical_prof)

                else:
                    other_prof: OtherColumnProfile = {
                        **base,
                        "inferred_kind": "OTHER",
                        "summary": _other_summary(s, sample_distinct=sample_distinct),
                    }
                    profiles.append(other_prof)

            except DatasetProfilingError:
                if strict:
                    raise
                # Non-strict: emit a valid profile with a note.
                fallback: OtherColumnProfile = {
                    "name": name,
                    "dtype": _dtype_to_str(dtypes, col_key),
                    "n_rows": n_rows,
                    "n_missing": 0,
                    "missing_rate": 0.0,
                    "distinct_count": None,
                    "inferred_kind": "OTHER",
                    "summary": {"distinct_values_sample": []},
                    "note": "Profiling failed for this column in non-strict mode.",
                }
                profiles.append(fallback)

        if strict and not profiles:
            raise DatasetProfilingError(
                ColumnProfileErrorDetails(
                    column=None,
                    reason="No columns could be profiled.",
                    hint="Verify the dataset is tabular and contains at least one named column.",
                )
            )

        return {"n_rows": n_rows, "profiles": profiles}
    
    @staticmethod
    def dataset_summary_to_json(
        summary: DatasetSummary,
          *,
        indent: int | None = None,
        sort_keys: bool = True,
    ) -> str:
        """
        Always returns STRICT valid JSON (no NaN/Inf).
        Deterministic output by default via sort_keys=True.
        """
        payload = json_sanitize(summary)
        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=indent,
            sort_keys=sort_keys,
            separators=(",", ":") if indent is None else None,
            allow_nan=False,  # enforce strict JSON
        )


# =============================================================================
# Internals
# =============================================================================

def _validate_params(*, max_categories: int, sample_distinct: int) -> None:
    if max_categories <= 0:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None, reason="max_categories must be > 0.", evidence={"max_categories": max_categories}
            )
        )
    if sample_distinct <= 0:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None, reason="sample_distinct must be > 0.", evidence={"sample_distinct": sample_distinct}
            )
        )


def _get_columns(df: Any, *, strict: bool) -> List[Any]:
    raw_cols = getattr(df, "columns", None)
    if raw_cols is None:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
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
            ColumnProfileErrorDetails(
                column=None,
                reason="Could not iterate dataset columns.",
                hint="Ensure df.columns is iterable.",
                evidence={"df_type": type(df).__name__, "error": repr(e)},
            )
        )
    if strict and not cols:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(column=None, reason="Dataset has zero columns.", hint="Provide at least one column.")
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
                ColumnProfileErrorDetails(
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
            ColumnProfileErrorDetails(
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
            ColumnProfileErrorDetails(
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
    # Conservative fallback:
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
            ColumnProfileErrorDetails(
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


def _numeric_summary(series: Any, *, compute_quantiles: bool) -> NumericSummary:
    try:
        s = series.dropna() if hasattr(series, "dropna") else series
        if hasattr(s, "astype"):
            try:
                s = s.astype(float)
            except Exception:
                # If coercion fails, treat as empty numeric summary rather than crashing.
                return {"min": None, "max": None, "mean": None, "std": None, "quantiles": None}

        mn = _as_float_or_none(getattr(s, "min", lambda: None)())
        mx = _as_float_or_none(getattr(s, "max", lambda: None)())
        mean = _as_float_or_none(getattr(s, "mean", lambda: None)())
        std = _as_float_or_none(getattr(s, "std", lambda: None)())

        quantiles: Optional[Dict[str, float]] = None
        if compute_quantiles and hasattr(s, "quantile"):
            qs = [0.05, 0.25, 0.5, 0.75, 0.95]
            qvals = s.quantile(qs)
            # pandas Series has .items()
            items = list(qvals.items()) if hasattr(qvals, "items") else []
            q_out: Dict[str, float] = {}
            for k, v in items:
                fv = _as_float_or_none(v)
                if fv is not None:
                    q_out[str(k)] = fv
            quantiles = q_out or None

        return {"min": mn, "max": mx, "mean": mean, "std": std, "quantiles": quantiles}
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None,
                reason="Numeric summary failed.",
                hint="Verify the column is numeric or coercible to float.",
                evidence={"error": repr(e)},
            )
        )


def _datetime_summary(series: Any) -> DatetimeSummary:
    try:
        s = series.dropna() if hasattr(series, "dropna") else series
        mn = getattr(s, "min", lambda: None)()
        mx = getattr(s, "max", lambda: None)()
        return {"min": _as_datetime_str(mn), "max": _as_datetime_str(mx)}
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None,
                reason="Datetime summary failed.",
                hint="Verify the column is datetime-like.",
                evidence={"error": repr(e)},
            )
        )


def _boolean_summary(series: Any) -> BooleanSummary:
    try:
        if hasattr(series, "value_counts"):
            vc = series.value_counts(dropna=True)
            items = list(vc.items()) if hasattr(vc, "items") else []
            counts: Dict[str, int] = {}
            for k, v in items:
                counts[str(k)] = int(v)
            return {"counts": counts}

        vals = list(series)
        counts = {"True": sum(1 for x in vals if x is True), "False": sum(1 for x in vals if x is False)}
        return {"counts": counts}
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None,
                reason="Boolean summary failed.",
                hint="Verify the column is boolean-like or has value_counts().",
                evidence={"error": repr(e)},
            )
        )


def _categorical_summary(series: Any, *, max_categories: int) -> CategoricalSummary:
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

        top: List[Dict[str, int | str]] = [{"value": str(k), "count": int(v)} for k, v in top_items]
        return {"top_categories": top, "other_count": int(other_count)}
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None,
                reason="Categorical summary failed.",
                hint="Verify the column is categorical/string-like or iterable.",
                evidence={"error": repr(e)},
            )
        )


def _other_summary(series: Any, *, sample_distinct: int) -> OtherSummary:
    try:
        s = series.dropna() if hasattr(series, "dropna") else [x for x in list(series) if x is not None]
        if hasattr(s, "unique"):
            uniq: List[Any] = list(s.unique()) # pyright: ignore[reportUnknownArgumentType, reportAttributeAccessIssue, reportUnknownMemberType]
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
        return {"distinct_values_sample": distinct}
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None,
                reason="Other-type summary failed.",
                hint="Verify the column is iterable or provides .unique().",
                evidence={"error": repr(e)},
            )
        )