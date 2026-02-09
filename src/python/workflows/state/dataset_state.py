from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from uuid import UUID

from python.workflows.utils.types import JSONDict


class DatasetState(TypedDict, total=False):
    """
    total=False because early stages only have `path`,
    later stages enrich with id/schema/summary.
    """
    id: Optional[UUID]
    raw_schema: Optional[JSONDict]
    summary: Optional[Dict[str, Dict[str, Any]]]
    load_error: Optional[str]
    get_file_last_user_msg_idx: Optional[int]


# ---------------------------
# Exceptions (user-actionable)
# ---------------------------

@dataclass(frozen=True)
class ColumnProfileErrorDetails:
    column: Optional[str]
    reason: str
    hint: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = None


class DatasetProfilingError(RuntimeError):
    """
    Raise this when dataset profiling cannot proceed and user action is needed.
    Keep it user-actionable: reason + hint + evidence.
    """

    def __init__(self, details: ColumnProfileErrorDetails):
        self.details = details
        msg = details.reason
        if details.column:
            msg = f"Column '{details.column}': {msg}"
        if details.hint:
            msg = f"{msg} Hint: {details.hint}"
        super().__init__(msg)


# ---------------------------
# Public API
# ---------------------------

class DatasetStateHelpers:
    @staticmethod
    def extract_column_profile(
        df: Any,
        *,
        max_categories: int = 30,
        sample_distinct: int = 50,
        compute_quantiles: bool = True,
        strict: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a type-aware profile for each column.

        Returns:
          Dict[column_name, profile]

        profile includes:
          - dtype (string | None)
          - inferred_kind: NUMERIC|CATEGORICAL|DATETIME|BOOLEAN|OTHER
          - n_rows, n_missing, missing_rate, distinct_count
          - summary (type-specific)

        Error policy:
          - strict=True (default): raise DatasetProfilingError on any hard failure
          - strict=False: best-effort; column failures become per-column "summary.note"
        """
        _validate_params(max_categories=max_categories, sample_distinct=sample_distinct)

        cols = _get_columns(df, strict=strict)
        n_rows = _safe_n_rows(df, strict=strict)
        dtypes = getattr(df, "dtypes", None)

        out: Dict[str, Dict[str, Any]] = {}

        for col in cols:
            col_name = str(col).strip()
            if not col_name:
                # empty column label is a real schema issue; raise in strict mode
                if strict:
                    raise DatasetProfilingError(
                        ColumnProfileErrorDetails(
                            column=None,
                            reason="Dataset contains an empty column name.",
                            hint="Rename the column to a non-empty string.",
                            evidence={"raw_column": repr(col)},
                        )
                    )
                continue

            try:
                series = _get_series(df, col, col_name, strict=strict)
                dtype_str = _dtype_to_str(dtypes, col)
                kind = _infer_kind(series, dtype_str)

                n_missing, missing_rate = _missingness(series, n_rows=n_rows, strict=strict)
                distinct_count = _distinct_count(series)

                base: Dict[str, Any] = {
                    "dtype": dtype_str,
                    "inferred_kind": kind,
                    "n_rows": n_rows,
                    "n_missing": n_missing,
                    "missing_rate": missing_rate,
                    "distinct_count": distinct_count,
                }

                if kind == "NUMERIC":
                    base["summary"] = _numeric_summary(series, compute_quantiles=compute_quantiles, strict=strict)
                elif kind == "DATETIME":
                    base["summary"] = _datetime_summary(series, strict=strict)
                elif kind == "BOOLEAN":
                    base["summary"] = _boolean_summary(series, strict=strict)
                elif kind == "CATEGORICAL":
                    base["summary"] = _categorical_summary(series, max_categories=max_categories, strict=strict)
                else:
                    base["summary"] = _other_summary(series, sample_distinct=sample_distinct, strict=strict)

                out[col_name] = base

            except DatasetProfilingError:
                # strict mode should bubble; non-strict should convert to note
                if strict:
                    raise
                out[col_name] = {
                    "dtype": _dtype_to_str(dtypes, col),
                    "inferred_kind": "OTHER",
                    "n_rows": n_rows,
                    "n_missing": None,
                    "missing_rate": None,
                    "distinct_count": None,
                    "summary": {"note": "Profiling failed for this column; continue in non-strict mode."},
                }

        if strict and not out:
            raise DatasetProfilingError(
                ColumnProfileErrorDetails(
                    column=None,
                    reason="No columns could be profiled.",
                    hint="Verify the dataset is tabular and contains at least one named column.",
                )
            )

        return out


# ---------------------------
# Validation / extraction
# ---------------------------

def _validate_params(*, max_categories: int, sample_distinct: int) -> None:
    if max_categories <= 0:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(column=None, reason="max_categories must be > 0.", evidence={"max_categories": max_categories})
        )
    if sample_distinct <= 0:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(column=None, reason="sample_distinct must be > 0.", evidence={"sample_distinct": sample_distinct})
        )


def _get_columns(df: Any, *, strict: bool) -> List[Any]:
    raw_cols = getattr(df, "columns", None)
    if raw_cols is None:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None,
                reason="Dataset object has no 'columns' attribute; not a DataFrame-like table.",
                hint="Load the dataset into a pandas DataFrame (or provide an object that exposes .columns and supports df[col]).",
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

    if strict and len(cols) == 0:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None,
                reason="Dataset has zero columns.",
                hint="Provide a tabular dataset with at least one column.",
            )
        )

    return cols


def _safe_n_rows(df: Any, *, strict: bool) -> int:
    # Prefer pandas-like df.shape[0]
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
                    reason="Could not read dataset row count (df.shape[0]).",
                    hint="Ensure df.shape is available and valid (pandas DataFrame recommended).",
                    evidence={"df_type": type(df).__name__, "error": repr(e)},
                )
            )

    # Fallback to len(df)
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
    # Must support __getitem__ access: df[col]
    try:
        return df[col_key]
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=col_name,
                reason="Could not access column data via df[col].",
                hint="Ensure the dataset is a DataFrame-like object with column access (pandas DataFrame recommended).",
                evidence={"col_key": repr(col_key), "df_type": type(df).__name__, "error": repr(e)},
            )
        )


def _dtype_to_str(dtypes: Any, col: Any) -> Optional[str]:
    try:
        if dtypes is None:
            return None
        return str(dtypes[col])
    except Exception:
        return None


# ---------------------------
# Type inference
# ---------------------------

def _infer_kind(series: Any, dtype_str: Optional[str]) -> str:
    ds = (dtype_str or "").lower()
    if "datetime" in ds or "date" in ds or "timestamp" in ds:
        return "DATETIME"
    if "bool" in ds:
        return "BOOLEAN"
    if any(x in ds for x in ("int", "float", "double", "numeric", "decimal")):
        return "NUMERIC"
    if any(x in ds for x in ("object", "string", "category")):
        return "CATEGORICAL"

    # Minimal fallback inference (not "heuristics about user intent"; just dtype class)
    try:
        s = series.dropna() if hasattr(series, "dropna") else series
        # empty -> OTHER
        if hasattr(s, "empty") and bool(getattr(s, "empty")):
            return "OTHER"
        # try numeric cast
        if hasattr(s, "astype"):
            try:
                s.astype(float)
                return "NUMERIC"
            except Exception:
                return "CATEGORICAL"
        return "OTHER"
    except Exception:
        return "OTHER"


# ---------------------------
# Missingness / distinct
# ---------------------------

def _missingness(series: Any, *, n_rows: int, strict: bool) -> Tuple[int, float]:
    if n_rows < 0:
        if strict:
            raise DatasetProfilingError(ColumnProfileErrorDetails(column=None, reason="Invalid row count.", evidence={"n_rows": n_rows}))
        return 0, 0.0

    try:
        if hasattr(series, "isna"):
            n_missing = int(series.isna().sum())
        elif hasattr(series, "isnull"):
            n_missing = int(series.isnull().sum())
        else:
            # Non-pandas fallback: treat None as missing
            vals = list(series)
            n_missing = sum(1 for x in vals if x is None)
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None,
                reason="Could not compute missingness for a column.",
                hint="Ensure the column supports isna()/isnull() or is iterable.",
                evidence={"error": repr(e)},
            )
        )

    missing_rate = (n_missing / n_rows) if n_rows > 0 else 0.0
    return n_missing, float(missing_rate)


def _distinct_count(series: Any) -> Optional[int]:
    try:
        if hasattr(series, "nunique"):
            return int(series.nunique(dropna=True))
    except Exception:
        return None
    return None


# ---------------------------
# Summaries
# ---------------------------

def _numeric_summary(series: Any, *, compute_quantiles: bool, strict: bool) -> Dict[str, Any]:
    try:
        s = series.dropna() if hasattr(series, "dropna") else series

        # coerce to float if possible (pandas)
        if hasattr(s, "astype"):
            s = s.astype(float)

        # empty
        if hasattr(s, "empty") and bool(getattr(s, "empty")):
            return {"min": None, "max": None, "mean": None, "std": None, "quantiles": None}

        out: Dict[str, Any] = {
            "min": _safe_scalar(getattr(s, "min", lambda: None)()),
            "max": _safe_scalar(getattr(s, "max", lambda: None)()),
            "mean": _safe_scalar(getattr(s, "mean", lambda: None)()),
            "std": _safe_scalar(getattr(s, "std", lambda: None)()),
        }

        if compute_quantiles:
            if not hasattr(s, "quantile"):
                if strict:
                    raise DatasetProfilingError(
                        ColumnProfileErrorDetails(column=None, reason="Numeric quantiles requested but column has no .quantile().")
                    )
                out["quantiles"] = None
            else:
                qs = [0.05, 0.25, 0.5, 0.75, 0.95]
                qvals = s.quantile(qs)
                out["quantiles"] = {str(k): _safe_scalar(v) for k, v in _iter_items(qvals)} # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        else:
            out["quantiles"] = None

        return out
    except DatasetProfilingError:
        raise
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None,
                reason="Numeric summary failed.",
                hint="Verify the column is numeric or coercible to float.",
                evidence={"error": repr(e)},
            )
        )


def _datetime_summary(series: Any, *, strict: bool) -> Dict[str, Any]:
    try:
        s = series.dropna() if hasattr(series, "dropna") else series
        if hasattr(s, "empty") and bool(getattr(s, "empty")):
            return {"min": None, "max": None}
        mn = getattr(s, "min", lambda: None)()
        mx = getattr(s, "max", lambda: None)()
        return {"min": _safe_datetime_str(mn), "max": _safe_datetime_str(mx)}
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None,
                reason="Datetime summary failed.",
                hint="Verify the column is a datetime type or parseable to datetime.",
                evidence={"error": repr(e)},
            )
        )


def _boolean_summary(series: Any, *, strict: bool) -> Dict[str, Any]:
    try:
        if hasattr(series, "value_counts"):
            vc = series.value_counts(dropna=True)
            counts = {str(k): int(v) for k, v in _iter_items(vc)} # type: ignore
            return {"counts": counts}

        # Non-pandas fallback: iterate values
        vals = list(series)
        return {
            "counts": {
                "True": sum(1 for x in vals if x is True),
                "False": sum(1 for x in vals if x is False),
            }
        }
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None,
                reason="Boolean summary failed.",
                hint="Verify the column is boolean or has value_counts().",
                evidence={"error": repr(e)},
            )
        )


def _categorical_summary(series: Any, *, max_categories: int, strict: bool) -> Dict[str, Any]:
    try:
        if hasattr(series, "value_counts"):
            vc = series.value_counts(dropna=True)
            items = [(str(k), int(v)) for k, v in _iter_items(vc)] # type: ignore
        else:
            vals = [x for x in list(series) if x is not None]
            freq: Dict[str, int] = {}
            for x in vals:
                key = str(x)
                freq[key] = freq.get(key, 0) + 1
            items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)

        top = items[:max_categories]
        other_count = sum(v for _, v in items[max_categories:]) if len(items) > max_categories else 0

        return {
            "top_categories": [{"value": k, "count": v} for k, v in top],
            "other_count": int(other_count),
        }
    except Exception as e:
        raise DatasetProfilingError(
            ColumnProfileErrorDetails(
                column=None,
                reason="Categorical summary failed.",
                hint="Verify the column is categorical/string-like or iterable.",
                evidence={"error": repr(e)},
            )
        )


def _other_summary(series: Any, *, sample_distinct: int, strict: bool) -> Dict[str, Any]:
    try:
        s = series.dropna() if hasattr(series, "dropna") else [x for x in list(series) if x is not None]

        if hasattr(s, "unique"):
            uniq = s.unique()  # pyright: ignore[reportUnknownVariableType, reportAttributeAccessIssue, reportUnknownMemberType]
            distinct = [str(x) for x in list(uniq)[:sample_distinct]] # type: ignore
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


# ---------------------------
# Serialization helpers
# ---------------------------

def _safe_scalar(v: Any) -> Any:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float, str, bool)):
            # normalize NaN/Inf to strings for JSON stability if needed
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return str(v)
            return v
        if hasattr(v, "item"):
            vv = v.item()
            return _safe_scalar(vv)
        return str(v)
    except Exception:
        return None


def _safe_datetime_str(v: Any) -> Optional[str]:
    try:
        if v is None:
            return None
        if hasattr(v, "isoformat"):
            return str(v.isoformat())
        return str(v)
    except Exception:
        return None


def _iter_items(obj: Any): # pyright: ignore[reportUnknownParameterType]
    if isinstance(obj, dict):
        return obj.items() # pyright: ignore[reportUnknownVariableType]
    if hasattr(obj, "items"):
        return obj.items()
    try:
        return list(obj)
    except Exception:
        return [] # pyright: ignore[reportUnknownVariableType]
