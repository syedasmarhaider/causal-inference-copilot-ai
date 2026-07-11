from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

EFFECT_ROW_COLUMN = "effect_row"
CATE_COLUMN = "cate"
CATE_LOWER_COLUMN = "cate_lower"
CATE_UPPER_COLUMN = "cate_upper"
CATE_REVERSE_COLUMN = "cate_reverse"
CATE_REVERSE_LOWER_COLUMN = "cate_reverse_lower"
CATE_REVERSE_UPPER_COLUMN = "cate_reverse_upper"
CATE_T0_COLUMN = "cate_t0"
CATE_T1_COLUMN = "cate_t1"


def build_all_row_cate_dataframe(
    *,
    dataframe: pd.DataFrame,
    cate_values: np.ndarray,
    lower_values: np.ndarray | None,
    upper_values: np.ndarray | None,
    for_treatment: Any = None,
) -> pd.DataFrame:
    query_df = dataframe.reset_index(drop=True).copy()
    query_df[EFFECT_ROW_COLUMN] = np.arange(1, len(query_df) + 1, dtype=int)
    query_df[CATE_COLUMN] = cate_values.astype(float, copy=False)
    query_df[CATE_LOWER_COLUMN] = _aligned_interval_column(
        interval_values=lower_values,
        length=len(query_df),
    )
    query_df[CATE_UPPER_COLUMN] = _aligned_interval_column(
        interval_values=upper_values,
        length=len(query_df),
    )
    query_df[CATE_REVERSE_COLUMN] = -query_df[CATE_COLUMN]
    if lower_values is not None and upper_values is not None:
        aligned_lower = _aligned_interval_column(interval_values=lower_values, length=len(query_df))
        aligned_upper = _aligned_interval_column(interval_values=upper_values, length=len(query_df))
        query_df[CATE_REVERSE_LOWER_COLUMN] = -aligned_upper
        query_df[CATE_REVERSE_UPPER_COLUMN] = -aligned_lower
    else:
        query_df[CATE_REVERSE_LOWER_COLUMN] = np.full(len(query_df), np.nan, dtype=float)
        query_df[CATE_REVERSE_UPPER_COLUMN] = np.full(len(query_df), np.nan, dtype=float)

    treatment_contrast = normalize_treatment_contrast(for_treatment)
    query_df[CATE_T0_COLUMN] = treatment_contrast.get("t0", np.nan)
    query_df[CATE_T1_COLUMN] = treatment_contrast.get("t1", np.nan)
    return query_df


def summarize_all_row_cate_dataframe(
    *,
    dataframe: pd.DataFrame,
    dataset_id: Any,
    effect_modifier_columns: Sequence[str],
    for_treatment: Any = None,
) -> dict[str, Any]:
    cate_values = pd.to_numeric(dataframe.get(CATE_COLUMN), errors="coerce")
    lower_values = pd.to_numeric(dataframe.get(CATE_LOWER_COLUMN), errors="coerce")
    upper_values = pd.to_numeric(dataframe.get(CATE_UPPER_COLUMN), errors="coerce")
    return {
        "status": "COMPLETED",
        "dataset_id": str(dataset_id),
        "row_count": int(len(dataframe)),
        "columns": [str(column) for column in dataframe.columns],
        "effect_modifier_columns": [str(column) for column in effect_modifier_columns],
        "for_treatment": normalize_treatment_contrast(for_treatment),
        "cate_summary": summarize_numeric_array(cate_values.to_numpy(dtype=float, copy=False)),
        "interval_summary": summarize_interval_arrays(
            lower_values.to_numpy(dtype=float, copy=False),
            upper_values.to_numpy(dtype=float, copy=False),
        ),
    }


def skipped_all_row_cate_summary(*, reason: str, warning: str) -> dict[str, Any]:
    return {
        "status": "SKIPPED",
        "reason": reason,
        "warning": warning,
    }


def failed_all_row_cate_summary(
    *,
    reason: str,
    warning: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "FAILED",
        "reason": reason,
        "warning": warning,
        "details": {str(key): json_safe_scalar(value) for key, value in dict(details or {}).items()},
    }


def normalize_treatment_contrast(for_treatment: Any) -> dict[str, Any]:
    if not isinstance(for_treatment, Mapping):
        return {}
    contrast: dict[str, Any] = {}
    if "t0" in for_treatment:
        contrast["t0"] = json_safe_scalar(for_treatment.get("t0"))
    if "t1" in for_treatment:
        contrast["t1"] = json_safe_scalar(for_treatment.get("t1"))
    return contrast


def summarize_numeric_array(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


def summarize_interval_arrays(
    lower: np.ndarray | None,
    upper: np.ndarray | None,
) -> dict[str, Any]:
    if lower is None or upper is None or lower.shape != upper.shape:
        return {"available": False}
    mask = np.isfinite(lower) & np.isfinite(upper)
    if not np.any(mask):
        return {"available": False}
    lower_f = lower[mask]
    upper_f = upper[mask]
    return {
        "available": True,
        "mean_lower": float(np.mean(lower_f)),
        "mean_upper": float(np.mean(upper_f)),
        "frac_crosses_zero": float(np.mean((lower_f <= 0.0) & (upper_f >= 0.0))),
    }


def json_safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe_scalar(value.item())
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, str):
        return value
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe_scalar(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe_scalar(item) for item in value]
    return str(value)


def _aligned_interval_column(
    *,
    interval_values: np.ndarray | None,
    length: int,
) -> np.ndarray:
    if interval_values is None or interval_values.size != length:
        return np.full(length, np.nan, dtype=float)
    return interval_values.astype(float, copy=False)
