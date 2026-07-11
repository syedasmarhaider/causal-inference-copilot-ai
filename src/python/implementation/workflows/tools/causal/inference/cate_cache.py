from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

EFFECT_ROW_COLUMN = "effect_row"
CATE_COLUMN = "cate"
CATE_LOWER_COLUMN = "cate_lower"
CATE_UPPER_COLUMN = "cate_upper"
CATE_STDERR_COLUMN = "cate_stderr"
CATE_REVERSE_COLUMN = "cate_reverse"
CATE_REVERSE_LOWER_COLUMN = "cate_reverse_lower"
CATE_REVERSE_UPPER_COLUMN = "cate_reverse_upper"
CATE_T0_COLUMN = "cate_t0"
CATE_T1_COLUMN = "cate_t1"
SHAP_COLUMN_PREFIX = "shap_"


def build_all_row_cate_dataframe(
    *,
    dataframe: pd.DataFrame,
    cate_values: np.ndarray,
    lower_values: np.ndarray | None,
    upper_values: np.ndarray | None,
    stderr_values: np.ndarray | None = None,
    shap_values: np.ndarray | None = None,
    shap_feature_names: Sequence[str] | None = None,
    for_treatment: Any = None,
) -> pd.DataFrame:
    query_df = dataframe.reset_index(drop=True).copy()
    query_df[CATE_COLUMN] = cate_values.astype(float, copy=False)
    query_df[CATE_LOWER_COLUMN] = _aligned_interval_column(
        interval_values=lower_values,
        length=len(query_df),
    )
    query_df[CATE_UPPER_COLUMN] = _aligned_interval_column(
        interval_values=upper_values,
        length=len(query_df),
    )
    query_df[CATE_STDERR_COLUMN] = _aligned_interval_column(
        interval_values=stderr_values,
        length=len(query_df),
    )
    _append_shap_columns(
        dataframe=query_df,
        shap_values=shap_values,
        shap_feature_names=shap_feature_names,
    )
    return query_df


def summarize_all_row_cate_dataframe(
    *,
    dataframe: pd.DataFrame,
    dataset_id: Any,
    effect_modifier_columns: Sequence[str],
    for_treatment: Any = None,
) -> dict[str, Any]:
    missing_series = pd.Series(np.nan, index=dataframe.index, dtype=float)
    cate_values = pd.to_numeric(dataframe.get(CATE_COLUMN, missing_series), errors="coerce")
    lower_values = pd.to_numeric(dataframe.get(CATE_LOWER_COLUMN, missing_series), errors="coerce")
    upper_values = pd.to_numeric(dataframe.get(CATE_UPPER_COLUMN, missing_series), errors="coerce")
    stderr_values = pd.to_numeric(
        dataframe.get(CATE_STDERR_COLUMN, missing_series),
        errors="coerce",
    )
    shap_columns = [
        str(column) for column in dataframe.columns if str(column).startswith(SHAP_COLUMN_PREFIX)
    ]
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
        "stderr_summary": summarize_numeric_array(stderr_values.to_numpy(dtype=float, copy=False)),
        "shap_summary": summarize_shap_columns(dataframe=dataframe, shap_columns=shap_columns),
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
        "details": {
            str(key): json_safe_scalar(value) for key, value in dict(details or {}).items()
        },
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


def summarize_shap_columns(
    *,
    dataframe: pd.DataFrame,
    shap_columns: Sequence[str],
) -> dict[str, Any]:
    if not shap_columns:
        return {"available": False, "columns": []}

    feature_summaries: list[dict[str, Any]] = []
    for column in shap_columns:
        values = pd.to_numeric(dataframe[column], errors="coerce").to_numpy(
            dtype=float,
            copy=False,
        )
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            feature_summaries.append(
                {
                    "column": str(column),
                    "n": 0,
                    "mean_shap": None,
                    "mean_abs_shap": None,
                    "std_shap": None,
                }
            )
            continue
        feature_summaries.append(
            {
                "column": str(column),
                "n": int(finite.size),
                "mean_shap": float(np.mean(finite)),
                "mean_abs_shap": float(np.mean(np.abs(finite))),
                "std_shap": float(np.std(finite)),
            }
        )

    ranked = sorted(
        feature_summaries,
        key=lambda item: (-1.0 if item["mean_abs_shap"] is None else -float(item["mean_abs_shap"])),
    )
    return {
        "available": True,
        "columns": [str(column) for column in shap_columns],
        "features": feature_summaries,
        "ranked_by_mean_abs_shap": ranked,
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


def _append_shap_columns(
    *,
    dataframe: pd.DataFrame,
    shap_values: np.ndarray | None,
    shap_feature_names: Sequence[str] | None,
) -> None:
    if shap_values is None or shap_feature_names is None:
        return
    try:
        values = np.asarray(shap_values, dtype=float)
    except Exception:
        return
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] != len(dataframe):
        return
    if values.shape[1] != len(shap_feature_names):
        return

    columns = _dedupe_column_names(
        [
            f"{SHAP_COLUMN_PREFIX}{_sanitize_column_suffix(feature_name)}"
            for feature_name in shap_feature_names
        ],
        existing_columns={str(column) for column in dataframe.columns},
    )
    for index, column in enumerate(columns):
        dataframe[column] = values[:, index].astype(float, copy=False)


def _sanitize_column_suffix(value: Any) -> str:
    suffix = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
    suffix = re.sub(r"_+", "_", suffix).strip("_").lower()
    return suffix or "feature"


def _dedupe_column_names(
    names: Sequence[str],
    *,
    existing_columns: set[str],
) -> list[str]:
    used = set(existing_columns)
    deduped: list[str] = []
    for name in names:
        candidate = str(name)
        if candidate not in used:
            used.add(candidate)
            deduped.append(candidate)
            continue
        index = 2
        while f"{candidate}_{index}" in used:
            index += 1
        resolved = f"{candidate}_{index}"
        used.add(resolved)
        deduped.append(resolved)
    return deduped
