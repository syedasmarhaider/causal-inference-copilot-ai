from __future__ import annotations

import re
from collections.abc import Mapping
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

_VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"
_DEFAULT_MAX_FEATURES = 10
_DEFAULT_MAX_POINTS_PER_FEATURE = 500
_JITTER_PATTERN = (0.0, -0.16, 0.16, -0.30, 0.30, -0.08, 0.08, -0.23, 0.23)


def build_shap_feature_importance_chart_spec(
    *,
    shap_dataframe: pd.DataFrame,
    value_dataframe: pd.DataFrame,
    shap_summary: Mapping[str, Any],
    max_features: int = _DEFAULT_MAX_FEATURES,
    max_points_per_feature: int = _DEFAULT_MAX_POINTS_PER_FEATURE,
) -> dict[str, Any] | None:
    """Build one signed SHAP summary chart with a zero reference line."""
    if max_features < 1:
        raise ValueError("max_features must be positive")
    if max_points_per_feature < 1:
        raise ValueError("max_points_per_feature must be positive")

    ranked = _ranked_features(shap_summary)
    if not ranked:
        return None

    records: list[dict[str, Any]] = []
    selected_features: list[str] = []
    for item in ranked:
        column = str(item.get("column", ""))
        if column not in shap_dataframe.columns:
            continue
        mean_abs_shap = _finite_float(item.get("mean_abs_shap"))
        if mean_abs_shap is None:
            continue
        selected_features.append(column)
        feature_name = _display_feature_name(column)
        feature_value_ranks = _feature_value_ranks(
            value_dataframe=value_dataframe,
            feature_name=feature_name,
        )
        values = pd.to_numeric(shap_dataframe[column], errors="coerce").to_numpy(
            dtype=float,
            copy=False,
        )
        finite_indexes = np.flatnonzero(np.isfinite(values))
        for position, row_index in enumerate(
            _sample_indexes(finite_indexes, limit=max_points_per_feature)
        ):
            shap_value = float(values[row_index])
            feature_value_rank = (
                feature_value_ranks[int(row_index)] if feature_value_ranks is not None else 0.5
            )
            records.append(
                {
                    "feature": _display_feature_name(column),
                    "shap_value": shap_value,
                    "mean_abs_shap": mean_abs_shap,
                    "direction": "negative" if shap_value < 0 else "positive",
                    "feature_value_rank": (
                        float(feature_value_rank) if np.isfinite(feature_value_rank) else 0.5
                    ),
                    "jitter": _JITTER_PATTERN[position % len(_JITTER_PATTERN)],
                    "feature_value": _feature_value_label(
                        value_dataframe=value_dataframe,
                        feature_name=feature_name,
                        row_index=int(row_index),
                    ),
                }
            )
        if len(selected_features) >= max_features:
            break

    if not records:
        return None

    return {
        "$schema": _VEGA_LITE_SCHEMA,
        "title": "SHAP summary: signed treatment-effect contributions",
        "width": 720,
        "height": max(220, min(620, 44 * len(selected_features))),
        "data": {"values": records},
        "layer": [
            {
                "mark": {"type": "rule", "color": "#666666", "strokeWidth": 1.5},
                "encoding": {"x": {"datum": 0}},
            },
            {
                "mark": {"type": "circle", "opacity": 0.42, "size": 34},
                "encoding": {
                    "x": {
                        "field": "shap_value",
                        "type": "quantitative",
                        "title": "SHAP value (signed treatment-effect contribution)",
                    },
                    "y": {
                        "field": "feature",
                        "type": "nominal",
                        "sort": {
                            "field": "mean_abs_shap",
                            "op": "max",
                            "order": "descending",
                        },
                        "title": "Effect modifier",
                    },
                    "yOffset": {
                        "field": "jitter",
                        "type": "quantitative",
                        "scale": {"domain": [-0.35, 0.35]},
                    },
                    "color": {
                        "field": "feature_value_rank",
                        "type": "quantitative",
                        "title": "Feature value",
                        "scale": {
                            "domain": [0, 0.5, 1],
                            "range": ["#2F80ED", "#8E24AA", "#E91E63"],
                        },
                        "legend": {
                            "title": "Feature value",
                            "values": [0, 1],
                            "labelExpr": "datum.value === 0 ? 'Low' : 'High'",
                        },
                    },
                    "tooltip": [
                        {"field": "feature", "type": "nominal", "title": "Effect modifier"},
                        {
                            "field": "shap_value",
                            "type": "quantitative",
                            "title": "Signed SHAP value",
                            "format": ".4f",
                        },
                        {
                            "field": "mean_abs_shap",
                            "type": "quantitative",
                            "title": "Mean absolute SHAP",
                            "format": ".4f",
                        },
                        {
                            "field": "direction",
                            "type": "nominal",
                            "title": "Push direction",
                        },
                        {
                            "field": "feature_value",
                            "type": "nominal",
                            "title": "Feature value",
                        },
                    ],
                },
            },
        ],
    }


def _ranked_features(shap_summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    importance = shap_summary.get("feature_importance")
    if not isinstance(importance, Mapping):
        return []
    ranked = importance.get("ranked")
    if not isinstance(ranked, list):
        return []
    return [item for item in ranked if isinstance(item, Mapping)]


def _sample_indexes(indexes: np.ndarray, *, limit: int) -> np.ndarray:
    if len(indexes) <= limit:
        return indexes
    positions = np.linspace(0, len(indexes) - 1, num=limit, dtype=int)
    return indexes[np.unique(positions)]


def _display_feature_name(column: str) -> str:
    if column.startswith("shap_"):
        return column[len("shap_") :] or column
    return column


def _feature_value_label(
    *,
    value_dataframe: pd.DataFrame,
    feature_name: str,
    row_index: int,
) -> str:
    column = _resolve_value_column(value_dataframe=value_dataframe, feature_name=feature_name)
    if column is None or row_index >= len(value_dataframe):
        return "Not available"
    value = value_dataframe.iloc[row_index][column]
    if value is None or pd.isna(value):
        return "Missing"
    if isinstance(value, (float, np.floating)) and isfinite(float(value)):
        return f"{float(value):.4g}"
    return str(value)


def _feature_value_ranks(
    *,
    value_dataframe: pd.DataFrame,
    feature_name: str,
) -> np.ndarray | None:
    column = _resolve_value_column(value_dataframe=value_dataframe, feature_name=feature_name)
    if column is None:
        return None

    series = value_dataframe[column]
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
    else:
        codes, _ = pd.factorize(series.astype("string"), sort=True)
        values = codes.astype(float, copy=False)
        values[values < 0] = np.nan

    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), 0.5, dtype=float)

    minimum = float(np.min(values[finite]))
    maximum = float(np.max(values[finite]))
    ranks = np.full(len(values), 0.5, dtype=float)
    if maximum > minimum:
        ranks[finite] = (values[finite] - minimum) / (maximum - minimum)
    return ranks


def _resolve_value_column(*, value_dataframe: pd.DataFrame, feature_name: str) -> str | None:
    if feature_name in value_dataframe.columns:
        return feature_name

    normalized_feature = _normalize_column_name(feature_name)
    normalized_columns = {
        _normalize_column_name(str(column)): str(column) for column in value_dataframe.columns
    }
    if normalized_feature in normalized_columns:
        return normalized_columns[normalized_feature]

    matches = [
        str(column)
        for column in value_dataframe.columns
        if normalized_feature.startswith(f"{_normalize_column_name(str(column))}_")
    ]
    return max(matches, key=len) if matches else None


def _normalize_column_name(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip()).strip("_").lower()
    return re.sub(r"_+", "_", normalized)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if isfinite(normalized) else None


__all__ = ["build_shap_feature_importance_chart_spec"]
