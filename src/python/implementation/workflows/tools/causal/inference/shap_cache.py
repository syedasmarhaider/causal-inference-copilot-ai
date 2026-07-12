from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

SHAP_COLUMN_PREFIX = "shap_"


def serialize_econml_shap_values_for_effect_modifiers(
    estimator: Any,
    *,
    X: Any,
    feature_names: Sequence[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    try:
        shap_method = getattr(estimator, "shap_values", None)
    except Exception as exc:
        return None, [f"SHAP_NOT_AVAILABLE: shap_values lookup failed: {repr(exc)}"]

    if not callable(shap_method):
        return None, ["SHAP_NOT_AVAILABLE: estimator does not expose shap_values"]

    try:
        raw_shap = shap_method(X)
    except Exception as exc:
        return None, [f"SHAP_NOT_AVAILABLE: shap_values failed: {repr(exc)}"]

    selected = _select_single_shap_payload(raw_shap)
    if selected is None:
        return None, ["SHAP_NOT_AVAILABLE: shap_values returned an unsupported shape"]

    values_raw, selector = selected
    values = _extract_shap_values_matrix(values_raw)
    if values is None:
        return None, ["SHAP_NOT_AVAILABLE: could not extract a 2D SHAP value matrix"]

    resolved_feature_names = _resolve_shap_feature_names(
        estimator=estimator,
        shap_payload=values_raw,
        input_feature_names=[str(name) for name in feature_names],
        width=int(values.shape[1]),
        warnings=warnings,
    )
    if len(resolved_feature_names) != values.shape[1]:
        return None, [
            "SHAP_NOT_AVAILABLE: SHAP feature-name count did not match value width "
            f"({len(resolved_feature_names)} != {values.shape[1]})"
        ]

    values, resolved_feature_names, aggregate_warnings = _aggregate_shap_by_effect_modifier(
        values=values,
        resolved_feature_names=resolved_feature_names,
        input_feature_names=[str(name) for name in feature_names],
    )
    warnings.extend(aggregate_warnings)

    return {
        "values": values,
        "feature_names": resolved_feature_names,
        "selector": selector,
    }, warnings


def build_shap_values_dataframe(
    *,
    dataframe: pd.DataFrame,
    identifier_column: str,
    effect_modifier_columns: Sequence[str],
    shap_values: np.ndarray,
    shap_feature_names: Sequence[str],
) -> pd.DataFrame:
    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] != len(dataframe):
        raise ValueError("SHAP values must be a 2D matrix with one row per source row")
    if values.shape[1] != len(shap_feature_names):
        raise ValueError("SHAP feature names must match SHAP value width")

    base_columns = [
        str(column)
        for column in [identifier_column, *effect_modifier_columns]
        if str(column) in dataframe.columns
    ]
    output = dataframe.reset_index(drop=True).loc[:, _dedupe_preserve_order(base_columns)].copy()
    shap_columns = _dedupe_column_names(
        [
            f"{SHAP_COLUMN_PREFIX}{_sanitize_column_suffix(feature_name)}"
            for feature_name in shap_feature_names
        ],
        existing_columns={str(column) for column in output.columns},
    )
    for index, column in enumerate(shap_columns):
        output[column] = values[:, index].astype(float, copy=False)
    return output


def summarize_shap_values_dataframe(
    *,
    dataframe: pd.DataFrame,
    dataset_id: Any,
    identifier_column: str,
    effect_modifier_columns: Sequence[str],
    selected_model: str,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    shap_columns = [
        str(column) for column in dataframe.columns if str(column).startswith(SHAP_COLUMN_PREFIX)
    ]
    feature_summaries = _summarize_shap_columns(dataframe=dataframe, shap_columns=shap_columns)
    ranked = sorted(
        feature_summaries,
        key=lambda item: (-1.0 if item["mean_abs_shap"] is None else -float(item["mean_abs_shap"])),
    )
    return {
        "status": "COMPLETED",
        "dataset_id": str(dataset_id),
        "row_count": int(len(dataframe)),
        "columns": [str(column) for column in dataframe.columns],
        "identifier_column": str(identifier_column),
        "effect_modifier_columns": [str(column) for column in effect_modifier_columns],
        "selected_model": str(selected_model),
        "shap_columns": shap_columns,
        "feature_importance": {
            "ranking_metric": "mean_abs_shap",
            "features": feature_summaries,
            "ranked": ranked,
        },
        "warnings": [str(warning) for warning in warnings],
    }


def _select_single_shap_payload(value: Any) -> tuple[Any, list[str]] | None:
    if isinstance(value, Mapping):
        current: Any = value
        selector: list[str] = []
        while isinstance(current, Mapping):
            if not current:
                return None
            key = next(iter(current.keys()))
            selector.append(str(key))
            current = current[key]
        return current, selector
    return value, []


def _extract_shap_values_matrix(value: Any) -> np.ndarray | None:
    raw_values = getattr(value, "values", value)
    try:
        arr = np.asarray(raw_values, dtype=float)
    except Exception:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        return None
    return arr.astype(float, copy=False)


def _resolve_shap_feature_names(
    *,
    estimator: Any,
    shap_payload: Any,
    input_feature_names: list[str],
    width: int,
    warnings: list[str],
) -> list[str]:
    payload_names = getattr(shap_payload, "feature_names", None)
    if payload_names is not None:
        names = [str(name) for name in list(payload_names)]
        if len(names) == width:
            return names

    cate_feature_names = getattr(estimator, "cate_feature_names", None)
    if callable(cate_feature_names):
        try:
            names = [
                str(name) for name in list(cate_feature_names(feature_names=input_feature_names))
            ]
            if len(names) == width:
                return names
            warnings.append(
                "SHAP_FEATURE_NAMES_FALLBACK: cate_feature_names width did not match SHAP values"
            )
        except Exception as exc:
            warnings.append(f"SHAP_FEATURE_NAMES_FALLBACK: cate_feature_names failed: {repr(exc)}")

    if len(input_feature_names) == width:
        return input_feature_names

    return [f"feature_{index + 1}" for index in range(width)]


def _aggregate_shap_by_effect_modifier(
    *,
    values: np.ndarray,
    resolved_feature_names: list[str],
    input_feature_names: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    if values.shape[1] == len(input_feature_names):
        return values, input_feature_names, []

    column_indexes_by_modifier: dict[str, list[int]] = {name: [] for name in input_feature_names}
    unmapped: list[str] = []
    for index, feature_name in enumerate(resolved_feature_names):
        modifier = _match_effect_modifier_feature_name(
            feature_name=feature_name,
            effect_modifier_names=input_feature_names,
        )
        if modifier is None:
            unmapped.append(feature_name)
            continue
        column_indexes_by_modifier[modifier].append(index)

    if unmapped or any(not indexes for indexes in column_indexes_by_modifier.values()):
        return (
            values,
            resolved_feature_names,
            [
                "SHAP_FEATURE_AGGREGATION_SKIPPED: could not map every expanded SHAP feature "
                "back to an effect modifier"
            ],
        )

    aggregated = np.column_stack(
        [
            np.sum(values[:, column_indexes_by_modifier[modifier]], axis=1)
            for modifier in input_feature_names
        ]
    )
    return aggregated.astype(float, copy=False), input_feature_names, []


def _match_effect_modifier_feature_name(
    *,
    feature_name: str,
    effect_modifier_names: Sequence[str],
) -> str | None:
    normalized_feature = str(feature_name)
    for modifier in effect_modifier_names:
        normalized_modifier = str(modifier)
        if normalized_feature == normalized_modifier:
            return normalized_modifier
        if normalized_feature.startswith(f"{normalized_modifier}_"):
            return normalized_modifier
        if normalized_feature.startswith(f"{normalized_modifier}["):
            return normalized_modifier
        if normalized_feature.startswith(f"{normalized_modifier}="):
            return normalized_modifier
    return None


def _summarize_shap_columns(
    *,
    dataframe: pd.DataFrame,
    shap_columns: Sequence[str],
) -> list[dict[str, Any]]:
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
    return feature_summaries


def _sanitize_column_suffix(value: Any) -> str:
    suffix = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
    suffix = re.sub(r"_+", "_", suffix).strip("_").lower()
    return suffix or "feature"


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


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
        suffix = 2
        while f"{candidate}_{suffix}" in used:
            suffix += 1
        unique = f"{candidate}_{suffix}"
        used.add(unique)
        deduped.append(unique)
    return deduped
