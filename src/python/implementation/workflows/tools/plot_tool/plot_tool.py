from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.models.models import NonEmptyStr
from python.domain.service.llm_service import AvailableModelsKey, LLMConfig, LLMService
from python.domain.workflows.tool import Tool
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.tools.plot_tool.plot_tool_prompts import (
    PLOT_SPECS_SYSTEM_PROMPT,
    PLOT_SPECS_USER_PROMPT_TEMPLATE,
)

_KIND_TO_VEGA_TYPE: dict[str, str] = {
    "NUMERIC": "quantitative",
    "DATETIME": "temporal",
    "CATEGORICAL": "nominal",
    "BOOLEAN": "nominal",
    "OTHER": "nominal",
}

log = get_app_logger(__name__, component="plot_tool", log_type="tool")


class VegaLiteSpecTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    spec: dict[str, Any]


class PlotSpecsPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ALLOWED_FIELD_NAMES: ClassVar[tuple[str, ...] | None] = None
    FIELD_KINDS: ClassVar[dict[str, str] | None] = None

    charts: list[VegaLiteSpecTemplate] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _validate_chart_count(self) -> PlotSpecsPlan:
        if not self.charts:
            raise ValueError("charts must contain at least one chart spec")

        allowed_field_names = type(self).ALLOWED_FIELD_NAMES
        field_kinds = type(self).FIELD_KINDS
        if allowed_field_names is None:
            return self

        for idx, chart in enumerate(self.charts, start=1):
            _validate_template_spec(spec=chart.spec, chart_index=idx)
            if not _has_visual_definition(chart.spec):
                raise ValueError(
                    f"chart {idx} must define a visual grammar "
                    "(mark or composition keys like layer/vconcat/hconcat/concat/facet/repeat)"
                )
            _validate_fields_exist(
                used_fields=_collect_field_names(chart.spec),
                dataframe_columns=allowed_field_names,
                chart_index=idx,
            )
            _validate_field_types_against_summary(
                spec=chart.spec,
                field_kinds=field_kinds,
                chart_index=idx,
            )
        return self

    @classmethod
    def for_summary(cls, summary: DatasetSummaryModel) -> type[PlotSpecsPlan]:
        normalized_field_names = _extract_summary_field_names(summary)
        if not normalized_field_names:
            raise ValueError("data_summary must contain at least one non-empty header")

        return type(
            f"{cls.__name__}ForFields_{len(normalized_field_names)}",
            (cls,),
            {
                "__module__": cls.__module__,
                "ALLOWED_FIELD_NAMES": normalized_field_names,
                "FIELD_KINDS": _extract_summary_field_kinds(summary),
            },
        )


@dataclass(frozen=True)
class PlotTool(Tool):
    NAME: ClassVar[str] = "PLOT_TOOL"

    def get_tool_name(self) -> str:
        return self.NAME

    def get_tool_info(self) -> str:
        return """Tool for generating Vega-Lite visualization specifications
    based on a given dataframe and user intent. The tool uses an LLM to create template specs 
    that reference the dataframe's fields, then injects actual data values into the specs 
    while ensuring they are valid Vega-Lite specifications ready for rendering.""".strip()

    llm: LLMService
    model: AvailableModelsKey = "basic"
    warn_max_rows_for_values: int = 2000
    vega_schema_url: NonEmptyStr = "https://vega.github.io/schema/vega-lite/v5.json"

    def generate_specs(
        self,
        *,
        dataframe: pd.DataFrame,
        data_summary: DatasetSummaryModel,
        user_intent: str,
        max_attempts: int = 3,
    ) -> list[dict[str, Any]]:
        normalized_intent = user_intent.strip()
        if not normalized_intent:
            raise ValueError("user_intent must be non-empty")
        if len(dataframe.columns) == 0:
            raise ValueError("dataframe must have at least one column")

        dataframe_columns = tuple(str(c) for c in dataframe.columns)
        summary_field_names = _extract_summary_field_names(data_summary)
        _validate_summary_headers_against_dataframe(
            summary_field_names=summary_field_names,
            dataframe_columns=dataframe_columns,
        )
        field_guide = _build_field_guide(data_summary)
        user_prompt = PLOT_SPECS_USER_PROMPT_TEMPLATE.format(
            user_intent=normalized_intent,
            n_rows=data_summary.n_rows,
            field_guide=field_guide,
        )

        log.debug(
            "generating vega-lite spec templates",
            rows=len(dataframe),
            columns=len(dataframe.columns),
        )
        plan_schema = PlotSpecsPlan.for_summary(data_summary)
        plan = self.llm.generate_json(
            schema=plan_schema,
            system_prompt=PLOT_SPECS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            config=LLMConfig(model=self.model, temperature=0.5),
            history=None,
            max_attempts=max_attempts,
        )

        records_df = dataframe.copy()
        if len(dataframe) > self.warn_max_rows_for_values:
            log.warning(
                "vega-lite values injection exceeds warning threshold; injecting all rows",
                row_count=len(dataframe),
                warning_threshold=self.warn_max_rows_for_values,
            )

        final_specs: list[dict[str, Any]] = []
        for idx, chart in enumerate(plan.charts, start=1):
            spec = dict(chart.spec)
            used_fields = self._collect_field_names(spec)
            values = self._build_values(records_df=records_df, used_fields=used_fields)
            final_spec = self._inject_values(spec=spec, values=values, title=chart.title)
            self._validate_final_spec_without_render(
                spec=final_spec,
                chart_index=idx,
            )
            final_specs.append(final_spec)

        log.info(
            "vega-lite specs ready",
            charts_count=len(final_specs),
            rows_injected=len(dataframe),
        )

        return final_specs

    @staticmethod
    def _validate_template_spec(*, spec: dict[str, Any], chart_index: int) -> None:
        _validate_template_spec(spec=spec, chart_index=chart_index)

    def _inject_values(
        self,
        *,
        spec: dict[str, Any],
        values: list[dict[str, Any]],
        title: str | None,
    ) -> dict[str, Any]:
        out = dict(spec)
        out.setdefault("$schema", str(self.vega_schema_url))
        if title:
            out.setdefault("title", title)
        out["data"] = {"values": values}
        return out

    def _validate_final_spec_without_render(
        self,
        *,
        spec: dict[str, Any],
        chart_index: int,
    ) -> None:
        if not self._has_visual_definition(spec):
            raise ValueError(
                f"chart {chart_index} must define a visual grammar "
                "(mark or composition keys like layer/vconcat/hconcat/concat/facet/repeat)"
            )

        data_obj = spec.get("data")
        if not isinstance(data_obj, dict):
            raise ValueError(f"chart {chart_index} data must be an object")

        values = data_obj.get("values")
        if not isinstance(values, list):
            raise ValueError(f"chart {chart_index} data.values must be a list")
        if any(not isinstance(row, dict) for row in values):
            raise ValueError(f"chart {chart_index} data.values must contain only objects")

        self._validate_encoding_types(spec=spec, values=values, chart_index=chart_index)

    @staticmethod
    def _has_visual_definition(spec: dict[str, Any]) -> bool:
        return _has_visual_definition(spec)

    def _validate_encoding_types(
        self,
        *,
        spec: dict[str, Any],
        values: list[dict[str, Any]],
        chart_index: int,
    ) -> None:
        encoding_nodes = self._collect_encoding_nodes(spec)
        for encoding in encoding_nodes:
            field_name = encoding.get("field")
            field_type = encoding.get("type")
            if not isinstance(field_name, str) or not isinstance(field_type, str):
                continue
            self._validate_field_type_against_values(
                field_name=field_name,
                field_type=field_type,
                values=values,
                chart_index=chart_index,
            )

    @staticmethod
    def _collect_encoding_nodes(spec: dict[str, Any]) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                maybe_field = node.get("field")
                if isinstance(maybe_field, str):
                    nodes.append(node)
                for value in node.values():
                    _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(spec)
        return nodes

    def _validate_field_type_against_values(
        self,
        *,
        field_name: str,
        field_type: str,
        values: list[dict[str, Any]],
        chart_index: int,
    ) -> None:
        observed = [
            row.get(field_name)
            for row in values
            if isinstance(row, dict) and field_name in row and row.get(field_name) is not None
        ]
        if not observed:
            return

        normalized_type = field_type.strip().lower()

        if normalized_type == "quantitative":
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in observed):
                raise ValueError(
                    f"chart {chart_index} field '{field_name}' declared quantitative but "
                    "values are not all numeric"
                )
            return

        if normalized_type == "temporal":
            if not all(self._is_temporal_value(v) for v in observed):
                raise ValueError(
                    f"chart {chart_index} field '{field_name}' declared temporal but "
                    "values are not all datetime-like"
                )
            return

    @staticmethod
    def _is_temporal_value(value: Any) -> bool:
        if isinstance(value, (datetime, date)):
            return True
        if isinstance(value, str):
            parsed = pd.to_datetime(value, errors="coerce")
            return bool(pd.notna(parsed))
        return False

    @staticmethod
    def _collect_field_names(spec: dict[str, Any]) -> tuple[str, ...]:
        return _collect_field_names(spec)

    @staticmethod
    def _validate_fields_exist(
        *,
        used_fields: tuple[str, ...],
        dataframe_columns: tuple[str, ...],
        chart_index: int,
    ) -> None:
        _validate_fields_exist(
            used_fields=used_fields,
            dataframe_columns=dataframe_columns,
            chart_index=chart_index,
        )

    def _build_values(
        self,
        *,
        records_df: pd.DataFrame,
        used_fields: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        if used_fields:
            existing_fields = [f for f in used_fields if f in records_df.columns]
            missing_fields = [f for f in used_fields if f not in records_df.columns]
            if missing_fields:
                log.warning(
                    "chart spec references fields not found in dataframe — falling back to all columns",
                    missing_fields=missing_fields,
                )
            selected_df = (
                records_df[existing_fields].copy() if existing_fields else records_df.copy()
            )
        else:
            selected_df = records_df.copy()

        for col in selected_df.columns:
            if pd.api.types.is_datetime64_any_dtype(selected_df[col]):
                selected_df[col] = selected_df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")

        values = selected_df.to_dict(orient="records")
        return [{key: _json_safe_value(value) for key, value in row.items()} for row in values]


def _validate_template_spec(*, spec: dict[str, Any], chart_index: int) -> None:
    if "datasets" in spec:
        raise ValueError(f"chart {chart_index} must not contain datasets")

    data_obj = spec.get("data")
    if isinstance(data_obj, dict):
        if "values" in data_obj:
            raise ValueError(f"chart {chart_index} must not contain data.values in template")
        if "url" in data_obj:
            raise ValueError(f"chart {chart_index} must not contain external data.url")


def _has_visual_definition(spec: dict[str, Any]) -> bool:
    if "mark" in spec:
        return True
    composition_keys = ("layer", "vconcat", "hconcat", "concat", "facet", "repeat")
    return any(key in spec for key in composition_keys)


def _collect_field_names(spec: dict[str, Any]) -> tuple[str, ...]:
    fields: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            field_name = node.get("field")
            if isinstance(field_name, str) and field_name.strip():
                fields.append(field_name.strip())
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(spec)
    return tuple(dict.fromkeys(fields))


def _validate_fields_exist(
    *,
    used_fields: tuple[str, ...],
    dataframe_columns: tuple[str, ...],
    chart_index: int,
) -> None:
    if not used_fields:
        return

    columns_set = set(dataframe_columns)
    missing = [field for field in used_fields if field not in columns_set]
    if missing:
        raise ValueError(f"chart {chart_index} references unknown data_summary fields: {missing}")


def _validate_field_types_against_summary(
    *,
    spec: dict[str, Any],
    field_kinds: dict[str, str] | None,
    chart_index: int,
) -> None:
    if not field_kinds:
        return

    for encoding in PlotTool._collect_encoding_nodes(spec):
        field_name = encoding.get("field")
        field_type = encoding.get("type")
        if not isinstance(field_name, str) or not isinstance(field_type, str):
            continue

        inferred_kind = field_kinds.get(field_name)
        if inferred_kind is None:
            continue

        expected_type = _KIND_TO_VEGA_TYPE.get(inferred_kind, "nominal")
        normalized_type = field_type.strip().lower()
        if normalized_type != expected_type:
            raise ValueError(
                f"chart {chart_index} field '{field_name}' declared {normalized_type} "
                f"but data_summary inferred kind is {inferred_kind}"
            )


def _build_field_guide(summary: DatasetSummaryModel) -> str:
    lines: list[str] = []
    for profile in summary.profiles:
        name = str(profile.name).strip()
        if not name:
            continue
        vega_type = _KIND_TO_VEGA_TYPE.get(str(profile.inferred_kind), "nominal")
        lines.append(f'- "{name}": {vega_type}')
    return "\n".join(lines) if lines else "(no fields)"


def _extract_summary_field_names(summary: DatasetSummaryModel) -> tuple[str, ...]:
    field_names = tuple(
        dict.fromkeys(
            str(profile.name).strip() for profile in summary.profiles if str(profile.name).strip()
        )
    )
    return field_names


def _extract_summary_field_kinds(summary: DatasetSummaryModel) -> dict[str, str]:
    return {
        str(profile.name).strip(): str(profile.inferred_kind)
        for profile in summary.profiles
        if str(profile.name).strip()
    }


def _validate_summary_headers_against_dataframe(
    *,
    summary_field_names: tuple[str, ...],
    dataframe_columns: tuple[str, ...],
) -> None:
    if not summary_field_names:
        raise ValueError("data_summary must contain at least one non-empty header")

    dataframe_columns_set = set(dataframe_columns)
    missing = [
        field_name for field_name in summary_field_names if field_name not in dataframe_columns_set
    ]
    if missing:
        raise ValueError(f"data_summary references unknown dataframe headers: {missing}")


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _json_safe_value(value.item())
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value
