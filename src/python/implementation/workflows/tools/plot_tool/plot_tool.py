from __future__ import annotations

from datetime import date, datetime
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.models.models import NonEmptyStr
from python.domain.service.llm_service import AvailableModelsKey, LLMConfig, LLMService
from python.domain.workflows.tool import Tool
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.tools.plot_tool.plot_tool_prompts import (
    PLOT_SPECS_SYSTEM_PROMPT,
    PLOT_SPECS_USER_PROMPT_TEMPLATE,
)

log = get_app_logger(__name__, component="plot_specs_service", log_type="service")


class VegaLiteSpecTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    spec: dict[str, Any]


class PlotSpecsPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    charts: list[VegaLiteSpecTemplate] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _validate_chart_count(self) -> PlotSpecsPlan:
        if not self.charts:
            raise ValueError("charts must contain at least one chart spec")
        return self

class PlotTool(Tool):
    NAME: ClassVar[str] = "PLOT_TOOL"
    def get_tool_name(self) -> str:
        return self.NAME
    
    def get_tool_info(self) -> str:
        return "Tool for generating Vega-Lite visualization specifications based on a given dataframe and user intent. The tool uses an LLM to create template specs that reference the dataframe's fields, then injects actual data values into the specs while ensuring they are valid Vega-Lite specifications ready for rendering."
    
    llm: LLMService
    model: AvailableModelsKey = "basic"
    max_attempts: int = 2
    max_rows_for_values: int = 2000
    vega_schema_url: NonEmptyStr = "https://vega.github.io/schema/vega-lite/v5.json"

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be >= 1")
        if self.max_rows_for_values <= 0:
            raise ValueError("max_rows_for_values must be >= 1")

    def generate_specs(
        self,
        *,
        dataframe: pd.DataFrame,
        data_summary: str,
        user_intent: str,
    ) -> list[dict[str, Any]]:
        normalized_summary = data_summary.strip()
        normalized_intent = user_intent.strip()
        if not normalized_summary:
            raise ValueError("data_summary must be non-empty")
        if not normalized_intent:
            raise ValueError("user_intent must be non-empty")
        if len(dataframe.columns) == 0:
            raise ValueError("dataframe must have at least one column")

        user_prompt = PLOT_SPECS_USER_PROMPT_TEMPLATE.format(
            user_intent=normalized_intent,
            data_summary=normalized_summary,
        )

        log.info(
            "generating vega-lite spec templates",
            rows=len(dataframe),
            columns=len(dataframe.columns),
        )
        plan = self.llm.generate_json(
            schema=PlotSpecsPlan,
            system_prompt=PLOT_SPECS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            config=LLMConfig(model=self.model, temperature=0.0, top_p=1.0),
            history=None,
            max_attempts=self.max_attempts,
        )

        records_df = dataframe.head(self.max_rows_for_values).copy()
        if len(dataframe) > self.max_rows_for_values:
            log.warning(
                "truncating dataframe rows for vega-lite values injection",
                original_rows=len(dataframe),
                truncated_rows=len(records_df),
            )

        final_specs: list[dict[str, Any]] = []
        for idx, chart in enumerate(plan.charts, start=1):
            spec = dict(chart.spec)
            self._validate_template_spec(spec=spec, chart_index=idx)
            used_fields = self._collect_field_names(spec)
            self._validate_fields_exist(
                used_fields=used_fields,
                dataframe_columns=tuple(str(c) for c in dataframe.columns),
                chart_index=idx,
            )
            values = self._build_values(records_df=records_df, used_fields=used_fields)
            final_spec = self._inject_values(spec=spec, values=values)
            self._validate_final_spec_without_render(
                spec=final_spec,
                chart_index=idx,
            )
            final_specs.append(final_spec)

        log.info(
            "vega-lite specs ready",
            charts_count=len(final_specs),
            rows_injected=min(len(dataframe), self.max_rows_for_values),
        )
        return final_specs

    @staticmethod
    def _validate_template_spec(*, spec: dict[str, Any], chart_index: int) -> None:
        if "datasets" in spec:
            raise ValueError(f"chart {chart_index} must not contain datasets")

        data_obj = spec.get("data")
        if isinstance(data_obj, dict):
            if "values" in data_obj:
                raise ValueError(f"chart {chart_index} must not contain data.values in template")
            if "url" in data_obj:
                raise ValueError(f"chart {chart_index} must not contain external data.url")

    def _inject_values(self, *, spec: dict[str, Any], values: list[dict[str, Any]]) -> dict[str, Any]:
        out = dict(spec)
        out.setdefault("$schema", str(self.vega_schema_url))
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
        if "mark" in spec:
            return True
        composition_keys = ("layer", "vconcat", "hconcat", "concat", "facet", "repeat")
        return any(key in spec for key in composition_keys)

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
                    f"chart {chart_index} field '{field_name}' declared quantitative "
                    "but injected values are not all numeric"
                )
            return

        if normalized_type == "temporal":
            if not all(self._is_temporal_value(v) for v in observed):
                raise ValueError(
                    f"chart {chart_index} field '{field_name}' declared temporal "
                    "but injected values are not all datetime-like"
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
        deduped = tuple(dict.fromkeys(fields))
        return deduped

    @staticmethod
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
            raise ValueError(
                f"chart {chart_index} references unknown dataframe fields: {missing}"
            )

    def _build_values(
        self,
        *,
        records_df: pd.DataFrame,
        used_fields: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        if used_fields:
            selected_df = records_df.loc[:, list(used_fields)].copy()
        else:
            selected_df = records_df.copy()

        for col in selected_df.columns:
            if pd.api.types.is_datetime64_any_dtype(selected_df[col]):
                selected_df[col] = selected_df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")

        selected_df = selected_df.where(pd.notnull(selected_df), None)
        return selected_df.to_dict(orient="records")
