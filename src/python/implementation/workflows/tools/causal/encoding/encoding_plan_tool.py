from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from python.domain.workflows.tool import Tool
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.tools.causal.encoding.encoding_plan import (
    TransformPlan,
    validate_transform_payload_structured,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel

log = get_app_logger(__name__, component="encoding_plan_tool", log_type="tool")


@dataclass(frozen=True)
class EncodingPlanTool(Tool):
    NAME: ClassVar[str] = "ENCODING_PLAN"

    def get_tool_name(self) -> str:
        return self.NAME

    def get_tool_info(self) -> str:
        return (
            "Tool for building and applying dataset-summary-bound validation for "
            "encoding transform plans over covariate and effect_modifier columns only. "
            "It provides per-summary Pydantic schemas, structured validation helpers, "
            "and typed post-validation for plans generated elsewhere."
        )

    def build_encoding_schema(
        self,
        *,
        data_summary: DatasetSummaryModel,
        covariate_columns: Sequence[str] | None = None,
        effect_modifier_columns: Sequence[str] | None = None,
    ) -> type[TransformPlan]:
        schema = TransformPlan.for_dataset_summary(
            data_summary,
            covariate_columns=covariate_columns,
            effect_modifier_columns=effect_modifier_columns,
        )

        log.debug(
            "built encoding plan schema",
            columns_count=len(data_summary.profiles),
        )
        return schema

    def validate_encoding_payload_structured(
        self,
        *,
        payload: Mapping[str, Any],
        data_summary: DatasetSummaryModel,
        covariate_columns: Sequence[str] | None = None,
        effect_modifier_columns: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        model_dict, issues = validate_transform_payload_structured(
            payload,
            dataset_summary=data_summary,
            covariate_columns=covariate_columns,
            effect_modifier_columns=effect_modifier_columns,
        )
        if issues:
            log.info(
                "encoding payload validation failed",
                issues_count=len(issues),
            )
        else:
            log.debug("encoding payload validation passed")
        return model_dict, issues

    def validate_encoding_payload(
        self,
        *,
        payload: Mapping[str, Any],
        data_summary: DatasetSummaryModel,
        covariate_columns: Sequence[str] | None = None,
        effect_modifier_columns: Sequence[str] | None = None,
    ) -> TransformPlan:
        schema = self.build_encoding_schema(
            data_summary=data_summary,
            covariate_columns=covariate_columns,
            effect_modifier_columns=effect_modifier_columns,
        )
        model = schema.model_validate(payload)

        log.debug(
            "encoding payload validated",
            columns_count=len(model.columns),
        )
        return model

    def post_validate_encoding_plan(
        self,
        *,
        plan: TransformPlan,
        data_summary: DatasetSummaryModel,
        covariate_columns: Sequence[str] | None = None,
        effect_modifier_columns: Sequence[str] | None = None,
    ) -> TransformPlan:
        model = self.validate_encoding_payload(
            payload=plan.model_dump(mode="json"),
            data_summary=data_summary,
            covariate_columns=covariate_columns,
            effect_modifier_columns=effect_modifier_columns,
        )

        log.debug(
            "encoding plan post-validation passed",
            columns_count=len(model.columns),
        )
        return model


__all__ = ["EncodingPlanTool"]
