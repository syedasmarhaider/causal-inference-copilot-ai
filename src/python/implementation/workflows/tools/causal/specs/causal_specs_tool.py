from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from python.domain.workflows.tool import Tool
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.tools.causal.specs.causal_spec import (
    CausalSpec,
    validate_backdoor_payload_structured,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel

log = get_app_logger(__name__, component="causal_specs_tool", log_type="tool")


@dataclass(frozen=True)
class CausalSpecsTool(Tool):
    NAME: ClassVar[str] = "CAUSAL_BACKDOOR_SPEC"

    def get_tool_name(self) -> str:
        return self.NAME

    def get_tool_info(self) -> str:
        return (
            "Tool for building and applying dataset-summary-bound validation for "
            "backdoor-adjustment causal specs. "
            "provides the per-summary Pydantic schema, structured validation helpers, "
            "and typed post-validation for specs generated."
        )

    def build_backdoor_schema(
        self,
        *,
        data_summary: DatasetSummaryModel,
    ) -> type[CausalSpec]:
        validated_summary = self._require_dataset_summary(data_summary)
        schema = CausalSpec.for_dataset_summary(validated_summary)

        log.debug(
            "built backdoor causal schema",
            columns_count=len(validated_summary.profiles),
        )
        return schema

    def validate_backdoor_payload_structured(
        self,
        *,
        payload: Mapping[str, Any],
        data_summary: DatasetSummaryModel,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        validated_summary = self._require_dataset_summary(data_summary)

        model_dict, issues = validate_backdoor_payload_structured(
            payload,
            dataset_summary=validated_summary,
        )
        if issues:
            log.info(
                "backdoor causal payload validation failed",
                issues_count=len(issues),
            )
        else:
            log.debug("backdoor causal payload validation passed")
        return model_dict, issues

    def validate_backdoor_payload(
        self,
        *,
        payload: Mapping[str, Any],
        data_summary: DatasetSummaryModel,
    ) -> CausalSpec:
        validated_summary = self._require_dataset_summary(data_summary)
        schema = CausalSpec.for_dataset_summary(validated_summary)
        model = schema.model_validate(payload)

        log.debug(
            "backdoor causal payload validated",
            treatment_column=str(model.treatment_spec.column),
            outcome_column=str(model.outcome_spec.column),
        )
        return model

    def post_validate_backdoor_spec(
        self,
        *,
        causal_spec: CausalSpec,
        data_summary: DatasetSummaryModel,
    ) -> CausalSpec:
        validated_summary = self._require_dataset_summary(data_summary)
        model = self.validate_backdoor_payload(
            payload=causal_spec.model_dump(mode="json"),
            data_summary=validated_summary,
        )

        log.debug(
            "backdoor causal spec post-validation passed",
            experiment_type=model.experiment_type,
        )
        return model

    @staticmethod
    def _require_dataset_summary(data_summary: DatasetSummaryModel) -> DatasetSummaryModel:
        return data_summary


__all__ = ["CausalSpecsTool"]
