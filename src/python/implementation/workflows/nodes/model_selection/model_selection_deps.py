from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from python.domain.models.validation import ValidationIssueModel, ValidationStatus
from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.nodes.data_compilation.data_compilation_state import (
    DataCompilationState,
)
from python.implementation.workflows.nodes.data_validation.data_validation_state import (
    DataValidationState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)


@dataclass(frozen=True)
class ModelSelectionDeps:
    dataset_id: UUID | None
    dataset_summary: DatasetSummaryModel | None
    causal_spec: CausalSpec | None
    transformation_plan: TransformPlan | None
    validation_issues: list[ValidationIssueModel]
    validation_status: ValidationStatus | None

    @classmethod
    def from_request(cls, request: NodeRequest) -> ModelSelectionDeps:
        compilation_raw = request.orchestrator_state.get(DataCompilationState.NAME)
        validation_raw = request.orchestrator_state.get(DataValidationState.NAME)

        if compilation_raw is not None and not isinstance(compilation_raw, Mapping):
            raise TypeError(
                "DATA_COMPILATION payload must be a dict with compiled dataset and setup values"
            )
        if validation_raw is not None and not isinstance(validation_raw, Mapping):
            raise TypeError(
                "DATA_VALIDATION payload must be a dict with validation results"
            )

        compilation = cast(Mapping[str, Any] | None, compilation_raw)
        validation = cast(Mapping[str, Any] | None, validation_raw)

        dataset_id = cast(UUID | None, None if compilation is None else compilation.get("working_dataset_id"))

        dataset_summary_raw = None if compilation is None else compilation.get("latest_dataset_summary")
        if dataset_summary_raw is None:
            dataset_summary = None
        elif isinstance(dataset_summary_raw, DatasetSummaryModel):
            dataset_summary = dataset_summary_raw
        elif isinstance(dataset_summary_raw, str):
            dataset_summary = DatasetSummaryModel.model_validate_json(dataset_summary_raw)
        else:
            dataset_summary = DatasetSummaryModel.model_validate(dataset_summary_raw)

        causal_spec_raw = None if compilation is None else compilation.get("causal_spec")
        if causal_spec_raw is None:
            causal_spec = None
        elif isinstance(causal_spec_raw, CausalSpec):
            causal_spec = causal_spec_raw
        else:
            causal_spec = CausalSpec.model_validate(causal_spec_raw)

        transformation_plan_raw = (
            None
            if compilation is None
            else compilation.get(
                "data_transformation_plan",
            )
        )
        if transformation_plan_raw is None:
            transformation_plan = None
        elif isinstance(transformation_plan_raw, TransformPlan):
            transformation_plan = transformation_plan_raw
        else:
            transformation_plan = TransformPlan.model_validate(transformation_plan_raw)

        validation_issues_raw = [] if validation is None else list(validation.get("validation_issues") or [])
        validation_issues = [
            issue
            if isinstance(issue, ValidationIssueModel)
            else ValidationIssueModel.model_validate(issue)
            for issue in validation_issues_raw
        ]
        validation_status = None
        for issue in validation_issues:
            if issue.status == ValidationStatus.WARN:
                validation_status = ValidationStatus.WARN
            elif issue.status == ValidationStatus.FAIL:
                raise Exception(f"Validation failed: {issue}")

        
        return cls(
            dataset_id=dataset_id,
            dataset_summary=dataset_summary,
            causal_spec=causal_spec,
            transformation_plan=transformation_plan,
            validation_issues=validation_issues,
            validation_status=validation_status,
        )


__all__ = ["ModelSelectionDeps"]
