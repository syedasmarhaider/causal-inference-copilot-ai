from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from python.domain.models.validation import ValidationIssueModel, ValidationStatus
from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)


@dataclass(frozen=True)
class ModelSelectionDeps:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel
    causal_spec: CausalSpec
    transformation_plan: TransformPlan
    validation_issues: list[ValidationIssueModel]
    validation_status: ValidationStatus

    @classmethod
    def from_request(cls, request: NodeRequest) -> ModelSelectionDeps:
        transformation_plan_raw = request.orchestrator_state.get("data_transformation_plan")
        causal_spec_raw = request.orchestrator_state.get("causal_spec")
        dataset_id_raw = request.orchestrator_state.get("working_dataset_id")
        dataset_summary_raw = request.orchestrator_state.get("latest_dataset_summary")
        validation_issues_raw = request.orchestrator_state.get("validation_issues")

        if dataset_id_raw is None:
            raise ValueError("ModelSelectionDeps: dataset_id is required but was not found in compilation state")
        if dataset_summary_raw is None:
            raise ValueError("ModelSelectionDeps: dataset_summary is required but was not found in compilation state")
        if causal_spec_raw is None:
            raise ValueError("ModelSelectionDeps: causal_spec is required but was not found in compilation state")
        if transformation_plan_raw is None:
            raise ValueError("ModelSelectionDeps: transformation_plan is required but was not found in compilation state")
        
        if not isinstance(dataset_id_raw, UUID):
            raise TypeError("ModelSelectionDeps: dataset_id must be a UUID")
        if not isinstance(dataset_summary_raw, DatasetSummaryModel):
            raise TypeError("ModelSelectionDeps: dataset_summary must be of type DatasetSummaryModel")
        if not isinstance(causal_spec_raw, CausalSpec):
            raise TypeError("ModelSelectionDeps: causal_spec must be of type CausalSpec")
        if not isinstance(transformation_plan_raw, TransformPlan):
            raise TypeError("ModelSelectionDeps: transformation_plan must be of type TransformPlan")
        if validation_issues_raw is not None and not isinstance(validation_issues_raw, list):
            raise TypeError("ModelSelectionDeps: validation_issues must be a list of ValidationIssueModel")
        if validation_issues_raw is None:
            validation_issues_raw = []
        else:
            validated_issues: list[ValidationIssueModel] = []
            for issue in validation_issues_raw:
                if isinstance(issue, dict):
                    validated_issues.append(ValidationIssueModel(**issue))
                elif isinstance(issue, ValidationIssueModel):
                    validated_issues.append(issue)
                else:
                    raise TypeError("ModelSelectionDeps: each item in validation_issues must be a dict or ValidationIssueModel")
            validation_issues_raw = validated_issues
        

        validation_status = "PASS"
        for issue in validation_issues_raw:
            if issue.severity == "WARN":
                validation_status = "WARN"
                break
            elif issue.severity == "FAIL":
                raise Exception(f"Validation failed: {issue}")
            
        return cls(
            dataset_id=dataset_id_raw,
            dataset_summary=dataset_summary_raw,
            causal_spec=causal_spec_raw,
            transformation_plan=transformation_plan_raw,
            validation_issues=validation_issues_raw,
            validation_status=validation_status,
        )


__all__ = ["ModelSelectionDeps"]
