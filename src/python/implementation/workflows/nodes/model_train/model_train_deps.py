from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)


@dataclass(frozen=True)
class ModelTrainDeps:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel
    causal_spec: CausalSpec
    transformation_plan: TransformPlan
    selected_model: str

    @classmethod
    def from_request(cls, request: NodeRequest) -> ModelTrainDeps:
        transformation_plan_raw = request.orchestrator_state.get("data_transformation_plan")
        causal_spec_raw = request.orchestrator_state.get("causal_spec")
        dataset_id_raw = request.orchestrator_state.get("working_dataset_id")
        dataset_summary_raw = request.orchestrator_state.get("latest_dataset_summary")
        selected_model_raw = request.orchestrator_state.get("selected_model")

        if dataset_id_raw is None:
            raise ValueError("ModelTrainDeps: dataset_id is required but was not found in compilation state")
        if dataset_summary_raw is None:
            raise ValueError("ModelTrainDeps: dataset_summary is required but was not found in compilation state")
        if causal_spec_raw is None:
            raise ValueError("ModelTrainDeps: causal_spec is required but was not found in compilation state")
        if transformation_plan_raw is None:
            raise ValueError("ModelTrainDeps: transformation_plan is required but was not found in compilation state")
        if selected_model_raw is None:
            raise ValueError("ModelTrainDeps: selected_model is required but was not found in compilation state")

        if not isinstance(dataset_id_raw, UUID):
            raise TypeError("ModelSelectionDeps: dataset_id must be a UUID")
        if not isinstance(dataset_summary_raw, DatasetSummaryModel):
            raise TypeError("ModelSelectionDeps: dataset_summary must be of type DatasetSummaryModel")
        if not isinstance(causal_spec_raw, CausalSpec):
            raise TypeError("ModelSelectionDeps: causal_spec must be of type CausalSpec")
        if not isinstance(transformation_plan_raw, TransformPlan):
            raise TypeError("ModelSelectionDeps: transformation_plan must be of type TransformPlan")
        if not isinstance(selected_model_raw, str):
            raise TypeError("ModelTrainDeps: selected_model must be a string")
        
        return cls(
            dataset_id=dataset_id_raw,
            dataset_summary=dataset_summary_raw,
            causal_spec=causal_spec_raw,
            transformation_plan=transformation_plan_raw,
            selected_model=selected_model_raw,
        )


__all__ = ["ModelTrainDeps"]
