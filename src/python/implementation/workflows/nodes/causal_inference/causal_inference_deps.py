from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from python.domain.models.errors import StateDependencyError
from python.domain.workflows.ochestrator_state import ReadOnlyOchestratorState
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)


@dataclass(frozen=True, slots=True)
class CausalInferenceDeps:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel
    inference_ready_spec: InferenceReadyCausalSpec
    selected_model: str
    trained_model_id: UUID

    @classmethod
    def from_loaded(cls, read_only_ochestration_state: ReadOnlyOchestratorState) -> CausalInferenceDeps:
        dataset_id = read_only_ochestration_state.get("working_dataset_id")
        dataset_summary = read_only_ochestration_state.get("working_dataset_summary")
        inference_ready_spec = read_only_ochestration_state.get("inference_ready_spec")
        selected = read_only_ochestration_state.get("selected_model")
        trained_model_id = read_only_ochestration_state.get("model_training_id")
        
        if (dataset_id is None 
            or dataset_summary is None 
            or inference_ready_spec is None 
            or selected is None
            or trained_model_id is None
        ):
            raise StateDependencyError(
                "CAUSAL_INFERENCE",
                "CAUSAL_INFERENCE",
                [
                    "working_dataset_id", 
                    "working_dataset_summary", 
                    "inference_ready_spec", 
                    "selected_model",
                    "model_training_id",
                ],
            )
       

        return cls(
            dataset_id=dataset_id,
            dataset_summary=dataset_summary,
            inference_ready_spec=inference_ready_spec,
            selected_model=selected.selected_model,
            trained_model_id=trained_model_id,
        )
