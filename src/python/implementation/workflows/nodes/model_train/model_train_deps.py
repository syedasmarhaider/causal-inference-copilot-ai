from __future__ import annotations
from dataclasses import dataclass
from uuid import UUID

from python.domain.models.errors import StateDependencyError
from python.domain.workflows.ochestrator_state import ReadOnlyOchestratorState
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)


@dataclass(frozen=True, slots=True)
class ModelTrainDeps:
    dataset_id: UUID
    inference_ready_spec: InferenceReadyCausalSpec
    selected_model: str


    @classmethod
    def from_loaded(cls, readonly_orchestrator_state: ReadOnlyOchestratorState) -> ModelTrainDeps:
        dataset_id = readonly_orchestrator_state.get("working_dataset_id")
        inference_ready = readonly_orchestrator_state.get("inference_ready_spec")
        selected = readonly_orchestrator_state.get("selected_model")
        if dataset_id is None or inference_ready is None or selected is None:
            raise StateDependencyError(
                "MODEL_TRAIN",
                "MODEL_TRAIN",
                ["working_dataset_id", "inference_ready_spec", "selected_model"],
            )
    
    
        return cls(
            dataset_id=dataset_id,
            inference_ready_spec=inference_ready,
            selected_model=selected.selected_model,
        )
