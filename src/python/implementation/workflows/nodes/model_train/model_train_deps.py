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
        causal_spec = readonly_orchestrator_state.get("causal_spec")
        data_transformation_plan = readonly_orchestrator_state.get("data_transformation_plan")
        dataset_summary = readonly_orchestrator_state.get("working_dataset_summary")
        selected_model = readonly_orchestrator_state.get("selected_model")
        if (
            dataset_id is None
            or causal_spec is None
            or data_transformation_plan is None
            or dataset_summary is None
            or selected_model is None
        ):
            raise StateDependencyError(
                "MODEL_TRAIN",
                "MODEL_TRAIN",
                [
                    "working_dataset_id",
                    "working_dataset_summary",
                    "causal_spec",
                    "data_transformation_plan",
                    "selected_model",
                ],
            )

        inference_ready_spec = InferenceReadyCausalSpec(
            causal_spec=causal_spec,
            transformation_plan=data_transformation_plan,
            data_summary=dataset_summary,
        )

        return cls(
            dataset_id=dataset_id,
            inference_ready_spec=inference_ready_spec,
            selected_model=selected_model,
        )
