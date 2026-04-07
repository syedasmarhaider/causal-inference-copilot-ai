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
        causal_spec = read_only_ochestration_state.get("causal_spec")
        data_transformation_plan = read_only_ochestration_state.get("data_transformation_plan")
        selected_model = read_only_ochestration_state.get("selected_model")
        trained_model_id = read_only_ochestration_state.get("model_training_id")

        if (
            dataset_id is None
            or dataset_summary is None
            or causal_spec is None
            or data_transformation_plan is None
            or selected_model is None
            or trained_model_id is None
        ):
            raise StateDependencyError(
                "CAUSAL_INFERENCE",
                "CAUSAL_INFERENCE",
                [
                    "working_dataset_id",
                    "working_dataset_summary",
                    "causal_spec",
                    "data_transformation_plan",
                    "selected_model",
                    "model_training_id",
                ],
            )

        inference_ready_spec = InferenceReadyCausalSpec(
            causal_spec=causal_spec,
            transformation_plan=data_transformation_plan,
            data_summary=dataset_summary,
        )

        return cls(
            dataset_id=dataset_id,
            dataset_summary=dataset_summary,
            inference_ready_spec=inference_ready_spec,
            selected_model=selected_model,
            trained_model_id=trained_model_id,
        )
