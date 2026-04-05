from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from python.domain.models.errors import StateDependencyError
from python.domain.workflows.state import State
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_state import (
    CompileAndValidateState,
)
from python.implementation.workflows.nodes.model_selection.mode_selection_state import (
    ModelSelectionState,
)
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainState
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
    def pre_required_states_names(cls) -> Sequence[str]:
        return [
            CompileAndValidateState.NAME,
            ModelSelectionState.NAME,
            ModelTrainState.NAME,
        ]

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> CausalInferenceDeps:
        compile_state = loaded.get(CompileAndValidateState.NAME)
        if compile_state is None or not isinstance(compile_state, CompileAndValidateState):
            raise StateDependencyError(
                "CAUSAL_INFERENCE",
                "CAUSAL_INFERENCE",
                [CompileAndValidateState.NAME],
            )
        if compile_state.payload.phase != "CONFIRMED":
            raise StateDependencyError(
                "CAUSAL_INFERENCE",
                "CAUSAL_INFERENCE",
                [CompileAndValidateState.NAME],
            )

        inference_ready_spec = compile_state.payload.inference_ready_causal_spec
        dataset_id = compile_state.payload.dataset_id
        dataset_summary = compile_state.payload.dataset_summary
        if (
            inference_ready_spec is None
            or dataset_id is None
            or dataset_summary is None
        ):
            raise StateDependencyError(
                "CAUSAL_INFERENCE",
                "CAUSAL_INFERENCE",
                [CompileAndValidateState.NAME],
            )

        model_selection_state = loaded.get(ModelSelectionState.NAME)
        if model_selection_state is None or not isinstance(
            model_selection_state, ModelSelectionState
        ):
            raise StateDependencyError(
                "CAUSAL_INFERENCE",
                "CAUSAL_INFERENCE",
                [ModelSelectionState.NAME],
            )

        selected = model_selection_state.payload.confirmed_model_selection
        if selected is None or selected.selected_model is None:
            raise StateDependencyError(
                "CAUSAL_INFERENCE",
                "CAUSAL_INFERENCE",
                [ModelSelectionState.NAME],
            )

        model_train_state = loaded.get(ModelTrainState.NAME)
        if model_train_state is None or not isinstance(model_train_state, ModelTrainState):
            raise StateDependencyError(
                "CAUSAL_INFERENCE",
                "CAUSAL_INFERENCE",
                [ModelTrainState.NAME],
            )
        if model_train_state.payload.trained_model_id is None:
            raise StateDependencyError(
                "CAUSAL_INFERENCE",
                "CAUSAL_INFERENCE",
                [ModelTrainState.NAME],
            )

        return cls(
            dataset_id=dataset_id,
            dataset_summary=dataset_summary,
            inference_ready_spec=inference_ready_spec,
            selected_model=selected.selected_model,
            trained_model_id=model_train_state.payload.trained_model_id,
        )
