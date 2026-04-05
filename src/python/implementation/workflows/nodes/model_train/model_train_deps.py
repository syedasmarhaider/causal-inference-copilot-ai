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
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)


@dataclass(frozen=True, slots=True)
class ModelTrainDeps:
    dataset_id: UUID
    inference_ready_spec: InferenceReadyCausalSpec
    selected_model: str

    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return [CompileAndValidateState.NAME, ModelSelectionState.NAME]

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> ModelTrainDeps:
        compile_state = loaded.get(CompileAndValidateState.NAME)
        if compile_state is None or not isinstance(compile_state, CompileAndValidateState):
            raise StateDependencyError(
                "MODEL_TRAIN",
                "MODEL_TRAIN",
                [CompileAndValidateState.NAME],
            )

        if compile_state.payload.phase != "CONFIRMED":
            raise StateDependencyError(
                "MODEL_TRAIN",
                "MODEL_TRAIN",
                [CompileAndValidateState.NAME],
            )

        inference_ready = compile_state.payload.inference_ready_causal_spec
        dataset_id = compile_state.payload.dataset_id
        if inference_ready is None or dataset_id is None:
            raise StateDependencyError(
                "MODEL_TRAIN",
                "MODEL_TRAIN",
                [CompileAndValidateState.NAME],
            )

        model_selection_state = loaded.get(ModelSelectionState.NAME)
        if model_selection_state is None or not isinstance(
            model_selection_state, ModelSelectionState
        ):
            raise StateDependencyError(
                "MODEL_TRAIN",
                "MODEL_TRAIN",
                [ModelSelectionState.NAME],
            )

        selected = model_selection_state.payload.confirmed_model_selection
        if selected is None or selected.selected_model is None:
            raise StateDependencyError(
                "MODEL_TRAIN",
                "MODEL_TRAIN",
                [ModelSelectionState.NAME],
            )

        return cls(
            dataset_id=dataset_id,
            inference_ready_spec=inference_ready,
            selected_model=selected.selected_model,
        )
