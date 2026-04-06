from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from python.domain.models.errors import StateDependencyError
from python.domain.models.validation import ValidationIssueModel
from python.domain.workflows.state import State
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_state import (
    CompileAndValidateState,
)
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)


@dataclass(frozen=True, slots=True)
class ModelSelectionDeps:
    inference_ready_spec: InferenceReadyCausalSpec
    validation_warnings: list[ValidationIssueModel]

    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return [CompileAndValidateState.NAME]

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> ModelSelectionDeps:
        state = loaded.get(CompileAndValidateState.NAME)
        if state is None or not isinstance(state, CompileAndValidateState):
            raise StateDependencyError(
                "MODEL_SELECTION",
                "MODEL_SELECTION",
                [CompileAndValidateState.NAME],
            )

        if state.payload.phase != "CONFIRMED":
            raise StateDependencyError(
                "MODEL_SELECTION",
                "MODEL_SELECTION",
                [CompileAndValidateState.NAME],
            )

        inference_ready = state.payload.inference_ready_causal_spec
        if inference_ready is None:
            raise StateDependencyError(
                "MODEL_SELECTION",
                "MODEL_SELECTION",
                [CompileAndValidateState.NAME],
            )

        warnings = [issue for issue in state.payload.validation_issues if issue.severity == "WARN"]
        return cls(
            inference_ready_spec=inference_ready,
            validation_warnings=warnings,
        )

