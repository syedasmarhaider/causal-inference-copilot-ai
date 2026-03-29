from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from python.domain.models.errors import NodeExecutionError
from python.domain.workflows.state import State, StateMessage, Status
from python.implementation.workflows.nodes.model_selection.model_selection_deps import (
    ModelSelectionDeps,
)


class ConfirmedModelSelectionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    selected_model: str | None = None
    reasoning: str | None = None


class ModelSelectionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    confirmed_model_selection: ConfirmedModelSelectionPayload | None = None
    message: str | None = None
    error: str | None = None
    system_choice_message: str | None = None


@dataclass(frozen=True, slots=True)
class ModelSelectionState(State):
    NAME: ClassVar[str] = "MODEL_SELECTION"
    payload: ModelSelectionPayload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def error(self) -> NodeExecutionError | None:
        if self.payload.error is not None:
            return NodeExecutionError(state_name=self.NAME, error=self.payload.error)
        return None

    @property
    def status(self) -> Status:
        if self.error is not None:
            return "ABORTED"

        cms = self.payload.confirmed_model_selection
        if cms is not None and cms.selected_model is not None and cms.reasoning is not None:
            return "DONE"

        return "PENDING"

    @property
    def message(self) -> StateMessage:
        if self.payload.message is None:
            raise ValueError(
                "ModelSelectionState.message is required but missing. "
                "Don't access .message outside a node context where message is guaranteed."
            )
        action = "NONE" if self.status in ("ABORTED", "DONE") else "NEEDS_INPUT"    
        return StateMessage(txt_message=self.payload.message, action=action)

    def pre_required_states_names(self) -> Sequence[str]:
        return ModelSelectionDeps.pre_required_states_names()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ModelSelectionState:
        model = ModelSelectionPayload.model_validate(payload)
        return cls(payload=model)

    @classmethod
    def init_empty(cls) -> ModelSelectionState:
        return cls(payload=ModelSelectionPayload())