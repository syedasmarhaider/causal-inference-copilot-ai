from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from python.domain.models.errors import NodeExecutionError
from python.domain.workflows.state import State, StateMessage, Status


@dataclass(frozen=True)
class NoopDoneState(State):
    """
    Terminal no-op state: always DONE, no message, no error, no action.
    Useful as a sink node or placeholder stage implementation.
    """
    NAME: ClassVar[str] = "NOOP_DONE"

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        return "DONE"

    @property
    def message(self) -> StateMessage:
        return StateMessage(txt_message="done",action="NONE")

    @property
    def error(self) -> NodeExecutionError | None:
        return None

    def pre_required_states_names(self) -> Sequence[str]:
        return []

    def to_json_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> NoopDoneState:
        return cls()
    
    @classmethod
    def init_empty(cls) -> NoopDoneState:
        return cls()
