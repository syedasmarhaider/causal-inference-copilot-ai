from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Sequence, Any
from python.domain.workflows.state import ACTION, State, StateMessage, Status


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
        return StateMessage(txt_message="done")

    @property
    def error(self) -> Optional[str]:
        return None

    @property
    def needs_action(self) -> ACTION:
        return "NONE"

    def pre_required_states_names(self) -> Sequence[str]:
        return []

    def to_json_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "NoopDoneState":
        return cls()
    
    @classmethod
    def init_empty(cls) -> "NoopDoneState":
        return cls()
