from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from python.domain.models.errors import NodeExecutionError
from python.domain.models.models import ChatMessage
from python.domain.workflows.state import State, Status


class NoopDoneState(State):
    NAME: ClassVar[str] = "NOOP_DONE"

    def name(self) -> str:
        return self.NAME

    def status(self) -> Status:
        return "DONE"

    def set_status_freez(self) -> None:
        return None

    def set_status_pending(self) -> None:
        return None

    def messages(self) -> Sequence[ChatMessage]:
        return [ChatMessage(role="assistant", content="Workflow is complete.")]

    def error(self) -> NodeExecutionError | None:
        return None

    def pre_required_states_names(self) -> Sequence[str]:
        return ()

    def to_json_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> NoopDoneState:
        _ = payload
        return cls()

    @classmethod
    def init_empty(cls) -> NoopDoneState:
        return cls()
