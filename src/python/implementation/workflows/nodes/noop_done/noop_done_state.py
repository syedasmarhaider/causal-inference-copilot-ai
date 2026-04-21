from __future__ import annotations

from typing import Any, ClassVar

from python.domain.workflows.node_state import NodeState


class NoopDoneState(NodeState):
    NAME: ClassVar[str] = "NOOP_DONE"

    def name(self) -> str:
        return self.NAME

    def clear_state(self) -> None:
        return None

    def to_json_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> NoopDoneState:
        _ = payload
        return cls()

    @classmethod
    def init_empty(cls) -> NoopDoneState:
        return cls()


__all__ = ["NoopDoneState"]
