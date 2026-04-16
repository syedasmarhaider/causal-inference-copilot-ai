from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from python.domain.workflows.node_state import NodeState


class GeneralQueriesPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    assistant_message: str | None = None


class GeneralQueriesState(NodeState):
    NAME: ClassVar[str] = "GENERAL_QUERIES"

    def __init__(self, payload: GeneralQueriesPayloadModel) -> None:
        self.payload = payload

    def name(self) -> str:
        return self.NAME

    def clear_state(self) -> None:
        self.payload.assistant_message = None

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> GeneralQueriesState:
        return cls(GeneralQueriesPayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> GeneralQueriesState:
        return cls(GeneralQueriesPayloadModel())


__all__ = [
    "GeneralQueriesPayloadModel",
    "GeneralQueriesState",
]
