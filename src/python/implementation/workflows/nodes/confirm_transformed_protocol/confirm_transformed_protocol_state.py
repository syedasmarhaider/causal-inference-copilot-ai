from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_state import TransformProtocolState


class ConfirmTransformedProtocolPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_accepted: Optional[bool] = None
    user_message: Optional[str] = None
    error_message: Optional[str] = None


class ConfirmTransformedProtocolState(State):
    NAME: ClassVar[str] = "CONFIRM_TRANSFORMED_PROTOCOL"

    def __init__(self, payload: ConfirmTransformedProtocolPayloadModel) -> None:
        self.payload = payload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        if self.error is not None or (self.payload.user_accepted is not None and self.payload.user_accepted is False):
            return "ABORTED"
        if self.payload.user_accepted and self.payload.user_accepted is True:
            return "DONE"
        return "PENDING"

    @property
    def message(self) -> Optional[str]:
        return self.payload.user_message

    @property
    def error(self) -> Optional[str]:
        return self.payload.error_message

    @property
    def needs_action(self) -> ACTION:
        return "NEEDS_INPUT" if self.payload.user_accepted is None else "NONE"

    def required_states_keys(self) -> Sequence[str]:
        return [TransformProtocolState.NAME]

    def to_json_dict(self) -> Dict[str, Any]:
        return self.payload.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "ConfirmTransformedProtocolState":
        model = ConfirmTransformedProtocolPayloadModel.model_validate(payload)
        return cls(model)