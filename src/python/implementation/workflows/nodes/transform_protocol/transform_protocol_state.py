from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.compile_inference.compile_inference_state import (
    CompileInferenceState,
)
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import (
    CompileProtocolState,
)
from python.implementation.workflows.utils.validation import ValidationIssueModel


class TransformProtocolValidationPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    issues: List[ValidationIssueModel] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]
    validation_error: Optional[str] = None
    user_message: Optional[str] = None
    transform_dataset_id: Optional[str] = None


@dataclass(frozen=True)
class TransformProtocol(State):
    NAME: ClassVar[str] = "TRANSFORM_PROTOCOL"

    payload: Optional[TransformProtocolValidationPayloadModel] = None

    @property
    def status(self) -> Status:
        if self.payload and any(i.severity == "FAIL" for i in self.payload.issues):
            return "ABORTED"
        return "DONE"

    @property
    def message(self) -> Optional[str]:
        return self.payload.user_message if self.payload else None

    @property
    def error(self) -> Optional[str]:
        return self.payload.validation_error if self.payload else None

    @property
    def needs_action(self) -> ACTION:
        return "NONE"

    def required_states_keys(self) -> Sequence[str]:
        return [CompileProtocolState.NAME, CompileInferenceState.NAME]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "payload": self.payload.model_dump(mode="json") if self.payload else None,
        }

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "TransformProtocol":
        raw = payload.get("payload")
        model = (
            TransformProtocolValidationPayloadModel.model_validate(raw)
            if isinstance(raw, dict)
            else None
        )
        return cls(payload=model)