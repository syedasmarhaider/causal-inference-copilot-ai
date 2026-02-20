from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field
from pydantic.types import StringConstraints
from typing import Annotated

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.compile_inference.compile_inference_state import CompileInferenceState
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

ValidationSeverity = Literal["WARN", "FAIL"]
ValidationStatus = Literal["PASS", "WARN", "FAIL"]

class InferenceReadyValidationIssueModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    severity: ValidationSeverity
    message: NonEmptyStr
    evidence: Dict[str, Any] = Field(default_factory=dict)
    fix_hint: Optional[NonEmptyStr] = None


class InferenceReadyValidationPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    issues: List[InferenceReadyValidationIssueModel] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]
    validation_error: Optional[str] = None
    user_message: Optional[str] = None


@dataclass(frozen=True)
class ValidateCompiledInferenceState(State):
    NAME: ClassVar[str] = "VALIDATE_COMPILED_INFERENCE"
    payload: Optional[InferenceReadyValidationPayloadModel] = None

    @property
    def status(self) -> Status:
        if self.payload is not None:
            if any(issue.severity == "FAIL" for issue in self.payload.issues):
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
            "validation_error": self.error,
            "user_message": self.message,
        }

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "ValidateCompiledInferenceState":
        raw = payload.get("payload")
        model = InferenceReadyValidationPayloadModel.model_validate(raw) if isinstance(raw, dict) else None
        return cls(
                payload=model,
       )
