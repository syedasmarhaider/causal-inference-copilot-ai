from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.utils.validation import ValidationIssueModel


# =============================================================================
# Payloads (strict, payload-only; NO embedded "name")
# =============================================================================
class CleanProtocolValidationPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    issues: List[ValidationIssueModel] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    validation_error: Optional[str] = None
    user_message: Optional[str] = None

    @field_validator("validation_error", "user_message", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        raise TypeError("validation_error/user_message must be str|null")


class ValidateCleanProtocolPayloadModel(BaseModel):
    """
    Payload for VALIDATE_CLEAN_PROTOCOL state.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    validation: Optional[CleanProtocolValidationPayloadModel] = None


# =============================================================================
# State wrapper (payload-only storage; derives interface fields)
# =============================================================================
class ValidateCleanProtocolState(State):
    NAME: ClassVar[str] = "VALIDATE_CLEAN_PROTOCOL"

    def __init__(self, payload: ValidateCleanProtocolPayloadModel) -> None:
        self.payload = payload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        # Preserve your existing semantics:
        # - If validation exists and has any FAIL issue => ABORTED
        # - Otherwise DONE (including when validation is None)
        v = self.payload.validation
        if v is not None and any(i.severity == "FAIL" for i in v.issues):
            return "ABORTED"
        return "DONE"

    @property
    def message(self) -> Optional[str]:
        v = self.payload.validation
        return v.user_message if v is not None else None

    @property
    def error(self) -> Optional[str]:
        v = self.payload.validation
        return v.validation_error if v is not None else None

    @property
    def needs_action(self) -> ACTION:
        return "NONE"

    def required_states_keys(self) -> Sequence[str]:
        return (CompileProtocolState.NAME, CleanProtocolState.NAME)

    def to_json_dict(self) -> Dict[str, Any]:
        return self.payload.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "ValidateCleanProtocolState":
        model = ValidateCleanProtocolPayloadModel.model_validate(payload)
        return cls(model)