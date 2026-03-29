from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.models.errors import NodeExecutionError
from python.domain.workflows.state import State, StateMessage, Status
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_deps import (
    ValidateCleanProtocolDeps,
)
from python.implementation.workflows.utils.validation import ValidationIssueModel


class ValidateCleanProtocolPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    issues: list[ValidationIssueModel] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    validation_error: str | None = None
    user_acceptance: bool | None = None
    user_message: str | None = None

    @field_validator("validation_error", "user_message", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        raise TypeError("validation_error/user_message must be str|null")


class ValidateCleanProtocolState(State):
    NAME: ClassVar[str] = "VALIDATE_CLEAN_PROTOCOL"

    def __init__(self, payload: ValidateCleanProtocolPayloadModel) -> None:
        self.payload = payload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        # ABORTED if any FAIL issue; else DONE
        if any(i.severity == "FAIL" for i in self.payload.issues) or (self.payload.user_acceptance is not None and not self.payload.user_acceptance):
            return "ABORTED"
        if self.payload.user_acceptance is not None and self.payload.user_acceptance:
            return "DONE"
        return "PENDING"

    @property
    def message(self) -> StateMessage:
        if self.payload.user_message is None:
            raise ValueError("ValidateCleanProtocolState message is required but missing. State must have user message. Dont call this property if this is not runned in the node context where user_message is guaranteed to be set.")
        if self.status in ("ABORTED", "DONE"):
            action = "NONE"
        else:
            action = "NEEDS_INPUT"
        return StateMessage(txt_message=self.payload.user_message,action=action)

    @property
    def error(self) -> NodeExecutionError | None:
        if self.payload.validation_error is None:
            return None
        return NodeExecutionError(state_name=self.NAME, error=self.payload.validation_error)

    def pre_required_states_names(self) -> Sequence[str]:
        return ValidateCleanProtocolDeps.pre_required_states_names()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ValidateCleanProtocolState:
        model = ValidateCleanProtocolPayloadModel.model_validate(payload)
        return cls(model)
    
    @classmethod
    def init_empty(cls) -> ValidateCleanProtocolState:
        return cls(ValidateCleanProtocolPayloadModel())