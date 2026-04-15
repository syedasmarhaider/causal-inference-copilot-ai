from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from python.domain.models.errors import NodeExecutionError
from python.domain.models.models import ChatMessage
from python.domain.models.validation import ValidationIssueModel
from python.domain.workflows.node_state import Action, State, Status
from python.implementation.workflows.utils.utils import uuid_from_any

ValidatePhase = Literal["INIT", "REVIEW_READY", "CONFIRMED", "FAILED"]


class ValidatePayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_id: UUID | None = None
    validation_issues: list[ValidationIssueModel] = Field(default_factory=list)
    phase: ValidatePhase = "INIT"
    assistant_message: str | None = None
    system_message: str | None = None
    error_message: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_dataset_id(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        if "dataset_id" not in normalized and "data_set_id" in normalized:
            normalized["dataset_id"] = normalized.pop("data_set_id")
        return normalized

    @field_validator("dataset_id", mode="before")
    @classmethod
    def _parse_dataset_id(cls, value: Any) -> UUID | None:
        return uuid_from_any(value)

    @field_validator(
        "assistant_message",
        "system_message",
        "error_message",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        raise TypeError("text fields must be str|null")

    def bind_dataset(self, dataset_id: UUID) -> ValidatePayloadModel:
        return self.model_copy(update={"dataset_id": dataset_id})

    def reset_for_recompile(self, *, dataset_id: UUID | None) -> ValidatePayloadModel:
        return self.model_copy(
            update={
                "dataset_id": dataset_id,
                "validation_issues": [],
                "phase": "INIT",
                "assistant_message": None,
                "system_message": None,
                "error_message": None,
            }
        )


class ValidateState(State):
    NAME: ClassVar[str] = "VALIDATE"

    def __init__(self, payload: ValidatePayloadModel) -> None:
        self.payload = payload

    def name(self) -> str:
        return self.NAME

    def status(self) -> Status:
        if self.payload.phase == "CONFIRMED":
            return "DONE"
        if self.payload.phase == "FAILED":
            return "ABORTED"
        return "PENDING"

    def action(self) -> Action:
        if self.status() == "PENDING":
            return "NEEDS_INPUT"
        return "NONE"

    def set_status_pending(self) -> None:
        if self.payload.phase in ("CONFIRMED", "FAILED"):
            dataset_id = self.payload.dataset_id
            self.payload = self.payload.reset_for_recompile(dataset_id=dataset_id)

    def messages(self) -> Sequence[ChatMessage]:
        messages: list[ChatMessage] = []
        if self.payload.system_message:
            messages.append(ChatMessage(role="system", content=self.payload.system_message))
        if self.payload.assistant_message:
            messages.append(ChatMessage(role="assistant", content=self.payload.assistant_message))
        if messages:
            return messages
        return [
            ChatMessage(
                role="assistant",
                content=(
                    "I will now validate the accepted causal specification and accepted "
                    "preprocessing plan against the active scoped dataset, then ask for final "
                    "confirmation if there are no blocking failures."
                ),
            )
        ]

    def error(self) -> NodeExecutionError | None:
        if self.payload.error_message is None:
            return None
        return NodeExecutionError(state_name=self.NAME, error=self.payload.error_message)

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ValidateState:
        return cls(ValidatePayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> ValidateState:
        return cls(ValidatePayloadModel())


CompileAndValidatePhase = ValidatePhase
CompileAndValidatePayloadModel = ValidatePayloadModel
CompileAndValidateState = ValidateState


__all__ = [
    "CompileAndValidatePayloadModel",
    "CompileAndValidatePhase",
    "CompileAndValidateState",
    "ValidatePayloadModel",
    "ValidatePhase",
    "ValidateState",
]
