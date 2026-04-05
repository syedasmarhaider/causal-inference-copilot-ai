from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.errors import NodeExecutionError
from python.domain.models.models import ChatMessage
from python.domain.workflows.state import Action, State, Status
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_deps import (
    CompileAndValidateDeps,
)
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.domain.models.validation import ValidationIssueModel

CompileAndValidatePhase = Literal["INIT", "REVIEW_READY", "CONFIRMED", "FAILED"]


class CompileAndValidatePayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    compiled_causal_spec: CausalSpec | None = None
    transformation_plan: TransformPlan | None = None
    inference_ready_causal_spec: InferenceReadyCausalSpec | None = None
    validation_issues: list[ValidationIssueModel] = Field(default_factory=lambda: [])
    phase: CompileAndValidatePhase = "INIT"
    assistant_message: str | None = None
    system_message: str | None = None
    error_message: str | None = None
    freezed: bool = False

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


class CompileAndValidateState(State):
    NAME: ClassVar[str] = "COMPILE_AND_VALIDATE"

    def __init__(self, payload: CompileAndValidatePayloadModel) -> None:
        self.payload = payload

    def name(self) -> str:
        return self.NAME

    def status(self) -> Status:
        if self.payload.freezed:
            return "FREEZED"
        if self.payload.phase == "CONFIRMED":
            return "DONE"
        if self.payload.phase == "FAILED":
            return "ABORTED"
        return "PENDING"
    
    def action(self) -> Action:
        if self.status() == "PENDING":
            return "NEEDS_INPUT"
        return "NONE"

    def set_status_freez(self) -> None:
        if self.payload.phase in ("CONFIRMED"):
            self.payload.freezed = True
        else:
            raise ValueError(f"Can only freeze when phase is CONFIRMED, current phase: {self.payload.phase!r}")    

    def set_status_pending(self) -> None:
        if self.payload.phase in ("CONFIRMED", "FAILED"):
            self.payload.phase = "INIT"
            self.payload.error_message = None

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
                    "I will now compile the confirmed protocol into a causal specification "
                    "and a preprocessing plan, validate them against the active dataset, "
                    "and then ask for confirmation if everything is clinically coherent."
                ),
            )
        ]

    def error(self) -> NodeExecutionError | None:
        if self.payload.error_message is None:
            return None
        return NodeExecutionError(state_name=self.NAME, error=self.payload.error_message)

    def pre_required_states_names(self) -> Sequence[str]:
        return CompileAndValidateDeps.pre_required_states_names()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> CompileAndValidateState:
        return cls(CompileAndValidatePayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> CompileAndValidateState:
        return cls(CompileAndValidatePayloadModel())
