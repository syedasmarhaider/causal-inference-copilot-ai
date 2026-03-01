from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.causal_inference.causal_inference_deps import CausalInferenceDeps


class CausalInferencePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ate_result_raw_json_str: Optional[str] = None
    ate_result_summary: Optional[str] = None

    ate_inference_error: Optional[str] = None

    # workflow control
    should_abort: Optional[bool] = None
    abort_error_message: Optional[str] = None

    # UI / node-local
    user_message: Optional[str] = None


@dataclass(frozen=True, slots=True)
class CausalInferenceState(State):
    NAME: ClassVar[str] = "CAUSAL_INFERENCE"
    payload: CausalInferencePayload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def error(self) -> Optional[str]:
        if self.payload.should_abort:
            return self.payload.abort_error_message or self.payload.ate_inference_error
        return self.payload.ate_inference_error

    @property
    def status(self) -> Status:
        if self.payload.should_abort:
            return "ABORTED"
        if self.error is not None:
            return "ABORTED"
        return "PENDING"

    @property
    def message(self) -> str:
        if self.payload.user_message is None:
            raise ValueError(
                "CausalInferenceState.message is required but missing. "
                "Don't access .message outside the node/UI context where user_message is guaranteed."
            )
        return self.payload.user_message

    @property
    def needs_action(self) -> ACTION:
        if self.status in ("DONE", "ABORTED"):
            return "NONE"
        return "NEEDS_INPUT"

    def pre_required_states_names(self) -> Sequence[str]:
        return CausalInferenceDeps.pre_required_states_names()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "CausalInferenceState":
        model = CausalInferencePayload.model_validate(payload)
        return cls(payload=model)

    @classmethod
    def init_empty(cls) -> "CausalInferenceState":
        return cls(payload=CausalInferencePayload())