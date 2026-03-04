from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from python.domain.workflows.state import ACTION, State, StateMessage, Status
from python.implementation.workflows.nodes.causal_inference.causal_inference_deps import CausalInferenceDeps


class CausalInferencePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ate_result_raw_json_str: Optional[str] = None
    error: Optional[str] = None
    
    # workflow control
    should_abort: Optional[bool] = None

    # UI / node-local
    message: Optional[str] = None


@dataclass(frozen=True, slots=True)
class CausalInferenceState(State):
    NAME: ClassVar[str] = "CAUSAL_INFERENCE"
    payload: CausalInferencePayload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def error(self) -> Optional[str]:
        return self.payload.error

    # TODO: chnage
    @property
    def status(self) -> Status:
        if self.payload.should_abort:
            return "PENDING"
        if self.error is not None:
            return "PENDING"
        return "PENDING"

    @property
    def message(self) -> StateMessage:
        if self.payload.message is None:
            raise ValueError(
                "CausalInferenceState.message is required but missing. "
                "Don't access .message outside the node/UI context where user_message is guaranteed."
            )
        return StateMessage(txt_message=self.payload.message)

    # TODO: change later
    @property
    def needs_action(self) -> ACTION:
        if self.status in ("DONE", "ABORTED"):
            return "NEEDS_INPUT"
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