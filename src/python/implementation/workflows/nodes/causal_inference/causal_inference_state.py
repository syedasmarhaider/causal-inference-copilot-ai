from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from python.domain.models.errors import NodeExecutionError
from python.domain.workflows.state import State, StateMessage, Status
from python.implementation.workflows.nodes.causal_inference.causal_inference_deps import (
    CausalInferenceDeps,
)


class CausalInferencePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ate_result_raw_json_str: str | None = None
    error: str | None = None
    
    # workflow control
    should_abort: bool | None = None

    # UI / node-local
    message: str | None = None
    artifacts: list[UUID] | None = None


@dataclass(frozen=True, slots=True)
class CausalInferenceState(State):
    NAME: ClassVar[str] = "CAUSAL_INFERENCE"
    payload: CausalInferencePayload
    current_artifact_ids: list[UUID] | None = None

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def error(self) -> NodeExecutionError | None:
        if self.payload.error is not None:
            return NodeExecutionError(state_name=self.NAME, error=self.payload.error)
        return None

    @property
    def status(self) -> Status:
        if self.payload.should_abort:
            return "ABORTED"
        return "PENDING"

    @property
    def message(self) -> StateMessage:
        if self.payload.message is None:
            raise ValueError(
                "CausalInferenceState.message is required but missing. "
                "Don't access .message outside the node/UI context where user_message is guaranteed."
            )
        return StateMessage(
            txt_message=self.payload.message,
            action="NONE" if self.status == "ABORTED" else "NEEDS_INPUT",
            artifact_ids=[str(aid) for aid in self.current_artifact_ids] if self.current_artifact_ids else None,
        )
        
    def pre_required_states_names(self) -> Sequence[str]:
        return CausalInferenceDeps.pre_required_states_names()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> CausalInferenceState:
        model = CausalInferencePayload.model_validate(payload)
        return cls(payload=model)

    @classmethod
    def init_empty(cls) -> CausalInferenceState:
        return cls(payload=CausalInferencePayload())
