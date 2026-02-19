from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Sequence
from uuid import UUID

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.compile_inference.inference_ready_payload import InferenceReadyPayloadModel
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState 

@dataclass(frozen=True)
class InferenceReadyState(State):
    NAME: ClassVar[str] = "INFERENCE_READY"

    id: Optional[UUID] = None
    payload: Optional[InferenceReadyPayloadModel] = None
    inference_error: Optional[str] = None
    user_message: Optional[str] = None

    @property
    def status(self) -> Status:
        if self.payload is not None:
            return "DONE"
        return "ABORTED"

    @property
    def message(self) -> Optional[str]:
        return self.user_message

    @property
    def error(self) -> Optional[str]:
        return self.inference_error

    @property
    def needs_action(self) -> ACTION:
        return "NONE"

    def required_states_keys(self) -> Sequence[str]:
        return [LoadDatasetState.NAME, CompileProtocolState.NAME]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "id": str(self.id) if self.id else None,
            "payload": self.payload.model_dump(mode="json") if self.payload else None,
            "inference_error": self.inference_error,
            "user_message": self.user_message,
        }

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "InferenceReadyState":
        rid = payload.get("id")
        state_payload = payload.get("payload")

        model: Optional[InferenceReadyPayloadModel] = None
        if isinstance(state_payload, dict):
            model = InferenceReadyPayloadModel.model_validate(state_payload)

        return cls(
            id=_parse_uuid(rid),
            payload=model,
            inference_error=_as_opt_str(payload.get("inference_error")),
            user_message=_as_opt_str(payload.get("user_message")),
        )


def _as_opt_str(v: Any) -> Optional[str]:
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return None


def _parse_uuid(v: Any) -> Optional[UUID]:
    if v is None:
        return None
    if isinstance(v, UUID):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return UUID(s)
        except Exception:
            return None
    return None
