from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Sequence
from uuid import UUID

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState


@dataclass(frozen=True)
class InferenceReadyState(State):
    """
    Minimal inference-ready marker state for your pipeline stage "INFERENCE_READY".

    Semantics:
      - DONE:     clean_dataset_id is set and cleaning_error is not set
      - ABORTED:  cleaning_error is set (regardless of id)
      - PENDING:  neither id nor error is set (defensive)
    """

    NAME: ClassVar[str] = "INFERENCE_READY"

    clean_dataset_id: Optional[UUID] = None
    cleaning_error: Optional[str] = None
    user_message: Optional[str] = None

    @property
    def status(self) -> Status:
        if self.cleaning_error is not None:
            return "ABORTED"
        if self.clean_dataset_id is not None:
            return "DONE"
        return "PENDING"

    @property
    def message(self) -> Optional[str]:
        return self.user_message

    @property
    def error(self) -> Optional[str]:
        return self.cleaning_error

    @property
    def needs_action(self) -> ACTION:
        # If this state aborted, the pipeline typically needs user correction/clarification.
        if self.status == "ABORTED":
            return "NEEDS_INPUT"
        return "NONE"

    def required_states_keys(self) -> Sequence[str]:
        # Dependency keys are always State.NAME values.
        return [LoadDatasetState.NAME, CompileProtocolState.NAME]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "clean_dataset_id": str(self.clean_dataset_id) if self.clean_dataset_id else None,
            "cleaning_error": self.cleaning_error,
            "user_message": self.user_message,
        }

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "InferenceReadyState":
        return cls(
            clean_dataset_id=_parse_uuid(payload.get("clean_dataset_id")),
            cleaning_error=_as_opt_str(payload.get("cleaning_error")),
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
