from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Sequence

from python.domain.workflows.state import ACTION, Status, State
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState


@dataclass(frozen=True)
class ProtocolDiscussionState(State):
    NAME: ClassVar[str] = "PROTOCOL_DISCUSSION"

    discussion: str = ""
    node_message: Optional[str] = None
    error_message: Optional[str] = None
    action: ACTION = "NONE"
    node_status: Status = "PENDING"

    @property
    def status(self) -> Status:
        return self.node_status

    @property
    def message(self) -> Optional[str]:
        if self.node_message is None:
            return None
        msg = self.node_message.strip()
        return msg or None

    @property
    def error(self) -> Optional[str]:
        if self.error_message is None:
            return None
        err = self.error_message.strip()
        return err or None

    @property
    def needs_action(self) -> ACTION:
        return self.action
    
    def required_states_keys(self) -> Sequence[str]:
        return [LoadDatasetState.NAME]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "discussion": self.discussion,
            "node_message": self.node_message,
            "error_message": self.error_message,
            "action": self.action,
            "node_status": self.node_status,
        }

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "ProtocolDiscussionState":
        name = payload.get("name")
        if name is not None and name != cls.NAME:
            raise ValueError(f"ProtocolDiscussionState.from_json_dict: name mismatch: {name!r} != {cls.NAME!r}")

        discussion = payload.get("discussion", "")
        if not isinstance(discussion, str):
            raise ValueError("ProtocolDiscussionState.from_json_dict: discussion must be str")

        node_message = payload.get("node_message")
        if node_message is not None and not isinstance(node_message, str):
            raise ValueError("ProtocolDiscussionState.from_json_dict: node_message must be str|null")

        error_message = payload.get("error_message")
        if error_message is not None and not isinstance(error_message, str):
            raise ValueError("ProtocolDiscussionState.from_json_dict: error_message must be str|null")

        action = payload.get("action", "NONE")
        if action not in ("NONE", "NEEDS_INPUT"):
            raise ValueError("ProtocolDiscussionState.from_json_dict: action must be NONE|NEEDS_INPUT")

        node_status = payload.get("node_status", "PENDING")
        if node_status not in ("PENDING", "DONE", "ABORTED"):
            raise ValueError("ProtocolDiscussionState.from_json_dict: node_status must be PENDING|DONE|ABORTED")

        return cls(
            discussion=discussion,
            node_message= node_message,
            error_message= error_message,
            action= action,
            node_status=node_status,
        )
