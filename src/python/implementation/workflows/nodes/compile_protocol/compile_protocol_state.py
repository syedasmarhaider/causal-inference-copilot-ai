from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, cast

from python.domain.workflows.state import ACTION, Status, State
from python.implementation.workflows.nodes.compile_protocol.protocol_specs import ProtocolSpec
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import ProtocolDiscussionState
from python.implementation.workflows.utils.utils import json_sanitize


@dataclass(frozen=True)
class CompileProtocolState(State):
    NAME: ClassVar[str] = "COMPILE_PROTOCOL"
    protocol: Optional[ProtocolSpec] = None
    compile_error: Optional[str] = None
    compile_issues: Optional[List[Dict[str, Any]]] = None
    user_message: Optional[str] = None

    @property
    def status(self) -> Status:
        if self.protocol is not None:
            return "DONE"
        return "ABORTED"

    @property
    def message(self) -> Optional[str]:
        msg = (self.user_message or "").strip()
        return msg or None

    @property
    def error(self) -> Optional[str]:
        err = (self.compile_error or "").strip()
        return err or None

    @property
    def needs_action(self) -> ACTION:
        return "NONE"

    def required_states_keys(self) -> Sequence[str]:
        return [LoadDatasetState.NAME, ProtocolDiscussionState.NAME]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "protocol": None if self.protocol is None else json_sanitize(self.protocol),
            "compile_error": self.compile_error,
            "compile_issues": None if self.compile_issues is None else json_sanitize(self.compile_issues),
            "user_message": self.user_message,
        }

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "CompileProtocolState":
        name = payload.get("name")
        if name is not None and name != cls.NAME:
            raise ValueError(f"CompileProtocolState.from_json_dict: name mismatch: {name!r} != {cls.NAME!r}")

        protocol_val = payload.get("protocol")
        compile_error_val = payload.get("compile_error")
        compile_issues_val = payload.get("compile_issues")
        user_message_val = payload.get("user_message")

        if compile_error_val is not None and not isinstance(compile_error_val, str):
            raise ValueError("CompileProtocolState.from_json_dict: compile_error must be str|null")

        if user_message_val is not None and not isinstance(user_message_val, str):
            raise ValueError("CompileProtocolState.from_json_dict: user_message must be str|null")

        if compile_issues_val is not None:
            if not isinstance(compile_issues_val, list) or any(not isinstance(x, dict) for x in compile_issues_val): # pyright: ignore[reportUnknownVariableType]
                raise ValueError("CompileProtocolState.from_json_dict: compile_issues must be list[dict]|null")

        protocol_spec: Optional[ProtocolSpec]
        if protocol_val is None:
            protocol_spec = None
        else:
            if not isinstance(protocol_val, dict):
                raise ValueError("CompileProtocolState.from_json_dict: protocol must be object|null")
            protocol_spec = cast(ProtocolSpec, protocol_val)

        return cls(
            protocol=protocol_spec,
            compile_error=compile_error_val,
            compile_issues=cast(Optional[List[Dict[str, Any]]], compile_issues_val),
            user_message=user_message_val,
        )
