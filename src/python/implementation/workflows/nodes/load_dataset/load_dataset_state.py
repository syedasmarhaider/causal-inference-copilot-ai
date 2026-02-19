from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Sequence, cast
from uuid import UUID

from python.domain.workflows.state import ACTION, Status, State
from python.implementation.workflows.nodes.load_dataset.load_dataset_utils import DatasetSummary
from python.implementation.workflows.utils.utils import json_sanitize, uuid_from_any, uuid_to_str


@dataclass(frozen=True)
class LoadDatasetState(State):
    NAME: ClassVar[str] = "LOAD_DATASET"
    id: Optional[UUID] = None
    summary: Optional[DatasetSummary] = None
    load_error: Optional[str] = None
    user_message: Optional[str] = None
    

    @property
    def status(self) -> Status:
        if self.load_error:
            return "ABORTED"
        if self.summary is not None:
            return "DONE"
        return "PENDING"

    @property
    def message(self) -> Optional[str]:
        if self.user_message is None:
            return None
        msg = self.user_message.strip()
        return msg or None

    @property
    def error(self) -> Optional[str]:
        if self.load_error is None:
            return None
        err = self.load_error.strip()
        return err or None

    @property
    def needs_action(self) -> ACTION:
        return "NEEDS_INPUT" if self.error is not None else "NONE"
    
    def required_states_keys(self) -> Sequence[str]:
        return []


    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "name": self.NAME,
            "id": uuid_to_str(self.id),
            "summary": None if self.summary is None else json_sanitize(self.summary),
            "load_error": self.load_error,
            "user_message": self.user_message,
        }

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "LoadDatasetState":
        name = payload.get("name")
        if name is not None and name != cls.NAME:
            raise ValueError(f"LoadDatasetState.from_json_dict: name mismatch: {name!r} != {cls.NAME!r}")

        id_val = payload.get("id")
        summary_val = payload.get("summary")
        load_error_val = payload.get("load_error")
        user_message_val = payload.get("user_message")

        if load_error_val is not None and not isinstance(load_error_val, str):
            raise ValueError("LoadDatasetState.from_json_dict: load_error must be str|null")

        if summary_val is not None and not isinstance(summary_val, dict):
            raise ValueError("LoadDatasetState.from_json_dict: summary must be object|null")

        if user_message_val is not None and not isinstance(user_message_val, str):
            raise ValueError("LoadDatasetState.from_json_dict: user_message must be str|null")
        
        summary = cast(Optional[DatasetSummary], summary_val)

        return cls(
            id=uuid_from_any(id_val),
            summary=summary,
            load_error= load_error_val,
            user_message= user_message_val,
        )
