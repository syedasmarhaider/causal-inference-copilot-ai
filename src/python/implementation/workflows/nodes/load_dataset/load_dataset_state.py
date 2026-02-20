from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Sequence, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.load_dataset.load_dataset_utils import DatasetSummary
from python.implementation.workflows.utils.utils import json_sanitize, uuid_from_any

class LoadDatasetPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: Optional[UUID] = None
    summary: Optional[DatasetSummary] = None
    load_error: Optional[str] = None
    user_message: Optional[str] = None

    @field_validator("load_error", "user_message", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return None

    @field_validator("id", mode="before")
    @classmethod
    def _parse_uuid(cls, v: Any) -> Optional[UUID]:
        # accept UUID | str | None
        return uuid_from_any(v)


class LoadDatasetState(State):
    NAME: ClassVar[str] = "LOAD_DATASET"

    def __init__(self, payload: LoadDatasetPayloadModel) -> None:
        self.payload = payload

    # ---- required by State ABC ----
    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        if self.error is not None:
            return "ABORTED"
        if self.payload.summary is not None:
            return "DONE"
        return "PENDING"

    @property
    def message(self) -> Optional[str]:
        return self.payload.user_message

    @property
    def error(self) -> Optional[str]:
        return self.payload.load_error

    @property
    def needs_action(self) -> ACTION:
        return "NEEDS_INPUT" if self.error is not None else "NONE"

    def required_states_keys(self) -> Sequence[str]:
        return ()

    def to_json_dict(self) -> Dict[str, Any]:
        d = self.payload.model_dump(mode="json")
        if self.payload.summary is not None:
            d["summary"] = json_sanitize(cast(Any, self.payload.summary))
        return d

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "LoadDatasetState":
        model = LoadDatasetPayloadModel.model_validate(payload)
        return cls(model)