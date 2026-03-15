from __future__ import annotations

import uuid
from typing import Any, ClassVar, Dict, Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from python.domain.workflows.state import State, StateMessage, Status
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import DatasetSummaryModel
from python.implementation.workflows.utils.utils import uuid_from_any


class LoadDatasetPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: Optional[UUID] = None
    summary: Optional[DatasetSummaryModel] = None
    load_error: Optional[str] = None
    graph_picture_ids: Optional[Sequence[UUID]] = None
    user_message: Optional[str] = "Not run yet"

    @field_validator("load_error", "user_message", mode="before")
    @classmethod
    def _empty_str_to_none(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return None

    @field_validator("id", mode="before")
    @classmethod
    def _parse_uuid(cls, value: Any) -> Optional[UUID]:
        return uuid_from_any(value)


class LoadDatasetState(State):
    INIT_DATA_ID = uuid.UUID("486f4975-6cd9-4261-a122-e6b0fc46462d")
    NAME: ClassVar[str] = "LOAD_DATASET"

    def __init__(self, payload: LoadDatasetPayloadModel) -> None:
        self.payload = payload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        if self.payload.load_error is not None:
            return "ABORTED"
        if self.payload.summary is not None:
            return "DONE"
        return "PENDING"

    @property
    def message(self) -> StateMessage:
        if self.payload.user_message is None:
            raise ValueError(
                "LoadDatasetState message is required but missing. State must have user message."
            )
        artifact_ids = (
            [str(picture_id) for picture_id in self.payload.graph_picture_ids]
            if self.payload.graph_picture_ids
            else None
        )
        action = "NEEDS_DATA" if self.payload.id is None else "NONE"
        return StateMessage(txt_message=self.payload.user_message, action=action, artifact_ids=artifact_ids)  

    @property
    def error(self) -> Optional[str]:
        return self.payload.load_error

    def pre_required_states_names(self) -> Sequence[str]:
        return ()

    def to_json_dict(self) -> Dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "LoadDatasetState":
        return cls(LoadDatasetPayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> "LoadDatasetState":
        return cls(
            LoadDatasetPayloadModel(
                id=None,
                summary=None,
                load_error=None,
                graph_picture_ids=None,
                user_message=(
                    "I do not have your dataset yet. Please upload a CSV dataset so I can "
                    "summarize it and continue."
                ),
            )
        )
