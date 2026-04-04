from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.models import ChatMessage
from python.domain.workflows.state import State
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetSummaryModel,
)


class DatasetIterationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    dataset_id: UUID
    summary: DatasetSummaryModel | None = None
    saved_vega_lite_specs: list[dict[str, Any]] = Field(default_factory=lambda: [])


class DatasetPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    dataset_iterations: list[DatasetIterationModel] = Field(default_factory=lambda: [])
    freezed: bool = False

class DatasetState(State):
    INIT_DATA_ID = uuid.UUID("486f4975-6cd9-4261-a122-e6b0fc46462d")
    NAME: ClassVar[str] = "DATASET"
    chat_message: ChatMessage | None = None

    def __init__(self, payload: DatasetPayloadModel) -> None:
        self.payload = payload

    def name(self) -> str:
        return self.NAME

    def status(self) -> str:
        if self.payload.freezed:
            return "FREEZED"
        return "PENDING"

    def messages(self) -> Sequence[ChatMessage]:
        if self.chat_message:
            return [self.chat_message]
        return [ChatMessage(role="assistant", content="Data set is not uploaded yet. Please upload the dataset to proceed.")]

    def freeze_status(self) -> None:
        self.payload.freezed = True    

    def error(self) -> None:
        return None

    @property
    def latest_iteration(self) -> DatasetIterationModel | None:
        if not self.payload.dataset_iterations:
            return None
        return self.payload.dataset_iterations[-1]

    def pre_required_states_names(self) -> Sequence[str]:
        return ()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DatasetState:
        return cls(DatasetPayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> DatasetState:
        return cls(payload=DatasetPayloadModel())


__all__ = [
    "DatasetIterationModel",
    "DatasetPayloadModel",
    "DatasetState",
]
