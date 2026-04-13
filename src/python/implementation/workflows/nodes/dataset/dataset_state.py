from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.models import ArtifactRef, ChatMessage, WorkingDatasetInfo
from python.domain.workflows.state import Action, State, Status
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetSummaryModel,
)


class DatasetPayloadModel(BaseModel):
    # Ignore legacy payload fields like persisted message artifact refs during migration.
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    dataset_iterations: list[UUID] = Field(default_factory=lambda: [])
    latest_summary: DatasetSummaryModel | None = None
    user_message: str | None = None


class DatasetState(State):
    INIT_DATA_ID = uuid.UUID("486f4975-6cd9-4261-a122-e6b0fc46462d")
    NAME: ClassVar[str] = "DATASET"

    def __init__(
        self,
        payload: DatasetPayloadModel,
        *,
        response_message_artifact_refs: Sequence[ArtifactRef] | None = None,
    ) -> None:
        self.payload = payload
        # Artifact refs are response-only. They are useful for the latest turn but should not
        # become part of long-lived workflow state.
        self._response_message_artifact_refs = list(response_message_artifact_refs or [])

    def name(self) -> str:
        return self.NAME

    def get_working_dataset_info(self) -> WorkingDatasetInfo | None:
        if not self.payload.dataset_iterations or len(self.payload.dataset_iterations) == 0:
            return None
        return WorkingDatasetInfo(self.payload.dataset_iterations[-1], False)

    def status(self) -> Status:
        return "PENDING"

    def action(self) -> Action:
        if not self.payload.dataset_iterations or len(self.payload.dataset_iterations) == 0:
            return "NEEDS_DATA"
        if self.payload.user_message is None:
            return "NONE"
        return "NEEDS_INPUT"

    def messages(self) -> Sequence[ChatMessage]:
        if self.payload.user_message:
            return [
                ChatMessage(
                    role="assistant",
                    content=self.payload.user_message,
                    artifact_refs=list(self._response_message_artifact_refs) or None,
                )
            ]
        return [
            ChatMessage(
                role="assistant",
                content="Data set is not uploaded yet. Please upload the dataset to proceed.",
            )
        ]
    
    def set_status_pending(self) -> None:
        self._status = "PENDING"    

    def error(self) -> None:
        return None

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DatasetState:
        return cls(DatasetPayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> DatasetState:
        return cls(payload=DatasetPayloadModel())
