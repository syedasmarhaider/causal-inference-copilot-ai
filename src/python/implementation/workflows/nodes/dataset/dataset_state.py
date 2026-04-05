from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.models import ArtifactRef, ChatMessage
from python.domain.workflows.state import Action, State, Status
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetSummaryModel,
)


class DatasetIterationModel(BaseModel):
    # Ignore legacy persisted fields like per-iteration summaries while migrating to the
    # leaner "IDs only" iteration history.
    model_config = ConfigDict(extra="ignore")

    dataset_id: UUID


class DatasetPayloadModel(BaseModel):
    # Ignore legacy payload fields like persisted message artifact refs during migration.
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    dataset_iterations: list[DatasetIterationModel] = Field(default_factory=lambda: [])
    latest_summary: DatasetSummaryModel | None = None
    freezed: bool = False
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
        self._response_message_artifact_refs = [
            dict(ref) for ref in (response_message_artifact_refs or [])
        ]

    def name(self) -> str:
        return self.NAME

    def status(self) -> Status:
        if self.payload.freezed:
            return "FREEZED"
        return "PENDING"

    def action(self) -> Action:
        # Dataset is the only stage that can genuinely require data upload. Once at least one
        # dataset iteration exists, the user can keep interacting with this node conversationally.
        if not self.payload.dataset_iterations:
            return "NEEDS_DATA"
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

    def set_status_freez(self) -> None:
        self.payload.freezed = True

    def set_status_pending(self) -> None:
        self.payload.freezed = False

    def error(self) -> None:
        return None

    @property
    def latest_iteration(self) -> DatasetIterationModel | None:
        if not self.payload.dataset_iterations:
            return None
        return self.payload.dataset_iterations[-1]

    @property
    def latest_summary(self) -> DatasetSummaryModel | None:
        return self.payload.latest_summary

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
