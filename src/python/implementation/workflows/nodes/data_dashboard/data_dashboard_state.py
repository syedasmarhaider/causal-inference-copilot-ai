from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.errors import NodeExecutionError
from python.domain.models.models import ArtifactRef, ChatMessage, WorkingDatasetInfo
from python.domain.workflows.node_state import Action, State, Status
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetSummaryModel,
)


class DataDashboardPayloadModel(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    dataset_iterations: list[UUID] = Field(default_factory=list)
    latest_summary: DatasetSummaryModel | None = None
    user_message: str | None = None


class DataDashboardState(State):
    INIT_DATA_ID: ClassVar[UUID] = uuid.UUID("386f4975-6cd9-4261-a122-e6b0fc46462d")
    NAME: ClassVar[str] = "DATA_DASHBOARD"

    def __init__(
        self,
        payload: DataDashboardPayloadModel,
        *,
        response_message_artifact_refs: Sequence[ArtifactRef] | None = None,
    ) -> None:
        self.payload = payload
        self._response_message_artifact_refs = list(response_message_artifact_refs or [])

    # -- State ABC ------------------------------------------------------------

    def name(self) -> str:
        return self.NAME

    def get_working_dataset_info(self) -> WorkingDatasetInfo | None:
        if not self.payload.dataset_iterations:
            return None
        return WorkingDatasetInfo(self.payload.dataset_iterations[-1], False)

    def status(self) -> Status:
        return "PENDING"

    def action(self) -> Action:
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
                content="No dataset uploaded yet. Please upload a CSV to get started.",
            )
        ]

    def set_status_pending(self) -> None:
        pass  # always PENDING

    def error(self) -> NodeExecutionError | None:
        return None

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DataDashboardState:
        return cls(DataDashboardPayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> DataDashboardState:
        return cls(payload=DataDashboardPayloadModel())


__all__ = ["DataDashboardPayloadModel", "DataDashboardState"]
