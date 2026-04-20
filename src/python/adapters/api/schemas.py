"""Pydantic request and response schemas for the HTTP adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from python.domain.models.models import (
    ArtifactFormat,
    ArtifactKind,
    MessageRole,
    NonEmptyStr,
)
from python.domain.repo.workflow_state_repo import ConversationType
from python.domain.workflows.node import Action, Status

ConversationName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
]


class ArtifactRefResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(description="Artifact UUID.")
    kind: ArtifactKind = Field(description="Artifact kind.")
    format: ArtifactFormat = Field(description="Artifact format.")
    artifact_meta: dict[str, str] | None = Field(
        default=None,
        description="Optional metadata emitted by the workflow node.",
    )


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole = Field(description="Message role.")
    content: str = Field(description="Message content.")
    id: str | None = Field(default=None, description="Optional message identifier.")
    artifact_refs: Sequence[ArtifactRefResponse] | None = Field(
        default=None,
        description="Optional artifact references attached to the message.",
    )


class WorkingDatasetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: UUID = Field(description="Current working dataset UUID.")
    is_frozen: bool = Field(description="Whether the dataset is frozen for editing.")


class ConversationSummaryResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "conversation_id": "22222222-2222-2222-2222-222222222222",
                "conversation_type": "causal",
                "conversation_name": "Hypertension cohort review",
                "last_updated_at_utc": 1712345678.123,
            }
        },
    )

    conversation_id: UUID = Field(description="Conversation UUID.")
    conversation_type: ConversationType = Field(description="Conversation type.")
    conversation_name: str | None = Field(
        default=None,
        description="Optional conversation display name.",
    )
    last_updated_at_utc: float = Field(
        description="Last update time as a UTC Unix timestamp in seconds.",
    )


class CreateConversationRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "conversation_type": "causal",
                "conversation_name": "Hypertension cohort review",
            }
        },
    )

    conversation_type: ConversationType = Field(description="Conversation type to create.")
    conversation_name: ConversationName | None = Field(
        default=None,
        description="Optional conversation display name.",
    )


class UploadDatasetResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "conversation_id": "22222222-2222-2222-2222-222222222222",
                "conversation_type": "data",
                "dataset_id": "33333333-3333-3333-3333-333333333333",
            }
        },
    )

    conversation_id: UUID = Field(description="Conversation UUID.")
    conversation_type: ConversationType = Field(description="Conversation type.")
    dataset_id: UUID = Field(description="Stored dataset UUID.")


class ConversationMessageCreateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "user_text": "Please summarize the uploaded dataset and tell me what to check first.",
            }
        },
    )

    user_text: str | None = Field(
        default=None,
        description=(
            "Optional user message forwarded to the current workflow stage. "
            "Send `revert_data_changes` to request a dataset-history revert inside the workflow."
        ),
    )


class ConversationStateReversionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"state_name": "MODEL_SELECTION"}},
    )

    state_name: NonEmptyStr = Field(description="Workflow state name to revert to.")


class ConversationSnapshotResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "conversation_id": "22222222-2222-2222-2222-222222222222",
                "conversation_type": "causal",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I loaded your dataset and summarized the main columns.",
                        "artifact_refs": None,
                    }
                ],
                "states": ["DATASET", "DATA_COMPILATION"],
                "working_dataset": {
                    "dataset_id": "33333333-3333-3333-3333-333333333333",
                    "is_frozen": False,
                },
            }
        },
    )

    conversation_id: UUID = Field(description="Conversation UUID.")
    conversation_type: ConversationType = Field(description="Conversation type.")
    messages: Sequence[ChatMessageResponse] = Field(
        description="Conversation messages returned by the workflow."
    )
    states: list[str] = Field(
        description="Ordered workflow states: completed states plus the current pending state."
    )
    working_dataset: WorkingDatasetResponse | None = Field(
        default=None,
        description="Current working dataset metadata, when a dataset exists.",
    )


class ConversationExecutionResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "conversation_id": "22222222-2222-2222-2222-222222222222",
                "conversation_type": "causal",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I summarized the dataset.",
                        "artifact_refs": [
                            {
                                "id": "44444444-4444-4444-4444-444444444444",
                                "kind": "graph",
                                "format": "json",
                            }
                        ],
                    }
                ],
                "action": "NEEDS_INPUT",
                "current_stage_name": "DATA_COMPILATION",
                "current_stage_status": "PENDING",
                "working_dataset": {
                    "dataset_id": "33333333-3333-3333-3333-333333333333",
                    "is_frozen": False,
                },
            }
        },
    )

    conversation_id: UUID = Field(description="Conversation UUID.")
    conversation_type: ConversationType = Field(description="Conversation type.")
    messages: Sequence[ChatMessageResponse] = Field(
        description="Messages produced by the workflow step."
    )
    action: Action = Field(description="Next action requested by the workflow.")
    current_stage_name: str = Field(description="Current workflow stage name.")
    current_stage_status: Status = Field(description="Current workflow stage status.")
    working_dataset: WorkingDatasetResponse | None = Field(
        default=None,
        description="Current working dataset metadata, when a dataset exists.",
    )
