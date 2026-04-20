"""Pydantic request and response schemas for the HTTP adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any
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
from python.implementation.workflows.utils.diff_util import DataFrameDiff

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


class DatasetPageResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "conversation_id": "22222222-2222-2222-2222-222222222222",
                "conversation_type": "data",
                "dataset_id": "33333333-3333-3333-3333-333333333333",
                "start": 20,
                "limit": 10,
                "row_count": 2,
                "columns": ["patient_id", "age", "bp_sys"],
                "rows": [
                    {"patient_id": "P021", "age": 64, "bp_sys": 132},
                    {"patient_id": "P022", "age": 59, "bp_sys": 128},
                ],
            }
        },
    )

    conversation_id: UUID = Field(description="Conversation UUID.")
    conversation_type: ConversationType = Field(description="Conversation type.")
    dataset_id: UUID = Field(description="Dataset UUID.")
    start: int = Field(description="Zero-based row offset applied to the dataset.")
    limit: int | None = Field(
        default=None,
        description="Requested maximum number of rows after applying `start`.",
    )
    row_count: int = Field(description="Number of rows returned in this page.")
    columns: list[str] = Field(description="Dataset columns in source order.")
    rows: list[dict[str, Any]] = Field(description="Dataset rows for the requested page.")


class DatasetDiffCreateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"key_columns": ["record_id"]}},
    )

    key_columns: list[NonEmptyStr] = Field(
        default_factory=list,
        description=(
            "Optional columns used to match rows across dataset versions. "
            "When omitted, rows are compared by position."
        ),
    )


class DatasetDiffResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "conversation_id": "22222222-2222-2222-2222-222222222222",
                "conversation_type": "data",
                "previous_dataset_id": "33333333-3333-3333-3333-333333333333",
                "current_dataset_id": "44444444-4444-4444-4444-444444444444",
                "diff": {
                    "identity_mode": "position",
                    "key_columns": [],
                    "schema_diff": {
                        "columns_added": [],
                        "columns_removed": [],
                        "column_type_changes": [],
                    },
                    "row_changes": [],
                    "summary": {
                        "old_row_count": 10,
                        "new_row_count": 10,
                        "inserted_rows": 0,
                        "deleted_rows": 0,
                        "updated_rows": 0,
                        "total_changed_rows": 0,
                        "total_changed_cells": 0,
                    },
                },
            }
        },
    )

    conversation_id: UUID = Field(description="Conversation UUID.")
    conversation_type: ConversationType = Field(description="Conversation type.")
    previous_dataset_id: UUID = Field(description="Previous working dataset UUID.")
    current_dataset_id: UUID = Field(description="Current working dataset UUID.")
    diff: DataFrameDiff = Field(
        description="Structured diff from the previous working dataset version to the current one."
    )


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
