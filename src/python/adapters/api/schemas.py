from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.models import ArtifactFormat, ArtifactKind, MessageRole
from python.domain.workflows.node_state import Action, Status


class ArtifactRefResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(description="Artifact UUID.")
    kind: ArtifactKind = Field(description="Artifact kind enum: `graph` or `data`.")
    format: ArtifactFormat = Field(description="Artifact format enum: `json` or `csv`.")
    artifact_meta: dict[str, str] | None = Field(
        default=None,
        description="Optional artifact metadata attached by the workflow node.",
    )


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole = Field(
        description="Message role. Workflow API responses return assistant messages."
    )
    content: str = Field(description="Assistant-visible message text.")
    id: str | None = Field(default=None, description="Optional message identifier.")
    artifact_refs: Sequence[ArtifactRefResponse] | None = Field(
        default=None,
        description="Optional artifact references emitted with this assistant message.",
    )


class WorkingDatasetInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: UUID = Field(description="Current working dataset UUID.")
    is_freezed: bool = Field(description="Whether the current working dataset is frozen/read-only.")


class CreateConversationResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "user_id": "11111111-1111-1111-1111-111111111111",
                "conversation_id": "22222222-2222-2222-2222-222222222222",
            }
        },
    )

    user_id: UUID = Field(
        description="Authenticated internal user UUID derived from the Firebase token."
    )
    conversation_id: UUID = Field(description="Newly created conversation UUID.")


class UploadDatasetResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "user_id": "11111111-1111-1111-1111-111111111111",
                "conversation_id": "22222222-2222-2222-2222-222222222222",
                "dataset_id": "33333333-3333-3333-3333-333333333333",
            }
        },
    )

    user_id: UUID = Field(
        description="Authenticated internal user UUID derived from the Firebase token."
    )
    conversation_id: UUID = Field(description="Conversation UUID that owns the dataset.")
    dataset_id: UUID = Field(description="Stored dataset UUID.")


class InvokeRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "user_text": "Please summarize the uploaded dataset and tell me what I should check first.",
            }
        },
    )

    user_text: str | None = Field(
        default=None,
        description=(
            "User message forwarded to the current workflow stage. "
            "To trigger dataset-history revert behavior inside workflow execution, "
            "send exactly `revert_data_changes`."
        ),
    )


class RevertStateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "state_name": "MODEL_SELECTION",
            }
        },
    )

    state_name: str | None = Field(
        default=None,
        description="Workflow state name to revert to. This is not the dataset-history revert mechanism.",
    )


class InvokeResponse(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "conversation_id": "22222222-2222-2222-2222-222222222222",
                "user_id": "11111111-1111-1111-1111-111111111111",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "I loaded your dataset and summarized the main columns.",
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
                "current_stage_name": "DATASET",
                "current_stage_status": "PENDING",
                "latest_working_dataset": {
                    "dataset_id": "33333333-3333-3333-3333-333333333333",
                    "is_freezed": False,
                },
            }
        },
    )

    conversation_id: UUID = Field(description="Conversation UUID.")
    user_id: UUID = Field(
        description="Authenticated internal user UUID derived from the Firebase token."
    )
    messages: Sequence[ChatMessageResponse] = Field(
        description="Assistant-visible messages returned by the workflow.",
    )
    action: Action = Field(description="Workflow action requested by the current stage.")
    current_stage_name: str = Field(description="Current workflow stage name.")
    current_stage_status: Status = Field(description="Current workflow stage status.")
    latest_working_dataset: WorkingDatasetInfoResponse | None = Field(
        default=None,
        description="Latest working dataset info from DataflowApp, if a dataset exists for this conversation.",
    )
