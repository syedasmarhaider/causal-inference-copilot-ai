from __future__ import annotations

from typing import Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CreateConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateConversationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    conversation_id: UUID


class UploadDatasetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    conversation_id: UUID
    dataset_id: UUID


class InvokeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_text: Optional[str] = Field(default=None)


class InvokeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    user_id: UUID
    node_message: Optional[str]
    needs_input: bool
    needs_data: bool
    artifact_ids: Optional[Sequence[str]] = None
    current_stage: str
    current_stage_status: str
