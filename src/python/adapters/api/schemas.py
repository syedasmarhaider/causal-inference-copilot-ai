from __future__ import annotations

from typing import Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, Field


class CreateConversationRequest(BaseModel):
    user_id: Optional[UUID] = None


class CreateConversationResponse(BaseModel):
    user_id: UUID
    conversation_id: UUID


class InvokeRequest(BaseModel):
    user_id: UUID
    user_text: Optional[str] = Field(default=None)


class InvokeResponse(BaseModel):
    conversation_id: UUID
    user_id: UUID
    node_message: Optional[str]
    needs_input: bool
    artifact_ids: Optional[Sequence[str]] = None
    current_stage: str
    current_stage_status: str