from __future__ import annotations

from typing import Optional
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
    current_stage: str
    current_stage_status: str
from __future__ import annotations

from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class CreateConversationRequest(BaseModel):
    user_id: Optional[UUID] = None
    max_steps: int = Field(default=16, ge=1, le=256)


class TurnRequest(BaseModel):
    user_id: UUID
    text: str = Field(min_length=1)
    max_steps: int = Field(default=16, ge=1, le=256)


class TurnResponse(BaseModel):
    outputs: list[str]
    needs_input: bool
    current_stage: str
    current_stage_status: str
    conversation_id: UUID
    user_id: UUID


class WsClientMessage(BaseModel):
    type: Literal["kick", "turn"]
    text: Optional[str] = None
    max_steps: int = Field(default=16, ge=1, le=256)
