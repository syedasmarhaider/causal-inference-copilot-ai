from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from python.domain.models.models import ChatMessage
from python.domain.workflows.ochestrator_state import OchestratorState
from python.domain.workflows.node_state import NodeState
from pydantic import Field

ConversationType = Literal["causal", "data"]


class Conversation(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    conversation_id: UUID
    last_updated_at_utc: float
    conversation_type: ConversationType
    
class WorkflowStateRepo(ABC):
    # -----------------------
    # Conversation persistence
    # -----------------------

    @abstractmethod
    def save_conversation(self, *, user_id: UUID, conversation: Conversation) -> None:
        raise NotImplementedError
    
    # -----------------------
    # Active stage pointer
    # -----------------------

    @abstractmethod
    def get_conversations(self, *, user_id: UUID) -> Sequence[Conversation]:
        raise NotImplementedError

    @abstractmethod
    def is_conversation_id_for_user_id_exists(
        self, *, user_id: UUID, conversation: Conversation
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_ochestrator_state(self, *, user_id: UUID, conversation_id: UUID) -> OchestratorState | None:
        raise NotImplementedError

    @abstractmethod
    def store_ochestrator_state(
        self, *, user_id: UUID, conversation_id: UUID, state: OchestratorState 
    ) -> None:
        raise NotImplementedError

    # -----------------------
    # Per-state persistence
    # -----------------------

    @abstractmethod
    def load_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> NodeState | None:
        raise NotImplementedError

    @abstractmethod
    def store_state(self, *, user_id: UUID, conversation_id: UUID, state: NodeState) -> None:
        raise NotImplementedError

    # Optional convenience
    @abstractmethod
    def delete_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> None:
        raise NotImplementedError

    # -----------------------
    # Message history
    # -----------------------

    @abstractmethod
    def append_message(self, *, user_id: UUID, conversation_id: UUID, message: ChatMessage) -> None:
        raise NotImplementedError

    @abstractmethod
    def append_messages(
        self, *, user_id: UUID, conversation_id: UUID, messages: Sequence[ChatMessage]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_message_history(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        limit: int = 20,
    ) -> Sequence[ChatMessage]:
        raise NotImplementedError

    @abstractmethod
    def clear_message_history(self, *, user_id: UUID, conversation_id: UUID) -> None:
        raise NotImplementedError
