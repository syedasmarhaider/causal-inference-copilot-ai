from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import UUID

from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.state import State


class WorkflowStateRepo(ABC):
    # -----------------------
    # Conversation persistence
    # -----------------------

    @abstractmethod
    def save_conversation_id(self, *, user_id: UUID, conversation_id: UUID) -> None:
        raise NotImplementedError

    # -----------------------
    # Active stage pointer
    # -----------------------

    @abstractmethod
    def get_conversation_ids_for_user(self, *, user_id: UUID) -> Sequence[UUID]:
        raise NotImplementedError
    
    
    @abstractmethod
    def is_conversation_id_for_user_id_exists(self, *, user_id: UUID, conversation_id: UUID) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_active_state_name(self, *, user_id: UUID, conversation_id: UUID) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def store_active_state_name(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> None:
        raise NotImplementedError

    # -----------------------
    # Per-state persistence
    # -----------------------

    @abstractmethod
    def load_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> State | None:
        raise NotImplementedError

    @abstractmethod
    def store_state(self, *, user_id: UUID, conversation_id: UUID, state: State) -> None:
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
    def append_messages(self, *, user_id: UUID, conversation_id: UUID, messages: Sequence[ChatMessage]) -> None:
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
