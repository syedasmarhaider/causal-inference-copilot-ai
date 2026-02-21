from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence
from uuid import UUID

from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.state import State


class WorkflowStateRepo(ABC):
    # -----------------------
    # Active stage pointer
    # -----------------------

    @abstractmethod
    def load_active_state_name(self, *, user_id: UUID, conversation_id: UUID) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def store_active_state_name(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> None:
        raise NotImplementedError

    # -----------------------
    # Per-state persistence
    # -----------------------

    @abstractmethod
    def load_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> Optional[State]:
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