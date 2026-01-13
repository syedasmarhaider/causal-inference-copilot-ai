from abc import ABC, abstractmethod
from uuid import UUID
from typing import Optional

from python.workflows.state.conversation_state import ConversationState


class ConversationRepo(ABC):
    @abstractmethod
    def load(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> Optional[ConversationState]:
        ...

    @abstractmethod
    def save(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: ConversationState,
    ) -> None:
        ...