from typing import Dict, Tuple
from uuid import UUID

from python.workflows.state.conversation_state import ConversationState
from python.domain.repo.conversation_repo import ConversationRepo


# TODO: will change it later to the S3 or some fast storage conversation repo
class InMemoryConversationRepo(ConversationRepo):
    def __init__(self) -> None:
        self._store: Dict[Tuple[UUID, UUID], ConversationState] = {}

    def load(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> ConversationState | None:
        return self._store.get((user_id, conversation_id))

    def save(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: ConversationState,
    ) -> None:
        self._store[(user_id, conversation_id)] = state
