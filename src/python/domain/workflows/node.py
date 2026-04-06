from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import UUID

from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.ochestrator_state import ReadOnlyOchestratorState
from python.domain.workflows.state import State


class Node(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_info(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        readonly_orchestrator_state: ReadOnlyOchestratorState,
        messages_history: Sequence[ChatMessage] | None,
    ) -> State:
        raise NotImplementedError
