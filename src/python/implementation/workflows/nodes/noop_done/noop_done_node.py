from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar
from uuid import UUID

from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.implementation.workflows.nodes.noop_done.noop_done_state import NoopDoneState


class NoopDoneNode(Node):
    NAME: ClassVar[str] = NoopDoneState.NAME

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return "No-op terminal node: immediately returns DONE."

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        previous_state_dependencies: Mapping[str, State],
        messages_history: Sequence[ChatMessage] | None,
    ) -> State:
        _ = user_id
        _ = conversation_id
        _ = state
        _ = previous_state_dependencies
        _ = messages_history
        return NoopDoneState()
