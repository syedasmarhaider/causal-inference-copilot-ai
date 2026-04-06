from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.state import State


class NextDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state_name: str | None = None
    router_confirmation_message_for_user: str | None = None
    should_persists_by_workflow= bool| None


class Router(ABC):
    @abstractmethod
    def get_initial_state_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_done_state_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_next_state_names(self, current_state_name: str) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def decide_next(
        self,
        *,
        current_state: State | None,
        messages_history: Sequence[ChatMessage],
    ) -> NextDecision:
        raise NotImplementedError
