from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from pydantic import BaseModel, ConfigDict

from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.state import State


class NextDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    state_name: str
    router_message_for_node: Optional[str] = None


class Router(ABC):
    @abstractmethod
    def get_initial_state_name(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def get_done_state_name(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def decide_next(
        self,
        *,
        current_state: Optional[State],
        messages_history: Sequence[ChatMessage],
    ) -> NextDecision:
        raise NotImplementedError