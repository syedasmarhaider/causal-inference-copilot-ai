from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Protocol

from python.domain.service.llm_service import ChatMessage, LLMService, LLMConfig
from python.domain.workflows.state import State


@dataclass(frozen=True)
class NextDecision:
    state_name: Optional[str]                 # None => blocked/finished
    router_message_for_node: Optional[str]    # message for node/UI, or None


class Router(Protocol):
    def decide_next(
        self,
        *,
        current_state: Optional[State],
        user_message: Optional[str],
        messages_history: Sequence[ChatMessage],
    ) -> NextDecision: ...


class LLMAssistedRouter:
    def __init__(
        self,
        *,
        llm: LLMService,
        llm_config: Optional[LLMConfig] = None,
    ) -> None:
        self._llm = llm
        self._llm_config = llm_config
        
    def decide_next(
        self,
        *,
        current_state: Optional[State],
        user_message: Optional[str],
        messages_history: Sequence[ChatMessage],
    ) -> NextDecision:
        raise NotImplementedError