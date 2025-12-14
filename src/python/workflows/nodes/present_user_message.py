from __future__ import annotations

from typing import Callable, List

from langchain_core.messages import  BaseMessage

from python.domain.service.llm_service import LLMService
from python.workflows.state.conversation_state import ConversationState
from python.workflows.utils.user_message_builder import build_user_message_with_llm


def make_present_user_message_node(
    llm: LLMService,
    *,
    model_name: str = "gemini-1.5-flash",
    history_window: int = 12,
) -> Callable[[ConversationState], ConversationState]:
    def present(state: ConversationState) -> ConversationState:
        # Build ONE assistant message from state (+ history)
        ai = build_user_message_with_llm(
            llm=llm,
            state=state,
            model_name=model_name,
            history_window=history_window,
        )

        out_messages: List[BaseMessage] = [ai]

        # Only presenter writes messages
        return {
            **state,
            "messages": out_messages,
        }

    return present
