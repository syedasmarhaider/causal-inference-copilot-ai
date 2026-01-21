from __future__ import annotations

from typing import Callable, Sequence, TypedDict, List, cast
from uuid import UUID
from langchain_core.messages import BaseMessage

from python.domain.service.llm_service import ChatMessage
from python.domain.service.llm_service import ChatMessage
from python.workflows.state.protocol_state import ProtocolState
from python.workflows.state.control_state import ControlState
from python.workflows.state.dataset_state import DatasetState
from python.workflows.state.metadata_state import MetadataState

class ConversationState(TypedDict):
    control: ControlState
    dataset: DatasetState   | None
    metadata: MetadataState | None
    protocol: ProtocolState | None
    messages: List[BaseMessage]

CallableNodeFunc = Callable[[UUID,UUID,ConversationState], ConversationState]

def to_chat_history_last_k(
    state: ConversationState,
    *,
    k: int,
    drop_last_user: bool,
) -> List[ChatMessage]:
    messages = cast(Sequence[BaseMessage], state.get("messages", []))
    msgs = list(messages)

    if drop_last_user and msgs:
        last = msgs[-1]
        if getattr(last, "type", None) == "human" or "human" in last.__class__.__name__.lower():
            msgs = msgs[:-1]

    tail = msgs[-k:] if k > 0 else []

    out: List[ChatMessage] = []
    for m in tail:
        content = str(getattr(m, "content", "") or "").strip()
        if not content:
            continue

        mtype = getattr(m, "type", None)
        cls = m.__class__.__name__.lower()

        if mtype == "human" or "human" in cls or "user" in cls:
            out.append(ChatMessage(role="user", content=content))
        elif mtype == "ai" or "ai" in cls or "assistant" in cls:
            out.append(ChatMessage(role="assistant", content=content))
        elif mtype == "system" or "system" in cls:
            out.append(ChatMessage(role="system", content=content))    
        else:
            continue

    return out


def last_human_text(state: ConversationState) -> str | None:
    messages = cast(Sequence[BaseMessage], state.get("messages", []))
    for m in reversed(list(messages)):
        if getattr(m, "type", None) == "human":
            return str(getattr(m, "content", None) or None)
        name = m.__class__.__name__.lower()
        if "human" in name or "user" in name:
            return str(getattr(m, "content", None) or None)
    return None