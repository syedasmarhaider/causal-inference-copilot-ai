from __future__ import annotations

import json
import logging
from typing import Any,  cast
from uuid import UUID

from python.domain.service.llm_service import  LLMConfig, LLMService
from python.workflows.nodes.prompts.protocol_discussion import (
    get_protocol_discussion_system_prompt,
    get_protocol_discussion_confirmation_prompt,
    get_protocol_discussion_readiness_prompt,
    get_questions,
)
from python.workflows.state.conversation_state import (
    CallableNodeFunc,
    ConversationState,
    ConversationStateHelpers,
)
from python.workflows.state.protocol_discussion_state import ProtocolDiscussionState
log = logging.getLogger(__name__)


# =============================================================================
# Public factory
# =============================================================================
def make_protocol_discussion_node(
    *,
    llm: LLMService,
    model_name: str,
) -> CallableNodeFunc:
    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        return _run(
            user_id=user_id,
            conversation_id=conversation_id,
            state=state,
            llm=llm,
            model_name=model_name,
        )

    return node


# =============================================================================
# internals
# =============================================================================
def _run(
    *,
    user_id: UUID,
    conversation_id: UUID,
    state: ConversationState,
    llm: LLMService,
    model_name: str,
) -> ConversationState:
    control = state["control"]

    pd = state.get("protocol_discussion")
    if pd is None:
        pd = ProtocolDiscussionState() 
        state["protocol_discussion"] = pd

    dataset = state.get("dataset")
    dataset_id = dataset.get("id")
    if dataset_id is None:
        ConversationStateHelpers.append_ai_message(state=state, content="Dataset id is missing. Reload dataset is required")
        return  ConversationStateHelpers.set_abort(state=state, action="NEEDS_INPUT",msg="Dataset id is missing. Reload dataset is required")
    
    summary = dataset["summary"] # pyright: ignore[reportTypedDictNotRequiredAccess, reportUnknownVariableType]
    if summary is None:
        ConversationStateHelpers.append_ai_message(state=state, content="Data summary missing. Reload dataset is required")
        return  ConversationStateHelpers.set_abort(state=state, action="NONE",msg="Data summary missing. Reload dataset is required")
    
            
    chat_history_messages = ConversationStateHelpers.chat_history_to_payload(state, k=7)

    payload: dict[str, Any] = { 
        "prev_questions_answers_discussion_state": get_questions(),
        "conversation_messages_till_now":chat_history_messages,
        "dataset_columns_summary": summary,
    }

    # -------------------------
    # LLM #1: Update discussion
    # -------------------------
    try:
        pd.discussion = _llm_call_text(
            llm=llm,
            model_name=model_name,
            temperature=0.0,
            system_prompt=get_protocol_discussion_system_prompt(),
            user_payload=payload,
            empty_err="LLM#1 returned empty discussion",
        )
        state["protocol_discussion"] = pd
    except Exception as e:
        log.exception("PROTOCOL_DISCUSSION: LLM#1 failed")
        ConversationStateHelpers.append_ai_message(state=state, content=f"Protocol discussion update failed: {e}")
        return  ConversationStateHelpers.set_abort(state=state, action="NONE",msg=f"Protocol discussion update failed: {e}")

    # -------------------------
    # LLM #2: User-facing message
    # -------------------------
    try:
        node_msg = _llm_call_text(
            llm=llm,
            model_name=model_name,
            temperature=0.3,
            system_prompt=get_protocol_discussion_confirmation_prompt(),
            user_payload=payload,
            empty_err="LLM#2 returned empty message",
        )
    except Exception:
        log.exception("PROTOCOL_DISCUSSION: LLM#2 failed, using fallback")
        node_msg = "I updated the protocol draft. Tell me what to change, or reply 'confirm' if it is correct."

    control["node_message"] = node_msg
    ConversationStateHelpers.append_ai_message(state, node_msg)

    # -------------------------
    # LLM #3: Readiness token
    # -------------------------
    try:
        token = _llm_call_text(
            llm=llm,
            model_name=model_name,
            temperature=0.0,
            system_prompt=get_protocol_discussion_readiness_prompt(),
            user_payload=payload,
            empty_err="LLM#3 returned empty token",
        )
        token = (token or "").strip().splitlines()[0].strip().split()[0].strip().upper()
    except Exception:
        log.exception("PROTOCOL_DISCUSSION: LLM#3 failed; defaulting to PENDING")
        token = "PENDING"

    if token == "READY":
        return ConversationStateHelpers.set_done(state=state,action="NONE",msg=node_msg)
    
    if token == "ABORT":
        return ConversationStateHelpers.set_abort(state=state,action="NONE",msg=node_msg)    
    else:
        return ConversationStateHelpers.set_pending(state=state,action="NEEDS_INPUT",msg=node_msg)


# =============================================================================
# LLM helper
# =============================================================================
def _llm_call_text(
    *,
    llm: LLMService,
    model_name: str,
    temperature: float,
    system_prompt: str,
    user_payload: dict[str, Any],
    empty_err: str,
) -> str:
    cfg = LLMConfig(model=model_name, temperature=temperature)
    raw = _llm_text(
        llm,
        config=cfg,
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=False),
    )
    out = (raw or "").strip()
    if not out:
        raise ValueError(empty_err)
    return out


def _llm_text(
    llm: LLMService,
    *,
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
) -> str:
        resp = llm.generate(  
            config=config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=None
        )
        return cast(Any, resp).content
