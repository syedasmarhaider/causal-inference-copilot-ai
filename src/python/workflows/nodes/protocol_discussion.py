# src/python/workflows/nodes/propose_and_confirm_protocol_discussion.py
from __future__ import annotations

import json
import logging
from typing import Any, List, cast
from uuid import UUID

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.workflows.nodes.prompts.protocol_discussion import (
    get_protocol_discussion_system_prompt,
    get_protocol_discussion_confirmation_prompt,
    get_protocol_discussion_readiness_prompt,
)
from python.workflows.state.conversation_state import (
    CallableNodeFunc,
    ConversationState,
    ConversationStateHelpers,
)
from python.workflows.state.control_state import ControlState
from python.workflows.state.dataset_state import DatasetStateHelpers
from python.workflows.state.protocol_discussion_state import ProtocolDiscussionState

_DEFAULT_PREVIEW_LIMIT = 5
log = logging.getLogger(__name__)


# =============================================================================
# Public factory
# =============================================================================
def make_protocol_discussion_node(
    *,
    data_repo: DataRepo,
    llm: LLMService,
    model_name: str,
) -> CallableNodeFunc:
    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        return _run(
            user_id=user_id,
            conversation_id=conversation_id,
            state=state,
            data_repo=data_repo,
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
    data_repo: DataRepo,
    llm: LLMService,
    model_name: str,
) -> ConversationState:
    control = state["control"]

    pd = state.get("protocol_discussion")
    if pd is None:
        pd = ProtocolDiscussionState() 
        state["protocol_discussion"] = pd

    dataset = state.get("dataset")
    dataset_id = dataset.get("id") if isinstance(dataset, dict) else None
    if dataset_id is None:
        return _fatal(state, "Dataset id is missing. Reload dataset is required.")
    
    try:
        df = data_repo.get_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
            limit=_DEFAULT_PREVIEW_LIMIT,
        )
    except Exception as e:
        return _fatal(state, f"Failed to load dataset preview: {type(e).__name__}: {e}")

    dataset_cols = DatasetStateHelpers.extract_columns_from_df(df)
    
    if not getattr(pd, "discussion", ""):
        pd.discussion = _build_discussion_template()
        
    chat_history_messages = ConversationStateHelpers.chat_history_to_payload(state, k=7)

    payload: dict[str, Any] = { 
        "protocol_discussion_important_context": pd.discussion,
        "conversation_messages_till_now":chat_history_messages,
        "dataset_columns_preview": dataset_cols,
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
        return _fatal(state, f"Protocol discussion update failed: {e}")

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
    ConversationStateHelpers.append_ai_message(state, node_msg, stage=control["current_stage"])

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
        _set_done(control, node_msg)
    else:
        _set_pending(control, control.get("node_message") or "")

    return state


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
    try:
        resp = llm.generate(  # type: ignore[call-arg]
            config=config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return cast(Any, resp).content
    except TypeError:
        msgs: List[ChatMessage] = []
        if system_prompt:
            msgs.append(ChatMessage(role="system", content=system_prompt))
        msgs.append(ChatMessage(role="user", content=user_prompt))
        resp = llm.generate(config=config, history=msgs)  # type: ignore[arg-type]
        return cast(Any, resp).content


# =============================================================================
# helpers
# =============================================================================
def _set_pending(control: ControlState, msg: str) -> None:
    control["current_stage_status"] = "PENDING"
    control["action_required"] = "NEEDS_INPUT"
    control["node_message"] = msg


def _set_done(control: ControlState,msg :str) -> None:
    control["current_stage_status"] = "DONE"
    control["action_required"] = "NONE"
    control["node_message"] = msg


def _fatal(state: ConversationState, msg: str) -> ConversationState:
    control = state["control"]
    control["current_stage_status"] = "ABORTED"
    control["action_required"] = "NONE"
    control["node_message"] = msg
    ConversationStateHelpers.append_ai_message(state, msg, stage=control["current_stage"])
    return state

def _build_discussion_template() -> str:
    qs = ProtocolDiscussionState.get_questions()
    lines: List[str] = []
    lines.append("PROTOCOL_DISCUSSION_CONTEXT")
    lines.append("Instruction: Only edit the A: parts. Do not reorder, delete, or rename questions.")
    lines.append("")
    for i, q in enumerate(qs, start=1):
        lines.append(f"Q{i}: {q}")
        lines.append("A: ")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
