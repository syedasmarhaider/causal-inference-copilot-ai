from __future__ import annotations

import json
import logging
import re
from typing import Any, List, cast
from uuid import UUID

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.workflows.nodes.prompts.model_selection import get_econml_allowed_estimators
from python.workflows.nodes.prompts.model_selection_discussion import (
    get_model_selection_discussion_extractor_system_prompt,
    get_model_selection_discussion_system_prompt,
)
from python.workflows.state.conversation_state import (
    CallableNodeFunc,
    ConversationState,
    ConversationStateHelpers,
)
from python.workflows.state.model_selection_discussion_state import ModelSelectionDiscussionState

log = logging.getLogger(__name__)


def make_model_selection_discussion_node(*, llm: LLMService, model_name: str) -> CallableNodeFunc:
    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        return _run(user_id=user_id, conversation_id=conversation_id, state=state, llm=llm, model_name=model_name)

    return node


def _run(
    *,
    user_id: UUID,
    conversation_id: UUID,
    state: ConversationState,
    llm: LLMService,
    model_name: str,
) -> ConversationState:
    allowed = tuple(get_econml_allowed_estimators())
    allowed_set = set(allowed)

    ms = state.get("model_selection")
    if ms is None:
        return _abort(state, "ModelSelectionState missing. Run MODEL_SELECTION first.")

    selected_top3 =  ms.get("selected_top3") or []

    mds =  state.get("model_selection_discussion") or ModelSelectionDiscussionState()
    state["model_selection_discussion"] = mds

    already = (mds.get("selected_model_fqcn") or "").strip()
    if already:
        msg = f"Model already selected: {already}"
        ConversationStateHelpers.append_ai_message(state=state, content=msg)
        return ConversationStateHelpers.set_done(state=state, action="NONE", msg=msg)

    chat_history = ConversationStateHelpers.chat_history_to_payload(state, k=12)

    # -------------------------
    # LLM #2: extractor (first)
    # -------------------------
    try:
        extractor_payload: dict[str, Any] = {
            "allowed_estimators": list(allowed),
            "selected_top3": selected_top3,
            "last_user_message": json.dumps(chat_history, ensure_ascii=False),
        }

        extracted = _llm_call_text(
            llm=llm,
            model_name=model_name,
            temperature=0.0,
            system_prompt=get_model_selection_discussion_extractor_system_prompt(),
            user_payload=extractor_payload,
            empty_err="LLM extractor returned empty output",
        )

        chosen = _resolve_choice_minimal(extracted, selected_top3=selected_top3, allowed=allowed_set)
        if chosen is not None:
            state["model_selection_discussion"] = ModelSelectionDiscussionState(selected_model_fqcn=chosen)
            msg = f"Model confirmed: {chosen}"
            ConversationStateHelpers.append_ai_message(state=state, content=msg)
            return ConversationStateHelpers.set_done(state=state, action="NONE", msg=msg)

    except Exception as e:
        log.exception("MODEL_SELECTION_DISCUSSION: extractor failed: %s", e)
        # Continue to LLM#1 discussion message.

    # -------------------------
    # LLM #1: discussion message (only if not selected)
    # -------------------------
    try:
        discussion_payload: dict[str, Any] = {
            "allowed_estimators": list(allowed),
            "model_selection_output": {
                "selected_top3": selected_top3,
                "selection_notes": ms.get("selection_notes"),
                "unknowns": ms.get("unknowns"),
                "rationale_text": ms.get("rationale_text"),
            },
            "chat_history": chat_history,
        }

        assistant_msg = _llm_call_text(
            llm=llm,
            model_name=model_name,
            temperature=0.3,
            system_prompt=get_model_selection_discussion_system_prompt(),
            user_payload=discussion_payload,
            empty_err="LLM discussion returned empty message",
        )

        ConversationStateHelpers.append_ai_message(state=state, content=assistant_msg)
        state["control"]["node_message"] = assistant_msg  # type: ignore[index]
        return ConversationStateHelpers.set_pending(state=state, action="NEEDS_INPUT", msg=assistant_msg)

    except Exception as e:
        log.exception("MODEL_SELECTION_DISCUSSION: discussion failed")
        return _abort(state, f"Model selection discussion failed: {e}")


def _abort(state: ConversationState, msg: str) -> ConversationState:
    ConversationStateHelpers.append_ai_message(state=state, content=msg)
    return ConversationStateHelpers.set_abort(state=state, action="NONE", msg=msg)


def _resolve_choice_minimal(extracted: str, *, selected_top3: List[str], allowed: set[str]) -> str | None:
    s = (extracted or "").strip()
    if not s or s.upper() == "NONE":
        return None

    # Exact fqcn only
    if s in allowed:
        return s

    # Allow "SELECT: <fqcn>" format
    m = re.match(r"(?i)^\s*select\s*:\s*(.+)\s*$", s)
    if m:
        candidate = m.group(1).strip()
        return candidate if candidate in allowed else None

    # Allow rank mapping only if selected_top3 present
    m2 = re.search(r"\b#?([1-3])\b", s)
    if m2 and selected_top3:
        idx = int(m2.group(1)) - 1
        if 0 <= idx < len(selected_top3):
            candidate = selected_top3[idx]
            return candidate if candidate in allowed else None

    return None


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
