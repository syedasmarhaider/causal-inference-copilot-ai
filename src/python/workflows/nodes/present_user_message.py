# src/python/workflows/nodes/present_user_message.py
from __future__ import annotations

from typing import Callable, List, cast
import hashlib
import json
import logging

from langchain_core.messages import BaseMessage

from python.domain.service.llm_service import LLMService
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState, Need
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI, JSONDict
from python.workflows.utils.user_message_builder import build_user_message_with_llm

log = logging.getLogger("causal_copilot.workflow")


def _control(state: ConversationState) -> ControlState:
    return cast(ControlState, state["control"])  # type: ignore


def _error_signature(stage: str, conversation_id: str, err: JSONDict) -> str:
    # stable-ish signature for dedupe; does not leak huge payloads into logs
    payload = json.dumps(
        {"stage": stage, "conversation_id": conversation_id, "error": err},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_present_user_message_node(
    llm: LLMService,
    *,
    model_name: str = DEFAULT_MODEL_GEMNI,
    history_window: int = 12,
) -> Callable[[ConversationState], ConversationState]:
    def present(state: ConversationState) -> ConversationState:
        c = _control(state)
        need: Need = c["need"]

        stage = str(c.get("stage") or "")
        conversation_id = str(c.get("conversation_id") or "")

        # ---- log last_error (with dedupe) ----
        err = c.get("last_error")
        if isinstance(err, dict) and err:
            try:
                sig = _error_signature(stage, conversation_id, err)
            except Exception:
                sig = ""

            already = ""
            try:
                already = str(state.get("_logged_error_sig") or "")
            except Exception:
                already = ""

            if sig and sig != already:
                try:
                    log.error(
                        "workflow_error stage=%s conversation_id=%s error=%s",
                        stage,
                        conversation_id,
                        json.dumps(err, ensure_ascii=False),
                    )
                except Exception:
                    log.error(
                        "workflow_error stage=%s conversation_id=%s error=%r",
                        stage,
                        conversation_id,
                        err,
                    )
                # store signature so we don't re-log the same error every PRESENT
                state = {**state, "_logged_error_sig": sig} # pyright: ignore[reportAssignmentType]

        # ---- build assistant message ----
        ai = build_user_message_with_llm(
            llm=llm,
            state=state,
            model_name=model_name,
            history_window=history_window,
        )

        # append to history (do not overwrite)
        prior: List[BaseMessage] = cast(List[BaseMessage], state.get("messages", [])) # pyright: ignore[reportUnnecessaryCast]
        out_messages: List[BaseMessage] = [*prior, ai]

        # ---- need transition semantics ----
        if need == "PRESENT_AND_USER_INPUT":
            next_need: Need = "NEEDS_INPUT"
        elif need == "PRESENT":
            next_need = "NONE"
        else:
            # if PRESENT was called unexpectedly, do not destroy state
            next_need = need

        # clear node_message to prevent re-present loop
        c2: ControlState = cast(
            ControlState,
            {
                **c,
                "need": next_need,
                "node_message": "",
            },
        )

        return {**state, "control": c2, "messages": out_messages}

    return present
