from __future__ import annotations

import json
import logging
from typing import Any, Dict, Final,  Optional, Tuple, cast
from uuid import UUID

from python.domain.service.llm_service import LLMConfig, LLMService
from python.workflows.nodes.prompts.validate_protocol_discussion import get_validate_protocol_discussion_system_prompt
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState, ConversationStateHelpers
from python.workflows.state.control_state import ACTION

log = logging.getLogger(__name__)

MAX_ATTEMPTS: Final[int] = 2


def make_validate_protocol_discussion_node(*, llm: LLMService, model_name: str) -> CallableNodeFunc:
    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        return _run(state=state, llm=llm, model_name=model_name)

    return node


def _run(*, state: ConversationState, llm: LLMService, model_name: str) -> ConversationState:
    report = _get_validation_report(state)
    if report is None:
        msg = "ProtocolStaticValidationState.report missing; run VALIDATE_PROTOCOL_STATIC first."
        # This is an infrastructure failure; OK to message.
        ConversationStateHelpers.append_ai_message(state=state, content=msg)
        return ConversationStateHelpers.set_abort(state=state, action=cast(ACTION, "NONE"), msg=msg)

    status = str(report.get("status") or "").upper().strip()

    # PASS => no discussion stage work
    if status == "PASS":
        state["control"]["node_message"] = None  # type: ignore[index]
        return ConversationStateHelpers.set_done(state=state, action=cast(ACTION, "NONE"), msg="Protocol static validation PASSED with no issues.")

    chat_history = ConversationStateHelpers.chat_history_to_payload(state, k=12)

    payload: Dict[str, Any] = {
        "validation_report": report,
        "protocol_state": state.get("protocol") or {},
        "dataset_summary": (state.get("dataset") or {}).get("summary"),
        "chat_history": chat_history,
    }

    token, discuss_msg = _llm_route(llm=llm, model_name=model_name, payload=payload)

    # Enforce hard rule: FAIL cannot proceed
    if status == "FAIL" and token == "DONE":
        token = "DISCUSS"
        discuss_msg = ""

    if token == "DONE":
        state["control"]["node_message"] = None  # type: ignore[index]
        # Per your requirement: no assistant message on DONE
        return ConversationStateHelpers.set_done(state=state, action=cast(ACTION, "NONE"), msg="Protocol static validation PASSED with no issues.")

    if token == "ABORT":
        state["control"]["node_message"] = None  # type: ignore[index]
        # Per your requirement: no assistant message on ABORT
        return ConversationStateHelpers.set_abort(state=state, action=cast(ACTION, "NONE"), msg="ABORT")

    # DISCUSS => must produce a user-facing message
    msg = (discuss_msg or "").strip()
    if not msg:
        msg = _fallback_discuss_message(status=status, report=report)

    ConversationStateHelpers.append_ai_message(state=state, content=msg)
    state["control"]["node_message"] = msg  
    return ConversationStateHelpers.set_pending(state=state, action= "NEEDS_INPUT", msg=msg)


# =============================================================================
# TypedDict-safe getters
# =============================================================================

def _get_validation_report(state: ConversationState) -> Optional[Dict[str, Any]]:
    psv = state.get("protocol_static_validation")
    if not isinstance(psv, dict):
        return None
    rep = psv.get("report")
    if not isinstance(rep, dict):
        return None
    return cast(Dict[str, Any], rep)


# =============================================================================
# LLM routing
# =============================================================================

def _llm_route(*, llm: LLMService, model_name: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    sys = get_validate_protocol_discussion_system_prompt()
    user_prompt = json.dumps(payload, ensure_ascii=False)

    # Prevent the exact backend error you hit earlier (empty content)
    if not sys.strip():
        sys = "You are a routing controller."
    if not user_prompt.strip():
        user_prompt = "{}"

    cfg = LLMConfig(model=model_name, temperature=0.0)
    last_err: Optional[Exception] = None

    for _ in range(MAX_ATTEMPTS):
        try:
            resp = llm.generate(config=cfg, system_prompt=sys, user_prompt=user_prompt, history=None)
            raw = str(cast(Any, resp).content or "").strip()
            if not raw:
                last_err = ValueError("Empty LLM output")
                continue

            return _parse_route_output(raw)
        except Exception as e:
            last_err = e

    log.exception("VALIDATE_PROTOCOL_DISCUSSION: LLM failed: %s", last_err)
    return "DISCUSS", ""


def _parse_route_output(raw: str) -> Tuple[str, str]:
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return "DISCUSS", ""

    head = lines[0].upper()
    if head == "DONE":
        return "DONE", ""
    if head == "ABORT":
        return "ABORT", ""
    if head == "DISCUSS":
        msg = "\n".join(lines[1:]).strip()
        return ("DISCUSS", msg) if msg else ("DISCUSS", "")

    # Salvage if model didn’t follow format
    if "ABORT" in head:
        return "ABORT", ""
    if "DONE" in head:
        return "DONE", ""
    return "DISCUSS", ""


def _fallback_discuss_message(*, status: str, report: Dict[str, Any]) -> str:
    issues = report.get("issues") or [] # pyright: ignore[reportUnknownVariableType]
    n_warn = sum(1 for it in issues if isinstance(it, dict) and str(it.get("severity")).upper() == "WARN") # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportUnknownMemberType]
    n_fail = sum(1 for it in issues if isinstance(it, dict) and str(it.get("severity")).upper() == "FAIL") # type: ignore

    if status == "FAIL":
        return (
            f"Static validation FAILED ({n_fail} fail issues). "
            "Fix the FAIL items in the protocol/dataset, then re-run validation. "
            "Ask me about a specific rule_id if you want details."
        )

    return (
        f"Static validation has WARNINGS ({n_warn} warn issues). "
        "Reply with: proceed / abort (or ask a question)."
    )
