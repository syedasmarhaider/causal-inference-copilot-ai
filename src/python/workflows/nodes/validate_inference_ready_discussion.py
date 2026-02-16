from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Final, List, Literal, Optional, Tuple, cast
from uuid import UUID

from python.domain.service.llm_service import LLMConfig, LLMService
from python.workflows.nodes.prompts.validate_protocol_discussion_node import get_validate_inference_ready_discussion_system_prompt
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState, ConversationStateHelpers
from python.workflows.state.control_state import ACTION

log = logging.getLogger(__name__)

RouteToken = Literal["DONE", "ABORT", "DISCUSS"]
ValidationStatus = Literal["PASS", "WARN", "FAIL"]

MAX_ATTEMPTS: Final[int] = 2


@dataclass(frozen=True)
class ValidateInferenceReadyDiscussionConfig:
    history_k: int = 12
    max_attempts: int = MAX_ATTEMPTS
    temperature: float = 0.0


def make_validate_inference_ready_discussion_node(*, llm: LLMService, model_name: str) -> CallableNodeFunc:
    cfg = ValidateInferenceReadyDiscussionConfig()

    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        return _run(state=state, llm=llm, model_name=model_name, cfg=cfg)

    return node


def _run(*, state: ConversationState, llm: LLMService, model_name: str, cfg: ValidateInferenceReadyDiscussionConfig) -> ConversationState:
    report = _get_inference_ready_validation_report(state)
    if report is None:
        msg = "InferenceReadyStaticValidationState.report missing; run VALIDATE_INFERENCE_READY_STATIC first."
        ConversationStateHelpers.append_ai_message(state=state, content=msg)
        return ConversationStateHelpers.set_abort(state=state, action=cast(ACTION, "NONE"), msg=msg)

    status = _normalize_status(report.get("status"))

    # PASS => no discussion, no user-facing output
    if status == "PASS":
        return ConversationStateHelpers.set_done(state=state, action=cast(ACTION, "NONE"), msg="")

    payload = _build_payload(state=state, report=report, history_k=cfg.history_k)

    token, discuss_msg = _llm_route(llm=llm, model_name=model_name, payload=payload, cfg=cfg)

    # Hard rule: FAIL cannot proceed
    if status == "FAIL" and token == "DONE":
        token = "DISCUSS"
        discuss_msg = ""

    if token == "DONE":
        return ConversationStateHelpers.set_done(state=state, action=cast(ACTION, "NONE"), msg="")

    if token == "ABORT":
        return ConversationStateHelpers.set_abort(state=state, action=cast(ACTION, "NONE"), msg="")

    msg = (discuss_msg or "").strip()
    if not msg:
        msg = _fallback_discuss_message(status=status, report=report)

    ConversationStateHelpers.append_ai_message(state=state, content=msg)
    state["control"]["node_message"] = msg
    return ConversationStateHelpers.set_pending(state=state, action=cast(ACTION, "NEEDS_INPUT"), msg=msg)


# =============================================================================
# Typed getters + payload
# =============================================================================

def _get_inference_ready_validation_report(state: ConversationState) -> Optional[Dict[str, Any]]:
    """
    Expected shape (you should store it in state somewhere):
      state["inference_ready_validation"] = {"report": {...}}
    """
    irv = state.get("inference_ready_validation")  # type: ignore[index]
    if not isinstance(irv, dict):
        return None
    rep = irv.get("report")
    if not isinstance(rep, dict):
        return None
    return cast(Dict[str, Any], rep)


def _build_payload(*, state: ConversationState, report: Dict[str, Any], history_k: int) -> Dict[str, Any]:
    chat_history = ConversationStateHelpers.chat_history_to_payload(state, k=history_k)
    last_user = ConversationStateHelpers.last_human_text(state) or ""

    ir_state = state.get("inference_ready") or {}
    if not isinstance(ir_state, dict):
        ir_state = {}

    # Prefer prepared dataset summary if present, else fall back to raw dataset summary.
    dataset_summary: Any = None
    prep_ds = ir_state.get("prepared_dataset")
    if isinstance(prep_ds, dict) and "summary" in prep_ds:
        dataset_summary = prep_ds.get("summary")
    else:
        ds = state.get("dataset") or {}
        if isinstance(ds, dict):
            dataset_summary = ds.get("summary")

    return {
        "validation_report": report,
        "inference_ready_state": ir_state,
        "dataset_summary": dataset_summary,
        "chat_history": chat_history,
        "last_user_message": last_user,
    }


def _normalize_status(x: Any) -> ValidationStatus:
    s = str(x or "").upper().strip()
    if s in ("PASS", "WARN", "FAIL"):
        return cast(ValidationStatus, s)
    return "WARN"


# =============================================================================
# LLM routing
# =============================================================================

def _llm_route(*, llm: LLMService, model_name: str, payload: Dict[str, Any], cfg: ValidateInferenceReadyDiscussionConfig) -> Tuple[RouteToken, str]:
    sys = (get_validate_inference_ready_discussion_system_prompt() or "").strip()
    if not sys:
        sys = "You are a routing controller. Output only: DONE | ABORT | DISCUSS (+ message)."

    user_prompt = json.dumps(payload, ensure_ascii=False)
    if not user_prompt.strip():
        user_prompt = "{}"

    llm_cfg = LLMConfig(model=model_name, temperature=float(cfg.temperature))

    last_err: Optional[Exception] = None
    for _ in range(int(cfg.max_attempts)):
        try:
            resp = llm.generate(config=llm_cfg, system_prompt=sys, user_prompt=user_prompt, history=None)
            raw = str(cast(Any, resp).content or "").strip()
            if not raw:
                last_err = ValueError("Empty LLM output")
                continue
            return _parse_route_output(raw)
        except Exception as e:
            last_err = e

    log.exception("VALIDATE_INFERENCE_READY_DISCUSSION: LLM routing failed: %s", last_err)
    return "DISCUSS", ""


def _parse_route_output(raw: str) -> Tuple[RouteToken, str]:
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
        return "DISCUSS", msg

    # Salvage if model violated format
    if "ABORT" in head:
        return "ABORT", ""
    if "DONE" in head:
        return "DONE", ""
    return "DISCUSS", ""


# =============================================================================
# Fallback discuss message (no LLM)
# =============================================================================

def _fallback_discuss_message(*, status: ValidationStatus, report: Dict[str, Any]) -> str:
    issues_any = report.get("issues")
    issues = issues_any if isinstance(issues_any, list) else []

    n_warn = 0
    n_fail = 0
    top: List[str] = []

    for it in issues:
        if not isinstance(it, dict):
            continue
        sev = str(it.get("severity") or "").upper().strip()
        rid = str(it.get("rule_id") or "").strip()
        msg = str(it.get("message") or "").strip()

        if sev == "WARN":
            n_warn += 1
        elif sev == "FAIL":
            n_fail += 1

        if rid and msg and len(top) < 3:
            top.append(f"{rid}: {msg}")

    bullets = "\n".join(f"- {x}" for x in top) if top else "- (no issue details found)"

    if status == "FAIL":
        return (
            f"Inference-ready validation FAILED ({n_fail} fail, {n_warn} warn).\n"
            "Fix the FAIL items, then re-run validation. Top items:\n"
            f"{bullets}\n"
            "Reply with the rule_id you want to fix first."
        )

    # WARN
    return (
        f"Inference-ready validation has WARNINGS ({n_warn} warn).\n"
        "Top items:\n"
        f"{bullets}\n"
        "Reply with: proceed / abort / explain <rule_id>."
    )
