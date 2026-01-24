from __future__ import annotations
import json
import logging
from uuid import UUID

from python.domain.service.llm_service import LLMConfig, LLMService
from python.workflows.nodes.prompts.review_protocol_static_validation_prompt import fatal_validation_prompt, warning_negotiation_prompt
from python.workflows.state.conversation_state import (
    CallableNodeFunc,
    ConversationState,
    ConversationStateHelpers,
)
from python.workflows.state.validate_protocol_state import ProtocolValidationReport

log = logging.getLogger(__name__)


def make_review_protocol_static_validation_node(
    *,
    llm: LLMService,
    model_name: str,
) -> CallableNodeFunc:
    def _run(
        user_id: UUID,
        conversation_id: UUID,
        state: ConversationState,
    ) -> ConversationState:
        stage = "REVIEW_PROTOCOL_STATIC_VALIDATION"

        report = _require_report(state)
        fatal = _has_fatal(report)
        
        if fatal:
            llm_text = _call_fatal_prompt(
                llm=llm,
                model_name= model_name,
                report=report
            )
            
            ConversationStateHelpers.append_ai_message(
                state,
                llm_text,
                stage=stage,
            )

            _set_fatal(state, stage)
            return state

        llm_text = _call_warning_prompt(
            llm=llm,
            model_name=model_name,
            state=state,
            report=report,
            chat_k=8,
        )

        llm_text = llm_text.strip()

        # ACCEPTED is the ONLY success signal
        if llm_text == "ACCEPTED":
            ConversationStateHelpers.append_ai_message(
                state,
                "Your choices have been accepted. Proceeding to the next step.",
                stage=stage,
            )
            _set_done(state, "Your choices have been accepted. Proceeding to the next step.")
            return state

        ConversationStateHelpers.append_ai_message(
            state,
            llm_text,
            stage=stage,
        )

        _set_pending(state, llm_text)
        return state

    return _run


# ---------------------------------------------------------------------
# Helpers (pure structure, no semantics)
# ---------------------------------------------------------------------


def _require_report(state: ConversationState) -> ProtocolValidationReport:
    psv = state.get("protocol_static_validation")
    if not psv:
        raise RuntimeError("protocol_static_validation missing from ConversationState")

    report = psv.get("report")
    if not report:
        raise RuntimeError("protocol_static_validation.report missing or invalid")

    return report


def _has_fatal(report: ProtocolValidationReport) -> bool:
    if str(report.get("status")).upper() == "FAIL":
        return True

    for issue in report.get("issues", []):
        if str(issue.get("severity")).upper() == "FAIL":
            return True

    return False


def _call_fatal_prompt(
    *,
    llm: LLMService,
    model_name: str,
    report: ProtocolValidationReport,
) -> str:
    prompt = fatal_validation_prompt().replace(
        "{{REPORT_JSON}}",
        json.dumps(report, ensure_ascii=False),
    )
    
    config = LLMConfig(model=model_name, temperature=0.5)

    return llm.generate(config=config,
                        system_prompt= "Listen to user :)"
                        ,user_prompt=prompt,
                        history=None).content


def _call_warning_prompt(
    *,
    llm: LLMService,
    model_name: str,
    state: ConversationState,
    report: ProtocolValidationReport,
    chat_k: int,
) -> str:
    user_last = ConversationStateHelpers.last_human_text(state)

    chat_history = ConversationStateHelpers.chat_history_to_payload(
        state,
        k=chat_k,
        drop_last_user=False,
        drop_system=True,
    )

    prompt = (
        warning_negotiation_prompt()
        .replace("{{USER_LAST_MESSAGE_JSON}}", json.dumps(user_last, ensure_ascii=False))
        .replace("{{CHAT_HISTORY_JSON}}", json.dumps(chat_history, ensure_ascii=False))
        .replace("{{REPORT_JSON}}", json.dumps(report, ensure_ascii=False))
    )
    config = LLMConfig(model=model_name, temperature=0.5)
    return llm.generate(config=config,
                        system_prompt= "Listen to user :)"
                        ,user_prompt=prompt,
                        history=None).content


def _set_fatal(state: ConversationState, message: str) -> None:
    control = state["control"]
    control["current_stage_status"] = "ABORTED"
    control["node_message"] = message
    state["control"] = control

def _set_done(state: ConversationState, message: str) -> None:
    control = state["control"]
    control["current_stage_status"] = "DONE"
    control["node_message"] = message
    state["control"] = control


def _set_pending(state: ConversationState, message: str) -> None:
    control = state["control"]
    control["current_stage_status"] = "DONE"
    control["node_message"] = message
    control["action_required"] = "NEEDS_INPUT"
    state["control"] = control

