from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, cast
from uuid import UUID

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.workflows.nodes.prompts.compile_protocol import (
    get_compile_protocol_system_prompt,
    get_compile_protocol_repair_system_prompt,
)
from python.workflows.state.conversation_state import (
    CallableNodeFunc,
    ConversationState,
    ConversationStateHelpers,
)
from python.workflows.state.control_state import ControlState
from python.workflows.state.dataset_state import DatasetStateHelpers
from python.workflows.state.protocol_state import ProtocolState

_DEFAULT_PREVIEW_LIMIT = 60

log = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


# =============================================================================
# Public factory
# =============================================================================
def make_compile_protocol_node(
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
# Node runner
# =============================================================================
def _run(*,    
    user_id: UUID,
    conversation_id: UUID,
    state: ConversationState,
    data_repo: DataRepo,
    llm: LLMService,
    model_name: str,) -> ConversationState:
    control = state["control"]

    pd = state.get("protocol_discussion")
    discussion = getattr(pd, "discussion", "") if pd is not None else ""
    if not discussion.strip():
        return _abort_with_feedback(
            state,
            "FEEDBACK: Protocol discussion is empty. Please answer the protocol questions first.",
        )
     
    dataset = state.get("dataset")
    dataset_id = dataset.get("id") if isinstance(dataset, dict) else None
    if dataset_id is None:
        return _abort_with_feedback(state, "Dataset id is missing. Reload dataset is required.")   

    try:
        df = data_repo.get_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
            limit=_DEFAULT_PREVIEW_LIMIT,
        )
    except Exception as e:
        return _abort_with_feedback(state, f"Failed to load dataset preview: {type(e).__name__}: {e}")

    dataset_cols = DatasetStateHelpers.extract_columns_from_df(df)
    chat_history = ConversationStateHelpers.to_chat_history_last_k(state, k=16, drop_last_user=False)

    payload: Dict[str, Any] = {
        "protocol_discussion": discussion,
        "dataset_columns_preview": dataset_cols[:60],
    }

    # -------------
    # LLM #1 compile
    # -------------
    out = _llm_call_text(
        llm=llm,
        model_name=model_name,
        temperature=0.0,
        system_prompt=get_compile_protocol_system_prompt(),
        user_payload=payload,
        history=chat_history,
        empty_err="CompileProtocol LLM returned empty output",
    )

    # Success path: JSON
    obj = _try_parse_protocol_json(out)
    if obj is not None:
        try:
            protocol = _validate_protocol_state(obj)
        except Exception as e:
            # treat as "needs repair"
            log.warning("COMPILE_PROTOCOL: JSON parsed but failed validation: %s", e)
            protocol = None

        if protocol is not None:
            state["protocol"] = protocol
            _set_done(control)
            # optional: append a small message for traceability
            ConversationStateHelpers.append_ai_message(
                state,
                "Protocol compiled successfully.",
                stage=control["current_stage"],
            )
            return state

    # Failure path: feedback
    if out.strip().upper().startswith("FEEDBACK:"):
        return _abort_with_feedback(state, out.strip())

    # -------------
    # LLM #2 repair
    # -------------
    repaired = _llm_call_text(
        llm=llm,
        model_name=model_name,
        temperature=0.0,
        system_prompt=get_compile_protocol_repair_system_prompt(),
        user_payload={
            "previous_output": out,
            "protocol_discussion": discussion,
            "dataset_columns_preview": dataset_cols,
        },
        history=None,
        empty_err="Repair LLM returned empty output",
    )

    obj2 = _try_parse_protocol_json(repaired)
    if obj2 is not None:
        try:
            protocol2 = _validate_protocol_state(obj2)
        except Exception as e:
            log.warning("COMPILE_PROTOCOL: repaired JSON invalid: %s", e)
            protocol2 = None

        if protocol2 is not None:
            state["protocol"] = protocol2
            _set_done(control)
            ConversationStateHelpers.append_ai_message(
                state,
                "Protocol compiled successfully (after repair).",
                stage=control["current_stage"],
            )
            return state

    if repaired.strip().upper().startswith("FEEDBACK:"):
        return _abort_with_feedback(state, repaired.strip())

    return _abort_with_feedback(
        state,
        "FEEDBACK: Cannot compile protocol yet. Please review your answers; some required fields are missing or inconsistent.",
    )


# =============================================================================
# Parsing + validation
# =============================================================================
def _try_parse_protocol_json(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.upper().startswith("FEEDBACK:"):
        return None

    m = _JSON_FENCE_RE.search(raw)
    if m:
        raw = m.group(1).strip()

    try:
        obj = json.loads(raw)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None
    return cast(Dict[str, Any], obj)


def _validate_protocol_state(obj: Dict[str, Any]) -> ProtocolState:
    """
    Strict runtime validation (no inference):
    - required keys present
    - enums valid
    - types correct-ish
    """
    required_keys = [
        "population",
        "time_zero_type",
        "time_zero",
        "time_zero_definition",
        "treatment",
        "treatment_window_start",
        "treatment_window_end",
        "treatment_window_unit",
        "comparator",
        "outcome",
        "outcome_is_duration",
        "outcome_window",
        "outcome_window_unit",
        "covariates",
        "effect_modifiers",
        "censoring_rules",
        "experiment_type",
    ]
    for k in required_keys:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")

    def s(x: Any) -> str:
        if not isinstance(x, str):
            raise TypeError("expected string")
        return x.strip()

    def b(x: Any) -> bool:
        if isinstance(x, bool):
            return x
        raise TypeError("expected bool")

    def ls(x: Any) -> List[str]:
        if isinstance(x, list) and all(isinstance(i, str) for i in x): # pyright: ignore[reportUnknownVariableType]
            return [cast(str, i).strip() for i in x if cast(str, i).strip()] # pyright: ignore[reportUnknownVariableType]
        raise TypeError("expected list[str]")

    out: Dict[str, Any] = {}
    out["population"] = s(obj["population"])

    tzt = s(obj["time_zero_type"])
    if tzt not in {"COLUMN", "CONCEPTUAL"}:
        raise ValueError("time_zero_type must be COLUMN or CONCEPTUAL")
    out["time_zero_type"] = cast(Any, tzt)

    out["time_zero"] = s(obj["time_zero"])
    out["time_zero_definition"] = s(obj["time_zero_definition"])

    out["treatment"] = s(obj["treatment"])
    out["treatment_window_start"] = s(obj["treatment_window_start"])
    out["treatment_window_end"] = s(obj["treatment_window_end"])

    twu = s(obj["treatment_window_unit"])
    if twu not in {"minutes", "hours", "days", "weeks", "months", "years"}:
        raise ValueError("invalid treatment_window_unit")
    out["treatment_window_unit"] = cast(Any, twu)

    out["comparator"] = s(obj["comparator"])

    out["outcome"] = s(obj["outcome"])
    out["outcome_is_duration"] = b(obj["outcome_is_duration"])
    out["outcome_window"] = s(obj["outcome_window"])

    owu = s(obj["outcome_window_unit"])
    if owu not in {"minutes", "hours", "days", "weeks", "months", "years"}:
        raise ValueError("invalid outcome_window_unit")
    out["outcome_window_unit"] = cast(Any, owu)

    out["covariates"] = ls(obj["covariates"])
    out["effect_modifiers"] = ls(obj["effect_modifiers"])
    out["censoring_rules"] = ls(obj["censoring_rules"])

    out["experiment_type"] = s(obj["experiment_type"])
    if out["experiment_type"] not in {"RCT", "OBSERVATIONAL", "Unknown", "UNKNOWN"}:
        # don't hard-fail on casing; normalize to canonical
        et = out["experiment_type"].upper()
        if et in {"RCT", "OBSERVATIONAL", "UNKNOWN"}:
            out["experiment_type"] = "Unknown" if et == "UNKNOWN" else et
        else:
            raise ValueError("experiment_type must be RCT or OBSERVATIONAL or Unknown")

    # additional invariant: time_zero may be empty only when CONCEPTUAL
    if out["time_zero_type"] == "COLUMN" and not out["time_zero"]:
        raise ValueError("time_zero required when time_zero_type=COLUMN")

    return cast(ProtocolState, out)


# =============================================================================
# Control + abort handling
# =============================================================================
def _set_done(control: ControlState) -> None:
    control["current_stage_status"] = "DONE"
    control["action_required"] = "NONE"


def _abort_with_feedback(state: ConversationState, feedback: str) -> ConversationState:
    control = state["control"]
    control["current_stage_status"] = "ABORTED"
    control["action_required"] = "NEEDS_INPUT"
    control["node_message"] = feedback
    # Append feedback message for UI trace
    ConversationStateHelpers.append_ai_message(
        state,
        feedback,
        stage=control["current_stage"],
    )
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
    user_payload: Dict[str, Any],
    history: Optional[Sequence[ChatMessage]],
    empty_err: str,
) -> str:
    cfg = LLMConfig(model=model_name, temperature=temperature)
    raw = _llm_text(
        llm,
        config=cfg,
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=False),
        history=history,
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
    history: Optional[Sequence[ChatMessage]],
) -> str:
    try:
        resp = llm.generate(  # type: ignore[call-arg]
            config=config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history,
        )
        return cast(Any, resp).content
    except TypeError:
        msgs: List[ChatMessage] = []
        if system_prompt:
            msgs.append(ChatMessage(role="system", content=system_prompt))
        if history:
            msgs.extend(list(history))
        msgs.append(ChatMessage(role="user", content=user_prompt))
        resp = llm.generate(config=config, history=msgs)  # type: ignore[arg-type]
        return cast(Any, resp).content
