from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import UUID

from python.domain.service.llm_service import LLMConfig, LLMService
from python.workflows.nodes.prompts.leakage_report import get_leakage_scan_repair_prompt, get_leakage_scan_system_prompt, get_leakage_scan_user_message_prompt
from python.workflows.state.control_state import ACTION, ControlState
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState, ConversationStateHelpers
from python.workflows.state.leakage_report_state import LeakageFinding, LeakageReportState
from python.workflows.state.protocol_state import ProtocolState

log = logging.getLogger(__name__)


_ALLOWED_RISKS: set[str] = {"LOW", "MED", "HIGH"}


# =============================================================================
# public factory
# =============================================================================
def make_leakage_report_node(
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

    dataset = state.get("dataset")
    if not dataset:
        return _set_fatal(state, "Dataset is missing. Reload dataset is required.","NEEDS_INPUT")

    dataset_id = dataset.get("id")
    if dataset_id is None:
        return _set_fatal(state, "Dataset id is missing. Reload dataset is required.","NEEDS_INPUT")

    summary = dataset.get("summary")
    if not isinstance(summary, dict) or not summary:
        return _set_fatal(state, "Dataset summary missing. Reload dataset is required.","NONE")

    protocol = state.get("protocol")
    if protocol is None:
        return _set_fatal(state, "Protocol is missing. Compile/confirm protocol first.","NONE")

    all_cols = list(summary.keys())
    z_candidates = _build_z_candidates(protocol=protocol, all_columns=all_cols)

    if not z_candidates:
        msg = "No covariates/effect modifiers were provided, so there is nothing to leakage-scan."
        control["node_message"] = msg
        ConversationStateHelpers.append_ai_message(state, msg, stage=control["current_stage"])
        _set_done(control, msg)
        state["leakage_report"] = LeakageReportState(
            z_candidates=[],
            findings=[],
            danger_list=[],
            n_low=0,
            n_med=0,
            n_high=0,
            model_name=model_name,
            created_at=_now_iso(),
            notes="No Z candidates",
            raw_llm_output=None,
        )
        return state

    z_profiles: List[Dict[str, Any]] = [{"name": z, "profile": summary.get(z, {})} for z in z_candidates]
    chat_history_messages = ConversationStateHelpers.chat_history_to_payload(state, k=7)

    payload: Dict[str, Any] = {
        "protocol": protocol,
        "protocol_string": _safe_protocol_string(protocol),
        "z_candidates": z_candidates,
        "z_candidate_profiles": z_profiles,
        "recent_conversation": chat_history_messages,
    }

    # -------------------------
    # LLM #1: produce report JSON
    # -------------------------
    raw_llm: Optional[str] = None
    parsed: Optional[Dict[str, Any]] = None

    try:
        raw_llm = _llm_call_text(
            llm=llm,
            model_name=model_name,
            temperature=0.0,
            system_prompt=get_leakage_scan_system_prompt(),
            user_payload=payload,
            empty_err="LLM leakage scan returned empty output",
        )
        parsed = _parse_json_strict(raw_llm)
        _validate_leakage_json(parsed, z_candidates)
    except Exception as e:
        log.exception("LEAKAGE_SCAN: LLM#1 invalid; attempting repair")
        # -------------------------
        # LLM #1b: repair JSON
        # -------------------------
        try:
            repair_payload = { # pyright: ignore[reportUnknownVariableType]
                "z_candidates": z_candidates,
                "required_schema": {
                    "findings": [{"z": "string", "risk": "LOW|MED|HIGH", "reason": "string", "action": "string"}],
                    "notes": "string",
                },
                "previous_output": raw_llm or "",
                "validation_error": str(e),
            }
            repaired = _llm_call_text(
                llm=llm,
                model_name=model_name,
                temperature=0.0,
                system_prompt=get_leakage_scan_repair_prompt(),
                user_payload=repair_payload, # pyright: ignore[reportUnknownArgumentType]
                empty_err="LLM repair returned empty output",
            )
            parsed = _parse_json_strict(repaired)
            _validate_leakage_json(parsed, z_candidates)
            raw_llm = repaired  # store repaired as final raw
        except Exception as e2:
            log.exception("LEAKAGE_SCAN: repair failed")
            return _set_fatal(state, f"Leakage scan failed (invalid JSON). Error: {e2} run this state again","NONE")

    assert parsed is not None

    findings = _coerce_findings(parsed, z_candidates)
    danger_list = [f["z"] for f in findings if f.get("risk") == "HIGH"]
    n_low, n_med, n_high = _count_risks(findings)
    notes = str(parsed.get("notes", "") or "").strip()

    report: LeakageReportState = {
        "z_candidates": z_candidates,
        "findings": [
            LeakageFinding(
                z=f["z"],
                risk_final=f["risk"],  # keep compatibility: risk_final is the LLM output # pyright: ignore[reportArgumentType]
                reason=f.get("reason", ""),
                action=f.get("action", ""),
            )
            for f in findings
        ],
        "danger_list": danger_list,
        "n_low": n_low,
        "n_med": n_med,
        "n_high": n_high,
        "model_name": model_name,
        "created_at": _now_iso(),
        "notes": notes,
        "raw_llm_output": raw_llm,
    }

    state["leakage_report"] = report

    # -------------------------
    # LLM #2: user-facing message
    # -------------------------
    user_msg_payload = { # pyright: ignore[reportUnknownVariableType]
        "protocol_string": _safe_protocol_string(protocol),
        "summary_counts": {"low": n_low, "med": n_med, "high": n_high},
        "danger_list": danger_list,
        "findings": [
            {
                "z": f["z"],
                "risk": f["risk"],
                "reason": f.get("reason", ""),
                "action": f.get("action", ""),
            }
            for f in findings
        ],
    }

    
    node_msg = _llm_call_text(
            llm=llm,
            model_name=model_name,
            temperature=0.3,
            system_prompt=get_leakage_scan_user_message_prompt(),
            user_payload=user_msg_payload, # pyright: ignore[reportUnknownArgumentType]
            empty_err="LLM user message returned empty output",
        )

    control["node_message"] = node_msg
    ConversationStateHelpers.append_ai_message(state, node_msg, stage=control["current_stage"])

    _set_done(control, node_msg)
    return state


# =============================================================================
# candidate building
# =============================================================================
def _build_z_candidates(*, protocol: ProtocolState, all_columns: List[str]) -> List[str]:
    cov = protocol.get("covariates", []) or []
    em = protocol.get("effect_modifiers", []) or []
    censor = protocol.get("censoring_rules", []) or []

    out: List[str] = []
    out.extend([c for c in cov if isinstance(c, str) and c.strip()])
    out.extend([c for c in em if isinstance(c, str) and c.strip()])
    out.extend(_extract_columns_from_rules(censor, all_columns))

    seen = set()
    deduped: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _extract_columns_from_rules(rules: Sequence[str], all_columns: List[str]) -> List[str]:
    cols = set(all_columns)
    found: List[str] = []
    for r in rules:
        if not isinstance(r, str) or not r.strip():
            continue
        for c in all_columns:
            if c in cols and c in r:
                found.append(c)
    return found


# =============================================================================
# JSON parsing + validation (structural only)
# =============================================================================
def _parse_json_strict(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty output")

    # strictly expect JSON; if model returned code fences, this is an LLM formatting error
    # (repair prompt will correct it). We keep parsing strict here.
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON must be an object")
    return obj


def _validate_leakage_json(obj: Mapping[str, Any], z_candidates: List[str]) -> None:
    findings = obj.get("findings")
    if not isinstance(findings, list):
        raise ValueError("Missing or invalid 'findings' list")

    seen: set[str] = set()
    for i, item in enumerate(findings):
        if not isinstance(item, dict):
            raise ValueError(f"findings[{i}] must be an object")

        z = str(item.get("z", "") or "").strip()
        if not z:
            raise ValueError(f"findings[{i}].z is missing/empty")
        if z not in z_candidates:
            raise ValueError(f"findings[{i}].z='{z}' is not in Z_candidates")
        if z in seen:
            raise ValueError(f"Duplicate z in findings: '{z}'")
        seen.add(z)

        risk = str(item.get("risk", "") or "").strip().upper()
        if risk not in _ALLOWED_RISKS:
            raise ValueError(f"Invalid risk for z='{z}': '{risk}' (must be LOW|MED|HIGH)")

        reason = str(item.get("reason", "") or "").strip()
        action = str(item.get("action", "") or "").strip()
        if not reason:
            raise ValueError(f"Missing reason for z='{z}'")
        if not action:
            raise ValueError(f"Missing action for z='{z}'")

    missing = [z for z in z_candidates if z not in seen]
    if missing:
        raise ValueError(f"Missing findings for Z_candidates: {missing}")


def _coerce_findings(obj: Mapping[str, Any], z_candidates: List[str]) -> List[Dict[str, str]]:
    # validated already; now normalize ordering to z_candidates
    by_z: Dict[str, Dict[str, str]] = {}
    for it in obj.get("findings", []):  # type: ignore[assignment]
        z = str(it.get("z", "") or "").strip()
        by_z[z] = {
            "z": z,
            "risk": str(it.get("risk", "") or "").strip().upper(),
            "reason": str(it.get("reason", "") or "").strip(),
            "action": str(it.get("action", "") or "").strip(),
        }
    return [by_z[z] for z in z_candidates if z in by_z]


def _count_risks(findings: Sequence[Mapping[str, Any]]) -> Tuple[int, int, int]:
    n_low = sum(1 for f in findings if str(f.get("risk", "")).upper() == "LOW")
    n_med = sum(1 for f in findings if str(f.get("risk", "")).upper() == "MED")
    n_high = sum(1 for f in findings if str(f.get("risk", "")).upper() == "HIGH")
    return n_low, n_med, n_high


# =============================================================================
# control helpers
# =============================================================================
def _set_done(control: ControlState, msg: str) -> None:
    control["current_stage_status"] = "DONE"
    control["action_required"] = "NONE"
    control["node_message"] = msg

 
def _set_fatal(state: ConversationState, msg: str,action_required :ACTION) -> ConversationState:
    control = state["control"]
    control["current_stage_status"] = "ABORTED"
    control["action_required"] = action_required
    ConversationStateHelpers.append_ai_message(state, msg, stage=control["current_stage"])
    return state


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_protocol_string(protocol: ProtocolState) -> str:
    parts = [
        f"Population={protocol.get('population', '')}",
        f"TimeZeroType={protocol.get('time_zero_type', '')}",
        f"TimeZero={protocol.get('time_zero', '')}",
        f"Treatment={protocol.get('treatment', '')}",
        f"Comparator={protocol.get('comparator', '')}",
        f"Outcome={protocol.get('outcome', '')}",
        f"Covariates={', '.join(protocol.get('covariates', []) or [])}",
        f"EffectModifiers={', '.join(protocol.get('effect_modifiers', []) or [])}",
    ]
    return " | ".join(parts)

# =============================================================================
# LLM call adapter (same as before)
# =============================================================================
def _llm_call_text(
    *,
    llm: LLMService,
    model_name: str,
    temperature: float,
    system_prompt: str,
    user_payload: Dict[str, Any],
    empty_err: str,
) -> str:
    user_content = json.dumps(user_payload, ensure_ascii=False)
    config = LLMConfig(model=model_name, temperature=0.5)

    text = llm.generate(
        config=config,
        system_prompt= system_prompt,
        user_prompt= user_content,
        history=None
    ).content
    
    if not text:
        raise ValueError(empty_err)
    return text
