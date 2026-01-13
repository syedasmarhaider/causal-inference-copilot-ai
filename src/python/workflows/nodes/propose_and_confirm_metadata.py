# src/python/workflows/nodes/propose_and_confirm_metadata.py
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, cast
from uuid import UUID

from langchain_core.messages import AIMessage

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.workflows.nodes.prompts.propose_and_confirm_metadata import (
    bad_metadata_edit_system_prompt,
    compose_node_message_system_prompt,
    edit_metadata_system_prompt,
    kickoff_system_prompt,
)
from python.workflows.state.conversation_state import (
    CallableNodeFunc,
    ConversationState,
    last_human_text,
    to_chat_history_last_k,
)
from python.workflows.state.control_state import ControlState
from python.workflows.state.metadata_state import MetadataState, empty_metadata

log = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_ALLOWED_STRATEGIES = {"USER_LIST", "ALL_EXCEPT_TY", "NONE"}


# =============================================================================
# Public factory
# =============================================================================
def make_propose_and_confirm_metadata_node(
    *,
    llm: LLMService,
    model_name: str,
) -> CallableNodeFunc:
    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        return _run_propose_and_confirm_metadata(
            user_id=user_id,
            conversation_id=conversation_id,
            state=state,
            llm=llm,
            model_name=model_name,
        )

    return node


# =============================================================================
# Node runner
# =============================================================================
def _run_propose_and_confirm_metadata(
    *,
    user_id: UUID,
    conversation_id: UUID,
    state: ConversationState,
    llm: LLMService,
    model_name: str,
) -> ConversationState:
    control = _require_control(state)
    md_old = _require_metadata(state)

    dataset_cols = _extract_dataset_columns(state.get("dataset")) or []
    now_iso = datetime.now(timezone.utc).isoformat()

    user_text = last_human_text(state)
    if user_text is None or not user_text.strip():
        # Kickoff
        node_msg = _llm_initial_prompt(llm=llm, model_name=model_name, dataset_columns=dataset_cols)
        control["node_message"] = node_msg
        control["current_stage"] = "PROPOSE_AND_CONFIRM_METADATA"
        control["current_stage_status"] = "PENDING"
        control["action_required"] = "NEEDS_INPUT"
        _append_ai_message(state, node_msg, stage="PROPOSE_AND_CONFIRM_METADATA")
        return state

    # last 10 messages, excluding the last user (we provide that separately as user_text)
    chat_history = to_chat_history_last_k(state, k=10, drop_last_user=True)

    # ---- LLM #1: strict JSON (LLM is responsible for FULL schema) ----
    try:
        obj1 = _llm_edit_metadata_json(
            llm=llm,
            model_name=model_name,
            temperature=0.0,
            current_metadata=md_old,
            user_text=user_text,
            dataset_columns=dataset_cols,
            now_iso=now_iso,
            chat_history=chat_history,
        )

        md_candidate = obj1.get("metadata")
        if not isinstance(md_candidate, dict):
            raise ValueError("LLM#1 output must contain object field 'metadata'")

        md_new = _sanitize_metadata(
            md_candidate, # type: ignore
            dataset_columns=dataset_cols,
        )

        # provenance (node-owned)
        prov = md_new.get("provenance")
        provenance: Dict[str, Any] = prov if isinstance(prov, dict) else {} # type: ignore
        provenance["metadata_llm1_updated_utc"] = now_iso
        provenance["metadata_llm1_raw"] = json.dumps(obj1, ensure_ascii=False)[:8000]
        provenance["user_id"] = str(user_id)
        provenance["conversation_id"] = str(conversation_id)
        md_new["provenance"] = provenance

        state["metadata"] = md_new

    except Exception as e:
        _abort(control, f"Metadata update failed: {e}")
        log.exception("PROPOSE_AND_CONFIRM_METADATA: LLM#1 failed")
        raise

    md = _require_metadata(state)

    # ---- LLM #2: user-facing message ----
    try:
        node_msg = _llm_compose_node_message(
            llm=llm,
            model_name=model_name,
            user_text=user_text,
            metadata=md,
            dataset_columns=dataset_cols,
            now_iso=now_iso,
            chat_history=chat_history,
        )
    except Exception as e:
        log.exception("PROPOSE_AND_CONFIRM_METADATA: LLM#2 failed, using fallback. err=%s", e)
        node_msg = _compose_plain_human_fallback(md)

    control["node_message"] = node_msg

    if bool(md.get("accepted", False)):
        control["current_stage"] = "DONE"
        control["current_stage_status"] = "DONE"
        control["action_required"] = "NONE"
    else:
        control["current_stage"] = "PROPOSE_AND_CONFIRM_METADATA"
        control["current_stage_status"] = "PENDING"
        control["action_required"] = "NEEDS_INPUT"

    _append_ai_message(state, node_msg, stage="PROPOSE_AND_CONFIRM_METADATA")
    return state


# =============================================================================
# LLM calls
# =============================================================================
def _llm_initial_prompt(
    *,
    llm: LLMService,
    model_name: str,
    dataset_columns: Optional[List[str]] = None,
) -> str:
    cols_preview = (dataset_columns or [])[:25]
    system = kickoff_system_prompt()
    payload = {"dataset_columns_preview": cols_preview}

    cfg = LLMConfig(model=model_name, temperature=0.4)
    msg = _llm_text(
        llm,
        config=cfg,
        system_prompt=system,
        user_prompt=json.dumps(payload, ensure_ascii=False),
        history=None,
    )
    msg = (msg or "").strip()
    if not msg:
        raise ValueError("Initial prompt LLM returned empty message")
    return msg


def _llm_edit_metadata_json(
    *,
    llm: LLMService,
    model_name: str,
    temperature: float,
    current_metadata: MetadataState,
    user_text: str,
    dataset_columns: List[str],
    now_iso: str,
    chat_history: Sequence[ChatMessage],
) -> Dict[str, Any]:
    schema_json = _metadata_schema_json()
    system = edit_metadata_system_prompt(schema_json=schema_json, now_iso=now_iso)

    payload: Dict[str, Any] = {
        "user_message": user_text,
        "current_metadata": current_metadata,
        "dataset_columns": dataset_columns,
    }

    return _llm_json_object_or_raise(
        llm=llm,
        model_name=model_name,
        system_prompt=system,
        temperature=temperature,
        user_payload=payload,
        expected_top_keys={"metadata"},
        schema_json=schema_json,
        chat_history=chat_history,
    )


def _llm_compose_node_message(
    *,
    llm: LLMService,
    model_name: str,
    user_text: str,
    metadata: MetadataState,
    dataset_columns: List[str],
    now_iso: str,
    chat_history: Sequence[ChatMessage],
) -> str:
    system = compose_node_message_system_prompt(now_iso=now_iso)
    payload: Dict[str, Any] = {
        "user_message": user_text,
        "metadata": metadata,
        "dataset_columns_preview": dataset_columns[:30],
    }

    cfg = LLMConfig(model=model_name, temperature=0.7)
    msg = _llm_text(
        llm,
        config=cfg,
        system_prompt=system,
        user_prompt=json.dumps(payload, ensure_ascii=False),
        history=chat_history,
    )
    msg = (msg or "").strip()
    if not msg:
        raise ValueError("Node message LLM returned empty message")
    return msg


# =============================================================================
# LLM adapter (supports both signatures)
# =============================================================================
def _llm_text(
    llm: LLMService,
    *,
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    history: Optional[Sequence[ChatMessage]],
) -> str:
    """
    Support both implementations:
    A) generate(config=..., system_prompt=..., user_prompt=..., history=...)
    B) generate(config=..., history=[ChatMessage...]) with system messages inside history
    """
    try:
        # Newer signature
        resp = llm.generate(  # type: ignore[call-arg]
            config=config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history,
        )
        return cast(Any, resp).content
    except TypeError:
        # Fallback signature
        msgs: List[ChatMessage] = []
        if system_prompt:
            msgs.append(ChatMessage(role="system", content=system_prompt))
        if history:
            msgs.extend(list(history))
        msgs.append(ChatMessage(role="user", content=user_prompt))
        resp = llm.generate(config=config, history=msgs)  # type: ignore[arg-type]
        return cast(Any, resp).content


# =============================================================================
# Strict JSON parsing + one repair attempt
# =============================================================================
def _parse_json_object_strict(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty LLM response")

    m = _JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON root must be an object")
    return cast(Dict[str, Any], obj)


def _llm_repair_json_once(
    *,
    llm: LLMService,
    model_name: str,
    schema_json: str,
    bad_text: str,
) -> str:
    cfg = LLMConfig(model=model_name, temperature=0.0)
    return _llm_text(
        llm,
        config=cfg,
        system_prompt=bad_metadata_edit_system_prompt(schema_json),
        user_prompt=bad_text,
        history=None,
    )


def _llm_json_object_or_raise(
    *,
    llm: LLMService,
    model_name: str,
    system_prompt: str,
    temperature: float,
    user_payload: Dict[str, Any],
    expected_top_keys: set[str],
    schema_json: str,
    chat_history: Sequence[ChatMessage],
) -> Dict[str, Any]:
    cfg = LLMConfig(model=model_name, temperature=temperature)

    raw = _llm_text(
        llm,
        config=cfg,
        system_prompt=system_prompt,
        user_prompt=json.dumps(user_payload, ensure_ascii=False),
        history=chat_history,
    )

    try:
        obj = _parse_json_object_strict(raw)
    except Exception:
        repaired = _llm_repair_json_once(
            llm=llm,
            model_name=model_name,
            schema_json=schema_json,
            bad_text=raw,
        )
        obj = _parse_json_object_strict(repaired)

    if set(obj.keys()) != expected_top_keys:
        raise ValueError(
            f"LLM JSON must have exactly keys: {sorted(expected_top_keys)} (got {sorted(obj.keys())})"
        )
    return obj


def _metadata_schema_json() -> str:
    schema_obj: Dict[str, Any] = {
        "metadata": {
            "treatment": "",
            "outcome": "",
            "confounder_strategy": "NONE",  # USER_LIST | ALL_EXCEPT_TY | NONE
            "confounders": [],
            "controls": [],
            "effect_modifiers": [],
            "causal_question": "",
            "accepted": False,
            "dataset_summary": "",
            "notes": [],
            "warnings": [],
            "provenance": {},
        }
    }
    return json.dumps(schema_obj, indent=2, ensure_ascii=False)


# =============================================================================
# Sanitization (types + invariants only; NO lock/intent gating)
# =============================================================================
def _sanitize_metadata(md_any: Dict[str, Any], *, dataset_columns: Optional[List[str]]) -> MetadataState:
    """
    Minimal sanitization only:
    - Ensure schema keys exist, drop unknown keys
    - Type coercion (strings/lists/bools)
    - Enum enforcement for confounder_strategy
    - Invariants: NONE => confounders=[]
    - Accepted validity: requires treatment+outcome
    This does NOT try to infer intent or revert changes.
    """
    base: MetadataState = empty_metadata()

    # keep only known keys
    for k in list(md_any.keys()):
        if k not in base:
            md_any.pop(k, None)

    # overlay md_any on top of base
    for k, v in md_any.items():
        if k in base:
            base[k] = v  # type: ignore[literal-required]

    def s(x: Any) -> str:
        if not isinstance(x, str):
            return ""
        return x.replace("\n", " ").replace("\r", " ").strip()

    def ls(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return _dedupe([str(i).strip() for i in x if str(i).strip()]) # type: ignore
        if isinstance(x, str):
            return _dedupe([p.strip() for p in x.split(",") if p.strip()])
        return []

    def b(x: Any) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            t = x.strip().lower()
            if t in {"true", "yes", "y", "1"}:
                return True
            if t in {"false", "no", "n", "0"}:
                return False
        if isinstance(x, (int, float)):
            return bool(x)
        return False

    base["treatment"] = s(base.get("treatment"))
    base["outcome"] = s(base.get("outcome"))
    base["causal_question"] = s(base.get("causal_question"))
    base["dataset_summary"] = s(base.get("dataset_summary"))

    base["confounders"] = ls(base.get("confounders"))
    base["controls"] = ls(base.get("controls"))
    base["effect_modifiers"] = ls(base.get("effect_modifiers"))
    base["notes"] = ls(base.get("notes"))
    base["warnings"] = ls(base.get("warnings"))
    base["accepted"] = b(base.get("accepted"))

    prov = base.get("provenance")
    base["provenance"] = prov if isinstance(prov, dict) else {} # pyright: ignore[reportUnnecessaryIsInstance]

    strat = s(base.get("confounder_strategy")) or "NONE"
    if strat not in _ALLOWED_STRATEGIES:
        strat = "NONE"
    base["confounder_strategy"] = cast(Any, strat)

    if base["confounder_strategy"] == "NONE":
        base["confounders"] = []

    # accepted validity (runnable)
    if base["accepted"] and (not base["treatment"] or not base["outcome"]):
        base["accepted"] = False
        base["warnings"].append("Cannot accept: treatment/outcome is missing.")

    # optional soft validation (never blocks)
    if dataset_columns:
        colset = {c.strip() for c in dataset_columns if isinstance(c, str) and c.strip()} # pyright: ignore[reportUnnecessaryIsInstance]

        def warn_unknown(kind: str, col: str) -> None:
            if col and col not in colset:
                base["warnings"].append(f"{kind} references unknown column: {col}")

        if base["treatment"]:
            warn_unknown("treatment", base["treatment"])
        if base["outcome"]:
            warn_unknown("outcome", base["outcome"])
        for c in base["confounders"]:
            warn_unknown("confounders", c)

    base["warnings"] = _dedupe([w for w in base["warnings"] if w.strip()])
    base["notes"] = _dedupe([n for n in base["notes"] if n.strip()])

    return base


def _dedupe(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


# =============================================================================
# Control + Messages
# =============================================================================
def _require_control(state: ConversationState) -> ControlState:
    c = state.get("control")
    if not isinstance(c, dict): # type: ignore
        raise ValueError("ConversationState.control must exist and be a dict")
    return cast(ControlState, c) # type: ignore


def _require_metadata(state: ConversationState) -> MetadataState:
    md_any = state.get("metadata")
    md = cast(Dict[str, Any], md_any) if isinstance(md_any, dict) else {} # type: ignore
    sanitized = _sanitize_metadata(md, dataset_columns=None)
    state["metadata"] = sanitized
    return sanitized


def _abort(control: ControlState, msg: str) -> None:
    control["current_stage"] = "PROPOSE_AND_CONFIRM_METADATA"
    control["current_stage_status"] = "ABORTED"
    control["action_required"] = "NEEDS_INPUT"
    control["node_message"] = msg


def _append_ai_message(state: ConversationState, content: str, *, stage: str) -> None:
    msgs = state.get("messages")
    if not isinstance(msgs, list): # type: ignore
        state["messages"] = []
        msgs = state["messages"]
    msgs.append(AIMessage(content=content, additional_kwargs={"source": "node", "stage": stage}))


def _compose_plain_human_fallback(md: MetadataState) -> str:
    t = md.get("treatment") or "not set"
    y = md.get("outcome") or "not set"
    q = md.get("causal_question") or "not set"
    conf = md.get("confounders") or []
    conf_txt = ", ".join(conf) if conf else "none yet"
    return (
        f"Okay — current draft: treatment={t}, outcome={y}, causal question={q}, confounders={conf_txt}. "
        "What do you want to set or change next?"
    )


# =============================================================================
# Dataset columns (best effort)
# =============================================================================
def _extract_dataset_columns(dataset_state: Any) -> Optional[List[str]]:
    if not isinstance(dataset_state, dict):
        return None

    raw_schema = dataset_state.get("raw_schema") # type: ignore
    if isinstance(raw_schema, dict):
        cols = raw_schema.get("columns") # type: ignore
        if isinstance(cols, list):
            names: List[str] = []
            for c in cols: # pyright: ignore[reportUnknownVariableType]
                if isinstance(c, dict):
                    n = c.get("name") # type: ignore
                    if isinstance(n, str) and n.strip():
                        names.append(n.strip())
            if names:
                return names

    for key in ("columns", "col_names", "column_names", "schema_columns"):
        v = dataset_state.get(key) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if isinstance(v, list) and all(isinstance(x, str) for x in v): # pyright: ignore[reportUnknownVariableType]
            out = [c.strip() for c in v if c.strip()] # type: ignore
            return out or None # type: ignore

    schema = dataset_state.get("schema") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if isinstance(schema, dict):
        out2 = [str(k).strip() for k in schema.keys() if str(k).strip()] # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        return out2 or None

    return None
