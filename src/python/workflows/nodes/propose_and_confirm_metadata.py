# src/python/workflows/nodes/propose_and_confirm_metadata.py
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, cast
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.workflows.nodes.prompts.propose_and_confirm_metadata import (
    compose_node_message_system_prompt,
    edit_metadata_system_prompt,
    kickoff_system_prompt,
)
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState
from python.workflows.state.control_state import ControlState
from python.workflows.state.metadata_state import MetadataField, MetadataState, empty_metadata

log = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

_ALLOWED_STRATEGIES = {"USER_LIST", "ALL_EXCEPT_TY", "NONE"}

# IMPORTANT: we use confounders naming (not covariates)
_ALLOWED_LOCK_FIELDS: set[str] = {
    "dataset_summary",
    "treatment",
    "outcome",
    "confounder_strategy",
    "confounders",
    "controls",
    "effect_modifiers",
    "causal_question",
}


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
    user_text = _last_human_text(cast(Sequence[BaseMessage], state.get("messages", []))).strip()

    # Kickoff
    if not user_text:
        try:
            node_msg = _llm_initial_prompt(
                llm=llm,
                model_name=model_name,
                dataset_columns=dataset_cols,
            )
        except Exception as e:
            _abort(control, f"Initial prompt LLM failed: {e}")
            log.exception("PROPOSE_AND_CONFIRM_METADATA: initial prompt failed")
            raise ValueError(f"PROPOSE_AND_CONFIRM_METADATA: initial prompt failed: {e}") from e

        control["node_message"] = node_msg
        control["current_stage"] = "PROPOSE_AND_CONFIRM_METADATA"
        control["current_stage_status"] = "PENDING"
        control["action_required"] = "NEEDS_INPUT"
        _append_ai_message(state, node_msg, stage="PROPOSE_AND_CONFIRM_METADATA")
        return state

    # ---- LLM #1: strict JSON metadata edit ----
    try:
        obj1 = _llm_edit_metadata_json(
            llm=llm,
            model_name=model_name,
            temperature=0.0,
            current_metadata=md_old,
            user_text=user_text,
            dataset_columns=dataset_cols,
            now_iso=now_iso,
        )

        md_candidate = obj1.get("metadata")
        if not isinstance(md_candidate, dict):
            raise ValueError("LLM#1 output must contain object field 'metadata'")

        md_new = _harden_metadata(
            old=md_old,
            new=cast(Dict[str, Any], md_candidate),
            dataset_columns=dataset_cols,
            user_text=user_text,
        )

        prov_in = md_new.get("provenance")
        provenance: Dict[str, Any] = prov_in if isinstance(prov_in, dict) else {}
        provenance["metadata_llm1_updated_utc"] = now_iso
        provenance["metadata_llm1_raw"] = json.dumps(obj1, ensure_ascii=False)[:8000]
        provenance["user_id"] = str(user_id)
        provenance["conversation_id"] = str(conversation_id)
        md_new["provenance"] = provenance

        state["metadata"] = md_new

    except Exception as e:
        _abort(control, f"Metadata update failed: {e}")
        log.exception("PROPOSE_AND_CONFIRM_METADATA: LLM#1 failed")
        raise ValueError(f"PROPOSE_AND_CONFIRM_METADATA: LLM#1 failed: {e}") from e

    md = _require_metadata(state)

    # ---- LLM #2: natural user-facing message ----
    try:
        node_msg = _llm_compose_node_message(
            llm=llm,
            model_name=model_name,
            user_text=user_text,
            metadata=md,
            dataset_columns=dataset_cols,
            now_iso=now_iso,
        )
    except Exception as e:
        _abort(control, f"Message generation failed: {e}")
        log.exception("PROPOSE_AND_CONFIRM_METADATA: LLM#2 failed")
        raise ValueError(f"PROPOSE_AND_CONFIRM_METADATA: LLM#2 failed: {e}") from e

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
    history: Sequence[ChatMessage] = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
    ]
    resp = llm.generate(config=cfg, history=history)
    msg = (resp.content or "").strip()
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
    )


def _llm_compose_node_message(
    *,
    llm: LLMService,
    model_name: str,
    user_text: str,
    metadata: MetadataState,
    dataset_columns: List[str],
    now_iso: str,
) -> str:
    system = compose_node_message_system_prompt(now_iso=now_iso)

    payload: Dict[str, Any] = {
        "user_message": user_text,
        "metadata": metadata,
        "dataset_columns_preview": dataset_columns[:30],
    }

    cfg = LLMConfig(model=model_name, temperature=0.7)
    history: Sequence[ChatMessage] = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
    ]
    resp = llm.generate(config=cfg, history=history)
    msg = (resp.content or "").strip()
    if not msg:
        raise ValueError("Node message LLM returned empty message")
    return msg


# =============================================================================
# Strict JSON parsing + one LLM repair attempt
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
    system = f"""
You repair JSON.

Return ONLY one VALID JSON object.
No markdown, no commentary.
It must conform exactly to this example schema:
{schema_json}

Rules:
- Must be parseable by json.loads()
- Do not add extra keys
""".strip()

    cfg = LLMConfig(model=model_name, temperature=0.0)
    history: Sequence[ChatMessage] = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=bad_text),
    ]
    return llm.generate(config=cfg, history=history).content


def _llm_json_object_or_raise(
    *,
    llm: LLMService,
    model_name: str,
    system_prompt: str,
    temperature: float,
    user_payload: Dict[str, Any],
    expected_top_keys: set[str],
    schema_json: str,
) -> Dict[str, Any]:
    cfg = LLMConfig(model=model_name, temperature=temperature)
    history: Sequence[ChatMessage] = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=json.dumps(user_payload, ensure_ascii=False)),
    ]
    raw = llm.generate(config=cfg, history=history).content

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
            "locked_fields": [],
            "notes": [],
            "warnings": [],
            "provenance": {},
        }
    }
    return json.dumps(schema_obj, indent=2, ensure_ascii=False)


# =============================================================================
# Reliability layer (deterministic hardening)
# =============================================================================
def _harden_metadata(
    *,
    old: MetadataState,
    new: Dict[str, Any],
    dataset_columns: Optional[List[str]],
    user_text: str,
) -> MetadataState:
    """
    Deterministic hardening ONLY:
    - shape completeness
    - single-line strings
    - enum enforcement
    - locked fields enforcement
    - acceptance validity checks (T/Y required, USER_LIST requires non-empty confounders)
    - soft warnings for unknown columns
    """
    base: MetadataState = empty_metadata()
    base.update(_normalize_legacy_fields(old))  # type: ignore[arg-type]

    incoming = _normalize_legacy_fields(new)
    for k, v in incoming.items():
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
            out = [str(i).strip() for i in x if str(i).strip()]
            return _dedupe(out)
        if isinstance(x, str):
            out = [p.strip() for p in x.split(",") if p.strip()]
            return _dedupe(out)
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

    locked = _clean_locked_fields(base.get("locked_fields"))
    if user_text.strip():
        locked = _apply_lock_intent_from_user_text(locked, user_text)

    treatment = s(base.get("treatment"))
    outcome = s(base.get("outcome"))
    causal_question = s(base.get("causal_question"))
    dataset_summary = s(base.get("dataset_summary"))

    confounder_strategy = s(base.get("confounder_strategy")) or "NONE"
    if confounder_strategy not in _ALLOWED_STRATEGIES:
        confounder_strategy = "NONE"

    confounders = ls(base.get("confounders"))
    controls = ls(base.get("controls"))
    effect_modifiers = ls(base.get("effect_modifiers"))
    notes = ls(base.get("notes"))
    warnings = ls(base.get("warnings"))
    accepted = b(base.get("accepted"))

    prov_in = base.get("provenance")
    provenance: Dict[str, Any] = prov_in if isinstance(prov_in, dict) else {}

    # Enforce locks by restoring OLD values.
    def restore(field: MetadataField, current: Any) -> Any:
        if field in locked:
            return old.get(field) if isinstance(old, dict) else current
        return current

    treatment = cast(str, restore("treatment", treatment))
    outcome = cast(str, restore("outcome", outcome))
    confounder_strategy = cast(str, restore("confounder_strategy", confounder_strategy))
    confounders = cast(List[str], restore("confounders", confounders))
    controls = cast(List[str], restore("controls", controls))
    effect_modifiers = cast(List[str], restore("effect_modifiers", effect_modifiers))
    causal_question = cast(str, restore("causal_question", causal_question))
    dataset_summary = cast(str, restore("dataset_summary", dataset_summary))

    # Strategy consistency
    if confounder_strategy == "NONE":
        confounders = []

    # Acceptance validity (only validity, no intent heuristics)
    if accepted and (not treatment or not outcome):
        accepted = False
        warnings.append("Cannot accept: treatment/outcome is missing.")

    if accepted and confounder_strategy == "USER_LIST" and not confounders:
        accepted = False
        warnings.append("Cannot accept: confounder_strategy=USER_LIST but confounders is empty.")

    # Soft column checks
    if dataset_columns:
        colset = {c.strip() for c in dataset_columns if isinstance(c, str) and c.strip()}

        def warn_unknown(kind: str, col: str) -> None:
            if col and col not in colset:
                warnings.append(f"{kind} references unknown column: {col}")

        if treatment:
            warn_unknown("treatment", treatment)
        if outcome:
            warn_unknown("outcome", outcome)
        for c in confounders:
            warn_unknown("confounders", c)
        for c in controls:
            warn_unknown("controls", c)
        for c in effect_modifiers:
            warn_unknown("effect_modifiers", c)

    warnings = _dedupe([w for w in warnings if w.strip()])
    notes = _dedupe([n for n in notes if n.strip()])

    out: MetadataState = {
        "treatment": treatment,
        "outcome": outcome,
        "confounder_strategy": cast(Any, confounder_strategy),
        "confounders": confounders,
        "controls": controls,
        "effect_modifiers": effect_modifiers,
        "causal_question": causal_question,
        "accepted": accepted,
        "dataset_summary": dataset_summary,
        "locked_fields": locked,
        "notes": notes,
        "warnings": warnings,
        "provenance": provenance,
    }
    return out


def _normalize_legacy_fields(d: Any) -> Dict[str, Any]:
    """
    Back-compat: map older covariate_* keys into confounder_* keys.
    This prevents losing information if older states exist.
    """
    if not isinstance(d, dict):
        return {}
    out = dict(d)

    if "confounder_strategy" not in out and "covariate_strategy" in out:
        out["confounder_strategy"] = out.get("covariate_strategy")
    if "confounders" not in out and "covariates" in out:
        out["confounders"] = out.get("covariates")

    # If old schema stored locks referencing old names, keep them (cleaner will map them).
    return out


def _clean_locked_fields(x: Any) -> List[MetadataField]:
    items: List[str]
    if x is None:
        items = []
    elif isinstance(x, list):
        items = [str(i).strip() for i in x if str(i).strip()]
    elif isinstance(x, str):
        items = [p.strip() for p in x.split(",") if p.strip()]
    else:
        items = []

    cleaned: List[str] = []
    for it in items:
        # map old -> new
        if it == "covariate_strategy":
            it = "confounder_strategy"
        if it == "covariates":
            it = "confounders"
        if it in _ALLOWED_LOCK_FIELDS:
            cleaned.append(it)

    return cast(List[MetadataField], _dedupe(cleaned))


def _apply_lock_intent_from_user_text(locked: List[MetadataField], user_text: str) -> List[MetadataField]:
    """
    Only explicit lock/unlock intent. Also supports covariates/confounders wording.
    """
    t = (user_text or "").lower()
    locked_set = set(locked)

    def mentions(field: str) -> bool:
        # direct
        if field in t or field.replace("_", " ") in t:
            return True
        # synonyms
        if field == "confounders" and ("covariates" in t or "confounders" in t):
            return True
        if field == "confounder_strategy" and ("covariate strategy" in t or "confounder strategy" in t):
            return True
        return False

    if "unlock" in t or "you can change" in t:
        for f in list(locked_set):
            if mentions(f):
                locked_set.discard(f)

    if ("don't change" in t) or ("do not change" in t) or ("leave" in t) or ("keep" in t):
        for f in _ALLOWED_LOCK_FIELDS:
            if mentions(f):
                locked_set.add(cast(MetadataField, f))

    return sorted(locked_set)


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
    if not isinstance(c, dict):
        raise ValueError("ConversationState.control must exist and be a dict")
    return cast(ControlState, c)


def _require_metadata(state: ConversationState) -> MetadataState:
    md_any = state.get("metadata")
    if isinstance(md_any, dict):
        md = cast(MetadataState, _normalize_legacy_fields(md_any))
    else:
        md = empty_metadata()

    hardened = _harden_metadata(old=md, new={}, dataset_columns=None, user_text="")
    state["metadata"] = hardened
    return hardened


def _abort(control: ControlState, msg: str) -> None:
    control["current_stage"] = "PROPOSE_AND_CONFIRM_METADATA"
    control["current_stage_status"] = "ABORTED"
    control["action_required"] = "NEEDS_INPUT"
    control["node_message"] = msg


def _append_ai_message(state: ConversationState, content: str, *, stage: str) -> None:
    msgs = state.get("messages")
    if not isinstance(msgs, list):
        state["messages"] = []
        msgs = state["messages"]

    msgs.append(AIMessage(content=content, additional_kwargs={"source": "node", "stage": stage}))


def _last_human_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(list(messages)):
        if getattr(m, "type", None) == "human":
            return str(getattr(m, "content", "") or "")
        name = m.__class__.__name__.lower()
        if "human" in name or "user" in name:
            return str(getattr(m, "content", "") or "")
    return ""


# =============================================================================
# Dataset columns (best effort)
# =============================================================================
def _extract_dataset_columns(dataset_state: Any) -> Optional[List[str]]:
    if not isinstance(dataset_state, dict):
        return None

    raw_schema = dataset_state.get("raw_schema")
    if isinstance(raw_schema, dict):
        cols = raw_schema.get("columns")
        if isinstance(cols, list):
            names: List[str] = []
            for c in cols:
                if isinstance(c, dict):
                    n = c.get("name")
                    if isinstance(n, str) and n.strip():
                        names.append(n.strip())
            if names:
                return names

    for key in ("columns", "col_names", "column_names", "schema_columns"):
        v = dataset_state.get(key)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            out = [c.strip() for c in v if c.strip()]
            return out or None

    schema = dataset_state.get("schema")
    if isinstance(schema, dict):
        out2 = [str(k).strip() for k in schema.keys() if str(k).strip()]
        return out2 or None

    return None
