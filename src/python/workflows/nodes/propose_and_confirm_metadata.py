from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any,  Dict, List, Optional, Sequence, cast

from langchain_core.messages import AIMessage, BaseMessage

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState
from python.workflows.state.control_state import ControlState
from python.workflows.state.metadata_state import MetadataField, MetadataState, empty_metadata

log = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

_ALLOWED_STRATEGIES = {"USER_LIST", "ALL_EXCEPT_TY", "NONE"}

_ALLOWED_LOCK_FIELDS: set[str] = {
    "dataset_summary",
    "treatment",
    "outcome",
    "covariate_strategy",
    "covariates",
    "controls",
    "effect_modifiers",
    "causal_question",
}


# -----------------------------------------------------------------------------
# Public factory: returns a node callable
# -----------------------------------------------------------------------------


def make_propose_and_confirm_metadata_node(
    *,
    llm: LLMService,
    model_name: str,
) -> CallableNodeFunc:

    def node(state: ConversationState) -> ConversationState:
        return _run_propose_and_confirm_metadata(
            state=state,
            llm=llm,
            model_name=model_name,
        )

    return node


# -----------------------------------------------------------------------------
# Node runner (pure wrt deps)
# -----------------------------------------------------------------------------

def _run_propose_and_confirm_metadata(
    *,
    state: ConversationState,
    llm: LLMService,
    model_name: str,
) -> ConversationState:
    """
    PROPOSE_AND_CONFIRM_METADATA node (2x LLM):
      1) LLM edits metadata (strict JSON: {"metadata": ...})
      2) LLM writes user-facing node message (strict JSON: {"node_message": "..."})

    Invariants:
      - NEVER append LLM prompt messages to state["messages"].
      - ALWAYS append ONLY the final user-facing AIMessage (node_message).
      - NO deterministic fallbacks: if LLM output is missing/invalid -> raise ValueError.
      - ControlState is updated:
          accepted -> DONE/DONE/NONE
          else     -> PROPOSE_AND_CONFIRM_METADATA/PENDING/NEEDS_INPUT
          errors   -> PROPOSE_AND_CONFIRM_METADATA/ABORTED/NEEDS_INPUT + raise
    """
    control = _require_control(state)
    md_old = _require_metadata(state)

    user_text = _last_human_text(cast(Sequence[BaseMessage], state.get("messages", [])))
    if not user_text.strip():
        _abort(control, "No user message found for metadata confirmation.")
        raise ValueError("propose_and_confirm_metadata: no user message found")

    dataset_cols = _extract_dataset_columns(state.get("dataset"))
    now_iso = datetime.now(timezone.utc).isoformat()

    # ---- LLM #1: metadata edit ----
    try:
        raw1 = _llm_edit_metadata(
            llm=llm,
            model_name=model_name,
            current_metadata=md_old,
            user_text=user_text,
            dataset_columns=dataset_cols or [],
            now_iso=now_iso,
        )
        obj1 = _parse_json_object(raw1)
        md_candidate = obj1.get("metadata")
        if not isinstance(md_candidate, dict):
            raise ValueError("LLM#1 output must contain object field 'metadata'")

        md_new = _harden_metadata(
            old=md_old,
            new=cast(Dict[str, Any], md_candidate),
            dataset_columns=dataset_cols,
            user_text=user_text,
        )

        prov = md_new.get("provenance")
        provenance: Dict[str, Any] = prov
        provenance["metadata_llm1_updated_utc"] = now_iso
        provenance["metadata_llm1_raw"] = raw1[:8000]
        md_new["provenance"] = provenance

        state["metadata"] = md_new
    except Exception as e:
        _abort(control, f"Metadata update failed: {e}")
        log.exception("PROPOSE_AND_CONFIRM_METADATA: LLM#1 failed")
        raise ValueError(f"PROPOSE_AND_CONFIRM_METADATA: LLM#1 failed: {e}") from e

    md = _require_metadata(state)

    # ---- LLM #2: node message ----
    try:
        raw2 = _llm_compose_node_message(
            llm=llm,
            model_name=model_name,
            user_text=user_text,
            metadata=md,
            dataset_columns=dataset_cols or [],
            now_iso=now_iso,
        )
        obj2 = _parse_json_object(raw2)
        node_msg = obj2.get("node_message")
        if not isinstance(node_msg, str) or not node_msg.strip():
            raise ValueError("LLM#2 output must contain non-empty string field 'node_message'")

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

        prov2 = md.get("provenance")
        provenance2: Dict[str, Any] = prov2 
        provenance2["metadata_llm2_raw"] = raw2[:8000]
        md["provenance"] = provenance2
        state["metadata"] = md

        return state

    except Exception as e:
        _abort(control, f"Message generation failed: {e}")
        log.exception("PROPOSE_AND_CONFIRM_METADATA: LLM#2 failed")
        raise ValueError(f"PROPOSE_AND_CONFIRM_METADATA: LLM#2 failed: {e}") from e


# =============================================================================
# LLM calls (PROMPTS NEVER appended to state["messages"])
# =============================================================================

def _llm_edit_metadata(
    *,
    llm: LLMService,
    model_name: str,
    current_metadata: MetadataState,
    user_text: str,
    dataset_columns: List[str],
    now_iso: str,
) -> str:
    schema = _metadata_schema_json()

    system = f"""
You are a deterministic metadata editor for a causal inference copilot.

OUTPUT RULES (non-negotiable):
- Output ONLY a single JSON object (no markdown, no commentary).
- It MUST match EXACTLY this schema:
{schema}

EDITING RULES:
1) Start from current_metadata and apply the user's instruction.
2) Do NOT remove information unless the user explicitly requests it.
3) locked_fields enforcement:
   - If a field is in locked_fields, keep it unchanged unless the user explicitly changes/unlocks it.
   - If user says “leave/keep/don't change X”, add X to locked_fields.
   - If user says “unlock X / you can change X”, remove X from locked_fields.
4) accepted:
   - If user expresses acceptance (“accept”, “looks good”, “proceed”), set accepted=true,
     but only if treatment and outcome are non-empty.
   - If user requests changes, accepted must be false.
5) covariate_strategy must be one of USER_LIST / ALL_EXCEPT_TY / NONE.
6) Prefer dataset_columns; if user mentions unknown columns, keep them but add warnings.
7) notes: short, decision-y. warnings: only validation/ambiguity.

now_utc: {now_iso}
""".strip()

    user_payload = { # pyright: ignore[reportUnknownVariableType]
        "user_message": user_text,
        "current_metadata": current_metadata,
        "dataset_columns": dataset_columns,
    }
    
    config = LLMConfig( 
        model=model_name,   
        temperature=0.0,
    )

    history = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=json.dumps(user_payload, ensure_ascii=False)),
    ]
    return llm.generate(config=config, history=history).content


def _llm_compose_node_message(
    *,
    llm: LLMService,
    model_name: str,
    user_text: str,
    metadata: MetadataState,
    dataset_columns: List[str],
    now_iso: str,
) -> str:
    system = f"""
You write a user-facing message for the PROPOSE_AND_CONFIRM_METADATA step.

OUTPUT RULES (non-negotiable):
- Output ONLY one JSON object with EXACTLY:
  {{ "node_message": string }}
- No markdown. No extra keys.

CONTENT RULES:
- If metadata.accepted == true:
  - Confirm treatment, outcome, covariate_strategy.
  - Mention controls/covariates briefly (if present).
  - Say we will proceed to the next stage.
- If metadata.accepted == false:
  - If treatment/outcome missing: ask for them explicitly.
  - Else: ask user to either accept or specify what to change.
  - If user asked for suggestions, propose plausible treatment/outcome based on dataset_columns.
- If warnings exist: mention the most important 1–2 briefly.

now_utc: {now_iso}
""".strip()

    user_payload = { # pyright: ignore[reportUnknownVariableType]
        "user_message": user_text,
        "metadata": metadata,
        "dataset_columns": dataset_columns,
    }
    
    config = LLMConfig(
        model=model_name,
        temperature=1.0,
    )

    history = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=json.dumps(user_payload, ensure_ascii=False)),
    ]
    return llm.generate(config=config, history=history).content


def _metadata_schema_json() -> str:
    schema_obj = { # pyright: ignore[reportUnknownVariableType]
        "metadata": {
            "treatment": "string",
            "outcome": "string",
            "covariate_strategy": "USER_LIST | ALL_EXCEPT_TY | NONE",
            "controls": ["string"],
            "covariates": ["string"],
            "effect_modifiers": ["string"],
            "causal_question": "string",
            "accepted": "boolean",
            "dataset_summary": "string",
            "locked_fields": [
                "dataset_summary | treatment | outcome | covariate_strategy | covariates | controls | effect_modifiers | causal_question"
            ],
            "notes": ["string"],
            "warnings": ["string"],
            "provenance": "object",
        }
    }
    return json.dumps(schema_obj, indent=2)


# =============================================================================
# Reliability layer
# =============================================================================

def _harden_metadata(
    *,
    old: MetadataState,
    new: Dict[str, Any],
    dataset_columns: Optional[List[str]],
    user_text: str,
) -> MetadataState:
    base = empty_metadata()
    base.update(old)
    for k, v in new.items():
        if k in base:
            base[k] = v

    def s(x: Any) -> str:
        return x.strip() if isinstance(x, str) else ""

    def ls(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            out = [str(i).strip() for i in x if str(i).strip()] # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
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
    locked = _apply_lock_intent_from_user_text(locked, user_text)

    treatment = s(base.get("treatment"))
    outcome = s(base.get("outcome"))

    covariate_strategy = s(base.get("covariate_strategy")) or "NONE"
    if covariate_strategy not in _ALLOWED_STRATEGIES:
        covariate_strategy = "NONE"

    controls = ls(base.get("controls"))
    covariates = ls(base.get("covariates"))
    effect_modifiers = ls(base.get("effect_modifiers"))
    causal_question = s(base.get("causal_question"))
    dataset_summary = s(base.get("dataset_summary"))
    notes = ls(base.get("notes"))
    warnings = ls(base.get("warnings"))
    accepted = b(base.get("accepted"))

    prov_in = base.get("provenance")
    provenance: Dict[str, Any] = prov_in 

    # enforce locks by restoring OLD values
    def restore(field: MetadataField, current: Any) -> Any:
        if field in locked:
            return old.get(field)
        return current

    treatment = cast(str, restore("treatment", treatment))
    outcome = cast(str, restore("outcome", outcome))
    covariate_strategy = cast(str, restore("covariate_strategy", covariate_strategy))
    controls = cast(List[str], restore("controls", controls))
    covariates = cast(List[str], restore("covariates", covariates))
    effect_modifiers = cast(List[str], restore("effect_modifiers", effect_modifiers))
    causal_question = cast(str, restore("causal_question", causal_question))
    dataset_summary = cast(str, restore("dataset_summary", dataset_summary))

    # acceptance constraints (hard)
    if accepted and (not treatment or not outcome):
        accepted = False
        warnings.append("Cannot accept: treatment/outcome is missing.")

    if accepted and covariate_strategy == "USER_LIST" and not covariates:
        accepted = False
        warnings.append("Cannot accept: covariate_strategy=USER_LIST but covariates is empty.")

    # soft column checks (warn only)
    if dataset_columns:
        colset = {c.strip() for c in dataset_columns if c.strip()}

        def warn_unknown(kind: str, col: str) -> None:
            if col and col not in colset:
                warnings.append(f"{kind} references unknown column: {col}")

        if treatment:
            warn_unknown("treatment", treatment)
        if outcome:
            warn_unknown("outcome", outcome)
        for c in controls:
            warn_unknown("controls", c)
        for c in covariates:
            warn_unknown("covariates", c)
        for c in effect_modifiers:
            warn_unknown("effect_modifiers", c)

    warnings = _dedupe([w for w in warnings if w.strip()])

    out: MetadataState = {
        "treatment": treatment,
        "outcome": outcome,
        "covariate_strategy": cast(Any, covariate_strategy),
        "controls": controls,
        "covariates": covariates,
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


def _clean_locked_fields(x: Any) -> List[MetadataField]:
    items: List[str]
    if x is None:
        items = []
    elif isinstance(x, list):
        items = [str(i).strip() for i in x if str(i).strip()] # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    elif isinstance(x, str):
        items = [p.strip() for p in x.split(",") if p.strip()]
    else:
        items = []

    cleaned = [i for i in items if i in _ALLOWED_LOCK_FIELDS]
    return cast(List[MetadataField], _dedupe(cleaned))


def _apply_lock_intent_from_user_text(locked: List[MetadataField], user_text: str) -> List[MetadataField]:
    t = user_text.lower()
    locked_set = set(locked)

    def mentions(field: str) -> bool:
        return field in t or field.replace("_", " ") in t

    # unlock intent
    if "unlock" in t or "you can change" in t:
        for f in list(locked_set):
            if mentions(f):
                locked_set.discard(f)

    # lock intent
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
    return c


def _require_metadata(state: ConversationState) -> MetadataState:
    md = state.get("metadata")
    # ensure shape-complete even if partially set (no user intent)
    hardened = _harden_metadata(old= md, new={}, dataset_columns=None, user_text="")
    state["metadata"] = hardened
    return hardened


def _abort(control: ControlState, msg: str) -> None:
    control["current_stage"] = "PROPOSE_AND_CONFIRM_METADATA"
    control["current_stage_status"] = "ABORTED"
    control["action_required"] = "NEEDS_INPUT"
    control["node_message"] = msg


def _append_ai_message(state: ConversationState, content: str, *, stage: str) -> None:
    msgs = state.get("messages")
    if not isinstance(msgs, list): # pyright: ignore[reportUnnecessaryIsInstance]
        state["messages"] = []
        msgs = state["messages"]

    msgs.append(
        AIMessage(
            content=content,
            additional_kwargs={"source": "node", "stage": stage},
        )
    )


def _last_human_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(list(messages)):
        if getattr(m, "type", None) == "human":
            return str(getattr(m, "content", "") or "")
        name = m.__class__.__name__.lower()
        if "human" in name or "user" in name:
            return str(getattr(m, "content", "") or "")
    return ""


# =============================================================================
# JSON parsing (strict)
# =============================================================================

def _parse_json_object(raw: str) -> Dict[str, Any]:
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


# =============================================================================
# Dataset columns (best effort; supports your raw_schema format)
# =============================================================================

def _extract_dataset_columns(dataset_state: Any) -> Optional[List[str]]:
    if not isinstance(dataset_state, dict):
        return None

    # raw_schema: {"columns":[{"name": "...", "dtype":"..."}]}
    raw_schema = dataset_state.get("raw_schema") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if isinstance(raw_schema, dict):
        cols = raw_schema.get("columns") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if isinstance(cols, list):
            names: List[str] = []
            for c in cols: # pyright: ignore[reportUnknownVariableType]
                if isinstance(c, dict):
                    n = c.get("name") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                    if isinstance(n, str) and n.strip():
                        names.append(n.strip())
            if names:
                return names

    for key in ("columns", "col_names", "column_names", "schema_columns"):
        v = dataset_state.get(key) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if isinstance(v, list) and all(isinstance(x, str) for x in v): # pyright: ignore[reportUnknownVariableType]
            out = [c.strip() for c in v if c.strip()] # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            return out or None # pyright: ignore[reportUnknownVariableType]

    schema = dataset_state.get("schema") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if isinstance(schema, dict):
        out2 = [str(k).strip() for k in schema.keys() if str(k).strip()] # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        return out2 or None

    return None
