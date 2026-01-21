# src/python/workflows/nodes/compile_protocol_state.py
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage
from pandas import DataFrame

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMConfig, LLMService
from python.workflows.nodes.prompts.compile_state_protocol import (
    load_compile_protocol_repair_system_prompt,
    load_compile_protocol_system_prompt,
    load_protocol_user_message_repair_system_prompt,
    load_protocol_user_message_system_prompt,
)
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState, to_chat_history_last_k
from python.workflows.state.dataset_state import DatasetState
from python.workflows.state.metadata_state import MetadataState
from python.workflows.state.protocol_state import ProtocolState

log = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)

_WRAPPER_KEYS = {"protocol", "ready_for_accept", "user_accepted"}

# For protocol compilation we only need a preview. Validation node should load full data.
_DEFAULT_PREVIEW_LIMIT = 200


def make_compile_protocol_state_node(
    data_repo: DataRepo,
    llm: LLMService,
    *,
    model_name: str,
    message_model_name: Optional[str] = None,
    preview_limit: int = _DEFAULT_PREVIEW_LIMIT,
) -> CallableNodeFunc:
    """
    Compiles a ProtocolState using an LLM under a strict JSON contract.

    Key design choices:
    - Treatment/outcome are HARD-LOCKED from metadata.
    - Acceptance is decided ONLY by the LLM contract fields:
        accepted := ready_for_accept AND user_accepted
      (no explicit accept parsing in code).
    - Protocol container is NEVER reset destructively:
      we merge any partial protocol into the template so accepted never "regresses"
      due to missing keys.
    - If the compiler returns a protocol with missing required execution fields,
      we run the same repair prompt with a synthetic "parse_error" describing
      what is missing (still delegating the fixing to the LLM).
    """
    msg_model = message_model_name or model_name

    def _node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        _ensure_control_container(state)

        dataset, meta, err = _require_dataset_and_meta(state)
        if err:
            return _fatal(state, err)

        _ensure_protocol_container(state, meta)

        # Dataset must be loadable (upstream loader should set load_error)
        load_error = dataset.get("load_error")
        if isinstance(load_error, str) and load_error.strip():
            return _fatal(state, f"Dataset could not be loaded ({load_error}). Please reload and retry.")

        dataset_id = dataset.get("id")
        if dataset_id is None:
            return _fatal(state, "Dataset ID is missing. Please reload the dataset.")

        # Preview rows (not full dataset). Validation node will load full data.
        try:
            df = data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=dataset_id,
                limit=preview_limit,
            )
        except Exception as e:
            return _fatal(state, f"Failed to load dataset preview: {type(e).__name__}: {e}")

        # Columns: prefer dataset schema; fallback to df.columns for robustness.
        columns = _dataset_columns(dataset)
        if not columns:
            columns = _dataset_columns_from_df(df)

        if not columns:
            return _fatal(state, "Dataset schema is missing (no columns detected). Please reload the dataset.")

        var_dict = _build_variable_dictionary(dataset)
        preview_rows: List[Dict[str, Any]] = _dataset_preview_rows(df=df)

        treatment = _as_nonempty_str(meta.get("treatment"))
        outcome = _as_nonempty_str(meta.get("outcome"))
        if not treatment or not outcome:
            return _fatal(state, "Missing treatment/outcome in metadata. Please confirm metadata first.")

        # Hard-lock treatment/outcome in protocol.
        protocol = cast(ProtocolState, state["protocol"])
        protocol["treatment"] = treatment
        protocol["outcome"] = outcome
        protocol = _normalize_protocol(protocol)
        state["protocol"] = protocol

        # Idempotent lock: if already accepted, never regress or call compiler again.
        if protocol.get("accepted") is True:
            msg = _llm_messenger_with_repair(
                llm=llm,
                model_name=msg_model,
                state=state,
                mode="LOCKED",
                dataset_columns=columns,
                variable_dictionary=var_dict,
                preview_rows=preview_rows,
                meta=meta,
                protocol=protocol,
                ready_for_accept=True,
                user_accepted=True,
                note=None,
            )
            _append_ai(state, msg)
            return _succeed(state, msg)

        last_user = _last_human_text(cast(Sequence[BaseMessage], state.get("messages", [])))

        # Provide observed values to help with treated/comparator mapping without extra questions.
        observed = _observed_values_from_preview(
            preview_rows=preview_rows,
            cols=[treatment, outcome],
            max_per_col=12,
        )

        wrapper, parse_error = _llm_compile_protocol_with_repair(
            llm=llm,
            model_name=model_name,
            state=state,
            dataset_columns=columns,
            variable_dictionary=var_dict,
            preview_rows=preview_rows,
            meta=meta,
            current_protocol=protocol,
            last_user_message=last_user,
            observed_values=observed,
        )

        if wrapper is None:
            # Not fatal: keep pending, ask user to retry.
            protocol = cast(ProtocolState, state["protocol"])
            protocol["accepted"] = False
            protocol["open_questions"] = [
                "I couldn’t parse the protocol compiler output. Please resend your last message."
            ]
            state["protocol"] = _normalize_protocol(protocol)

            msg = _llm_messenger_with_repair(
                llm=llm,
                model_name=msg_model,
                state=state,
                mode="NEEDS_INPUT",
                dataset_columns=columns,
                variable_dictionary=var_dict,
                preview_rows=preview_rows,
                meta=meta,
                protocol=cast(ProtocolState, state["protocol"]),
                ready_for_accept=False,
                user_accepted=False,
                note={"parse_error": parse_error or "unknown"},
            )
            _append_ai(state, msg)
            return _need_input(state, msg)

        nxt = _coerce_protocol(wrapper.get("protocol"))
        if nxt is None:
            # Not fatal: keep pending, ask user to retry.
            protocol = cast(ProtocolState, state["protocol"])
            protocol["accepted"] = False
            protocol["open_questions"] = [
                "The compiler returned a protocol with an invalid shape. Please resend your last message."
            ]
            state["protocol"] = _normalize_protocol(protocol)

            msg = _llm_messenger_with_repair(
                llm=llm,
                model_name=msg_model,
                state=state,
                mode="NEEDS_INPUT",
                dataset_columns=columns,
                variable_dictionary=var_dict,
                preview_rows=preview_rows,
                meta=meta,
                protocol=cast(ProtocolState, state["protocol"]),
                ready_for_accept=False,
                user_accepted=False,
                note={"parse_error": "invalid protocol shape"},
            )
            _append_ai(state, msg)
            return _need_input(state, msg)

        # Normalize + hard-lock treatment/outcome post-hoc.
        nxt = _normalize_protocol(nxt)
        nxt["treatment"] = treatment
        nxt["outcome"] = outcome

        # If compiler produced an execution-incomplete protocol, delegate fixing back to LLM repair.
        missing = _missing_required_execution_fields(nxt)
        if missing:
            repaired_wrapper, repair_err = _llm_repair_missing_fields(
                llm=llm,
                model_name=model_name,
                meta=meta,
                bad_protocol_wrapper=wrapper,
                missing_fields=missing,
            )
            if repaired_wrapper is not None:
                repaired_protocol = _coerce_protocol(repaired_wrapper.get("protocol"))
                if repaired_protocol is not None:
                    nxt = _normalize_protocol(repaired_protocol)
                    nxt["treatment"] = treatment
                    nxt["outcome"] = outcome
                    wrapper = repaired_wrapper
                    missing = _missing_required_execution_fields(nxt)

        # If still missing, force NOT-ready and ask minimally (static fallback).
        if missing:
            nxt["accepted"] = False
            nxt["open_questions"] = [
                f"Protocol is missing required fields ({', '.join(missing)}). "
                "Please clarify these protocol details so I can lock the protocol."
            ]
            nxt = _normalize_protocol(nxt)
            state["protocol"] = nxt

            msg = _llm_messenger_with_repair(
                llm=llm,
                model_name=msg_model,
                state=state,
                mode="NEEDS_INPUT",
                dataset_columns=columns,
                variable_dictionary=var_dict,
                preview_rows=preview_rows,
                meta=meta,
                protocol=nxt,
                ready_for_accept=False,
                user_accepted=False,
                note={"parse_error": "missing required execution fields"},
            )
            _append_ai(state, msg)
            return _need_input(state, msg)

        # Delegate acceptance to LLM ONLY (no explicit accept parsing here).
        ready = bool(wrapper.get("ready_for_accept") is True) and len(nxt.get("open_questions", [])) == 0
        accepted = bool(ready and (wrapper.get("user_accepted") is True))

        nxt["accepted"] = accepted
        if accepted:
            nxt["open_questions"] = []

        nxt = _normalize_protocol(nxt)
        state["protocol"] = nxt

        mode = "LOCKED" if accepted else ("READY" if ready else "NEEDS_INPUT")
        msg = _llm_messenger_with_repair(
            llm=llm,
            model_name=msg_model,
            state=state,
            mode=mode,
            dataset_columns=columns,
            variable_dictionary=var_dict,
            preview_rows=preview_rows,
            meta=meta,
            protocol=nxt,
            ready_for_accept=ready,
            user_accepted=accepted,
            note=None,
        )
        _append_ai(state, msg)

        return _succeed(state, msg) if accepted else _need_input(state, msg)

    return _node


# ----------------------------
# LLM #1: compile + repair
# ----------------------------

def _llm_compile_protocol_with_repair(
    *,
    llm: LLMService,
    model_name: str,
    state: ConversationState,
    dataset_columns: List[str],
    variable_dictionary: List[Dict[str, Any]],
    preview_rows: List[Dict[str, Any]],
    meta: MetadataState,
    current_protocol: ProtocolState,
    last_user_message: str,
    observed_values: Dict[str, List[str]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    payload: Dict[str, Any] = {
        "conversation_snapshot": {
            "control": state.get("control", {}),
            "last_user_message": last_user_message or "",
        },
        "dataset": {
            "columns": dataset_columns,
            "variable_dictionary": variable_dictionary,
            "preview_rows": preview_rows,
            "observed_values_from_preview": observed_values,
        },
        "metadata_locked": {
            "treatment": meta.get("treatment", ""),
            "outcome": meta.get("outcome", ""),
            "causal_question": meta.get("causal_question", ""),
            "confounder_strategy": meta.get("confounder_strategy", ""),
            "confounders": meta.get("confounders", []),
            "controls": meta.get("controls", []),
            "effect_modifiers": meta.get("effect_modifiers", []),
            "dataset_summary": meta.get("dataset_summary", ""),
        },
        "current_protocol": current_protocol,
        "protocol_template": _empty_protocol(),
        "contract": {"top_level_keys": sorted(_WRAPPER_KEYS)},
    }

    history = to_chat_history_last_k(state, k=12, drop_last_user=False)
    raw = llm.generate(
        system_prompt=load_compile_protocol_system_prompt(),
        user_prompt=json.dumps(payload, ensure_ascii=False),
        config=LLMConfig(model=model_name, temperature=0.0),
        history=history,
    ).content

    obj, parse_error = _parse_strict_wrapper(raw)
    if obj is not None:
        return obj, None

    # Repair ONLY from parse_error (delegated to LLM).
    repair_payload: Dict[str, Any] = {
        "bad_output": raw,
        "parse_error": parse_error or "Unknown parse error",
        "required_top_level_keys": sorted(_WRAPPER_KEYS),
        "protocol_template": _empty_protocol(),
        "metadata_locked": {
            "treatment": meta.get("treatment", ""),
            "outcome": meta.get("outcome", ""),
        },
    }

    repaired = llm.generate(
        system_prompt=load_compile_protocol_repair_system_prompt(),
        user_prompt=json.dumps(repair_payload, ensure_ascii=False),
        config=LLMConfig(model=model_name, temperature=0.0),
        history=None,
    ).content

    obj2, parse_error2 = _parse_strict_wrapper(repaired)
    if obj2 is not None:
        return obj2, None

    return None, (parse_error2 or parse_error or "Repair parse failed")


def _llm_repair_missing_fields(
    *,
    llm: LLMService,
    model_name: str,
    meta: MetadataState,
    bad_protocol_wrapper: Dict[str, Any],
    missing_fields: List[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Uses the SAME repair prompt, but with a synthetic parse_error describing missing required fields.
    This keeps the "fixing" responsibility inside the LLM (as requested).
    """
    repair_payload: Dict[str, Any] = {
        "bad_output": json.dumps(bad_protocol_wrapper, ensure_ascii=False),
        "parse_error": f"Protocol missing required execution fields: {', '.join(missing_fields)}",
        "required_top_level_keys": sorted(_WRAPPER_KEYS),
        "protocol_template": _empty_protocol(),
        "metadata_locked": {
            "treatment": meta.get("treatment", ""),
            "outcome": meta.get("outcome", ""),
        },
    }

    repaired = llm.generate(
        system_prompt=load_compile_protocol_repair_system_prompt(),
        user_prompt=json.dumps(repair_payload, ensure_ascii=False),
        config=LLMConfig(model=model_name, temperature=0.0),
        history=None,
    ).content

    obj, err = _parse_strict_wrapper(repaired)
    return obj, err


def _parse_strict_wrapper(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    s = (text or "").strip()
    if not s:
        return None, "Empty output"

    m = _JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return None, f"Top-level JSON is not an object (type={type(obj).__name__})"
        keys = set(obj.keys())
        if keys != _WRAPPER_KEYS:
            return None, f"Top-level keys mismatch: got={sorted(keys)}, expected={sorted(_WRAPPER_KEYS)}"
        return cast(Dict[str, Any], obj), None
    except Exception as e:
        direct_err = f"Direct JSON parse failed: {type(e).__name__}: {e}"

    m2 = _JSON_OBJ_RE.search(s)
    if not m2:
        return None, direct_err

    try:
        obj2 = json.loads(m2.group(0))
        if not isinstance(obj2, dict):
            return None, f"Extracted JSON is not an object (type={type(obj2).__name__})"
        keys2 = set(obj2.keys())
        if keys2 != _WRAPPER_KEYS:
            return None, f"Extracted keys mismatch: got={sorted(keys2)}, expected={sorted(_WRAPPER_KEYS)}"
        return cast(Dict[str, Any], obj2), None
    except Exception as e2:
        return None, f"{direct_err}; Extracted parse failed: {type(e2).__name__}: {e2}"


# ----------------------------
# LLM #2: messenger + repair (user-facing)
# ----------------------------

def _llm_messenger_with_repair(
    *,
    llm: LLMService,
    model_name: str,
    state: ConversationState,
    mode: str,
    dataset_columns: List[str],
    variable_dictionary: List[Dict[str, Any]],
    preview_rows: List[Dict[str, Any]],
    meta: MetadataState,
    protocol: ProtocolState,
    ready_for_accept: bool,
    user_accepted: bool,
    note: Optional[Dict[str, Any]],
) -> str:
    payload: Dict[str, Any] = {
        "mode": mode,
        "flags": {"ready_for_accept": ready_for_accept, "user_accepted": user_accepted},
        "note": note or {},
        "dataset": {
            "columns_count": len(dataset_columns),
            "columns": dataset_columns,
            "variable_dictionary": variable_dictionary,
            "preview_rows": preview_rows,
        },
        "metadata_locked": {
            "treatment": meta.get("treatment", ""),
            "outcome": meta.get("outcome", ""),
            "causal_question": meta.get("causal_question", ""),
            "confounders": meta.get("confounders", []),
            "controls": meta.get("controls", []),
            "effect_modifiers": meta.get("effect_modifiers", []),
            "dataset_summary": meta.get("dataset_summary", ""),
        },
        "protocol": protocol,
    }

    history = to_chat_history_last_k(state, k=8, drop_last_user=False)
    out = llm.generate(
        system_prompt=load_protocol_user_message_system_prompt(),
        user_prompt=json.dumps(payload, ensure_ascii=False),
        config=LLMConfig(model=model_name, temperature=0.0),
        history=history,
    ).content.strip()

    if out:
        return out

    # Repair (still LLM-only).
    repair_payload = {"bad_output": out, "payload": payload}
    repaired = llm.generate(
        system_prompt=load_protocol_user_message_repair_system_prompt(),
        user_prompt=json.dumps(repair_payload, ensure_ascii=False),
        config=LLMConfig(model=model_name, temperature=0.0),
        history=None,
    ).content.strip()

    if not repaired:
        raise ValueError("Messenger produced empty output after repair")
    return repaired


# ----------------------------
# Protocol helpers (template-safe merge)
# ----------------------------

def _empty_protocol() -> ProtocolState:
    # Template MUST stay shape-complete (single source of truth).
    return cast(
        ProtocolState,
        {
            "accepted": False,
            "clarified": [],
            "open_questions": [],
            "population": "",
            "time_zero_type": "CONCEPTUAL",
            "time_zero": "",
            "time_zero_definition": "",
            "treatment": "",
            "treatment_window_start": "",
            "treatment_window_end": "",
            "treatment_window_unit": "days",
            "comparator": "",
            "outcome": "",
            "outcome_is_duration": False,
            "outcome_window": "",
            "outcome_window_unit": "days",
            "covariates": [],
            "effect_modifiers": [],
            "censoring_rules": [],
        },
    )


def _merge_protocol_into_template(existing: Any) -> ProtocolState:
    """
    Never "reset" protocol due to missing keys.
    Instead, start from template and overlay any compatible values.
    This prevents accepted=True from regressing to False because one key was missing.
    """
    tpl = _empty_protocol()
    if not isinstance(existing, dict):
        return tpl

    out: Dict[str, Any] = dict(tpl)

    # bool fields
    for k in ["accepted", "outcome_is_duration"]:
        v = existing.get(k)
        if isinstance(v, bool):
            out[k] = v

    # str fields
    for k in [
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
        "outcome_window",
        "outcome_window_unit",
    ]:
        v = existing.get(k)
        if isinstance(v, str):
            out[k] = v

    # list[str] fields
    for k in ["clarified", "open_questions", "covariates", "effect_modifiers", "censoring_rules"]:
        v = existing.get(k)
        if isinstance(v, list) and all(isinstance(z, str) for z in v):
            out[k] = v

    return cast(ProtocolState, out)


def _coerce_protocol(x: Any) -> Optional[ProtocolState]:
    """
    Strict check for a FULL ProtocolState (post-LLM).
    """
    if not isinstance(x, dict):
        return None

    required = set(_empty_protocol().keys())
    if not required.issubset(x.keys()):
        return None

    if not isinstance(x.get("accepted"), bool):
        return None
    if not isinstance(x.get("outcome_is_duration"), bool):
        return None

    for k in [
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
        "outcome_window",
        "outcome_window_unit",
    ]:
        if not isinstance(x.get(k), str):
            return None

    for k in ["clarified", "open_questions", "covariates", "effect_modifiers", "censoring_rules"]:
        v = x.get(k)
        if not isinstance(v, list) or not all(isinstance(z, str) for z in v):
            return None

    return cast(ProtocolState, x)


def _missing_required_execution_fields(p: ProtocolState) -> List[str]:
    """
    These are the fields your downstream static validator expects to be non-empty.
    Keep this aligned with your validator rules.
    """
    missing: List[str] = []

    def is_empty_str(v: Any) -> bool:
        return not isinstance(v, str) or not v.strip()

    if is_empty_str(p.get("population")):
        missing.append("population")
    if is_empty_str(p.get("treatment_window_start")):
        missing.append("treatment_window_start")
    if is_empty_str(p.get("treatment_window_end")):
        missing.append("treatment_window_end")
    if is_empty_str(p.get("outcome_window")):
        missing.append("outcome_window")
    # comparator is also required for many flows
    if is_empty_str(p.get("comparator")):
        missing.append("comparator")

    # time zero definition is required when conceptual
    if p.get("time_zero_type") == "CONCEPTUAL" and is_empty_str(p.get("time_zero_definition")):
        missing.append("time_zero_definition")

    return missing


def _normalize_protocol(p: ProtocolState) -> ProtocolState:
    def s(v: Any) -> str:
        return v.strip() if isinstance(v, str) else ""

    def uniq(xs: Any) -> List[str]:
        if not isinstance(xs, list):
            return []
        seen: set[str] = set()
        out: List[str] = []
        for it in xs:
            if not isinstance(it, str):
                continue
            t = it.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    p["population"] = s(p.get("population"))
    p["time_zero_type"] = cast(Any, s(p.get("time_zero_type")))
    p["time_zero"] = s(p.get("time_zero"))
    p["time_zero_definition"] = s(p.get("time_zero_definition"))

    p["treatment"] = s(p.get("treatment"))
    p["treatment_window_start"] = s(p.get("treatment_window_start"))
    p["treatment_window_end"] = s(p.get("treatment_window_end"))
    p["treatment_window_unit"] = cast(Any, s(p.get("treatment_window_unit")))

    p["comparator"] = s(p.get("comparator"))

    p["outcome"] = s(p.get("outcome"))
    p["outcome_window"] = s(p.get("outcome_window"))
    p["outcome_window_unit"] = cast(Any, s(p.get("outcome_window_unit")))

    p["clarified"] = uniq(p.get("clarified"))
    p["open_questions"] = uniq(p.get("open_questions"))
    p["covariates"] = uniq(p.get("covariates"))
    p["effect_modifiers"] = uniq(p.get("effect_modifiers"))
    p["censoring_rules"] = uniq(p.get("censoring_rules"))

    # hard invariant: never include T/Y in covariates/modifiers
    tcol = p.get("treatment", "")
    ycol = p.get("outcome", "")
    p["covariates"] = [c for c in p["covariates"] if c not in (tcol, ycol)]
    p["effect_modifiers"] = [m for m in p["effect_modifiers"] if m not in (tcol, ycol)]

    # If conceptual time zero, we should not set a column name.
    if p.get("time_zero_type") == "CONCEPTUAL":
        p["time_zero"] = ""
    # If column time zero, definition should be empty (definition is for conceptual).
    if p.get("time_zero_type") == "COLUMN":
        p["time_zero_definition"] = ""

    return p


def _ensure_protocol_container(state: ConversationState, meta: MetadataState) -> None:
    """
    Ensure protocol is ALWAYS shape-complete and never regresses due to missing keys.
    """
    cur = _merge_protocol_into_template(state.get("protocol"))
    cur = _normalize_protocol(cur)

    # Lock T/Y if present in metadata
    t = meta.get("treatment")
    y = meta.get("outcome")
    if isinstance(t, str) and t.strip():
        cur["treatment"] = t.strip()
    if isinstance(y, str) and y.strip():
        cur["outcome"] = y.strip()

    state["protocol"] = _normalize_protocol(cur)


# ----------------------------
# Observed values helper
# ----------------------------

def _observed_values_from_preview(
    *,
    preview_rows: List[Dict[str, Any]],
    cols: List[str],
    max_per_col: int,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not preview_rows:
        return out
    for col in cols:
        if not col:
            continue
        seen: set[str] = set()
        vals: List[str] = []
        for r in preview_rows:
            if not isinstance(r, dict) or col not in r:
                continue
            v = r.get(col)
            if v is None:
                continue
            sv = str(v).strip()
            if not sv or sv in seen:
                continue
            seen.add(sv)
            vals.append(sv)
            if len(vals) >= max_per_col:
                break
        if vals:
            out[col] = vals
    return out


# ----------------------------
# Dataset helpers
# ----------------------------

def _dataset_columns_from_df(df: DataFrame) -> List[str]:
    try:
        return [str(c).strip() for c in df.columns if str(c).strip()]
    except Exception:
        return []


def _dataset_columns(dataset: DatasetState) -> List[str]:
    cols: List[str] = []

    raw = dataset.get("raw_schema")
    if isinstance(raw, dict):
        c = raw.get("columns")
        if isinstance(c, list):
            for it in c:
                if isinstance(it, dict) and isinstance(it.get("name"), str):
                    name = it["name"].strip()
                    if name:
                        cols.append(name)

    summary = dataset.get("summary")
    if isinstance(summary, dict):
        c2 = summary.get("columns")
        if isinstance(c2, dict):
            for name in c2.keys():
                if isinstance(name, str):
                    n = name.strip()
                    if n:
                        cols.append(n)

    seen: set[str] = set()
    out: List[str] = []
    for x in cols:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _build_variable_dictionary(dataset: DatasetState) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    raw = dataset.get("raw_schema")
    if isinstance(raw, dict):
        cols = raw.get("columns")
        if isinstance(cols, list):
            for c in cols:
                if not isinstance(c, dict):
                    continue
                name = c.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                dtype = c.get("dtype") or c.get("type") or "unknown"
                out.append({"name": name.strip(), "type": str(dtype)})

    summary = dataset.get("summary")
    if isinstance(summary, dict):
        cols2 = summary.get("columns")
        if isinstance(cols2, dict):
            existing = {d["name"] for d in out if isinstance(d.get("name"), str)}
            for name, info in cols2.items():
                if not isinstance(name, str) or not name.strip() or name in existing:
                    continue
                dtype = "unknown"
                if isinstance(info, dict):
                    dtype = str(info.get("dtype") or info.get("type") or "unknown")
                out.append({"name": name.strip(), "type": dtype})

    return out


def _dataset_preview_rows(df: DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        rec: Dict[str, Any] = {}
        for col in df.columns:
            rec[str(col)] = row[col]
        out.append(rec)
    return out


# ----------------------------
# Conversation helpers
# ----------------------------

def _last_human_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(list(messages or [])):
        if getattr(m, "type", None) == "human":
            return str(getattr(m, "content", "") or "").strip()
        name = m.__class__.__name__.lower()
        if "human" in name or "user" in name:
            return str(getattr(m, "content", "") or "").strip()
    return ""


# ----------------------------
# State/control helpers
# ----------------------------

def _ensure_control_container(state: ConversationState) -> None:
    """
    Router/logging often assumes state['control'] exists.
    Make this node safe even if upstream forgot to initialize it.
    """
    c = state.get("control")
    if isinstance(c, dict):
        return
    state["control"] = {
        "current_stage_status": "PENDING",
        "action_required": "NONE",
        "node_message": None,
    }


def _require_dataset_and_meta(state: ConversationState) -> Tuple[DatasetState, MetadataState, Optional[str]]:
    dataset = state.get("dataset")
    meta = state.get("metadata")
    if not isinstance(dataset, dict) or not isinstance(meta, dict):
        return cast(DatasetState, {}), cast(MetadataState, {}), "Metadata or dataset state is missing."
    return cast(DatasetState, dataset), cast(MetadataState, meta), None


def _append_ai(state: ConversationState, content: str) -> None:
    msgs = cast(List[BaseMessage], state.get("messages") or [])
    msgs = list(msgs)
    msgs.append(AIMessage(content=content))
    state["messages"] = msgs


def _succeed(state: ConversationState, msg: Optional[str]) -> ConversationState:
    c = cast(Dict[str, Any], state["control"])
    c["current_stage_status"] = "DONE"
    c["action_required"] = "NONE"
    c["node_message"] = msg
    state["control"] = c
    return state


def _need_input(state: ConversationState, msg: str) -> ConversationState:
    c = cast(Dict[str, Any], state["control"])
    c["current_stage_status"] = "PENDING"
    c["action_required"] = "NEEDS_INPUT"
    c["node_message"] = msg
    state["control"] = c
    return state


def _fatal(state: ConversationState, msg: str) -> ConversationState:
    c = cast(Dict[str, Any], state["control"])
    c["current_stage_status"] = "ABORTED"
    c["action_required"] = "NEEDS_INPUT"
    c["node_message"] = msg
    state["control"] = c
    return state


def _as_nonempty_str(x: Any) -> Optional[str]:
    if isinstance(x, str):
        t = x.strip()
        return t or None
    return None
