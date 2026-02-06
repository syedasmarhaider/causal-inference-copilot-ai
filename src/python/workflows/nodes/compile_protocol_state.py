from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple, cast, get_args
from uuid import UUID

from python.domain.service.llm_service import LLMConfig, LLMService
from python.workflows.nodes.prompts.compile_protocol import (
    compile_protocol_prompt,
    compile_protocol_repair_prompt,
)
from python.workflows.state.conversation_state import (
    CallableNodeFunc,
    ConversationState,
    ConversationStateHelpers,
)
from python.workflows.state.control_state import ACTION
from python.workflows.state.protocol_state import (
    REQUIRED_KEYS,
    FilterOp,
    ProtocolState,
    TimeZeroType,
    WindowUnit,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


def make_compile_protocol_state_node(*, llm: LLMService, model_name: str) -> CallableNodeFunc:
    def _run(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        dataset = state.get("dataset") or {}
        ds_summary = dataset.get("summary")
        if ds_summary is None:
            ConversationStateHelpers.append_ai_message(state, "Dataset summary missing; cannot compile ProtocolState.")
            return ConversationStateHelpers.set_abort(state,  "NONE", "Dataset summary missing; cannot compile ProtocolState.")

        dataset_cols = _extract_columns(ds_summary)

        protocol_text = _extract_protocol_text(state)
        if not protocol_text.strip():
            ConversationStateHelpers.append_ai_message(state, "Protocol summary/discussion missing; cannot compile ProtocolState.")
            return ConversationStateHelpers.set_abort(state,  "NONE", "Protocol summary/discussion missing; cannot compile ProtocolState.")

        last_raw_json = ""
        last_errors: List[str] = []

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                if attempt == 1:
                    prompt = (
                        compile_protocol_prompt()
                        .replace("{{PROTOCOL_TEXT}}", protocol_text)
                        .replace("{{DATASET_SUMMARY_JSON}}", json.dumps(ds_summary, ensure_ascii=False))
                    )
                else:
                    prompt = (
                        compile_protocol_repair_prompt()
                        .replace("{{PROTOCOL_TEXT}}", protocol_text)
                        .replace("{{DATASET_SUMMARY_JSON}}", json.dumps(ds_summary, ensure_ascii=False))
                        .replace("{{PREVIOUS_JSON}}", last_raw_json)
                        .replace("{{VALIDATION_ERRORS}}", json.dumps(last_errors or ["Unknown compiler error"], ensure_ascii=False))
                    )

                raw = _llm_json_only(llm=llm, model_name=model_name, prompt=prompt)
                last_raw_json = raw

                obj = _parse_json_object(raw)

                ok, errs = _validate_protocol_state(obj, dataset_cols)
                if not ok:
                    last_errors = errs
                    continue

                protocol = _normalize_protocol_state(obj)
                state["protocol"] = cast(ProtocolState, protocol)

                msg = "ProtocolState compiled successfully."
                ConversationStateHelpers.append_ai_message(state, msg)
                return ConversationStateHelpers.set_done(state, cast(ACTION, "NONE"), msg)

            except Exception as e:
                last_errors = [f"Attempt {attempt} exception: {e}"]
                continue

        err_text = _format_errors(last_errors, last_raw_json)
        final_msg = f"Failed to compile a valid ProtocolState after {MAX_ATTEMPTS} attempts.\n{err_text}"
        ConversationStateHelpers.append_ai_message(state, final_msg)
        return ConversationStateHelpers.set_abort(state,  "NONE", final_msg)

    return _run


# ----------------------------
# LLM + parsing
# ----------------------------

def _llm_json_only(*, llm: LLMService, model_name: str, prompt: str) -> str:
    cfg = LLMConfig(model=model_name, temperature=0.0)
    resp = llm.generate(
        config=cfg,
        system_prompt="Return JSON only. No extra text.",
        user_prompt=prompt,
        history=None,
    )
    return cast(Any, resp).content or ""


def _parse_json_object(raw: str) -> Dict[str, Any]:
    txt = (raw or "").strip()

    # Minimal recovery: find first '{' and last '}' (handles accidental prose or ```json fences)
    if not txt.startswith("{"):
        i, j = txt.find("{"), txt.rfind("}")
        if i >= 0 and j > i:
            txt = txt[i : j + 1]

    obj = json.loads(txt)
    if not isinstance(obj, dict):
        raise ValueError("Compiler output is not a JSON object.")
    return cast(Dict[str, Any], obj)


# ----------------------------
# Validation (ProtocolState only)
# ----------------------------

def _validate_protocol_state(obj: Dict[str, Any], dataset_columns: List[str] | None) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    for k in REQUIRED_KEYS:
        if k not in obj:
            errors.append(f"Missing required key: {k}")
    if errors:
        return False, errors

    allowed_time_zero = set(get_args(TimeZeroType))          # {"COLUMN","CONCEPTUAL"}
    allowed_units = set(get_args(WindowUnit))                # {"minutes","hours",...}
    allowed_ops = set(get_args(FilterOp))                    # {"==","!=","in",...}

    if not isinstance(obj["time_zero_type"], str) or obj["time_zero_type"] not in allowed_time_zero:
        errors.append("time_zero_type must be 'COLUMN' or 'CONCEPTUAL'")

    if not isinstance(obj["treatment_window_unit"], str) or obj["treatment_window_unit"] not in allowed_units:
        errors.append("treatment_window_unit must be a valid unit enum")

    if not isinstance(obj["outcome_window_unit"], str) or obj["outcome_window_unit"] not in allowed_units:
        errors.append("outcome_window_unit must be a valid unit enum")

    if not isinstance(obj["outcome_is_duration"], bool):
        errors.append("outcome_is_duration must be boolean")

    for k in ("covariates", "effect_modifiers", "censoring_rules"):
        if not isinstance(obj[k], list) or any(not isinstance(x, str) for x in obj[k]):
            errors.append(f"{k} must be list[str]")

    if not isinstance(obj["exclusions"], list):
        errors.append("exclusions must be a list")
    else:
        for i, ex in enumerate(obj["exclusions"]): # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            if not isinstance(ex, dict):
                errors.append(f"exclusions[{i}] must be an object")
                continue

            for ek in ("column", "op", "values", "reason"):
                if ek not in ex:
                    errors.append(f"exclusions[{i}] missing key: {ek}")

            op = ex.get("op") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            if not isinstance(op, str) or op not in allowed_ops:
                errors.append(f"exclusions[{i}].op invalid: {op}")

            vals = ex.get("values") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            if not isinstance(vals, list) or any(not isinstance(v, str) for v in vals): # pyright: ignore[reportUnknownVariableType]
                errors.append(f"exclusions[{i}].values must be list[str]")

            col = ex.get("column") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            if dataset_columns is not None and isinstance(col, str) and col not in dataset_columns:
                errors.append(f"exclusions[{i}].column not in dataset: '{col}'")

    # Coherence: conceptual baseline cannot support duration outcome
    if obj["time_zero_type"] == "CONCEPTUAL" and obj.get("outcome_is_duration") is True:
        errors.append("Snapshot/CONCEPTUAL time_zero cannot have duration outcome without time support.")

    return (len(errors) == 0), errors


def _normalize_protocol_state(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Snapshot invariants
    if obj["time_zero_type"] == "CONCEPTUAL":
        obj["time_zero"] = str(obj.get("time_zero") or "CONCEPTUAL_BASELINE")
        obj["time_zero_definition"] = str(obj.get("time_zero_definition") or "shared conceptual baseline at data cut-off")

        obj["treatment_window_start"] = str(obj.get("treatment_window_start") or "0")
        obj["treatment_window_end"] = str(obj.get("treatment_window_end") or "0")
        obj["treatment_window_unit"] = str(obj.get("treatment_window_unit") or "days")

        obj["outcome_window"] = str(obj.get("outcome_window") or "0")
        obj["outcome_window_unit"] = str(obj.get("outcome_window_unit") or "days")
        obj["outcome_is_duration"] = False

    # Ensure lists are list[str]
    for k in ("covariates", "effect_modifiers", "censoring_rules"):
        obj[k] = [str(x) for x in (obj.get(k, []) or [])]

    # Ensure exclusions list type
    if not isinstance(obj.get("exclusions"), list):
        obj["exclusions"] = []

    return obj


# ----------------------------
# Extractors
# ----------------------------

def _extract_protocol_text(state: ConversationState) -> str:
    pd = state.get("protocol_discussion") or {} # pyright: ignore[reportUnknownVariableType]
    txt = str(pd.get("discussion") or "").strip() # type: ignore
    if txt:
        return txt

    # fallback: last assistant message text
    messages = state.get("messages", []) or []
    for m in reversed(list(messages)):
        content = str(getattr(m, "content", "") or "").strip()
        if not content:
            continue
        mtype = getattr(m, "type", None)
        cls = m.__class__.__name__.lower()
        if mtype == "ai" or "ai" in cls or "assistant" in cls:
            return content

    return ""


def _extract_columns(ds_summary: Any) -> List[str] | None:
    if not isinstance(ds_summary, dict):
        return None

    col_names = ds_summary.get("column_names") # type: ignore
    if isinstance(col_names, list):
        return [str(x) for x in col_names]

    cols = ds_summary.get("columns") # type: ignore
    if isinstance(cols, list):
        out: List[str] = []
        for c in cols:
            if isinstance(c, dict) and "name" in c:
                out.append(str(c["name"])) # type: ignore
        return out or None

    return None


def _format_errors(errors: List[str], raw_json: str) -> str:
    e = "\n".join([f"- {x}" for x in (errors or ["Unknown error"])])
    snippet = (raw_json or "").strip()
    if len(snippet) > 800:
        snippet = snippet[:800] + "…"
    return f"Validation/compile errors:\n{e}\n\nLast JSON snippet:\n{snippet}"
