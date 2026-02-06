from __future__ import annotations

import json
import logging
from typing import Any, Dict, Final, List, Mapping, MutableMapping, Sequence, Tuple, cast
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
    ALLOWED_OPS,
    ALLOWED_OUT_KINDS,
    ALLOWED_TIME_ZERO,
    ALLOWED_TREAT_KINDS,
    ALLOWED_UNITS,
    ProtocolState,
    REQUIRED_KEYS,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS: Final[int] = 1


def make_compile_protocol_state_node(*, llm: LLMService, model_name: str) -> CallableNodeFunc:
    def _run(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        ds_summary = _require_dataset_summary(state)
        if ds_summary is None:
            msg = "Dataset summary missing; cannot compile ProtocolState."
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

        dataset_cols = _extract_columns(ds_summary)

        protocol_text = _require_protocol_text(state)
        if protocol_text is None:
            msg = "Protocol summary/discussion missing; cannot compile ProtocolState."
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

        last_raw = ""
        last_errors: List[str] = []

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                prompt = _build_prompt(
                    attempt=attempt,
                    protocol_text=protocol_text,
                    dataset_summary=ds_summary,
                    previous_json=last_raw,
                    validation_errors=last_errors,
                )

                raw = _llm_json_only(llm=llm, model_name=model_name, prompt=prompt)
                last_raw = raw

                obj = _parse_json_object(raw)  # pyright: ignore[reportUnknownVariableType] # Dict[str, Any] but treated as untrusted

                ok, errs = _validate_protocol_object(obj, dataset_cols) # pyright: ignore[reportUnknownArgumentType]
                if not ok:
                    last_errors = errs
                    continue

                normalized = _normalize_protocol_object(obj) # type: ignore
                state["protocol"] = cast(ProtocolState, normalized)

                msg = "ProtocolState compiled successfully."
                ConversationStateHelpers.append_ai_message(state, msg)
                return ConversationStateHelpers.set_done(state, cast(ACTION, "NONE"), msg)

            except Exception as e:
                last_errors = [f"Attempt {attempt} exception: {e}"]
                continue

        err_text = _format_errors(last_errors, last_raw)
        final_msg = f"Failed to compile a valid ProtocolState after {MAX_ATTEMPTS} attempts.\n{err_text}"
        ConversationStateHelpers.append_ai_message(state, final_msg)
        return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), final_msg)

    return _run


# =============================================================================
# Required input extractors
# =============================================================================

def _require_dataset_summary(state: ConversationState) -> Mapping[str, Any] | None:
    dataset = state.get("dataset") or {}
    summary = dataset.get("summary")
    if isinstance(summary, dict):
        return cast(Mapping[str, Any], summary)
    return None


def _require_protocol_text(state: ConversationState) -> str | None:
    pd = state.get("protocol_discussion")
    txt = (pd.discussion if pd is not None else "").strip()
    return txt or None


# =============================================================================
# Prompt builder
# =============================================================================

def _build_prompt(
    *,
    attempt: int,
    protocol_text: str,
    dataset_summary: Mapping[str, Any],
    previous_json: str,
    validation_errors: List[str],
) -> str:
    ds_json = json.dumps(dict(dataset_summary), ensure_ascii=False)

    if attempt == 1:
        return (
            compile_protocol_prompt()
            .replace("{{PROTOCOL_TEXT}}", protocol_text)
            .replace("{{DATASET_SUMMARY_JSON}}", ds_json)
        )

    return (
        compile_protocol_repair_prompt()
        .replace("{{PROTOCOL_TEXT}}", protocol_text)
        .replace("{{DATASET_SUMMARY_JSON}}", ds_json)
        .replace("{{PREVIOUS_JSON}}", previous_json)
        .replace("{{VALIDATION_ERRORS}}", json.dumps(validation_errors or ["Unknown compiler error"], ensure_ascii=False))
    )


# =============================================================================
# LLM + parsing
# =============================================================================

def _llm_json_only(*, llm: LLMService, model_name: str, prompt: str) -> str:
    cfg = LLMConfig(model=model_name, temperature=0.0)
    resp = llm.generate(
        config=cfg,
        system_prompt="Return JSON only. No extra text.",
        user_prompt=prompt,
        history=None,
    )
    return str(cast(Any, resp).content or "")


def _parse_json_object(raw: str) -> Dict[str, Any]:
    txt = (raw or "").strip()

    if not txt.startswith("{"):
        i, j = txt.find("{"), txt.rfind("}")
        if i >= 0 and j > i:
            txt = txt[i : j + 1]

    obj = json.loads(txt)
    if not isinstance(obj, dict):
        raise ValueError("Compiler output is not a JSON object.")
    return cast(Dict[str, Any], obj)


# =============================================================================
# Validation (single-responsibility helpers)
# =============================================================================

def _validate_protocol_object(obj: Mapping[str, Any], dataset_columns: List[str] | None) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    errors.extend(_validate_required_keys(obj))
    errors.extend(_validate_enums(obj))
    errors.extend(_validate_string_list_fields(obj, fields=("covariates", "effect_modifiers", "censoring_rules")))
    errors.extend(_validate_exclusions(obj, dataset_columns))
    errors.extend(_validate_treatment_spec(obj, dataset_columns))
    errors.extend(_validate_outcome_spec(obj, dataset_columns))
    return (len(errors) == 0), errors


def _validate_required_keys(obj: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for k in REQUIRED_KEYS:
        if k not in obj:
            out.append(f"Missing required key: {k}")
    return out


def _validate_enums(obj: Mapping[str, Any]) -> List[str]:
    out: List[str] = []

    tz = _get_str(obj, "time_zero_type")
    if tz is None or tz not in ALLOWED_TIME_ZERO:
        out.append(f"time_zero_type must be one of: {sorted(ALLOWED_TIME_ZERO)}")

    twu = _get_str(obj, "treatment_window_unit")
    if twu is None or twu not in ALLOWED_UNITS:
        out.append(f"treatment_window_unit must be one of: {sorted(ALLOWED_UNITS)}")

    owu = _get_str(obj, "outcome_window_unit")
    if owu is None or owu not in ALLOWED_UNITS:
        out.append(f"outcome_window_unit must be one of: {sorted(ALLOWED_UNITS)}")

    return out


def _validate_string_list_fields(obj: Mapping[str, Any], *, fields: Sequence[str]) -> List[str]:
    out: List[str] = []
    for f in fields:
        v = obj.get(f)
        if not isinstance(v, list) or any(not isinstance(x, str) for x in v): # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            out.append(f"{f} must be list[str]") 
    return out


def _validate_exclusions(obj: Mapping[str, Any], dataset_columns: List[str] | None) -> List[str]:
    out: List[str] = []
    excls = obj.get("exclusions")

    if not isinstance(excls, list):
        return ["exclusions must be a list"]

    cols = set(dataset_columns or [])

    for i, ex in enumerate(excls): # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        if not isinstance(ex, dict):
            out.append(f"exclusions[{i}] must be an object")
            continue

        col = ex.get("column") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        op = ex.get("op") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        values = ex.get("values") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        reason = ex.get("reason") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        if not isinstance(col, str) or not col:
            out.append(f"exclusions[{i}].column must be a non-empty string")
        elif dataset_columns is not None and col not in cols:
            out.append(f"exclusions[{i}].column not in dataset: '{col}'")

        if not isinstance(op, str) or op not in ALLOWED_OPS:
            out.append(f"exclusions[{i}].op invalid: {op}")

        if not isinstance(values, list) or any(not isinstance(v, str) for v in values): # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            out.append(f"exclusions[{i}].values must be list[str]")

        if not isinstance(reason, str):
            out.append(f"exclusions[{i}].reason must be string")

    return out


def _validate_treatment_spec(obj: Mapping[str, Any], dataset_columns: List[str] | None) -> List[str]:
    out: List[str] = []
    ts = obj.get("treatment_spec")
    if not isinstance(ts, dict):
        return ["treatment_spec must be an object"]

    kind = ts.get("kind") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if not isinstance(kind, str) or kind not in ALLOWED_TREAT_KINDS:
        return [f"treatment_spec.kind must be one of: {sorted(ALLOWED_TREAT_KINDS)}"]

    col = ts.get("column") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if not isinstance(col, str) or not col:
        out.append("treatment_spec.column must be a non-empty string")
    elif dataset_columns is not None and col not in set(dataset_columns):
        out.append(f"treatment_spec.column not in dataset: '{col}'")

    if kind == "binary":
        if not isinstance(ts.get("treated"), str) or not isinstance(ts.get("control"), str): # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            out.append("binary treatment_spec requires 'treated' and 'control' strings")

    if kind == "categorical":
        lv = ts.get("levels") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if not isinstance(lv, list) or len(lv) < 2 or any(not isinstance(x, str) for x in lv): # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportUnknownMemberType]
            out.append("categorical treatment_spec requires levels: list[str] with len>=2")

    # continuous: optional numeric fields allowed; validation handled in static node
    return out


def _validate_outcome_spec(obj: Mapping[str, Any], dataset_columns: List[str] | None) -> List[str]:
    out: List[str] = []
    ys = obj.get("outcome_spec")
    if not isinstance(ys, dict):
        return ["outcome_spec must be an object"]

    kind = ys.get("kind") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if not isinstance(kind, str) or kind not in ALLOWED_OUT_KINDS:
        return [f"outcome_spec.kind must be one of: {sorted(ALLOWED_OUT_KINDS)}"]

    ds_cols = set(dataset_columns or [])

    if kind == "duration":
        dcol = ys.get("duration_column") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        ecol = ys.get("event_column") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if not isinstance(dcol, str) or not dcol:
            out.append("duration outcome_spec requires duration_column")
        elif dataset_columns is not None and dcol not in ds_cols:
            out.append(f"outcome_spec.duration_column not in dataset: '{dcol}'")

        if not isinstance(ecol, str) or not ecol:
            out.append("duration outcome_spec requires event_column")
        elif dataset_columns is not None and ecol not in ds_cols:
            out.append(f"outcome_spec.event_column not in dataset: '{ecol}'")

        if not isinstance(ys.get("event_value"), str) or not isinstance(ys.get("censor_value"), str): # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            out.append("duration outcome_spec requires event_value and censor_value strings")

        return out

    # non-duration: must have column
    col = ys.get("column") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if not isinstance(col, str) or not col:
        out.append("outcome_spec.column must be a non-empty string")
    elif dataset_columns is not None and col not in ds_cols:
        out.append(f"outcome_spec.column not in dataset: '{col}'")

    if kind == "binary":
        if not isinstance(ys.get("event"), str) or not isinstance(ys.get("non_event"), str): # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            out.append("binary outcome_spec requires 'event' and 'non_event' strings")

    if kind == "categorical":
        lv = ys.get("levels") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if not isinstance(lv, list) or len(lv) < 2 or any(not isinstance(x, str) for x in lv): # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportUnknownMemberType]
            out.append("categorical outcome_spec requires levels: list[str] with len>=2")

    return out


# =============================================================================
# Normalization (safe mutations only)
# =============================================================================

def _normalize_protocol_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    if obj.get("time_zero_type") == "CONCEPTUAL":
        obj["time_zero"] = str(obj.get("time_zero") or "CONCEPTUAL_BASELINE")
        obj["time_zero_definition"] = str(obj.get("time_zero_definition") or "shared conceptual baseline at data cut-off")

        obj["treatment_window_start"] = str(obj.get("treatment_window_start") or "0")
        obj["treatment_window_end"] = str(obj.get("treatment_window_end") or "0")
        obj["treatment_window_unit"] = str(obj.get("treatment_window_unit") or "days")

        obj["outcome_window"] = str(obj.get("outcome_window") or "0")
        obj["outcome_window_unit"] = str(obj.get("outcome_window_unit") or "days")

    _normalize_list_str(obj, "covariates")
    _normalize_list_str(obj, "effect_modifiers")
    _normalize_list_str(obj, "censoring_rules")

    if not isinstance(obj.get("exclusions"), list):
        obj["exclusions"] = []

    return obj


def _normalize_list_str(obj: MutableMapping[str, Any], key: str) -> None:
    v = obj.get(key)
    if not isinstance(v, list):
        obj[key] = []
        return
    obj[key] = [str(x) for x in v] # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


# =============================================================================
# Dataset column extraction
# =============================================================================

def _extract_columns(ds_summary: Mapping[str, Any]) -> List[str] | None:
    col_names = ds_summary.get("column_names")
    if isinstance(col_names, list) and all(isinstance(x, str) for x in col_names): # pyright: ignore[reportUnknownVariableType]
        return list(col_names) # pyright: ignore[reportUnknownArgumentType]

    cols = ds_summary.get("columns")
    if isinstance(cols, list):
        out: List[str] = []
        for c in cols: # pyright: ignore[reportUnknownVariableType]
            if isinstance(c, dict):
                name = c.get("name") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                if isinstance(name, str) and name:
                    out.append(name)
        return out or None

    return None


# =============================================================================
# Small access helpers
# =============================================================================

def _get_str(obj: Mapping[str, Any], key: str) -> str | None:
    v = obj.get(key)
    return v if isinstance(v, str) else None


def _format_errors(errors: List[str], raw_json: str) -> str:
    e = "\n".join([f"- {x}" for x in (errors or ["Unknown error"])])
    snippet = (raw_json or "").strip()
    if len(snippet) > 800:
        snippet = snippet[:800] + "…"
    return f"Validation/compile errors:\n{e}\n\nLast JSON snippet:\n{snippet}"
