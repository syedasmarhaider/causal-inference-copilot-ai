from __future__ import annotations

import json
import logging
from typing import Any, Dict, Final, List, Mapping, MutableMapping, Optional, Sequence, Tuple, cast
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
    ALLOWED_TIME_ZERO,
    ALLOWED_UNITS,
    ProtocolState,
    REQUIRED_KEYS,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS: Final[int] = 2


# =============================================================================
# Node
# =============================================================================
def make_compile_protocol_state_node(*, llm: LLMService, model_name: str) -> CallableNodeFunc:
    def _run(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        ds_summary = _require_dataset_summary(state)
        if ds_summary is None:
            msg = "Dataset summary missing; cannot compile ProtocolState."
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

        cols_in_order, profiles_by_name = _extract_columns_and_profiles(ds_summary)
        if not cols_in_order:
            msg = "Dataset summary has no column profiles; cannot compile ProtocolState."
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

        col_resolver, col_resolver_errs = _build_column_resolver(cols_in_order)
        if col_resolver_errs:
            msg = "Dataset has ambiguous column names after normalization:\n" + "\n".join(f"- {e}" for e in col_resolver_errs)
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

        value_vocab = _extract_value_vocab(ds_summary)  # may be partial by design
        token_resolvers = _build_token_resolvers(value_vocab)

        protocol_text = _require_protocol_text(state)
        if protocol_text is None:
            msg = "Protocol discussion missing; cannot compile ProtocolState."
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

        last_raw = ""
        last_errors: List[str] = []

        for attempt in range(1, MAX_ATTEMPTS + 1):
            prompt = _build_prompt(
                attempt=attempt,
                protocol_text=protocol_text,
                dataset_summary=ds_summary,
                dataset_columns=cols_in_order,
                value_vocab=value_vocab,
                previous_json=last_raw,
                validation_errors=last_errors,
            )

            try:
                raw = _llm_json_only(llm=llm, model_name=model_name, prompt=prompt)
                last_raw = raw
                obj = _parse_json_object(raw)

                # Canonicalize to EXACT df column names / EXACT token strings (where vocab exists)
                canon_errs = _canonicalize_protocol_object_inplace(
                    obj=obj,
                    col_resolver=col_resolver,
                    token_resolvers=token_resolvers,
                )
                if canon_errs:
                    last_errors = canon_errs
                    continue

                ok, errs = _validate_protocol_object(
                    obj=obj,
                    dataset_columns=cols_in_order,
                    value_vocab=value_vocab,
                    profiles_by_name=profiles_by_name,
                )
                if not ok:
                    last_errors = errs
                    continue

                normalized = _normalize_protocol_object(obj)
                state["protocol"] = cast(ProtocolState, normalized)

                msg = "ProtocolState compiled successfully."
                ConversationStateHelpers.append_ai_message(state, msg)
                return ConversationStateHelpers.set_done(state, cast(ACTION, "NONE"), msg)

            except Exception as e:
                last_errors = [f"Attempt {attempt} exception: {e!r}"]
                continue

        final_msg = (
            f"Failed to compile a valid ProtocolState after {MAX_ATTEMPTS} attempts.\n"
            f"{_format_errors(last_errors, last_raw)}"
        )
        ConversationStateHelpers.append_ai_message(state, final_msg)
        return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), final_msg)

    return _run


# =============================================================================
# Inputs (FIXED: actually returns the summary)
# =============================================================================
def _require_dataset_summary(state: ConversationState) -> Mapping[str, Any] | None:
    dataset = state.get("dataset") or {}
    summary = dataset.get("summary")
    return cast(Mapping[str, Any], summary) if isinstance(summary, dict) else None


def _require_protocol_text(state: ConversationState) -> str | None:
    pd = state.get("protocol_discussion")
    txt = (pd.discussion if pd is not None else "").strip()
    return txt or None


# =============================================================================
# Prompt builder (robust: appends cols+vocab even if prompt template lacks placeholders)
# =============================================================================
def _build_prompt(
    *,
    attempt: int,
    protocol_text: str,
    dataset_summary: Mapping[str, Any],
    dataset_columns: List[str],
    value_vocab: Dict[str, List[str]],
    previous_json: str,
    validation_errors: List[str],
) -> str:
    ds_json = json.dumps(dict(dataset_summary), ensure_ascii=False)
    cols_json = json.dumps(dataset_columns, ensure_ascii=False)
    vocab_json = json.dumps(value_vocab, ensure_ascii=False)

    appendix = (
        "\n\nDATASET_COLUMNS_JSON:\n" + cols_json +
        "\n\nDATASET_VALUE_VOCAB_JSON:\n" + vocab_json
    )

    if attempt == 1:
        return (
            compile_protocol_prompt()
            .replace("{{PROTOCOL_TEXT}}", protocol_text)
            .replace("{{DATASET_SUMMARY_JSON}}", ds_json)
            + appendix
        )

    return (
        compile_protocol_repair_prompt()
        .replace("{{PROTOCOL_TEXT}}", protocol_text)
        .replace("{{DATASET_SUMMARY_JSON}}", ds_json)
        .replace("{{PREVIOUS_JSON}}", previous_json)
        .replace("{{VALIDATION_ERRORS}}", json.dumps(validation_errors or ["Unknown compiler error"], ensure_ascii=False))
        + appendix
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
# Dataset summary extraction (YOUR new deterministic format: {"n_rows":..,"profiles":[...]} )
# =============================================================================
def _extract_columns_and_profiles(ds_summary: Mapping[str, Any]) -> Tuple[List[str], Dict[str, Mapping[str, Any]]]:
    """
    Returns:
      - columns in deterministic df.columns order
      - profiles_by_name[name] -> profile dict
    """
    profs = ds_summary.get("profiles")
    if not isinstance(profs, list):
        return [], {}

    cols: List[str] = []
    by_name: Dict[str, Mapping[str, Any]] = {}

    for p in profs: # pyright: ignore[reportUnknownVariableType]
        if not isinstance(p, dict):
            continue
        name = p.get("name") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if not isinstance(name, str):
            continue
        nm = name.strip()
        if not nm:
            continue
        cols.append(nm)
        by_name[nm] = cast(Mapping[str, Any], p)

    return cols, by_name


def _extract_value_vocab(ds_summary: Mapping[str, Any]) -> Dict[str, List[str]]:
    """
    Strict vocab used for exact-token matching.
    For categorical -> from summary.top_categories[].value
    For boolean -> from summary.counts keys
    For other -> from summary.distinct_values_sample
    """
    profs = ds_summary.get("profiles")
    if not isinstance(profs, list):
        return {}

    vocab: Dict[str, List[str]] = {}

    for p in profs: # pyright: ignore[reportUnknownVariableType]
        if not isinstance(p, dict):
            continue
        col = p.get("name") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        kind = p.get("inferred_kind")# pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        summ = p.get("summary") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if not isinstance(col, str) or not col.strip():
            continue
        if not isinstance(kind, str) or not isinstance(summ, dict):
            continue

        coln = col.strip()

        if kind == "CATEGORICAL":
            top = summ.get("top_categories") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            vals: List[str] = []
            if isinstance(top, list):
                for it in top: # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                    if isinstance(it, dict):
                        v = it.get("value") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                        if isinstance(v, str) and v != "":
                            vals.append(v)
            if vals:
                vocab[coln] = _dedup_preserve(vals)

        elif kind == "BOOLEAN":
            counts = summ.get("counts") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            if isinstance(counts, dict):
                keys = [k for k in counts.keys() if isinstance(k, str) and k != ""]
                if keys:
                    vocab[coln] = _dedup_preserve(keys)

        elif kind == "OTHER":
            dv = summ.get("distinct_values_sample") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            if isinstance(dv, list) and all(isinstance(x, str) for x in dv): # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                vals2 = [x for x in dv if x != ""] # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                if vals2:
                    vocab[coln] = _dedup_preserve(vals2) # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportUnknownMemberType]

    return vocab


def _dedup_preserve(xs: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =============================================================================
# Canonicalization (turn LLM strings into EXACT df column names / EXACT tokens)
# =============================================================================
def _norm_key(s: str) -> str:
    return " ".join(s.strip().split()).lower()


def _build_column_resolver(columns_in_order: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Returns:
      resolver[norm(col)] -> exact col
      errors: normalization collisions
    """
    resolver: Dict[str, str] = {}
    errors: List[str] = []

    for col in columns_in_order:
        k = _norm_key(col)
        if k in resolver and resolver[k] != col:
            errors.append(f"'{resolver[k]}' vs '{col}' collide under normalization '{k}'")
        else:
            resolver[k] = col

    return resolver, errors


def _build_token_resolvers(value_vocab: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
    """
    token_resolvers[col][norm(token)] -> exact token (only if unique under normalization).
    If collisions exist, we do NOT resolve for that normalized key.
    """
    out: Dict[str, Dict[str, str]] = {}

    for col, toks in value_vocab.items():
        m: Dict[str, str] = {}
        collisions: set[str] = set()

        for t in toks:
            k = _norm_key(t)
            if k in m and m[k] != t:
                collisions.add(k)
            else:
                m[k] = t

        for k in collisions:
            m.pop(k, None)

        out[col] = m

    return out


def _canonicalize_column(col: str, col_resolver: Dict[str, str]) -> Optional[str]:
    return col_resolver.get(_norm_key(col))


def _canonicalize_token(col: str, token: str, token_resolvers: Dict[str, Dict[str, str]]) -> Optional[str]:
    m = token_resolvers.get(col)
    if not m:
        return None
    return m.get(_norm_key(token))


def _canonicalize_protocol_object_inplace(
    *,
    obj: MutableMapping[str, Any],
    col_resolver: Dict[str, str],
    token_resolvers: Dict[str, Dict[str, str]],
) -> List[str]:
    """
    Canonicalizes:
      - exclusions[].column
      - treatment_spec.column / outcome_spec.column
      - covariates[] / effect_modifiers[]
      - categorical/boolean tokens wherever vocab exists
    Returns a list of errors (empty => ok).
    """
    errs: List[str] = []

    # helper: must exist and be resolvable
    def canon_col(field: str, v: Any) -> Optional[str]:
        if not isinstance(v, str) or not v.strip():
            errs.append(f"{field} must be a non-empty string")
            return None
        c = _canonicalize_column(v, col_resolver)
        if c is None:
            errs.append(f"{field} column not in dataset (after normalization): '{v}'")
            return None
        return c

    def canon_list_cols(field: str, v: Any) -> Optional[List[str]]:
        if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
            errs.append(f"{field} must be list[str]")
            return None
        out_list: List[str] = []
        for x in v:
            cx = _canonicalize_column(x, col_resolver)
            if cx is None:
                errs.append(f"{field} contains column not in dataset: '{x}'")
                continue
            out_list.append(cx)
        return out_list

    # covariates/effect_modifiers
    if "covariates" in obj:
        v = canon_list_cols("covariates", obj.get("covariates"))
        if v is not None:
            obj["covariates"] = v
    if "effect_modifiers" in obj:
        v = canon_list_cols("effect_modifiers", obj.get("effect_modifiers"))
        if v is not None:
            obj["effect_modifiers"] = v

    # exclusions
    excls = obj.get("exclusions")
    if isinstance(excls, list):
        for i, ex in enumerate(excls):
            if not isinstance(ex, dict):
                continue
            c = canon_col(f"exclusions[{i}].column", ex.get("column"))
            if c is not None:
                ex["column"] = c

            # token canonicalization (only if vocab exists for that col)
            values = ex.get("values")
            op = ex.get("op")
            if isinstance(c, str) and isinstance(values, list) and all(isinstance(x, str) for x in values) and isinstance(op, str):
                if op in ("==", "!=", "in", "not_in"):
                    fixed: List[str] = []
                    for t in values:
                        ct = _canonicalize_token(c, t, token_resolvers)
                        fixed.append(ct if ct is not None else t)
                    ex["values"] = fixed

    # treatment/outcome columns + tokens
    ts = obj.get("treatment_spec")
    if isinstance(ts, dict):
        c = canon_col("treatment_spec.column", ts.get("column"))
        if c is not None:
            ts["column"] = c
            _canonicalize_spec_tokens_inplace(
                spec=ts,
                column=c,
                token_resolvers=token_resolvers,
                fields=("treated_values", "control_values", "included_levels", "levels", "treated", "control"),
            )

    ys = obj.get("outcome_spec")
    if isinstance(ys, dict):
        c = canon_col("outcome_spec.column", ys.get("column"))
        if c is not None:
            ys["column"] = c
            _canonicalize_spec_tokens_inplace(
                spec=ys,
                column=c,
                token_resolvers=token_resolvers,
                fields=("event_values", "non_event_values", "included_levels", "levels", "event", "non_event"),
            )

    return errs


def _canonicalize_spec_tokens_inplace(
    *,
    spec: MutableMapping[str, Any],
    column: str,
    token_resolvers: Dict[str, Dict[str, str]],
    fields: Sequence[str],
) -> None:
    m = token_resolvers.get(column)
    if not m:
        return

    for f in fields:
        v = spec.get(f)
        if isinstance(v, str):
            spec[f] = m.get(_norm_key(v), v)
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            spec[f] = [m.get(_norm_key(x), x) for x in v]


# =============================================================================
# Validation
# =============================================================================
def _validate_protocol_object(
    *,
    obj: Mapping[str, Any],
    dataset_columns: List[str],
    value_vocab: Dict[str, List[str]],
    profiles_by_name: Mapping[str, Mapping[str, Any]],
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    errors.extend(_validate_required_keys(obj))
    errors.extend(_validate_enums(obj))
    errors.extend(_validate_windows(obj))
    errors.extend(_validate_feature_lists(obj, dataset_columns))
    errors.extend(_validate_exclusions(obj, dataset_columns, value_vocab, profiles_by_name))
    errors.extend(_validate_treatment_spec(obj, dataset_columns, value_vocab, profiles_by_name))
    errors.extend(_validate_outcome_spec(obj, dataset_columns, value_vocab, profiles_by_name))
    return (len(errors) == 0), errors


def _validate_required_keys(obj: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for k in REQUIRED_KEYS:
        if k not in obj:
            out.append(f"Missing required key: {k}")
    return out


def _validate_enums(obj: Mapping[str, Any]) -> List[str]:
    out: List[str] = []

    tz = obj.get("time_zero_type")
    if not isinstance(tz, str) or tz not in ALLOWED_TIME_ZERO:
        out.append(f"time_zero_type must be one of: {sorted(ALLOWED_TIME_ZERO)}")

    twu = obj.get("treatment_window_unit")
    if not isinstance(twu, str) or twu not in ALLOWED_UNITS:
        out.append(f"treatment_window_unit must be one of: {sorted(ALLOWED_UNITS)}")

    owu = obj.get("outcome_window_unit")
    if not isinstance(owu, str) or owu not in ALLOWED_UNITS:
        out.append(f"outcome_window_unit must be one of: {sorted(ALLOWED_UNITS)}")

    et = obj.get("experiment_type")
    if not isinstance(et, str) or et not in ("RCT", "Observational"):
        out.append("experiment_type must be 'RCT' or 'Observational'")

    return out


def _validate_windows(obj: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("treatment_window_start", "treatment_window_end", "outcome_window"):
        v = obj.get(key)
        if not isinstance(v, str) or not v.strip():
            out.append(f"{key} must be a non-empty string")
    return out


def _validate_feature_lists(obj: Mapping[str, Any], dataset_columns: List[str]) -> List[str]:
    cols = set(dataset_columns)
    out: List[str] = []
    for key in ("covariates", "effect_modifiers"):
        v = obj.get(key)
        if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
            out.append(f"{key} must be list[str]")
            continue
        bad = [x for x in v if x not in cols]
        if bad:
            out.append(f"{key} contains columns not in dataset: {bad}")
    return out


def _profile_kind(profiles_by_name: Mapping[str, Mapping[str, Any]], col: str) -> Optional[str]:
    p = profiles_by_name.get(col)
    if not isinstance(p, dict):
        return None
    k = p.get("inferred_kind")
    return k if isinstance(k, str) else None


def _validate_exclusions(
    obj: Mapping[str, Any],
    dataset_columns: List[str],
    value_vocab: Dict[str, List[str]],
    profiles_by_name: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    excls = obj.get("exclusions")
    if not isinstance(excls, list):
        return ["exclusions must be a list"]

    cols = set(dataset_columns)
    out: List[str] = []

    for i, ex in enumerate(excls):
        if not isinstance(ex, dict):
            out.append(f"exclusions[{i}] must be an object")
            continue

        col = ex.get("column")
        op = ex.get("op")
        values = ex.get("values")
        reason = ex.get("reason")

        if not isinstance(col, str) or not col:
            out.append(f"exclusions[{i}].column must be a non-empty string")
            continue
        if col not in cols:
            out.append(f"exclusions[{i}].column not in dataset: '{col}'")
            continue

        if not isinstance(op, str) or op not in ALLOWED_OPS:
            out.append(f"exclusions[{i}].op invalid: {op}")

        # is_null / not_null should have empty values list (strict)
        if isinstance(op, str) and op in ("is_null", "not_null"):
            if values != []:
                out.append(f"exclusions[{i}].values must be [] for op='{op}'")
        else:
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                out.append(f"exclusions[{i}].values must be list[str]")

        if not isinstance(reason, str) or not reason.strip():
            out.append(f"exclusions[{i}].reason must be a non-empty string")

        kind = _profile_kind(profiles_by_name, col)
        if kind in ("CATEGORICAL", "BOOLEAN") and isinstance(values, list) and isinstance(op, str) and op in ("==", "!=", "in", "not_in"):
            out.extend(_validate_vocab_list(col, f"exclusions[{i}].values", values, value_vocab))

    return out


def _validate_treatment_spec(
    obj: Mapping[str, Any],
    dataset_columns: List[str],
    value_vocab: Dict[str, List[str]],
    profiles_by_name: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    ts = obj.get("treatment_spec")
    if not isinstance(ts, dict):
        return ["treatment_spec must be an object"]

    cols = set(dataset_columns)
    kind = ts.get("kind")
    col = ts.get("column")

    if not isinstance(kind, str) or kind not in ("binary", "continuous", "categorical"):
        return ["treatment_spec.kind must be one of: ['binary','continuous','categorical']"]

    if not isinstance(col, str) or not col:
        return ["treatment_spec.column must be a non-empty string"]
    if col not in cols:
        return [f"treatment_spec.column not in dataset: '{col}'"]

    out: List[str] = []

    if kind == "binary":
        tv = ts.get("treated_values")
        cv = ts.get("control_values")
        if not isinstance(tv, list) or any(not isinstance(x, str) for x in tv) or not tv:
            out.append("treatment_spec.treated_values must be non-empty list[str]")
        if not isinstance(cv, list) or any(not isinstance(x, str) for x in cv) or not cv:
            out.append("treatment_spec.control_values must be non-empty list[str]")

        if isinstance(tv, list) and isinstance(cv, list):
            inter = set(tv).intersection(set(cv))
            if inter:
                out.append(f"treated_values and control_values must be disjoint; overlap={sorted(inter)}")

        if _profile_kind(profiles_by_name, col) in ("CATEGORICAL", "BOOLEAN"):
            if isinstance(tv, list):
                out.extend(_validate_vocab_list(col, "treatment_spec.treated_values", tv, value_vocab))
            if isinstance(cv, list):
                out.extend(_validate_vocab_list(col, "treatment_spec.control_values", cv, value_vocab))

    elif kind == "categorical":
        lv = ts.get("included_levels")
        if not isinstance(lv, list) or any(not isinstance(x, str) for x in lv) or len(lv) < 2:
            out.append("treatment_spec.included_levels must be list[str] with len>=2")
        if _profile_kind(profiles_by_name, col) in ("CATEGORICAL", "BOOLEAN") and isinstance(lv, list):
            out.extend(_validate_vocab_list(col, "treatment_spec.included_levels", lv, value_vocab))

    else:  # continuous
        vmin = ts.get("valid_min")
        vmax = ts.get("valid_max")
        if vmin is not None and not isinstance(vmin, (int, float)):
            out.append("treatment_spec.valid_min must be a number if provided")
        if vmax is not None and not isinstance(vmax, (int, float)):
            out.append("treatment_spec.valid_max must be a number if provided")
        if isinstance(vmin, (int, float)) and isinstance(vmax, (int, float)) and vmin > vmax:
            out.append("treatment_spec.valid_min must be <= valid_max")

    return out


def _validate_outcome_spec(
    obj: Mapping[str, Any],
    dataset_columns: List[str],
    value_vocab: Dict[str, List[str]],
    profiles_by_name: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    ys = obj.get("outcome_spec")
    if not isinstance(ys, dict):
        return ["outcome_spec must be an object"]

    cols = set(dataset_columns)
    kind = ys.get("kind")
    col = ys.get("column")

    if not isinstance(kind, str) or kind not in ("binary", "continuous", "categorical"):
        return ["outcome_spec.kind must be one of: ['binary','continuous','categorical']"]

    if not isinstance(col, str) or not col:
        return ["outcome_spec.column must be a non-empty string"]
    if col not in cols:
        return [f"outcome_spec.column not in dataset: '{col}'"]

    out: List[str] = []

    if kind == "binary":
        ev = ys.get("event_values")
        nev = ys.get("non_event_values")
        if not isinstance(ev, list) or any(not isinstance(x, str) for x in ev) or not ev:
            out.append("outcome_spec.event_values must be non-empty list[str]")
        if not isinstance(nev, list) or any(not isinstance(x, str) for x in nev) or not nev:
            out.append("outcome_spec.non_event_values must be non-empty list[str]")

        if isinstance(ev, list) and isinstance(nev, list):
            inter = set(ev).intersection(set(nev))
            if inter:
                out.append(f"event_values and non_event_values must be disjoint; overlap={sorted(inter)}")

        if _profile_kind(profiles_by_name, col) in ("CATEGORICAL", "BOOLEAN"):
            if isinstance(ev, list):
                out.extend(_validate_vocab_list(col, "outcome_spec.event_values", ev, value_vocab))
            if isinstance(nev, list):
                out.extend(_validate_vocab_list(col, "outcome_spec.non_event_values", nev, value_vocab))

    elif kind == "categorical":
        lv = ys.get("included_levels")
        if not isinstance(lv, list) or any(not isinstance(x, str) for x in lv) or len(lv) < 2:
            out.append("outcome_spec.included_levels must be list[str] with len>=2")
        if _profile_kind(profiles_by_name, col) in ("CATEGORICAL", "BOOLEAN") and isinstance(lv, list):
            out.extend(_validate_vocab_list(col, "outcome_spec.included_levels", lv, value_vocab))

    else:  # continuous
        vmin = ys.get("valid_min")
        vmax = ys.get("valid_max")
        if vmin is not None and not isinstance(vmin, (int, float)):
            out.append("outcome_spec.valid_min must be a number if provided")
        if vmax is not None and not isinstance(vmax, (int, float)):
            out.append("outcome_spec.valid_max must be a number if provided")
        if isinstance(vmin, (int, float)) and isinstance(vmax, (int, float)) and vmin > vmax:
            out.append("outcome_spec.valid_min must be <= valid_max")

    return out


def _validate_vocab_list(column: str, field_name: str, values: List[str], value_vocab: Dict[str, List[str]]) -> List[str]:
    vocab = value_vocab.get(column)
    if not vocab:
        return [
            f"Missing token vocabulary for column '{column}' required for strict matching of {field_name}. "
            f"Fix: store a fuller vocab in DatasetSummary (raise max_categories / add full unique levels when distinct_count is small)."
        ]
    allowed = set(vocab)
    missing = [x for x in values if x not in allowed]
    if missing:
        return [
            f"{field_name} contains tokens not present in dataset vocab for '{column}': {missing}. "
            f"Available tokens (from profiling): {vocab}"
        ]
    return []


# =============================================================================
# Normalization
# =============================================================================
def _normalize_protocol_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    _normalize_list_str(obj, "covariates")
    _normalize_list_str(obj, "effect_modifiers")

    if not isinstance(obj.get("exclusions"), list):
        obj["exclusions"] = []

    if obj.get("time_zero_type") == "CONCEPTUAL":
        obj["time_zero"] = str(obj.get("time_zero") or "CONCEPTUAL_BASELINE")
        obj["time_zero_definition"] = str(obj.get("time_zero_definition") or "shared conceptual baseline at data cut-off")

        obj["treatment_window_start"] = str(obj.get("treatment_window_start") or "0")
        obj["treatment_window_end"] = str(obj.get("treatment_window_end") or "0")
        obj["treatment_window_unit"] = str(obj.get("treatment_window_unit") or "days")

        obj["outcome_window"] = str(obj.get("outcome_window") or "0")
        obj["outcome_window_unit"] = str(obj.get("outcome_window_unit") or "days")

    return obj


def _normalize_list_str(obj: MutableMapping[str, Any], key: str) -> None:
    v = obj.get(key)
    if not isinstance(v, list):
        obj[key] = []
        return
    obj[key] = [str(x) for x in v]


# =============================================================================
# Errors
# =============================================================================
def _format_errors(errors: List[str], raw_json: str) -> str:
    e = "\n".join([f"- {x}" for x in (errors or ["Unknown error"])])
    snippet = (raw_json or "").strip()
    if len(snippet) > 800:
        snippet = snippet[:800] + "…"
    return f"Validation/compile errors:\n{e}\n\nLast JSON snippet:\n{snippet}"
