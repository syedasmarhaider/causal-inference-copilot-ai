# src/python/workflows/nodes/metadata_intake.py
from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, cast
from uuid import UUID

from langchain_core.messages import BaseMessage

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.workflows.state.control_state import ACTION, NEED_STAGE, ControlState, Stage, Status
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.dataset_state import DatasetState
from python.workflows.state.metadata_state import MetadataState
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI, JSONDict

log = logging.getLogger(__name__)

# JSON can come as raw, fenced, or with extra text around it.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

# Key=value fast path (humans type this a lot)
_KV_RE = re.compile(
    r"^\s*(treatment|outcome|covariates|controls|modifiers|question)\s*=\s*(.+?)\s*$",
    re.IGNORECASE,
)

# Common acceptance / reset phrases
_ACCEPT_RE = re.compile(
    r"\b(ok|okay|looks\s+good|good|fine|proceed|continue|confirm|yes|ship|go\s+ahead|leave\s+it|keep\s+it)\b",
    re.IGNORECASE,
)
_RESET_RE = re.compile(r"\b(reset|start\s+over|clear|wipe)\b", re.IGNORECASE)

# When user asks to “use suggestions”
_USE_SUGGESTED_CONTROLS_RE = re.compile(
    r"\b(use|take|apply)\s+(suggested|proposed|recommended)\s+(controls|covariates)\b",
    re.IGNORECASE,
)
_USE_SUGGESTED_MODIFIERS_RE = re.compile(
    r"\b(use|take|apply)\s+(suggested|proposed|recommended)\s+(modifiers|effect\s+modifiers)\b",
    re.IGNORECASE,
)

# “Add/remove …: a,b”
_ADD_REMOVE_RE = re.compile(
    r"\b(add|remove|drop|delete)\s+(controls|covariates|modifiers|effect\s+modifiers)\s*:\s*([^\n;]+)",
    re.IGNORECASE,
)

Delta = Dict[str, Any]


# =============================================================================
# State adapters
# =============================================================================
def _require_control(state: ConversationState) -> ControlState:
    return cast(ControlState, state["control"])  # type: ignore


def _as_dataset(state: ConversationState) -> DatasetState:
    return cast(DatasetState, state.get("dataset", {}))  # type: ignore


def _as_metadata(state: ConversationState) -> MetadataState:
    return cast(MetadataState, state.get("metadata", {}))  # type: ignore


def _role_from_langchain_msg(m: BaseMessage) -> str:
    t = getattr(m, "type", None)
    return {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(str(t), "user")


def _last_human_msg_idx(messages: Sequence[BaseMessage]) -> int:
    """
    Returns the index of the last non-empty human message, else -1.
    We use this to avoid re-processing old messages (e.g. dataset path) as metadata instructions.
    """
    last = -1
    for i, m in enumerate(messages):
        if getattr(m, "type", None) == "human":
            txt = str(getattr(m, "content", "") or "").strip()
            if txt:
                last = i
    return last


# =============================================================================
# Column utilities (robust matching)
# =============================================================================
def _columns_from_raw_schema(raw_schema: Any) -> List[str]:
    if not isinstance(raw_schema, dict):
        return []
    cols = raw_schema.get("columns")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if not isinstance(cols, list):
        return []
    out: List[str] = []
    for c in cols:  # pyright: ignore[reportUnknownVariableType]
        if isinstance(c, dict):
            name = c.get("name")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if isinstance(name, str) and name:
                out.append(name)
    return out


def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", s.strip().lower())


def _resolve_column(user_col: str, columns: Sequence[str]) -> str | None:
    if not user_col:
        return None
    if user_col in columns:
        return user_col

    lowered = {c.lower(): c for c in columns}
    if user_col.lower() in lowered:
        return lowered[user_col.lower()]

    normed = {_norm(c): c for c in columns}
    return normed.get(_norm(user_col))


def _suggest_columns(bad: str, columns: Sequence[str], *, k: int = 5) -> List[str]:
    if not bad:
        return []
    return difflib.get_close_matches(bad, list(columns), n=k, cutoff=0.55)


# =============================================================================
# JSON extraction / repair
# =============================================================================
def _extract_json_object(text: str) -> JSONDict:
    """
    Robustly extract a JSON object from:
      - fenced blocks
      - pure JSON
      - any text containing a JSON object substring
    """
    s = (text or "").strip()

    # 1) fenced
    m = _JSON_FENCE_RE.search(s)
    if m:
        fenced = m.group(1).strip()
        try:
            obj = json.loads(fenced)
            if isinstance(obj, dict):
                return cast(JSONDict, obj)
        except Exception:
            pass

    # 2) direct
    try:
        obj2 = json.loads(s)
        if isinstance(obj2, dict):
            return cast(JSONDict, obj2)
    except Exception:
        pass

    # 3) substring via JSONDecoder.raw_decode
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            obj3, _end = dec.raw_decode(s[i:])
            if isinstance(obj3, dict):
                return cast(JSONDict, obj3)
        except Exception:
            continue

    raise ValueError("No valid JSON object found in LLM output.")


def _repair_json_with_llm(
    llm: LLMService,
    *,
    model_name: str,
    broken_text: str,
    schema_name: str,
    schema_keys: List[str],
) -> JSONDict:
    """
    Uses the LLM as a strict JSON repair bot (only when parsing fails).
    Keeps orchestration resilient to “almost JSON” outputs.
    """
    keys = ", ".join(f'"{k}"' for k in schema_keys)
    sys = (
        "You are a JSON repair bot.\n"
        f"Return ONLY a valid JSON object for schema '{schema_name}'.\n"
        f"Must contain EXACTLY these keys: {keys}\n"
        "No markdown. No prose. No extra keys.\n"
    )
    cfg = LLMConfig(model=model_name, temperature=0.0)
    resp = llm.generate(
        config=cfg,
        history=[
            ChatMessage(role="system", content=sys),
            ChatMessage(role="user", content=broken_text),
        ],
    )
    return _extract_json_object(resp.content)


# =============================================================================
# Proposal (LLM) + normalization
# =============================================================================
def _default_proposal() -> Dict[str, Any]:
    return {
        "dataset_summary": "",
        "treatment_candidates": [],
        "outcome_candidates": [],
        "controls_candidates": [],
        "effect_modifier_candidates": [],
        "effect_examples": [],
        "questions_for_user": [
            "What is the main causal question you want to answer?",
            "Which column is the treatment?",
            "Which column is the outcome?",
            "Which columns are confounders/controls (or say 'all other columns' / 'none')?",
            "Optional: do you want heterogeneous effects? If yes, which columns define subgroups?",
        ],
    }


def _normalize_str_list(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for v in x:  # pyright: ignore[reportUnknownVariableType]
        if v is None:
            continue
        s = str(v).strip()  # pyright: ignore[reportUnknownArgumentType]
        if s and s not in out:
            out.append(s)
    return out


def _filter_allowed(values: Sequence[str], allowed: Sequence[str], *, k: int | None = None) -> List[str]:
    allowed_set = set(allowed)
    out: List[str] = []
    for v in values:
        if v in allowed_set and v not in out:
            out.append(v)
            if k is not None and len(out) >= k:
                break
    return out


def _sanitize_proposal(obj: JSONDict, columns: Sequence[str]) -> Dict[str, Any]:
    """
    Enforce hard constraints: candidates must be real column names.
    """
    p = _default_proposal()
    p["dataset_summary"] = str(obj.get("dataset_summary", "")).strip()

    p["treatment_candidates"] = _filter_allowed(_normalize_str_list(obj.get("treatment_candidates")), columns, k=12)
    p["outcome_candidates"] = _filter_allowed(_normalize_str_list(obj.get("outcome_candidates")), columns, k=12)
    p["controls_candidates"] = _filter_allowed(_normalize_str_list(obj.get("controls_candidates")), columns, k=24)
    p["effect_modifier_candidates"] = _filter_allowed(
        _normalize_str_list(obj.get("effect_modifier_candidates")), columns, k=24
    )

    p["effect_examples"] = _normalize_str_list(obj.get("effect_examples"))[:10]
    qs = _normalize_str_list(obj.get("questions_for_user"))
    p["questions_for_user"] = qs if qs else _default_proposal()["questions_for_user"]
    return p


def _propose_design_once(
    llm: LLMService,
    data_repo: DataRepo,
    *,
    dataset_id: UUID,
    raw_schema: Dict[str, Any],
    summary: Dict[str, Any],
    columns: List[str],
    messages: Sequence[BaseMessage],
    model_name: str,
    history_window: int,
    sample_rows: int,
    max_sample_chars: int,
) -> Tuple[Dict[str, Any], JSONDict | None]:
    """
    Propose candidate columns once (hard-constrained to AllowedColumns).
    """
    sample_json = "null"
    try:
        df_head = data_repo.get_csv_data(dataset_id, limit=sample_rows)
        df_head = df_head.where(df_head.notna(), None)  # type: ignore
        sample_json = df_head.to_json(orient="records", force_ascii=False)  # type: ignore
        if len(sample_json) > max_sample_chars:
            sample_json = sample_json[:max_sample_chars] + "…"
    except Exception:
        sample_json = "null"

    sys = (
        "You are a causal inference copilot.\n"
        "Goal: propose candidate columns for roles in a causal analysis.\n\n"
        "Hard constraints:\n"
        "- ONLY output column names that appear in AllowedColumns.\n"
        "- If unsure, return empty lists.\n\n"
        "Return ONLY one valid JSON object with EXACTLY these keys:\n"
        "{\n"
        '  "dataset_summary": string,\n'
        '  "treatment_candidates": [string],\n'
        '  "outcome_candidates": [string],\n'
        '  "controls_candidates": [string],\n'
        '  "effect_modifier_candidates": [string],\n'
        '  "effect_examples": [string],\n'
        '  "questions_for_user": [string]\n'
        "}\n"
        "No markdown. No extra keys."
    )

    tail = list(messages)[-history_window:]
    llm_history: List[ChatMessage] = [ChatMessage(role="system", content=sys)]
    for m in tail:
        llm_history.append(
            ChatMessage(
                role=cast(Any, _role_from_langchain_msg(m)),
                content=str(getattr(m, "content", "")),
            )
        )

    llm_history.append(
        ChatMessage(
            role="user",
            content=(
                "AllowedColumns (JSON array):\n"
                f"{json.dumps(columns, ensure_ascii=False)}\n\n"
                "Dataset schema (JSON):\n"
                f"{json.dumps(raw_schema, ensure_ascii=False)}\n\n"
                "Dataset summary (JSON):\n"
                f"{json.dumps(summary, ensure_ascii=False)}\n\n"
                "Sample rows (JSON array of records, may be truncated):\n"
                f"{sample_json}\n"
            ),
        )
    )

    cfg = LLMConfig(model=model_name, temperature=0.0)

    raw_out: Optional[str] = None
    try:
        resp = llm.generate(config=cfg, history=llm_history)
        raw_out = resp.content
        obj = _extract_json_object(resp.content)
    except Exception as e:
        try:
            obj = _repair_json_with_llm(
                llm,
                model_name=model_name,
                broken_text=str(raw_out or ""),
                schema_name="ProposedDesign",
                schema_keys=[
                    "dataset_summary",
                    "treatment_candidates",
                    "outcome_candidates",
                    "controls_candidates",
                    "effect_modifier_candidates",
                    "effect_examples",
                    "questions_for_user",
                ],
            )
            return _sanitize_proposal(obj, columns), {
                "code": "LLM_METADATA_PROPOSAL_REPAIRED",
                "detail": str(e),
                "raw_llm_output": (str(raw_out)[:1500] if raw_out else None),
            }
        except Exception as e2:
            return _default_proposal(), {"code": "LLM_METADATA_PROPOSAL_FAILED", "detail": f"{e} | repair={e2}"}

    return _sanitize_proposal(obj, columns), None


# =============================================================================
# Draft defaults / conversions
# =============================================================================
def _default_draft() -> Dict[str, Any]:
    return {
        "treatment": None,
        "outcome": None,
        "covariate_strategy": None,  # USER_LIST | ALL_EXCEPT_TY | NONE
        "covariates": [],
        "effect_modifiers": [],
        "causal_question": None,
        "accept": False,
    }


def _draft_from_final(final_design: Dict[str, Any]) -> Dict[str, Any]:
    """
    If user wants to edit after acceptance, we reopen into a draft.
    """
    return {
        "treatment": final_design.get("treatment"),
        "outcome": final_design.get("outcome"),
        "covariate_strategy": final_design.get("covariate_strategy"),
        "covariates": list(final_design.get("covariates", []) or []),
        "effect_modifiers": list(final_design.get("effect_modifiers", []) or []),
        "causal_question": final_design.get("causal_question"),
        "accept": False,  # require re-accept after edits
    }


def _iter_new_humans(messages: Sequence[BaseMessage], last_idx_seen: int) -> Iterator[Tuple[int, str]]:
    """
    Robustness: if multiple human messages landed before node runs,
    apply them in order (no lost instructions).
    """
    start = max(-1, int(last_idx_seen))
    for i in range(start + 1, len(messages)):
        m = messages[i]
        if getattr(m, "type", None) == "human":
            txt = str(getattr(m, "content", "")).strip()
            if txt:
                yield i, txt


# =============================================================================
# Delta parsing (heuristic first, LLM as fallback)
# =============================================================================
def _heuristic_delta(user_text: str, proposed: Dict[str, Any]) -> Delta:
    """
    Cheap deterministic parsing for common patterns.
    If message looks “free-form”, caller can fall back to LLM parsing.
    """
    txt = user_text.strip()
    low = txt.lower()

    d: Delta = {
        "accept": bool(_ACCEPT_RE.search(low)),
        "reset": bool(_RESET_RE.search(low)),
        "set_treatment": None,
        "set_outcome": None,
        "covariate_strategy": None,  # USER_LIST | ALL_EXCEPT_TY | NONE
        "set_covariates": None,  # list[str] | None
        "add_covariates": [],
        "remove_covariates": [],
        "add_modifiers": [],
        "remove_modifiers": [],
        "set_question": None,
        "clear_covariates": False,
        "clear_modifiers": False,
        "use_suggested_controls": bool(_USE_SUGGESTED_CONTROLS_RE.search(txt)),
        "use_suggested_modifiers": bool(_USE_SUGGESTED_MODIFIERS_RE.search(txt)),
    }

    if d["reset"]:
        return d

    # Clear commands
    if re.search(r"\b(clear|remove|drop)\s+(all\s+)?(covariates|controls)\b", low):
        d["clear_covariates"] = True
    if re.search(r"\b(clear|remove|drop)\s+(all\s+)?(modifiers|effect\s+modifiers)\b", low):
        d["clear_modifiers"] = True

    # Key=value lines
    for line in re.split(r"[\n;]+", txt):
        m = _KV_RE.match(line)
        if not m:
            continue
        key = m.group(1).lower()
        val = m.group(2).strip()

        if key == "treatment":
            d["set_treatment"] = val
        elif key == "outcome":
            d["set_outcome"] = val
        elif key in ("covariates", "controls"):
            vlow = val.lower()
            if "all other" in vlow or "all except" in vlow:
                d["covariate_strategy"] = "ALL_EXCEPT_TY"
                d["set_covariates"] = []
            elif vlow in ("none", "no", "null"):
                d["covariate_strategy"] = "NONE"
                d["set_covariates"] = []
            else:
                d["covariate_strategy"] = "USER_LIST"
                d["set_covariates"] = [x.strip() for x in val.split(",") if x.strip()]
        elif key == "modifiers":
            d["add_modifiers"] = [x.strip() for x in val.split(",") if x.strip()]
        elif key == "question":
            d["set_question"] = val

    # add/remove blocks
    for op, kind, items in _ADD_REMOVE_RE.findall(txt):
        raw_items = [x.strip() for x in items.split(",") if x.strip()]
        op_l = op.lower()
        kind_l = kind.lower()

        if "covariate" in kind_l or "control" in kind_l:
            if op_l == "add":
                d["add_covariates"].extend(raw_items)
            else:
                d["remove_covariates"].extend(raw_items)
        else:
            if op_l == "add":
                d["add_modifiers"].extend(raw_items)
            else:
                d["remove_modifiers"].extend(raw_items)

    # “use suggested …”
    if d["use_suggested_controls"]:
        d["covariate_strategy"] = "USER_LIST"
        d["set_covariates"] = list(proposed.get("controls_candidates", []) or [])
    if d["use_suggested_modifiers"]:
        d["add_modifiers"].extend(list(proposed.get("effect_modifier_candidates", []) or []))

    return d


def _llm_delta(
    llm: LLMService,
    *,
    model_name: str,
    user_text: str,
    columns: Sequence[str],
    proposed: Dict[str, Any],
    draft: Dict[str, Any],
) -> Tuple[Delta, JSONDict | None]:
    """
    LLM parser for messy user messages. Produces a delta (not a full design).
    """
    sys = (
        "You are a strict parser for a causal inference copilot.\n"
        "Convert the user's message into an incremental UPDATE (delta) to metadata.\n\n"
        "Return ONLY one JSON object with EXACTLY these keys:\n"
        "{\n"
        '  "accept": boolean,\n'
        '  "reset": boolean,\n'
        '  "set_treatment": string | null,\n'
        '  "set_outcome": string | null,\n'
        '  "covariate_strategy": "USER_LIST" | "ALL_EXCEPT_TY" | "NONE" | null,\n'
        '  "set_covariates": [string] | null,\n'
        '  "add_covariates": [string],\n'
        '  "remove_covariates": [string],\n'
        '  "add_modifiers": [string],\n'
        '  "remove_modifiers": [string],\n'
        '  "set_question": string | null,\n'
        '  "clear_covariates": boolean,\n'
        '  "clear_modifiers": boolean,\n'
        '  "use_suggested_controls": boolean,\n'
        '  "use_suggested_modifiers": boolean\n'
        "}\n"
        "No markdown. No extra keys. No prose.\n"
        "Rules:\n"
        "- Only choose columns from AllowedColumns.\n"
        "- If user says 'all other columns', use covariate_strategy=ALL_EXCEPT_TY.\n"
        "- If user says 'none', use covariate_strategy=NONE.\n"
        "- If user says 'use suggested controls/modifiers', set those booleans.\n"
        "- If user says 'leave as is' and wants to proceed, set accept=true.\n"
    )

    history = [
        ChatMessage(role="system", content=sys),
        ChatMessage(
            role="user",
            content=(
                "AllowedColumns (JSON array):\n"
                f"{json.dumps(list(columns), ensure_ascii=False)}\n\n"
                "Current draft (json):\n"
                f"{json.dumps(draft, ensure_ascii=False)}\n\n"
                "Proposed candidates (json):\n"
                f"{json.dumps(proposed, ensure_ascii=False)}\n\n"
                "User message:\n"
                f"{user_text}\n"
            ),
        ),
    ]

    cfg = LLMConfig(model=model_name, temperature=0.0)
    raw_out: Optional[str] = None
    try:
        resp = llm.generate(config=cfg, history=history)
        raw_out = resp.content
        obj = _extract_json_object(resp.content)
        return cast(Delta, obj), None  # pyright: ignore[reportUnnecessaryCast]
    except Exception as e:
        try:
            repaired = _repair_json_with_llm(
                llm,
                model_name=model_name,
                broken_text=str(raw_out or ""),
                schema_name="MetadataDelta",
                schema_keys=[
                    "accept",
                    "reset",
                    "set_treatment",
                    "set_outcome",
                    "covariate_strategy",
                    "set_covariates",
                    "add_covariates",
                    "remove_covariates",
                    "add_modifiers",
                    "remove_modifiers",
                    "set_question",
                    "clear_covariates",
                    "clear_modifiers",
                    "use_suggested_controls",
                    "use_suggested_modifiers",
                ],
            )
            return cast(Delta, repaired), {  # pyright: ignore[reportUnnecessaryCast]
                "code": "LLM_METADATA_DELTA_REPAIRED",
                "detail": str(e),
                "raw_llm_output": (str(raw_out)[:1500] if raw_out else None),
            }
        except Exception as e2:
            neutral: Delta = {
                "accept": False,
                "reset": False,
                "set_treatment": None,
                "set_outcome": None,
                "covariate_strategy": None,
                "set_covariates": None,
                "add_covariates": [],
                "remove_covariates": [],
                "add_modifiers": [],
                "remove_modifiers": [],
                "set_question": None,
                "clear_covariates": False,
                "clear_modifiers": False,
                "use_suggested_controls": False,
                "use_suggested_modifiers": False,
            }
            return neutral, {"code": "LLM_METADATA_DELTA_PARSE_FAILED", "detail": f"{e} | repair={e2}"}


def _delta_is_meaningful(d: Delta) -> bool:
    return any(
        [
            bool(d.get("reset")),
            d.get("set_treatment") is not None,
            d.get("set_outcome") is not None,
            d.get("covariate_strategy") is not None,
            d.get("set_covariates") is not None,
            bool(d.get("clear_covariates")),
            bool(d.get("clear_modifiers")),
            bool(d.get("use_suggested_controls")),
            bool(d.get("use_suggested_modifiers")),
            len(cast(List[Any], d.get("add_covariates", []))) > 0,
            len(cast(List[Any], d.get("remove_covariates", []))) > 0,
            len(cast(List[Any], d.get("add_modifiers", []))) > 0,
            len(cast(List[Any], d.get("remove_modifiers", []))) > 0,
            d.get("set_question") is not None,
            bool(d.get("accept")),
        ]
    )


# =============================================================================
# Apply delta (robust semantics)
# =============================================================================
def _default_covariates_all_except_ty(columns: Sequence[str], t: str, y: str) -> List[str]:
    id_like = re.compile(r"(?:^id$|uuid|guid|index|row_id|customer_id|user_id)", re.IGNORECASE)
    out: List[str] = []
    for c in columns:
        if c in (t, y):
            continue
        if id_like.search(c):
            continue
        out.append(c)
    return out


def _apply_delta(
    draft: Dict[str, Any],
    delta: Delta,
    columns: Sequence[str],
    proposed: Dict[str, Any],
) -> Tuple[Dict[str, Any], JSONDict | None]:
    """
    Important semantics:
      - resolve user-provided column names robustly
      - if user edits anything, accept is unset unless they explicitly accept in same message
      - if user edits covariates under ALL_EXCEPT_TY/NONE, we materialize + switch to USER_LIST
    """
    if bool(delta.get("reset")):
        return _default_draft(), None

    err: JSONDict | None = None
    changed_structurally = False

    def res_col(x: Any) -> Optional[str]:
        if not isinstance(x, str):
            return None
        return _resolve_column(x, columns)

    # Clear commands
    if bool(delta.get("clear_covariates")):
        draft["covariates"] = []
        changed_structurally = True

    if bool(delta.get("clear_modifiers")):
        draft["effect_modifiers"] = []
        changed_structurally = True

    # Treatment / outcome
    if delta.get("set_treatment") is not None:
        raw = delta.get("set_treatment")
        t_new = res_col(raw)
        if not t_new:
            err = {"code": "INVALID_TREATMENT", "detail": {"value": raw, "suggest": _suggest_columns(str(raw), columns)}}
        elif draft.get("treatment") != t_new:
            draft["treatment"] = t_new
            changed_structurally = True

    if delta.get("set_outcome") is not None:
        raw = delta.get("set_outcome")
        y_new = res_col(raw)
        if not y_new:
            err = {"code": "INVALID_OUTCOME", "detail": {"value": raw, "suggest": _suggest_columns(str(raw), columns)}}
        elif draft.get("outcome") != y_new:
            draft["outcome"] = y_new
            changed_structurally = True

    # Covariate strategy
    strat = delta.get("covariate_strategy")
    if isinstance(strat, str) and strat in ("USER_LIST", "ALL_EXCEPT_TY", "NONE"):
        if draft.get("covariate_strategy") != strat:
            draft["covariate_strategy"] = strat
            changed_structurally = True

        if strat == "NONE":
            if draft.get("covariates"):
                draft["covariates"] = []
                changed_structurally = True
        elif strat == "ALL_EXCEPT_TY":
            # Materialize later when t/y are known
            draft["covariates"] = []
            changed_structurally = True

    # “use suggested …” (explicit + safe)
    if bool(delta.get("use_suggested_controls")):
        sc = list(proposed.get("controls_candidates", []) or [])
        if sc:
            draft["covariate_strategy"] = "USER_LIST"
            draft["covariates"] = sc
            changed_structurally = True

    if bool(delta.get("use_suggested_modifiers")):
        em = list(proposed.get("effect_modifier_candidates", []) or [])
        if em:
            if "effect_modifiers" not in draft or not isinstance(draft.get("effect_modifiers"), list):
                draft["effect_modifiers"] = []
            for c in em:
                if c not in cast(List[str], draft["effect_modifiers"]):
                    cast(List[str], draft["effect_modifiers"]).append(c)
            changed_structurally = True

    # set covariates list (override)
    if isinstance(delta.get("set_covariates"), list):
        cleaned: List[str] = []
        for v in cast(List[Any], delta["set_covariates"]):
            if isinstance(v, str):
                r = _resolve_column(v, columns)
                if r and r not in cleaned:
                    cleaned.append(r)
        draft["covariate_strategy"] = "USER_LIST"
        draft["covariates"] = cleaned
        changed_structurally = True

    # add/remove covariates under ALL_EXCEPT_TY/NONE -> materialize and switch to USER_LIST
    add_covs = [x for x in cast(List[Any], delta.get("add_covariates", [])) if isinstance(x, str)]
    rem_covs = [x for x in cast(List[Any], delta.get("remove_covariates", [])) if isinstance(x, str)]
    if (add_covs or rem_covs) and draft.get("covariate_strategy") in ("ALL_EXCEPT_TY", "NONE"):
        t0 = draft.get("treatment")
        y0 = draft.get("outcome")
        if draft.get("covariate_strategy") == "ALL_EXCEPT_TY" and isinstance(t0, str) and isinstance(y0, str):
            draft["covariates"] = _default_covariates_all_except_ty(columns, t=t0, y=y0)
        else:
            draft["covariates"] = list(draft.get("covariates", []) or [])
        draft["covariate_strategy"] = "USER_LIST"
        changed_structurally = True

    # add covariates
    for v in add_covs:
        r = _resolve_column(v, columns)
        if r and r not in cast(List[str], draft.get("covariates", [])):
            cast(List[str], draft["covariates"]).append(r)
            changed_structurally = True

    # remove covariates
    for v in rem_covs:
        r = _resolve_column(v, columns)
        if r and r in cast(List[str], draft.get("covariates", [])):
            cast(List[str], draft["covariates"]).remove(r)
            changed_structurally = True

    # modifiers: ensure list exists
    if "effect_modifiers" not in draft or not isinstance(draft.get("effect_modifiers"), list):
        draft["effect_modifiers"] = []

    add_mods = [x for x in cast(List[Any], delta.get("add_modifiers", [])) if isinstance(x, str)]
    rem_mods = [x for x in cast(List[Any], delta.get("remove_modifiers", [])) if isinstance(x, str)]

    for v in add_mods:
        r = _resolve_column(v, columns)
        if r and r not in cast(List[str], draft["effect_modifiers"]):
            cast(List[str], draft["effect_modifiers"]).append(r)
            changed_structurally = True

    for v in rem_mods:
        r = _resolve_column(v, columns)
        if r and r in cast(List[str], draft["effect_modifiers"]):
            cast(List[str], draft["effect_modifiers"]).remove(r)
            changed_structurally = True

    # causal question
    q = delta.get("set_question")
    if isinstance(q, str):
        q2 = q.strip()
        if q2 and draft.get("causal_question") != q2:
            draft["causal_question"] = q2
            changed_structurally = True

    # enforce: covariates/modifiers cannot include t/y
    t = draft.get("treatment")
    y = draft.get("outcome")
    if isinstance(t, str) and isinstance(y, str):
        draft["covariates"] = [c for c in cast(List[str], draft.get("covariates", [])) if c not in (t, y)]
        draft["effect_modifiers"] = [
            c for c in cast(List[str], draft.get("effect_modifiers", [])) if c not in (t, y)
        ]

    # accept semantics
    if changed_structurally and not bool(delta.get("accept", False)):
        draft["accept"] = False
    if bool(delta.get("accept", False)):
        draft["accept"] = True

    return draft, err


# =============================================================================
# Prompt rendering (deterministic fallback)
# =============================================================================
def _render_prompt(
    columns: Sequence[str],
    proposed: Dict[str, Any],
    draft: Dict[str, Any],
    *,
    last_error: JSONDict | None,
) -> str:
    cols_preview = ", ".join(list(columns)[:18]) + (" ..." if len(columns) > 18 else "")

    t = draft.get("treatment")
    y = draft.get("outcome")
    strat = draft.get("covariate_strategy")
    covs = cast(List[str], draft.get("covariates", []) or [])
    mods = cast(List[str], draft.get("effect_modifiers", []) or [])
    q = draft.get("causal_question")
    accept = bool(draft.get("accept", False))

    msg = "🧩 Metadata setup\n\n"
    msg += "Current:\n"
    msg += f"- treatment: {t}\n"
    msg += f"- outcome: {y}\n"
    msg += f"- covariate_strategy: {strat}\n"
    msg += f"- covariates: {covs[:12]}{' ...' if len(covs) > 12 else ''}\n"
    msg += f"- effect_modifiers: {mods[:12]}{' ...' if len(mods) > 12 else ''}\n"
    if q:
        msg += f"- question: {q}\n"
    msg += f"- accepted: {accept}\n"

    if last_error:
        msg += "\n⚠️ Issue:\n"
        msg += f"- {last_error.get('code')}: {last_error.get('detail')}\n"

    if not isinstance(t, str) or not isinstance(y, str):
        msg += "\nPick treatment & outcome.\n"
        msg += f"Suggested treatment: {proposed.get('treatment_candidates', [])[:6]}\n"
        msg += f"Suggested outcome: {proposed.get('outcome_candidates', [])[:6]}\n"
        msg += "Reply like: treatment=<col>, outcome=<col>\n"
    elif strat is None:
        msg += "\nChoose covariates strategy:\n"
        msg += "- covariates=all other columns\n"
        msg += "- covariates=none\n"
        msg += "- covariates=a,b,c\n"
        msg += "Tip: you can also say 'use suggested controls'.\n"
    elif strat == "ALL_EXCEPT_TY":
        msg += "\nCovariates are set to: all other columns (excluding id-like + treatment/outcome).\n"
        msg += "If you want to exclude something, say: remove covariates: colA,colB\n"
    elif strat == "USER_LIST" and len(covs) == 0:
        msg += "\nYou chose USER_LIST but no covariates were provided.\n"
        msg += "Reply with covariates=a,b,c OR say 'covariates=all other columns' OR 'covariates=none'\n"
    else:
        msg += "\nIf this looks good, reply with 'ok' / 'proceed'.\n"
        msg += "You can still update, e.g.:\n"
        msg += "- add covariates: a,b\n"
        msg += "- remove modifiers: c\n"
        msg += "- clear covariates\n"
        msg += "- reset\n"

    msg += f"\nColumns preview: {cols_preview}"
    return msg


def _render_final_message(final_design: Dict[str, Any]) -> str:
    covs = cast(List[str], final_design.get("covariates", []) or [])
    mods = cast(List[str], final_design.get("effect_modifiers", []) or [])
    q = final_design.get("causal_question")

    msg = "✅ Confirmed metadata.\n"
    msg += f"- treatment: {final_design.get('treatment')}\n"
    msg += f"- outcome: {final_design.get('outcome')}\n"
    msg += f"- covariate_strategy: {final_design.get('covariate_strategy')}\n"
    msg += f"- covariates: {covs[:12]}{' ...' if len(covs) > 12 else ''}\n"
    msg += f"- effect_modifiers: {mods[:12]}{' ...' if len(mods) > 12 else ''}\n"
    if q:
        msg += f"- question: {q}\n"
    return msg


# =============================================================================
# Node-level presenter (LLM) — used ONLY at PRESENT boundary
# =============================================================================
_METADATA_NODE_SYSTEM_PROMPT = (
    "You are the user-facing voice of the metadata intake node in a causal inference copilot.\n"
    "You will get recent conversation + a compact state snapshot + a node-provided draft message.\n\n"
    "Write EXACTLY ONE message to the user.\n"
    "Rules:\n"
    "- Be concise, concrete, and actionable.\n"
    "- Do NOT mention internal field names, JSON, or implementation details.\n"
    "- If there is an error: explain it simply and state the next step.\n"
    "- If the user needs to pick columns: show a short helpful list of column names.\n"
    "- If the design is complete but not accepted: ask for explicit confirmation ('ok').\n"
    "- Prefer bullet points over long paragraphs.\n"
)

def _compact_list(x: Any, n: int) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for v in x[:n]:  # pyright: ignore[reportUnknownVariableType]
        s = str(v).strip()  # pyright: ignore[reportUnknownArgumentType]
        if s:
            out.append(s)
    return out

def _compact_proposed_for_prompt(proposed: Any) -> Dict[str, Any]:
    if not isinstance(proposed, dict):
        return {}
    return {
        "treatment_candidates": _compact_list(proposed.get("treatment_candidates"), 8),
        "outcome_candidates": _compact_list(proposed.get("outcome_candidates"), 8),
        "controls_candidates": _compact_list(proposed.get("controls_candidates"), 12),
        "effect_modifier_candidates": _compact_list(proposed.get("effect_modifier_candidates"), 12),
        "questions_for_user": _compact_list(proposed.get("questions_for_user"), 6),
    }

def _build_metadata_node_message(
    llm: LLMService,
    *,
    state: ConversationState,
    fallback: str,
    intent: str,
    model_name: str,
    temperature: float = 0.4,
    history_window: int = 10,
    max_error_chars: int = 1200,
) -> str:
    """
    Presentation layer for this node:
      - Call ONLY when the node is about to PRESENT.
      - Never raises; returns fallback on failures.
      - Produces exactly one user-facing string.
    """
    try:
        control = cast(Dict[str, Any], state.get("control", {}))
        dataset = cast(Dict[str, Any], state.get("dataset", {}))
        metadata = cast(Dict[str, Any], state.get("metadata", {}))

        last_error = control.get("last_error")
        if isinstance(last_error, dict):
            last_error = {
                "code": last_error.get("code"),
                "detail": str(last_error.get("detail", ""))[:max_error_chars],
            }

        cols = _columns_from_raw_schema(dataset.get("raw_schema"))

        payload: Dict[str, Any] = {
            "intent": intent,
            "stage": control.get("stage"),
            "post_action": control.get("post_action"),
            "status": control.get("status"),
            "last_error": last_error,
            "dataset": {
                "path": dataset.get("path"),
                "id": str(dataset.get("id")) if dataset.get("id") is not None else None,
                "summary": dataset.get("summary"),
                "columns_preview": cols[:24],
            },
            "proposed_preview": _compact_proposed_for_prompt(metadata.get("proposed_design")),
            "draft": metadata.get("draft"),
            "final_design": metadata.get("final_design"),
            "node_default_message": fallback[:2000],
        }

        prior: Sequence[BaseMessage] = cast(Sequence[BaseMessage], state.get("messages", []))
        tail = list(prior)[-history_window:] if isinstance(prior, list) else []

        history: List[ChatMessage] = [ChatMessage(role="system", content=_METADATA_NODE_SYSTEM_PROMPT)]
        for m in tail:
            history.append(
                ChatMessage(
                    role=cast(Any, _role_from_langchain_msg(m)),
                    content=str(getattr(m, "content", "")),
                )
            )

        user_prompt = (
            "You are a causal inference copilot.\n"
            "A workflow node is about to present a message to the user.\n\n"
            "Node draft/default message:\n"
            f"{fallback.strip()}\n\n"
            "State snapshot:\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
            "Follow the rules and write EXACTLY ONE final user-facing message."
        )
        history.append(ChatMessage(role="user", content=user_prompt))

        cfg = LLMConfig(model=model_name, temperature=temperature)
        resp = llm.generate(config=cfg, history=history)
        txt = resp.content.strip()
        return txt if txt else fallback
    except Exception:
        return fallback


# =============================================================================
# Node
# =============================================================================
def make_propose_and_confirm_metadata(
    llm: LLMService,
    data_repo: DataRepo,
    *,
    model_name: str = DEFAULT_MODEL_GEMNI,
    history_window: int = 12,
    sample_rows: int = 80,
    max_sample_chars: int = 10_000,
    force_covariates_decision: bool = True,
) -> Callable[[ConversationState], ConversationState]:
    """
    Stage-aware node:
      - PROPOSE_METADATA: generate proposed_design once and prompt user for inputs.
      - CONFIRM_METADATA: apply deltas from user messages until design is complete AND accepted.

    ControlState semantics:
      - post_action drives UI/runtime behavior
      - post_failure_suggested_stage is only meaningful on ABORTED
    """

    def node(state: ConversationState) -> ConversationState:
        control_in = _require_control(state)
        dataset_in = _as_dataset(state)
        metadata_in = _as_metadata(state)

        conversation_id = control_in["conversation_id"]
        stage: Stage = control_in["stage"]

        def mk_control(
            *,
            status: Status,
            post_action: ACTION,
            post_failure_suggested_stage: NEED_STAGE | None,
            node_message: str,
            last_error: JSONDict | None,
            pending_stage: Stage | None = None,
        ) -> ControlState:
            out: ControlState = cast(
                ControlState,
                {
                    **control_in,
                    "conversation_id": conversation_id,
                    "stage": stage,
                    "status": status,
                    "post_action": post_action,
                    "post_failure_suggested_stage": post_failure_suggested_stage,
                    "last_error": last_error,
                    "node_message": node_message,
                },
            )
            out["pending_stage"] = pending_stage
            return out

        # ---- dataset guardrails
        dataset_id = dataset_in.get("id")
        raw_schema = dataset_in.get("raw_schema")
        summary = dataset_in.get("summary")

        if not isinstance(dataset_id, UUID) or not isinstance(raw_schema, dict) or not isinstance(summary, dict):
            fallback = "Dataset is not loaded yet. Please load the dataset first."
            tmp_control = mk_control(
                status="ABORTED",
                post_action="PRESENT",
                post_failure_suggested_stage="LOAD_DATASET",
                last_error={
                    "code": "MISSING_DATASET",
                    "detail": "dataset.id/raw_schema/summary missing; run LOAD_DATASET first.",
                },
                node_message=fallback,
            )
            msg = _build_metadata_node_message(
                llm,
                state={**state, "control": tmp_control},
                fallback=fallback,
                intent="metadata_fatal_missing_dataset",
                model_name=model_name,
            )
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="LOAD_DATASET",
                    last_error={
                        "code": "MISSING_DATASET",
                        "detail": "dataset.id/raw_schema/summary missing; run LOAD_DATASET first.",
                    },
                    node_message=msg,
                ),
            }

        columns = _columns_from_raw_schema(raw_schema)
        if not columns:
            fallback = "Dataset schema looks empty. Please reload the dataset."
            tmp_control = mk_control(
                status="ABORTED",
                post_action="PRESENT",
                post_failure_suggested_stage="LOAD_DATASET",
                last_error={"code": "MISSING_SCHEMA", "detail": "dataset.raw_schema has no columns."},
                node_message=fallback,
            )
            msg = _build_metadata_node_message(
                llm,
                state={**state, "control": tmp_control},
                fallback=fallback,
                intent="metadata_fatal_missing_schema",
                model_name=model_name,
            )
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="LOAD_DATASET",
                    last_error={"code": "MISSING_SCHEMA", "detail": "dataset.raw_schema has no columns."},
                    node_message=msg,
                ),
            }

        messages: Sequence[BaseMessage] = cast(Sequence[BaseMessage], state.get("messages", []))

        # ---- metadata init (stable defaults)
        warnings = metadata_in.get("warnings")
        warnings = warnings if isinstance(warnings, list) else []

        proposed = metadata_in.get("proposed_design")
        draft = metadata_in.get("draft")
        final_design = metadata_in.get("final_design")

        last_idx_seen = metadata_in.get("last_user_msg_idx", -1)
        last_idx_seen = last_idx_seen if isinstance(last_idx_seen, int) else -1

        if not isinstance(draft, dict):
            draft = _default_draft()

        # ---- PROPOSE_METADATA
        if stage == "PROPOSE_METADATA":
            if not isinstance(proposed, dict):
                proposed_obj, err = _propose_design_once(
                    llm,
                    data_repo,
                    dataset_id=dataset_id,
                    raw_schema=raw_schema,
                    summary=summary,
                    columns=columns,
                    messages=messages,
                    model_name=model_name,
                    history_window=history_window,
                    sample_rows=sample_rows,
                    max_sample_chars=max_sample_chars,
                )
                proposed = proposed_obj
            else:
                err = None

            if not isinstance(draft, dict) or not draft:
                draft = _default_draft()

            fallback = (
                "🧠 Draft causal design proposal (auto)\n"
                f"- rows={summary.get('n_rows')} cols={summary.get('n_cols')}\n\n"
                f"Treatment candidates: {cast(List[str], proposed.get('treatment_candidates', []))[:8]}\n"
                f"Outcome candidates: {cast(List[str], proposed.get('outcome_candidates', []))[:8]}\n"
                f"Controls candidates: {cast(List[str], proposed.get('controls_candidates', []))[:10]}\n"
                f"Effect modifiers: {cast(List[str], proposed.get('effect_modifier_candidates', []))[:10]}\n\n"
                "Reply like:\n"
                "treatment=<col>\n"
                "outcome=<col>\n"
                "covariates=all other columns | none | a,b,c\n"
                "modifiers=a,b (optional)\n"
                "question=... (optional)\n"
                "You can also say: use suggested controls\n"
            )

            # Critical: advance last_user_msg_idx to avoid re-parsing old messages (e.g. dataset path)
            last_seen_now = _last_human_msg_idx(messages)

            metadata_out: MetadataState = {
                "proposed_design": cast(Any, proposed),
                "draft": cast(Any, draft),
                "final_design": None,
                "last_user_msg_idx": last_seen_now,
                "canonical_metadata": metadata_in.get("canonical_metadata"),
                "warnings": warnings,
            }

            tmp_control = mk_control(
                status="DONE",
                post_action="PRESENT_AND_USER_INPUT",
                post_failure_suggested_stage=None,
                last_error=err,
                node_message=fallback,
            )
            msg = _build_metadata_node_message(
                llm,
                state={**state, "control": tmp_control, "metadata": metadata_out},
                fallback=fallback,
                intent="metadata_propose",
                model_name=model_name,
            )

            return {
                **state,
                "control": mk_control(
                    status="DONE",
                    post_action="PRESENT_AND_USER_INPUT",
                    post_failure_suggested_stage=None,
                    last_error=err,
                    node_message=msg,
                ),
                "metadata": metadata_out,
            }

        # ---- only CONFIRM_METADATA supported beyond this point
        if stage != "CONFIRM_METADATA":
            fallback = f"Metadata node was called in an unexpected stage: {stage}"
            tmp_control = mk_control(
                status="ABORTED",
                post_action="PRESENT",
                post_failure_suggested_stage=None,
                last_error={"code": "WRONG_STAGE", "detail": f"metadata_intake called in stage={stage}"},
                node_message=fallback,
            )
            msg = _build_metadata_node_message(
                llm,
                state={**state, "control": tmp_control},
                fallback=fallback,
                intent="metadata_fatal_wrong_stage",
                model_name=model_name,
            )
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage=None,
                    last_error={"code": "WRONG_STAGE", "detail": f"metadata_intake called in stage={stage}"},
                    node_message=msg,
                ),
            }

        # If CONFIRM is entered without prior PROPOSE, skip all existing human messages and prompt fresh.
        if last_idx_seen < 0:
            last_idx_seen = _last_human_msg_idx(messages)

        # Ensure we have a proposal even in CONFIRM (robust to unusual routing)
        if not isinstance(proposed, dict):
            proposed_obj, err = _propose_design_once(
                llm,
                data_repo,
                dataset_id=dataset_id,
                raw_schema=raw_schema,
                summary=summary,
                columns=columns,
                messages=messages,
                model_name=model_name,
                history_window=history_window,
                sample_rows=sample_rows,
                max_sample_chars=max_sample_chars,
            )
            proposed = proposed_obj
        else:
            err = None

        # If already finalized: show final unless user sent new change messages
        if isinstance(final_design, dict) and final_design.get("accepted") is True:
            any_new_human = any(True for _i, _txt in _iter_new_humans(messages, last_idx_seen))
            if not any_new_human:
                metadata_out: MetadataState = {
                    "proposed_design": cast(Any, proposed),
                    "draft": cast(Any, _draft_from_final(final_design)),  # pyright: ignore[reportArgumentType]
                    "final_design": cast(Any, final_design),
                    "last_user_msg_idx": last_idx_seen,
                    "canonical_metadata": metadata_in.get("canonical_metadata"),
                    "warnings": warnings,
                }
                fallback = _render_final_message(final_design)  # pyright: ignore[reportArgumentType]
                tmp_control = mk_control(
                    status="DONE",
                    post_action="PRESENT",
                    post_failure_suggested_stage=None,
                    last_error=None,
                    node_message=fallback,
                )
                msg = _build_metadata_node_message(
                    llm,
                    state={**state, "control": tmp_control, "metadata": metadata_out},
                    fallback=fallback,
                    intent="metadata_final_repeat",
                    model_name=model_name,
                )
                return {
                    **state,
                    "control": mk_control(
                        status="DONE",
                        post_action="PRESENT",
                        post_failure_suggested_stage=None,
                        last_error=None,
                        node_message=msg,
                    ),
                    "metadata": metadata_out,
                }

            # Reopen for edits
            draft = _draft_from_final(final_design)  # pyright: ignore[reportArgumentType]
            final_design = None

        # Apply all new human messages since last_idx_seen
        last_error: JSONDict | None = err
        newest_idx_seen = last_idx_seen

        for idx, user_text in _iter_new_humans(messages, last_idx_seen):
            newest_idx_seen = max(newest_idx_seen, idx)

            # 1) heuristic parse first (fast)
            delta = _heuristic_delta(user_text, cast(Dict[str, Any], proposed))

            # 2) if heuristic yields no signal, use LLM delta parse
            if not _delta_is_meaningful(delta):
                delta_llm, parse_err = _llm_delta(
                    llm,
                    model_name=model_name,
                    user_text=user_text,
                    columns=columns,
                    proposed=cast(Dict[str, Any], proposed),
                    draft=cast(Dict[str, Any], draft),
                )
                if _delta_is_meaningful(delta_llm):
                    delta = delta_llm
                if parse_err:
                    last_error = parse_err

            # Apply delta
            draft, apply_err = _apply_delta(cast(Dict[str, Any], draft), delta, columns, cast(Dict[str, Any], proposed))
            if apply_err:
                last_error = apply_err

            # If ALL_EXCEPT_TY and t/y known: materialize now so downstream always sees a concrete list.
            if draft.get("covariate_strategy") == "ALL_EXCEPT_TY":
                t0 = draft.get("treatment")
                y0 = draft.get("outcome")
                if isinstance(t0, str) and isinstance(y0, str):
                    draft["covariates"] = _default_covariates_all_except_ty(columns, t=t0, y=y0)

        # Completeness checks
        t = draft.get("treatment")
        y = draft.get("outcome")
        strat = draft.get("covariate_strategy")
        covs = cast(List[str], draft.get("covariates", []) or [])

        missing_ty = not isinstance(t, str) or not isinstance(y, str)
        missing_strat = strat is None
        missing_covs_for_user_list = (
            force_covariates_decision and (not missing_ty) and strat == "USER_LIST" and len(covs) == 0
        )

        can_finalize = (not missing_ty) and (not missing_strat) and (not missing_covs_for_user_list)
        user_accepted = bool(draft.get("accept", False))

        if not can_finalize or not user_accepted:
            fallback = _render_prompt(
                columns,
                cast(Dict[str, Any], proposed),
                cast(Dict[str, Any], draft),
                last_error=last_error,
            )

            metadata_out: MetadataState = {
                "proposed_design": cast(Any, proposed),
                "draft": cast(Any, draft),
                "final_design": None,
                "last_user_msg_idx": newest_idx_seen,
                "canonical_metadata": metadata_in.get("canonical_metadata"),
                "warnings": warnings,
            }

            tmp_control = mk_control(
                status="PENDING",
                post_action="PRESENT_AND_USER_INPUT",
                post_failure_suggested_stage=None,
                last_error=last_error,
                node_message=fallback,
            )
            msg = _build_metadata_node_message(
                llm,
                state={**state, "control": tmp_control, "metadata": metadata_out},
                fallback=fallback,
                intent="metadata_confirm",
                model_name=model_name,
            )

            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    post_action="PRESENT_AND_USER_INPUT",
                    post_failure_suggested_stage=None,
                    last_error=last_error,
                    node_message=msg,
                ),
                "metadata": metadata_out,
            }

        # Final design (locked)
        final_design_out: Dict[str, Any] = {
            "treatment": cast(str, t),
            "outcome": cast(str, y),
            "covariate_strategy": cast(str, strat),
            "covariates": covs,
            "effect_modifiers": cast(List[str], draft.get("effect_modifiers", []) or []),
            "causal_question": draft.get("causal_question"),
            "accepted": True,
        }

        metadata_out_final: MetadataState = {
            "proposed_design": cast(Any, proposed),
            "draft": cast(Any, draft),
            "final_design": cast(Any, final_design_out),
            "last_user_msg_idx": newest_idx_seen,
            "canonical_metadata": metadata_in.get("canonical_metadata"),
            "warnings": warnings,
        }

        fallback = _render_final_message(final_design_out)
        tmp_control = mk_control(
            status="DONE",
            post_action="PRESENT",
            post_failure_suggested_stage=None,
            last_error=last_error,
            node_message=fallback,
        )
        msg = _build_metadata_node_message(
            llm,
            state={**state, "control": tmp_control, "metadata": metadata_out_final},
            fallback=fallback,
            intent="metadata_final",
            model_name=model_name,
        )

        return {
            **state,
            "control": mk_control(
                status="DONE",
                post_action="PRESENT",
                post_failure_suggested_stage=None,
                last_error=last_error,
                node_message=msg,
            ),
            "metadata": metadata_out_final,
        }

    return node
