# src/python/workflows/nodes/confirm_metadata.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, cast
import json
import re

from langchain_core.messages import BaseMessage

from python.domain.service.llm_service import LLMService, LLMConfig, ChatMessage
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState, JSONDict, Need, Outcome, Status
from python.workflows.state.dataset_state import DatasetState
from python.workflows.state.metadata_state import MetadataState

JSONValue = Any
JSONDictLocal = Dict[str, JSONValue]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _require_control(state: ConversationState) -> ControlState:
    # invariant: start node sets this
    return cast(ControlState, state["control"])  # pyright: ignore[reportUnnecessaryCast, reportTypedDictNotRequiredAccess]


def _as_dataset(state: ConversationState) -> DatasetState:
    return cast(DatasetState, state.get("dataset", {})) # type: ignore


def _as_metadata(state: ConversationState) -> MetadataState:
    return cast(MetadataState, state.get("metadata", {})) # pyright: ignore[reportUnnecessaryCast]


def _role_from_langchain_msg(m: BaseMessage) -> str:
    t = getattr(m, "type", None)
    return {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(str(t), "user")


def _last_user_text(messages: Sequence[BaseMessage]) -> str | None:
    for m in reversed(list(messages)):
        if getattr(m, "type", None) == "human":
            txt = str(getattr(m, "content", "")).strip()
            return txt or None
    return None


def _extract_json_object(text: str) -> JSONDictLocal:
    s = text.strip()

    m = _JSON_FENCE_RE.search(s)
    if m:
        obj = json.loads(m.group(1))
        if isinstance(obj, dict):
            return cast(JSONDictLocal, obj)

    try:
        obj2 = json.loads(s)
        if isinstance(obj2, dict):
            return cast(JSONDictLocal, obj2)
    except Exception:
        pass

    m2 = _JSON_OBJ_RE.search(s)
    if m2:
        obj3 = json.loads(m2.group(1))
        if isinstance(obj3, dict):
            return cast(JSONDictLocal, obj3)

    raise ValueError("LLM did not return a valid JSON object.")


def _columns_from_raw_schema(raw_schema: Any) -> List[str]:
    if not isinstance(raw_schema, dict):
        return []
    cols = raw_schema.get("columns") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if not isinstance(cols, list):
        return []
    out: List[str] = []
    for c in cols: # pyright: ignore[reportUnknownVariableType]
        if isinstance(c, dict):
            name = c.get("name") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
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
    key = _norm(user_col)
    return normed.get(key)


def _parse_optional_str_list(x: Any) -> List[str] | None:
    """
    Returns:
      - None if user didn't specify (null / missing / empty string)
      - [] if user explicitly said empty list
      - ["a","b"] if provided
    Accepts a list or a comma/semicolon-separated string as a robustness fallback.
    """
    if x is None:
        return None
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip() != ""] # type: ignore
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        parts = re.split(r"[,\n;]+", s)
        vals = [p.strip() for p in parts if p.strip()]
        return vals
    return None


def _dedupe_keep_order(xs: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _resolve_columns_list(user_cols: Sequence[str], columns: Sequence[str]) -> tuple[List[str], List[str]]:
    resolved: List[str] = []
    unresolved: List[str] = []
    for raw in user_cols:
        c = _resolve_column(raw, columns)
        if c is None:
            unresolved.append(raw)
        else:
            resolved.append(c)
    return _dedupe_keep_order(resolved), unresolved


def make_confirm_metadata_node(
    llm: LLMService,
    *,
    model_name: str = "gemini-1.5-flash",
    history_window: int = 12,
) -> Callable[[ConversationState], ConversationState]:
    """
    CONFIRM_METADATA node (T, Y, W, X).

    Extract from user:
      - treatment_column (T)
      - outcome_column (Y)
      - controls_columns (W)          [recommended]
      - effect_modifier_columns (X)   [optional]
      - causal_question (optional)
      - accept (bool)

    Validates all provided columns exist in dataset.raw_schema.

    Does NOT mutate control.stage (router owns transitions).
    Does NOT emit user-facing messages (presenter owns that).
    """

    def confirm_metadata(state: ConversationState) -> ConversationState:
        control_in = _require_control(state)
        dataset_in = _as_dataset(state)
        metadata_in = _as_metadata(state)

        conversation_id = control_in["conversation_id"]
        stage = control_in["stage"]  # invariant: router owns transitions

        def mk_control(
            *,
            status: Status,
            outcome: Outcome,
            need: Need,
            node_message: str,
            last_error: JSONDict | None,
            interrupt_type: str | None,
        ) -> ControlState:
            return {
                "conversation_id": conversation_id,
                "status": status,
                "stage": stage,
                "outcome": outcome,
                "need": need,
                "interrupt_type": interrupt_type,
                "last_error": last_error,
                "node_message": node_message,
            }

        raw_schema = dataset_in.get("raw_schema")
        columns = _columns_from_raw_schema(raw_schema)
        if not columns:
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    outcome="NEEDS_INPUT",
                    need="DATASET_PATH",
                    interrupt_type=None,
                    last_error={"code": "MISSING_SCHEMA", "detail": "dataset.raw_schema missing; run LOAD_DATASET first."},
                    node_message="Dataset schema is missing. Reload dataset so I can validate T/Y/W/X.",
                ),
            }

        prior_msgs: Sequence[BaseMessage] = cast(Sequence[BaseMessage], state.get("messages", []))
        user_text = _last_user_text(prior_msgs)
        if not user_text:
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    outcome="NEEDS_INPUT",
                    need="CONFIRM_METADATA",
                    interrupt_type="REVIEW_METADATA",
                    last_error={"code": "NO_USER_INPUT", "detail": "No recent user message found to confirm metadata."},
                    node_message="Tell me treatment (T), outcome (Y), and (ideally) controls/confounders (W).",
                ),
            }

        proposed = metadata_in.get("proposed_design")
        proposed_dict = proposed if isinstance(proposed, dict) else {}
        proposed_json = json.dumps(proposed_dict, ensure_ascii=False)
        cols_json = json.dumps(columns, ensure_ascii=False)

        # --- strict parse prompt ---
        sys = (
            "You are a strict parser for a causal inference copilot.\n"
            "Extract the user's confirmation/corrections for a causal design.\n\n"
            "Return ONLY one JSON object with EXACTLY these keys:\n"
            "{\n"
            '  "accept": boolean,\n'
            '  "treatment_column": string | null,\n'
            '  "outcome_column": string | null,\n'
            '  "controls_columns": [string] | null,\n'
            '  "effect_modifier_columns": [string] | null,\n'
            '  "causal_question": string | null\n'
            "}\n"
            "No markdown. No extra keys. No prose.\n"
            "If user did not specify something, use null.\n"
            "IMPORTANT: Only use column names from AllowedColumns.\n"
        )

        tail = list(prior_msgs)[-history_window:] if isinstance(prior_msgs, list) else []
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
                    "AllowedColumns (array):\n"
                    f"{cols_json}\n\n"
                    "Current proposed design (json):\n"
                    f"{proposed_json}\n\n"
                    "User message to parse:\n"
                    f"{user_text}\n"
                ),
            )
        )

        config = LLMConfig(model=model_name, temperature=0.0, max_tokens=500)

        parsed: JSONDictLocal
        parse_error: JSONDict | None = None
        try:
            resp = llm.generate(config=config, history=llm_history)
            parsed = _extract_json_object(resp.content)
        except Exception as e:
            parse_error = {"code": "LLM_CONFIRM_PARSE_FAILED", "detail": str(e)}
            parsed = {
                "accept": False,
                "treatment_column": None,
                "outcome_column": None,
                "controls_columns": None,
                "effect_modifier_columns": None,
                "causal_question": None,
            }

        accept = bool(parsed.get("accept", False))
        t_raw = parsed.get("treatment_column")
        y_raw = parsed.get("outcome_column")
        w_raw = parsed.get("controls_columns")
        x_raw = parsed.get("effect_modifier_columns")
        q_raw = parsed.get("causal_question")

        t = _resolve_column(str(t_raw), columns) if isinstance(t_raw, str) else None
        y = _resolve_column(str(y_raw), columns) if isinstance(y_raw, str) else None
        q = str(q_raw).strip() if isinstance(q_raw, str) and q_raw.strip() else None

        # --- required: T and Y ---
        if not t or not y:
            detail: JSONDict = {
                "code": "MISSING_OR_INVALID_T_Y",
                "detail": {
                    "parsed": {"treatment_column": t_raw, "outcome_column": y_raw},
                    "available_columns_count": len(columns),
                },
            }
            outcome: Outcome = "RETRYABLE_ERROR" if parse_error is not None else "NEEDS_INPUT"
            last_error: JSONDict | None = parse_error or detail
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    outcome=outcome,
                    need="TREATMENT_OUTCOME",
                    interrupt_type="REVIEW_METADATA",
                    last_error=last_error,
                    node_message="I still need valid treatment (T) and outcome (Y) column names from your dataset.",
                ),
                "metadata": {**metadata_in, "user_accepts": False},
            }

        w_list = _parse_optional_str_list(w_raw)
        x_list = _parse_optional_str_list(x_raw)

        proposed_w = proposed_dict.get("controls_candidates")
        proposed_x = proposed_dict.get("effect_modifier_candidates")

        proposed_w_list = [str(v) for v in proposed_w] if isinstance(proposed_w, list) else [] # type: ignore
        proposed_x_list = [str(v) for v in proposed_x] if isinstance(proposed_x, list) else [] # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]

        w_source: str
        x_source: str

        if w_list is None:
            w_source = "proposal"
            w_resolved, _ = _resolve_columns_list(proposed_w_list, columns)
        else:
            w_source = "user"
            w_resolved, w_unresolved = _resolve_columns_list(w_list, columns)
            if w_list and not w_resolved:
                # user tried to provide W but none matched -> ask again
                detail = {
                    "code": "INVALID_CONTROLS",
                    "detail": {"parsed": w_list, "unresolved": w_unresolved, "available_columns_count": len(columns)},
                }
                outcome = "RETRYABLE_ERROR" if parse_error is not None else "NEEDS_INPUT"
                return {
                    **state,
                    "control": mk_control(
                        status="PENDING",
                        outcome=outcome,
                        need="CONFIRM_METADATA",
                        interrupt_type="REVIEW_METADATA",
                        last_error=parse_error or detail,
                        node_message="Your controls/confounders (W) didn’t match dataset columns. Please copy-paste column names.",
                    ),
                    "metadata": {**metadata_in, "user_accepts": False},
                }

        if x_list is None:
            x_source = "proposal"
            x_resolved, _ = _resolve_columns_list(proposed_x_list, columns)
        else:
            x_source = "user"
            x_resolved, x_unresolved = _resolve_columns_list(x_list, columns)
            if x_list and not x_resolved:
                detail = {
                    "code": "INVALID_EFFECT_MODIFIERS",
                    "detail": {"parsed": x_list, "unresolved": x_unresolved, "available_columns_count": len(columns)},
                }
                outcome = "RETRYABLE_ERROR" if parse_error is not None else "NEEDS_INPUT"
                return {
                    **state,
                    "control": mk_control(
                        status="PENDING",
                        outcome=outcome,
                        need="CONFIRM_METADATA",
                        interrupt_type="REVIEW_METADATA",
                        last_error=parse_error or detail,
                        node_message="Your effect modifier columns (X) didn’t match dataset columns. Please copy-paste column names.",
                    ),
                    "metadata": {**metadata_in, "user_accepts": False},
                }

        # sanity: W/X should not include T/Y
        w_resolved = [c for c in w_resolved if c not in (t, y)]
        x_resolved = [c for c in x_resolved if c not in (t, y)]

        final_design: JSONDictLocal = {
            "treatment": {"column": t},
            "outcome": {"column": y},
            "controls": {"columns": w_resolved, "source": w_source},            # W
            "effect_modifiers": {"columns": x_resolved, "source": x_source},    # X
            "causal_question": q,
            "accepted_by_user": accept,
        }

        metadata_out: MetadataState = {
            **metadata_in,
            "final_design": final_design,
            "treatment_hint": t,
            "outcome_hint": y,
            "controls_hint": w_resolved,       
            "effect_modifiers_hint": x_resolved,  
            "user_accepts": True,                 
        }

        return {
            **state,
            "control": mk_control(
                status="OK",
                outcome="DONE",
                need="NONE",
                interrupt_type=None,
                last_error=parse_error,  # keep if you want observability; or drop it
                node_message="Confirmed causal design (T/Y/W/X). Ready to move to estimator selection.",
            ),
            "metadata": metadata_out,
        }

    return confirm_metadata
