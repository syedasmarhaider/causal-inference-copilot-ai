from __future__ import annotations

from typing import Any, Callable, List, Sequence, cast
import json
import re

from langchain_core.messages import BaseMessage

from python.domain.service.llm_service import LLMService, LLMConfig, ChatMessage
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState, JSONDict, Need, Outcome, Status
from python.workflows.state.dataset_state import DatasetState
from python.workflows.state.metadata_state import MetadataState
from python.workflows.utils.types import JSONDict



_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _require_control(state: ConversationState) -> ControlState:
    return cast(ControlState, state["control"]) # type: ignore


def _as_dataset(state: ConversationState) -> DatasetState:
    return cast(DatasetState, state.get("dataset", {})) # type: ignore


def _as_metadata(state: ConversationState) -> MetadataState:
    return cast(MetadataState, state.get("metadata", {})) # type: ignore


def _role_from_langchain_msg(m: BaseMessage) -> str:
    t = getattr(m, "type", None)
    return {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(str(t), "user")


def _last_user_text(messages: Sequence[BaseMessage]) -> str | None:
    for m in reversed(list(messages)):
        if getattr(m, "type", None) == "human":
            txt = str(getattr(m, "content", "")).strip()
            return txt or None
    return None


def _extract_json_object(text: str) -> JSONDict:
    s = text.strip()

    m = _JSON_FENCE_RE.search(s)
    if m:
        obj = json.loads(m.group(1))
        if isinstance(obj, dict):
            return cast(JSONDict, obj)

    try:
        obj2 = json.loads(s)
        if isinstance(obj2, dict):
            return cast(JSONDict, obj2)
    except Exception:
        pass

    m2 = _JSON_OBJ_RE.search(s)
    if m2:
        obj3 = json.loads(m2.group(1))
        if isinstance(obj3, dict):
            return cast(JSONDict, obj3)

    raise ValueError("LLM did not return a valid JSON object.")


def _columns_from_raw_schema(raw_schema: Any) -> List[str]:
    if not isinstance(raw_schema, dict):
        return []
    cols = raw_schema.get("columns") # type: ignore
    if not isinstance(cols, list):
        return []
    out: List[str] = []
    for c in cols: # type: ignore
        if isinstance(c, dict):
            name = c.get("name") # type: ignore
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


def _default_covariates_all_except_ty(columns: Sequence[str], t: str, y: str) -> List[str]:
    # minimal safe heuristic: exclude treatment/outcome + obvious identifiers
    id_like = re.compile(r"(?:^id$|uuid|guid|index|row_id|customer_id|user_id)", re.IGNORECASE)
    out: List[str] = []
    for c in columns:
        if c == t or c == y:
            continue
        if id_like.search(c):
            continue
        out.append(c)
    return out


def make_confirm_metadata_node(
    llm: LLMService,
    *,
    model_name: str = "gemini-1.5-flash",
    history_window: int = 12,
) -> Callable[[ConversationState], ConversationState]:
    def confirm_metadata(state: ConversationState) -> ConversationState:
        control_in = _require_control(state)
        dataset_in = _as_dataset(state)
        metadata_in = _as_metadata(state)

        conversation_id = control_in["conversation_id"]
        stage = control_in["stage"]  # router-owned

        def mk_control(
            *,
            status: Status,
            outcome: Outcome,
            need: Need,
            node_message: str,
            last_error: JSONDict | None,
            interrupt_type: str | None,
        ) -> ControlState:
            control_out: ControlState = {
                "conversation_id": conversation_id,
                "status": status,
                "stage": stage,
                "outcome": outcome,
                "need": need,
                "interrupt_type": interrupt_type,
                "last_error": last_error,
                "node_message": node_message,
            }
            return control_out

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
                    node_message="Dataset schema missing. Reload dataset so I can validate treatment/outcome/covariates.",
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
                    need="TREATMENT_OUTCOME",
                    interrupt_type="REVIEW_METADATA",
                    last_error={"code": "NO_USER_INPUT", "detail": "No recent user message to confirm metadata."},
                    node_message="Tell me the treatment column, outcome column, and (optionally) covariates.",
                ),
            }

        proposed = metadata_in.get("proposed_design")
        proposed_json = json.dumps(proposed, ensure_ascii=False) if isinstance(proposed, dict) else "{}"
        cols_json = json.dumps(columns, ensure_ascii=False)

        sys = (
            "You are a strict parser for a causal inference copilot.\n"
            "Extract user's confirmation/corrections.\n\n"
            "Return ONLY one JSON object with EXACTLY these keys:\n"
            "{\n"
            '  "accept": boolean,\n'
            '  "treatment_column": string | null,\n'
            '  "outcome_column": string | null,\n'
            '  "covariate_strategy": "USER_LIST" | "ALL_EXCEPT_TY" | "NONE" | null,\n'
            '  "covariate_columns": [string] | null,\n'
            '  "causal_question": string | null\n'
            "}\n"
            "No markdown. No extra keys. No prose.\n"
            "If user says 'use all other columns as controls/covariates', set covariate_strategy=ALL_EXCEPT_TY.\n"
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
                    "Allowed dataset columns (array):\n"
                    f"{cols_json}\n\n"
                    "Current proposed design (json):\n"
                    f"{proposed_json}\n\n"
                    "User message to parse:\n"
                    f"{user_text}\n"
                ),
            )
        )

        config = LLMConfig(model=model_name, temperature=0.0, max_tokens=450)

        parsed: JSONDict
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
                "covariate_strategy": None,
                "covariate_columns": None,
                "causal_question": None,
            }

        accept = bool(parsed.get("accept", False))
        t_raw = parsed.get("treatment_column")
        y_raw = parsed.get("outcome_column")
        cov_strategy = parsed.get("covariate_strategy")
        cov_cols_raw = parsed.get("covariate_columns")
        q_raw = parsed.get("causal_question")

        t = _resolve_column(str(t_raw), columns) if isinstance(t_raw, str) else None
        y = _resolve_column(str(y_raw), columns) if isinstance(y_raw, str) else None
        q = str(q_raw).strip() if isinstance(q_raw, str) and q_raw.strip() else None

        if not t or not y:
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    outcome="RETRYABLE_ERROR" if parse_error else "NEEDS_INPUT",
                    need="TREATMENT_OUTCOME",
                    interrupt_type="REVIEW_METADATA",
                    last_error=parse_error
                    or {
                        "code": "MISSING_OR_INVALID_TY",
                        "detail": {"parsed_treatment": t_raw, "parsed_outcome": y_raw},
                    },
                    node_message="I need valid treatment and outcome column names (copy-paste them from the schema).",
                ),
                "metadata": {**metadata_in, "user_accepts": False},
            }

        covariates: List[str] = []
        cov_strategy_s = str(cov_strategy) if isinstance(cov_strategy, str) else "USER_LIST"

        if cov_strategy_s == "NONE":
            covariates = []
        elif cov_strategy_s == "ALL_EXCEPT_TY":
            covariates = _default_covariates_all_except_ty(columns, t=t, y=y)
        else:
            if isinstance(cov_cols_raw, list):
                for v in cov_cols_raw: # type: ignore
                    if isinstance(v, str):
                        resolved = _resolve_column(v, columns)
                        if resolved and resolved not in covariates and resolved not in (t, y):
                            covariates.append(resolved)

        final_design: JSONDict = {
            "treatment": {"column": t},
            "outcome": {"column": y},
            "covariates": {"columns": covariates, "strategy": cov_strategy_s},
            "causal_question": q,
            "accepted": accept,
        }

        metadata_out: MetadataState = {
            **metadata_in,
            "final_design": final_design,
            "treatment_hint": t,
            "outcome_hint": y,
            "covariate_hint": covariates[0] if covariates else "",
            "user_accepts": True,
        }

        # If covariates are empty, you may still proceed, but it’s scientifically risky.
        # So: keep OK/DONE but set need=COVARIATES if you want to force the user to decide.
        if not covariates and cov_strategy_s != "NONE":
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    outcome="DONE",
                    need="COVARIATES",
                    interrupt_type="REVIEW_METADATA",
                    last_error=None,
                    node_message=(
                        "Treatment/outcome confirmed. Next: confirm covariates (confounders/controls). "
                        "You can list them, or say 'use all other columns'."
                    ),
                ),
                "metadata": metadata_out,
            }

        return {
            **state,
            "control": mk_control(
                status="OK",
                outcome="DONE",
                need="NONE",
                interrupt_type=None,
                last_error=parse_error,
                node_message="Confirmed treatment/outcome/covariates. Ready for estimator selection.",
            ),
            "metadata": metadata_out,
        }

    return confirm_metadata
