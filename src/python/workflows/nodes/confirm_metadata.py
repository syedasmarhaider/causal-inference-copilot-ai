# src/python/workflows/nodes/confirm_metadata.py
from __future__ import annotations

from typing import Any, Callable, List, Sequence, cast
import json
import re

from langchain_core.messages import BaseMessage

from python.domain.service.llm_service import LLMService, LLMConfig, ChatMessage
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState, Need, Status
from python.workflows.state.dataset_state import DatasetState
from python.workflows.state.metadata_state import MetadataState
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI, JSONDict


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)


def _require_control(state: ConversationState) -> ControlState:
    return cast(ControlState, state["control"])  # type: ignore


def _as_dataset(state: ConversationState) -> DatasetState:
    return cast(DatasetState, state.get("dataset", {}))  # type: ignore


def _as_metadata(state: ConversationState) -> MetadataState:
    return cast(MetadataState, state.get("metadata", {}))  # type: ignore


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
    history_window: int = 12,
    force_covariates_decision: bool = True,
) -> Callable[[ConversationState], ConversationState]:
    """
    CONFIRM_METADATA stage.

    - Reads the latest human message and parses:
      treatment, outcome, covariates (list / all-other-columns / none), optional causal question.
    - Validates column names against dataset schema.
    - If missing/invalid -> need=PRESENT_AND_USER_INPUT
    - If dataset schema missing -> status=ABORTED (router should bounce to LOAD_DATASET)
    """

    def confirm_metadata(state: ConversationState) -> ConversationState:
        control_in = _require_control(state)
        dataset_in = _as_dataset(state)
        metadata_in = _as_metadata(state)

        conversation_id = control_in["conversation_id"]
        stage = control_in["stage"]  # should be "CONFIRM_METADATA"

        def mk_control(
            *,
            status: Status,
            need: Need,
            node_message: str,
            last_error: JSONDict | None,
        ) -> ControlState:
            return cast(
                ControlState,
                {
                    **control_in,
                    "conversation_id": conversation_id,
                    "stage": stage,
                    "status": status,
                    "need": need,
                    "last_error": last_error,
                    "node_message": node_message,
                },
            )

        raw_schema = dataset_in.get("raw_schema")
        columns = _columns_from_raw_schema(raw_schema)
        if not columns:
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    need="PRESENT",
                    last_error={"code": "MISSING_SCHEMA", "detail": "dataset.raw_schema missing; run LOAD_DATASET first."},
                    node_message="Fatal: dataset schema missing. Returning to LOAD_DATASET.",
                ),
            }

        prior_msgs: Sequence[BaseMessage] = cast(Sequence[BaseMessage], state.get("messages", []))
        user_text = _last_user_text(prior_msgs)
        if not user_text:
            # We don't have a user answer yet.
            proposed = metadata_in.get("proposed_design")
            proposed_txt = ""
            if isinstance(proposed, dict):
                tcs = proposed.get("treatment_candidates", [])
                ycs = proposed.get("outcome_candidates", [])
                wcs = proposed.get("controls_candidates", [])
                proposed_txt = (
                    f"Proposed candidates:\n"
                    f"- treatment: {tcs[:5]}\n"
                    f"- outcome: {ycs[:5]}\n"
                    f"- controls: {wcs[:8]}\n\n"
                )

            cols_preview = ", ".join(columns[:15]) + (" ..." if len(columns) > 15 else "")
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    need="PRESENT_AND_USER_INPUT",
                    last_error={"code": "NO_USER_INPUT", "detail": "No user message to confirm metadata."},
                    node_message=(
                        f"{proposed_txt}"
                        "Reply with:\n"
                        "- treatment=<column>\n"
                        "- outcome=<column>\n"
                        "- covariates=<comma-separated columns> OR say 'all other columns' OR 'none'\n"
                        "- optional: question=<your causal question>\n\n"
                        f"Columns preview: {cols_preview}"
                    ),
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
            "Rules:\n"
            "- Only choose columns from AllowedColumns.\n"
            "- If user says 'use all other columns as controls/covariates', set covariate_strategy=ALL_EXCEPT_TY.\n"
            "- If user says 'no controls/covariates', set covariate_strategy=NONE.\n"
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
                    "AllowedColumns (JSON array):\n"
                    f"{cols_json}\n\n"
                    "Current proposed design (json):\n"
                    f"{proposed_json}\n\n"
                    "User message to parse:\n"
                    f"{user_text}\n"
                ),
            )
        )

        config = LLMConfig(model=DEFAULT_MODEL_GEMNI, temperature=1.0)

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
            cols_preview = ", ".join(columns[:20]) + (" ..." if len(columns) > 20 else "")
            return {
                **state,
                "control": mk_control(
                    status="RETRYABLE_ERROR" if parse_error else "PENDING",
                    need="PRESENT_AND_USER_INPUT",
                    last_error=parse_error
                    or {
                        "code": "MISSING_OR_INVALID_TY",
                        "detail": {"parsed_treatment": t_raw, "parsed_outcome": y_raw},
                    },
                    node_message=(
                        "I need valid treatment and outcome column names.\n"
                        "Reply like: treatment=<col>, outcome=<col>, covariates=<...>\n"
                        f"Columns preview: {cols_preview}"
                    ),
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
                for v in cov_cols_raw: # pyright: ignore[reportUnknownVariableType]
                    if isinstance(v, str):
                        resolved = _resolve_column(v, columns)
                        if resolved and resolved not in covariates and resolved not in (t, y):
                            covariates.append(resolved)

        # If user didn't explicitly choose NONE, and covariates ended up empty, force a decision (optional).
        if force_covariates_decision and not covariates and cov_strategy_s != "NONE":
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    need="PRESENT_AND_USER_INPUT",
                    last_error={"code": "MISSING_COVARIATES", "detail": "No covariates resolved from user input."},
                    node_message=(
                        "Treatment/outcome are valid.\n"
                        "Now decide covariates:\n"
                        "- list them (comma-separated), OR\n"
                        "- say 'all other columns', OR\n"
                        "- say 'none'\n"
                    ),
                ),
                "metadata": {
                    **metadata_in,
                    "treatment_hint": t,
                    "outcome_hint": y,
                    "user_accepts": False,
                },
            }

        final_design: JSONDict = {
            "treatment": {"column": t},
            "outcome": {"column": y},
            "covariates": {"columns": covariates, "strategy": cov_strategy_s},
            "causal_question": q,
            "accepted": accept,
        }

        metadata_out: MetadataState = cast(
            MetadataState,
            {
                **metadata_in,
                "final_design": final_design,
                "treatment_hint": t,
                "outcome_hint": y,
                "covariate_hint": covariates[0] if covariates else "",
                "user_accepts": True,
            },
        )

        # DONE -> orchestrator advances to SELECT_ESTIMATOR
        return {
            **state,
            "control": mk_control(
                status="DONE",
                need="PRESENT",
                last_error=parse_error,
                node_message=(
                    "✅ Confirmed metadata.\n"
                    f"- treatment: {t}\n"
                    f"- outcome: {y}\n"
                    f"- covariates: {covariates[:12]}{' ...' if len(covariates) > 12 else ''}\n"
                    + (f"- question: {q}\n" if q else "")
                    + ""
                ),
            ),
            "metadata": metadata_out,
        }

    return confirm_metadata
