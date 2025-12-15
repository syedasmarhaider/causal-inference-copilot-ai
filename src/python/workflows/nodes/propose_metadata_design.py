# src/python/workflows/nodes/propose_metadata_design.py
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Sequence, cast
from uuid import UUID
import json
import re

from langchain_core.messages import BaseMessage

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMService, LLMConfig, ChatMessage
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState, Need, Status
from python.workflows.state.dataset_state import DatasetState
from python.workflows.state.metadata_state import MetadataState
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI, JSONDict

JSONValue = Any
JSONDictLocal = Dict[str, JSONValue]

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


def _extract_json_object(text: str) -> JSONDictLocal:
    s = text.strip()

    m = _JSON_FENCE_RE.search(s)
    if m:
        obj = json.loads(m.group(1))
        if isinstance(obj, dict):
            return cast(JSONDictLocal, obj)

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return cast(JSONDictLocal, obj)
    except Exception:
        pass

    m2 = _JSON_OBJ_RE.search(s)
    if m2:
        obj2 = json.loads(m2.group(1))
        if isinstance(obj2, dict):
            return cast(JSONDictLocal, obj2)

    raise ValueError("LLM did not return a valid JSON object.")


def _normalize_str_list(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for v in x:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def _columns_from_raw_schema(raw_schema: Any) -> List[str]:
    if not isinstance(raw_schema, dict):
        return []
    cols = raw_schema.get("columns")
    if not isinstance(cols, list):
        return []
    out: List[str] = []
    for c in cols:
        if isinstance(c, dict):
            name = c.get("name")
            if isinstance(name, str) and name:
                out.append(name)
    return out


def _default_proposal(columns: Sequence[str]) -> JSONDictLocal:
    return {
        "dataset_summary": "",
        "treatment_candidates": [],
        "outcome_candidates": [],
        "controls_candidates": [],
        "effect_modifier_candidates": [],
        "effect_examples": [],
        "questions_for_user": [
            "What is the main causal question you want to answer?",
            "Which column is the treatment (the intervention/exposure)?",
            "Which column is the outcome (the result you care about)?",
            "Which columns are confounders/controls we should adjust for?",
            "Do you want heterogeneous effects by subgroup (optional)? If yes, which columns define subgroups?",
        ],
    }


def _filter_allowed(values: Sequence[str], allowed: Sequence[str], *, k: int | None = None) -> List[str]:
    allowed_set = set(allowed)
    out: List[str] = []
    for v in values:
        if v in allowed_set and v not in out:
            out.append(v)
            if k is not None and len(out) >= k:
                break
    return out


def _repair_to_strict_json(
    llm: LLMService,
    *,
    model_name: str,
    broken_text: str,
) -> JSONDictLocal:
    sys = (
        "You are a JSON repair bot.\n"
        "You will be given text that should represent a JSON object.\n"
        "Return ONLY a valid JSON object with EXACTLY these keys:\n"
        "{\n"
        '  "dataset_summary": string,\n'
        '  "treatment_candidates": [string],\n'
        '  "outcome_candidates": [string],\n'
        '  "controls_candidates": [string],\n'
        '  "effect_modifier_candidates": [string],\n'
        '  "effect_examples": [string],\n'
        '  "questions_for_user": [string]\n'
        "}\n"
        "No markdown. No extra keys. No prose."
    )
    cfg = LLMConfig(model=model_name, temperature=1.0)
    resp = llm.generate(
        config=cfg,
        history=[
            ChatMessage(role="system", content=sys),
            ChatMessage(role="user", content=broken_text),
        ],
    )
    return _extract_json_object(resp.content)


def make_propose_metadata_node(
    llm: LLMService,
    data_repo: DataRepo,
    *,
    sample_rows: int = 80,
    model_name: str = DEFAULT_MODEL_GEMNI,
    history_window: int = 10,
    max_sample_chars: int = 10_000,
) -> Callable[[ConversationState], ConversationState]:
    """
    PROPOSE_METADATA stage.

    Output behavior:
      - Always returns status="DONE" and need="PRESENT_AND_USER_INPUT"
      - last_error is populated if LLM failed or needed fallback/repair
      - metadata.proposed_design exists even on failure (default proposal)
    """

    def propose_metadata_design(state: ConversationState) -> ConversationState:
        control_in = _require_control(state)
        dataset_in = _as_dataset(state)
        metadata_in = _as_metadata(state)

        conversation_id = control_in["conversation_id"]
        stage = control_in["stage"]  # "PROPOSE_METADATA"

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

        dataset_id = dataset_in.get("id")
        raw_schema = dataset_in.get("raw_schema")
        summary = dataset_in.get("summary")

        if not isinstance(dataset_id, UUID) or not isinstance(raw_schema, dict) or not isinstance(summary, dict):
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    need="PRESENT",
                    last_error={
                        "code": "MISSING_DATA_FOR_METADATA_PROPOSAL",
                        "detail": "Missing dataset.id / dataset.raw_schema / dataset.summary. LOAD_DATASET must run first.",
                    },
                    node_message="Fatal: dataset is not loaded (missing schema/summary). Returning to LOAD_DATASET.",
                ),
            }

        columns = _columns_from_raw_schema(raw_schema)

        # sample rows (best effort)
        sample_json = "null"
        try:
            df_head = data_repo.get_csv_data(dataset_id, limit=sample_rows)
            df_head = df_head.where(df_head.notna(), None)  # type: ignore
            sample_json = df_head.to_json(orient="records", force_ascii=False)  # type: ignore
            if len(sample_json) > max_sample_chars:
                sample_json = sample_json[:max_sample_chars] + "…"
        except Exception:
            sample_json = "null"

        schema_json = json.dumps(raw_schema, ensure_ascii=False)
        summary_json = json.dumps(summary, ensure_ascii=False)
        columns_json = json.dumps(columns, ensure_ascii=False)

        # history tail
        prior_msgs: Sequence[BaseMessage] = cast(Sequence[BaseMessage], state.get("messages", []))
        tail = list(prior_msgs)[-history_window:] if isinstance(prior_msgs, list) else []

        llm_history: List[ChatMessage] = [
            ChatMessage(
                role="system",
                content=(
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
                ),
            )
        ]
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
                    f"{columns_json}\n\n"
                    "Dataset schema (JSON):\n"
                    f"{schema_json}\n\n"
                    "Dataset summary (JSON):\n"
                    f"{summary_json}\n\n"
                    "Sample rows (JSON array of records, may be truncated):\n"
                    f"{sample_json}\n"
                ),
            )
        )

        # IMPORTANT: temp=0.0 to reduce JSON breakage
        config = LLMConfig(model=model_name, temperature=1)

        proposal: JSONDictLocal = _default_proposal(columns)
        llm_error: JSONDict | None = None
        raw_out: str | None = None

        try:
            resp = llm.generate(config=config, history=llm_history)
            raw_out = resp.content
            logging.warning(f"LLM raw output for metadata proposal: {raw_out}")
            try:
                proposal = _extract_json_object(resp.content)
            except Exception as parse_e:
                # attempt repair
                try:
                    proposal = _repair_to_strict_json(llm, model_name=model_name, broken_text=resp.content)
                    llm_error = {
                        "code": "LLM_METADATA_PROPOSAL_REPAIRED",
                        "detail": str(parse_e),
                        "raw_llm_output": (resp.content[:2000] if isinstance(resp.content, str) else None),
                    }
                except Exception as repair_e:
                    proposal = _default_proposal(columns)
                    llm_error = {
                        "code": "LLM_METADATA_PROPOSAL_FAILED",
                        "detail": f"parse_error={parse_e} repair_error={repair_e}",
                        "raw_llm_output": (resp.content[:2000] if isinstance(resp.content, str) else None),
                    }
        except Exception as e:
            proposal = _default_proposal(columns)
            llm_error = {
                "code": "LLM_METADATA_PROPOSAL_FAILED",
                "detail": str(e),
                "raw_llm_output": (raw_out[:2000] if isinstance(raw_out, str) else None),
            }

        # normalize + filter to allowed columns (hard constraint enforcement)
        dataset_summary_text = str(proposal.get("dataset_summary", "")).strip()

        t_candidates = _filter_allowed(_normalize_str_list(proposal.get("treatment_candidates")), columns, k=12)
        y_candidates = _filter_allowed(_normalize_str_list(proposal.get("outcome_candidates")), columns, k=12)
        w_candidates = _filter_allowed(_normalize_str_list(proposal.get("controls_candidates")), columns, k=24)
        x_candidates = _filter_allowed(_normalize_str_list(proposal.get("effect_modifier_candidates")), columns, k=24)

        effect_examples = _normalize_str_list(proposal.get("effect_examples"))[:10]
        questions_for_user = _normalize_str_list(proposal.get("questions_for_user"))

        proposal_clean: JSONDictLocal = {
            "dataset_summary": dataset_summary_text,
            "treatment_candidates": t_candidates,
            "outcome_candidates": y_candidates,
            "controls_candidates": w_candidates,
            "effect_modifier_candidates": x_candidates,
            "effect_examples": effect_examples,
            "questions_for_user": questions_for_user or _default_proposal(columns)["questions_for_user"],
        }

        # hints
        t_hint_in = metadata_in.get("treatment_hint")
        y_hint_in = metadata_in.get("outcome_hint")
        new_t_hint = t_candidates[0] if t_candidates else (t_hint_in if isinstance(t_hint_in, str) else "")
        new_y_hint = y_candidates[0] if y_candidates else (y_hint_in if isinstance(y_hint_in, str) else "")

        metadata_out: MetadataState = cast(
            MetadataState,
            {
                **metadata_in,
                "proposed_design": proposal_clean,
                "treatment_hint": new_t_hint,
                "outcome_hint": new_y_hint,
                "controls_hint": w_candidates[:10],
                "effect_modifiers_hint": x_candidates[:10],
                "user_accepts": None,
            },
        )

        # message
        cols_preview = ", ".join(columns[:12]) + (" ..." if len(columns) > 12 else "")
        msg = (
            "🧠 Draft causal design proposal (auto):\n"
            f"- Dataset: rows={summary.get('n_rows')} cols={summary.get('n_cols')}\n"
            f"- Columns preview: {cols_preview}\n\n"
        )
        if dataset_summary_text:
            msg += f"Summary: {dataset_summary_text}\n\n"

        msg += (
            f"Treatment candidates: {t_candidates[:8]}\n"
            f"Outcome candidates: {y_candidates[:8]}\n"
            f"Controls candidates: {w_candidates[:10]}\n"
            f"Effect modifiers: {x_candidates[:10]}\n"
        )

        if effect_examples:
            msg += "\nExamples:\n" + "\n".join([f"- {e}" for e in effect_examples[:5]]) + "\n"

        msg += "\nQuestions:\n" + "\n".join([f"- {q}" for q in proposal_clean["questions_for_user"][:5]])
        msg += "\n\nNext: CONFIRM_METADATA."

        if llm_error is not None:
            msg += "\n\n(note: LLM proposal had a fallback/repair due to an error)"

        return {
            **state,
            "control": mk_control(
                status="DONE",
                need="PRESENT_AND_USER_INPUT",
                last_error=llm_error,
                node_message=msg,
            ),
            "metadata": metadata_out,
        }

    return propose_metadata_design
