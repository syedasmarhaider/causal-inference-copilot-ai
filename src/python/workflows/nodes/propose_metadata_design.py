# src/python/workflows/nodes/propose_metadata_design.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, cast
from uuid import UUID
import json
import re

from langchain_core.messages import BaseMessage

from python.domain.repo.data_repo import DataRepo
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
    return cast(DatasetState, state.get("dataset", {})) # pyright: ignore[reportUnnecessaryCast]


def _as_metadata(state: ConversationState) -> MetadataState:
    return cast(MetadataState, state.get("metadata", {})) # pyright: ignore[reportUnnecessaryCast]


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
    return [str(v) for v in x] # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]


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


def _default_proposal(columns: Sequence[str]) -> JSONDictLocal:
    # keep defaults aligned with the keys we enforce in the LLM contract
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
        # NOTE: we do NOT add extra keys beyond this contract
    }


def make_propose_metadata_node(
    llm: LLMService,
    data_repo: DataRepo,
    *,
    sample_rows: int = 100,
    model_name: str = "gemini-1.5-flash",
    history_window: int = 10,
    max_sample_chars: int = 12_000,
) -> Callable[[ConversationState], ConversationState]:
    """
    PROPOSE_METADATA node.

    Router-owned transitions:
      - DOES NOT mutate control.stage.
      - Reports outcome + need.

    Produces:
      - metadata.proposed_design with treatment/outcome/controls/effect_modifiers candidates
      - sets metadata hints (treatment_hint/outcome_hint and optional controls/effect_modifiers hints)
    """

    def propose_metadata_design(state: ConversationState) -> ConversationState:
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
            # ControlState is required-keys TypedDict -> always return full shape
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

        dataset_id = dataset_in.get("id")
        raw_schema = dataset_in.get("raw_schema")
        summary = dataset_in.get("summary")

        if not isinstance(dataset_id, UUID) or not isinstance(raw_schema, dict) or not isinstance(summary, dict):
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    outcome="NEEDS_INPUT",
                    need="DATASET_PATH",
                    interrupt_type=None,
                    last_error={
                        "code": "MISSING_DATA_FOR_METADATA_PROPOSAL",
                        "detail": "Missing dataset.id / dataset.raw_schema / dataset.summary. Run LOAD_DATASET first.",
                    },
                    node_message="Dataset info missing. Load a dataset first so I can propose roles (T/Y/W/X).",
                ),
            }

        columns = _columns_from_raw_schema(raw_schema)

        # --- sample rows (best-effort) ---
        sample_json = "null"
        try:
            df_head = data_repo.get_csv_data(dataset_id, limit=sample_rows)
            df_head = df_head.where(df_head.notna(), None)
            sample_json = df_head.to_json(orient="records", force_ascii=False) # pyright: ignore[reportUnknownMemberType]
            if len(sample_json) > max_sample_chars:
                sample_json = sample_json[:max_sample_chars] + "…"
        except Exception:
            sample_json = "null"

        schema_json = json.dumps(raw_schema, ensure_ascii=False)
        summary_json = json.dumps(summary, ensure_ascii=False)
        columns_json = json.dumps(columns, ensure_ascii=False)

        # optional hints coming in
        hints_payload: Dict[str, Any] = {}
        t_hint_in = metadata_in.get("treatment_hint")
        y_hint_in = metadata_in.get("outcome_hint")
        if isinstance(t_hint_in, str) and t_hint_in:
            hints_payload["treatment_hint"] = t_hint_in
        if isinstance(y_hint_in, str) and y_hint_in:
            hints_payload["outcome_hint"] = y_hint_in
        # if you add these fields to MetadataState later, they’ll be used automatically
        w_hint_in = metadata_in.get("controls_hint")
        x_hint_in = metadata_in.get("effect_modifiers_hint")
        if isinstance(w_hint_in, list):
            hints_payload["controls_hint"] = [str(v) for v in w_hint_in]
        if isinstance(x_hint_in, list):
            hints_payload["effect_modifiers_hint"] = [str(v) for v in x_hint_in]
        hints_json = json.dumps(hints_payload, ensure_ascii=False)

        # --- history tail ---
        prior_msgs: Sequence[BaseMessage] = cast(Sequence[BaseMessage], state.get("messages", []))
        tail = list(prior_msgs)[-history_window:] if isinstance(prior_msgs, list) else []

        llm_history: List[ChatMessage] = [
            ChatMessage(
                role="system",
                content=(
                    "You are a causal inference copilot.\n"
                    "Goal: propose candidate columns for roles in a causal analysis.\n\n"
                    "Important constraints:\n"
                    "  - You MUST ONLY output column names that appear in AllowedColumns.\n"
                    "  - If unsure, return empty lists.\n\n"
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
                    "No markdown. No extra keys. No prose outside JSON."
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

        user_prompt = (
            "AllowedColumns (JSON array):\n"
            f"{columns_json}\n\n"
            "Dataset schema (JSON):\n"
            f"{schema_json}\n\n"
            "Dataset summary (JSON):\n"
            f"{summary_json}\n\n"
            "Sample rows (JSON array of records, may be truncated):\n"
            f"{sample_json}\n\n"
            "Existing hints (JSON):\n"
            f"{hints_json}\n"
        )
        llm_history.append(ChatMessage(role="user", content=user_prompt))

        config = LLMConfig(model=model_name, temperature=0.2, max_tokens=900)

        proposal: JSONDictLocal
        llm_error: JSONDict | None = None
        try:
            resp = llm.generate(config=config, history=llm_history)
            proposal = _extract_json_object(resp.content)
        except Exception as e:
            proposal = _default_proposal(columns)
            llm_error = {"code": "LLM_METADATA_PROPOSAL_FAILED", "detail": str(e)}

        # --- normalize ---
        dataset_summary_text = str(proposal.get("dataset_summary", "")).strip()

        t_candidates = _normalize_str_list(proposal.get("treatment_candidates"))
        y_candidates = _normalize_str_list(proposal.get("outcome_candidates"))
        w_candidates = _normalize_str_list(proposal.get("controls_candidates"))
        x_candidates = _normalize_str_list(proposal.get("effect_modifier_candidates"))
        effect_examples = _normalize_str_list(proposal.get("effect_examples"))
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

        # light hinting (downstream confirm step still validates against schema)
        new_t_hint = t_candidates[0] if t_candidates else (t_hint_in if isinstance(t_hint_in, str) else "")
        new_y_hint = y_candidates[0] if y_candidates else (y_hint_in if isinstance(y_hint_in, str) else "")

        metadata_out: MetadataState = {
            **metadata_in,
            "proposed_design": proposal_clean,
            "treatment_hint": new_t_hint,
            "outcome_hint": new_y_hint,
            # optional but recommended fields in MetadataState (see below)
            "controls_hint": w_candidates[:10],
            "effect_modifiers_hint": x_candidates[:10],
            "user_accepts": None,
        }

        if llm_error is not None:
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    outcome="RETRYABLE_ERROR",
                    need="CONFIRM_METADATA",
                    interrupt_type="REVIEW_METADATA",
                    last_error=llm_error,
                    node_message=(
                        "Auto-inference for roles (T/Y/W/X) was unreliable. "
                        "Next: ask user to specify/correct treatment, outcome, and controls."
                    ),
                ),
                "metadata": metadata_out,
            }

        return {
            **state,
            "control": mk_control(
                status="PENDING",
                outcome="DONE",
                need="CONFIRM_METADATA",
                interrupt_type="REVIEW_METADATA",
                last_error=None,
                node_message=(
                    "Draft causal design is ready (treatment/outcome/controls/effect modifiers candidates). "
                    "Next: confirm/correct with the user."
                ),
            ),
            "metadata": metadata_out,
        }

    return propose_metadata_design
