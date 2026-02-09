from __future__ import annotations

import json
import logging
from uuid import UUID

from langchain_core.messages import AIMessage

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMConfig, LLMService
from python.workflows.nodes.prompts.load_dataset import load_dataset_system_prompt
from python.workflows.state.control_state import ACTION, ControlState, Stage, Status
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState
from python.workflows.state.dataset_state import DatasetState, DatasetStateHelpers, DatasetProfilingError
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI, JSONDict

log = logging.getLogger(__name__)


# TODO: refactor to latest design
def _mk_control(
    *,
    current_stage: Stage,
    current_stage_status: Status,
    action_required: ACTION,
    node_message: str | None,
) -> ControlState:
    return {
        "current_stage": current_stage,
        "current_stage_status": current_stage_status,
        "action_required": action_required,
        "node_message": node_message,
    }


def _append_final_ai_message(state: ConversationState, content: str) -> None:
    msgs = state.get("messages")
    if not isinstance(msgs, list):  # pyright: ignore[reportUnnecessaryIsInstance]
        state["messages"] = []
        msgs = state["messages"]
    msgs.append(AIMessage(content=content, additional_kwargs={"source": "node", "stage": "LOAD_DATASET"}))


def _llm_message_strict(
    llm: LLMService,
    *,
    model_name: str,
    snapshot: JSONDict,
) -> str:
    cfg = LLMConfig(model=model_name, temperature=0.5)
    msg = llm.generate(
        config=cfg,
        system_prompt=load_dataset_system_prompt(),
        user_prompt=json.dumps(snapshot, ensure_ascii=False),
        history=None,
    ).content
    if not msg:
        raise ValueError("LOAD_DATASET: LLM returned empty node message")
    return msg


def _format_columns_block(cols: list[str]) -> str:
    lines = [f"Columns ({len(cols)}):"]
    for i, c in enumerate(cols, start=1):
        lines.append(f"{i}. {c}")
    return "\n".join(lines)


def make_load_dataset_node(
    data_repo: DataRepo,
    llm: LLMService,
    *,
    model_name: str = DEFAULT_MODEL_GEMNI,
) -> CallableNodeFunc:
    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        dataset: DatasetState = state.get("dataset", {})  # type: ignore[assignment]
        dataset_id = dataset.get("id")
        if not isinstance(dataset_id, UUID):
            raise ValueError("LOAD_DATASET: dataset.id missing or invalid UUID")

        try:
            df = data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=dataset_id,
                limit=1000000,
            )
        except Exception as e:
            out_state: ConversationState = {
                **state,
                "dataset": {
                    **dataset,
                    "id": dataset_id,
                    "raw_schema": None,
                    "summary": None,
                    "load_error": "LOAD_FAILED",
                },
                "control": _mk_control(
                    current_stage="LOAD_DATASET",
                    current_stage_status="PENDING",
                    action_required="NEEDS_INPUT",
                    node_message=None,
                ),
            }

            snapshot: JSONDict = {
                "intent": "LOAD_FAILED",
                "error": str(e),
                "hint": "Verify the configured CSV file exists and is readable, or update the configured path in FileDataRepo.",
            }

            try:
                msg = _llm_message_strict(llm, model_name=model_name, snapshot=snapshot)
            except Exception as llm_e:
                out_state["control"] = _mk_control(
                    current_stage="LOAD_DATASET",
                    current_stage_status="ABORTED",
                    action_required="NEEDS_INPUT",
                    node_message=None,
                )
                raise ValueError(f"LOAD_DATASET: LLM message generation failed: {llm_e}") from llm_e

            out_state["control"] = {**out_state["control"], "node_message": msg}  # type: ignore[index]
            _append_final_ai_message(out_state, msg)
            return out_state

        # ---- Success: write schema + summary deterministically ----
        n_rows, n_cols = df.shape
        cols = [str(c) for c in df.columns.tolist()]

        raw_schema: JSONDict = {
            "columns": [{"name": str(col), "dtype": str(dtype)} for col, dtype in df.dtypes.items()]
        }

        # Summary: column profile (raises on real dataset problems)
        try:
            summary = DatasetStateHelpers.extract_column_profile(
                df,
                max_categories=1000,
                sample_distinct=1000,
                compute_quantiles=True,
                strict=True,
            )
        except DatasetProfilingError as pe:
            # This is a user-actionable dataset failure, not an internal crash.
            details = getattr(pe, "details", None)
            err_reason = str(pe)

            out_state_bad_summary: ConversationState = {
                **state,
                "dataset": {
                    **dataset,
                    "id": dataset_id,
                    "raw_schema": raw_schema,
                    "summary": None,
                    "load_error": "SUMMARY_FAILED",
                },
                "control": _mk_control(
                    current_stage="LOAD_DATASET",
                    current_stage_status="PENDING",
                    action_required="NEEDS_INPUT",
                    node_message=None,
                ),
            }

            snapshot_sum_fail: JSONDict = {
                "intent": "SUMMARY_FAILED",
                "error": err_reason,
                "details": (
                    {
                        "column": getattr(details, "column", None),
                        "reason": getattr(details, "reason", None),
                        "hint": getattr(details, "hint", None),
                        "evidence": getattr(details, "evidence", None),
                    }
                    if details is not None
                    else None
                ),
                "hint": "Fix the dataset format/schema (e.g., ensure it is tabular, readable, and columns are accessible), then reload.",
            }

            try:
                msg = _llm_message_strict(llm, model_name=model_name, snapshot=snapshot_sum_fail)
            except Exception as llm_e:
                out_state_bad_summary["control"] = _mk_control(
                    current_stage="LOAD_DATASET",
                    current_stage_status="ABORTED",
                    action_required="NEEDS_INPUT",
                    node_message=None,
                )
                raise ValueError(f"LOAD_DATASET: LLM message generation failed: {llm_e}") from llm_e

            out_state_bad_summary["control"] = {**out_state_bad_summary["control"], "node_message": msg}  # type: ignore[index]
            _append_final_ai_message(out_state_bad_summary, msg)
            return out_state_bad_summary

        out_state_ok: ConversationState = {
            **state,
            "dataset": {
                **dataset,
                "id": dataset_id,
                "raw_schema": raw_schema,
                "summary": summary,  # Dict[str, Dict[str, Any]]
                "load_error": None,
            },
            "control": _mk_control(
                current_stage="LOAD_DATASET",
                current_stage_status="DONE",
                action_required="NONE",
                node_message=None,
            ),
        }

        snapshot_ok: JSONDict = {
            "intent": "LOADED_OK",
            "dataset_preview": {"rows": int(n_rows), "cols": int(n_cols), "columns": cols},
            # Keep LLM payload small: provide only top-level counts here
            "summary_stats": {
                "profiled_columns": len(summary),
                "rows": int(n_rows),
                "cols": int(n_cols),
            },
        }

        try:
            msg_ok = _llm_message_strict(llm, model_name=model_name, snapshot=snapshot_ok)
        except Exception as llm_e:
            out_state_ok["control"] = _mk_control(
                current_stage="LOAD_DATASET",
                current_stage_status="ABORTED",
                action_required="NEEDS_INPUT",
                node_message=None,
            )
            raise ValueError(f"LOAD_DATASET: LLM message generation failed: {llm_e}") from llm_e

        columns_block = _format_columns_block(cols)
        final_msg_ok = f"{columns_block}\n\n{msg_ok}".strip()

        out_state_ok["control"] = {**out_state_ok["control"], "node_message": final_msg_ok}  # type: ignore[index]
        _append_final_ai_message(out_state_ok, final_msg_ok)

        return out_state_ok

    return node
