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
from python.workflows.state.dataset_state import DatasetState
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI, JSONDict

log = logging.getLogger(__name__)
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
    msgs.append(AIMessage(content=content, additional_kwargs={"source": "node", "stage": "LOAD_DATASET"}))


def _llm_message_strict(
    llm: LLMService,
    *,
    model_name: str,
    snapshot: JSONDict,
) -> str:
    cfg = LLMConfig(model=model_name, temperature=0.5)
    msg = llm.generate(config=cfg, system_prompt=load_dataset_system_prompt(), user_prompt=json.dumps(snapshot, ensure_ascii=False), history=None).content
    if not msg:
        raise ValueError("LOAD_DATASET: LLM returned empty node message")
    return msg


def make_load_dataset_node(
    data_repo: DataRepo,
    llm: LLMService,
    *,
    model_name: str = DEFAULT_MODEL_GEMNI,
    limit: int | None = None,
    append_ai_message: bool = True,
) -> CallableNodeFunc:
    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        dataset: DatasetState = state.get("dataset", {})  # type: ignore[assignment]
        dataset_id = dataset.get("id")
        if not isinstance(dataset_id, UUID):
            raise ValueError("LOAD_DATASET: dataset.id missing or invalid UUID")

        # ---- Load via DataRepo (no manual validation here) ----
        try:
            df = data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=dataset_id,
                limit=limit,
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
            if append_ai_message:
                _append_final_ai_message(out_state, msg)
            return out_state

        # ---- Success: write schema + summary deterministically ----
        n_rows, n_cols = df.shape
        cols = [str(c) for c in df.columns.tolist()]

        raw_schema: JSONDict = {
            "columns": [{"name": str(col), "dtype": str(dtype)} for col, dtype in df.dtypes.items()]
        }
        summary: JSONDict = {"n_rows": int(n_rows), "n_cols": int(n_cols)}

        out_state_ok: ConversationState = {
            **state,
            "dataset": {
                **dataset,
                "id": dataset_id,
                "raw_schema": raw_schema,
                "summary": summary,
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

        out_state_ok["control"] = {**out_state_ok["control"], "node_message": msg_ok}  # type: ignore[index]
        if append_ai_message:
            _append_final_ai_message(out_state_ok, msg_ok)

        return out_state_ok

    return node
