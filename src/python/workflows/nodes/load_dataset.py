# src/python/workflows/nodes/load_dataset.py
from __future__ import annotations

from typing import Any, Callable, Dict
from uuid import UUID

from python.domain.repo.data_repo import DataRepo
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState, JSONDict, Need, Outcome, Status
from python.workflows.state.dataset_state import DatasetState

JSONValue = Any
JSONDictLocal = Dict[str, JSONValue]


def _require_control(state: ConversationState) -> ControlState:
    return state["control"]  # type: ignore[typeddict-item]


def _as_dataset(state: ConversationState) -> DatasetState:
    return state.get("dataset", {}) 


def make_load_dataset_node(data_repo: DataRepo) -> Callable[[ConversationState], ConversationState]:
    def load_dataset(state: ConversationState) -> ConversationState:
        control_in = _require_control(state)
        dataset_in = _as_dataset(state)

        conversation_id: UUID = control_in["conversation_id"]
        stage = control_in["stage"] 

        def mk_control(
            *,
            status: Status,
            outcome: Outcome,
            need: Need,
            node_message: str,
            last_error: JSONDict | None,
            interrupt_type: str | None = None,
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

        dataset_path = dataset_in.get("path")

        if not isinstance(dataset_path, str) or not dataset_path:
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    outcome="NEEDS_INPUT",
                    need="DATASET_PATH",
                    last_error={"code": "NO_DATASET_PATH", "detail": "dataset.path is missing or empty."},
                    node_message="Provide a CSV path so I can load the dataset.",
                ),
                "dataset": {**dataset_in, "load_error": "NO_DATASET_PATH"},
            }

        # 2) trivial .csv check only
        if not dataset_path.lower().endswith(".csv"):
            return {
                **state,
                "control": mk_control(
                    status="PENDING",
                    outcome="NEEDS_INPUT",
                    need="DATASET_PATH",
                    last_error={"code": "INVALID_FORMAT", "detail": f"Path {dataset_path!r} is not a .csv file."},
                    node_message="Dataset path must end with .csv.",
                ),
                "dataset": {**dataset_in, "path": dataset_path, "load_error": "INVALID_FORMAT"},
            }

        # 3) register
        try:
            dataset_id = data_repo.register_csv_dataset(
                conversation_id=conversation_id,
                dataset_path=dataset_path,
            )
        except Exception as e:
            return {
                **state,
                "control": mk_control(
                    status="ERROR",
                    outcome="FAILED",
                    need="NONE",
                    last_error={"code": "REGISTER_DATASET_ERROR", "detail": str(e)},
                    node_message="Failed to register dataset in repository.",
                ),
                "dataset": {**dataset_in, "path": dataset_path, "load_error": "REGISTER_DATASET_ERROR"},
            }

        # 4) load (repo is responsible for file/CSV parse errors)
        try:
            df = data_repo.get_csv_data(dataset_id)
        except Exception as e:
            return {
                **state,
                "control": mk_control(
                    status="ERROR",
                    outcome="FAILED",
                    need="NONE",
                    last_error={"code": "DATA_LOAD_ERROR", "detail": str(e)},
                    node_message="Failed to load dataset from repository.",
                ),
                "dataset": {**dataset_in, "id": dataset_id, "path": dataset_path, "load_error": "DATA_LOAD_ERROR"},
            }

        # 5) minimal schema + summary (you said OK to keep this lightweight)
        n_rows, n_cols = df.shape
        raw_schema: JSONDictLocal = {
            "columns": [{"name": col, "dtype": str(dtype)} for col, dtype in df.dtypes.items()]
        }
        summary: JSONDictLocal = {"n_rows": int(n_rows), "n_cols": int(n_cols)}

        return {
            **state,
            "control": mk_control(
                status="OK",
                outcome="DONE",
                need="NONE",
                last_error=None,
                node_message="Dataset loaded. Next step is to propose treatment/outcome candidates.",
            ),
            "dataset": {
                **dataset_in,
                "id": dataset_id,
                "path": dataset_path,
                "load_error": None,
                "raw_schema": raw_schema,
                "summary": summary,
            },
        }

    return load_dataset
