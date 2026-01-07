from __future__ import annotations

from typing import Callable, cast
from uuid import UUID

from python.domain.repo.data_repo import DataRepo
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState, Status, Stage, NEED_STAGE, ACTION
from python.workflows.state.dataset_state import DatasetState
from python.workflows.utils.types import JSONDict


def _require_control(state: ConversationState) -> ControlState:
    return cast(ControlState, state["control"])  # type: ignore


def _as_dataset(state: ConversationState) -> DatasetState:
    return cast(DatasetState, state.get("dataset", {}))  # type: ignore


def make_load_dataset_node(data_repo: DataRepo) -> Callable[[ConversationState], ConversationState]:
    def load_dataset(state: ConversationState) -> ConversationState:
        control_in = _require_control(state)
        dataset_in = _as_dataset(state)

        conversation_id: UUID = control_in["conversation_id"]
        stage: Stage = control_in["stage"]  # expected: "LOAD_DATASET"

        def mk_control(
            *,
            status: Status,
            post_action: ACTION,
            post_failure_suggested_stage: NEED_STAGE | None,
            node_message: str,
            last_error: JSONDict | None,
            pending_stage: Stage | None = None,
        ) -> ControlState:
            # NOTE: Orchestrator decides next stage on non-failure.
            # post_failure_suggested_stage is only meaningful when status == "ABORTED".
            base: ControlState = cast(
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
            # pending_stage is NotRequired, but setting it explicitly keeps state predictable.
            base["pending_stage"] = pending_stage
            return base

        dataset_path = dataset_in.get("path")

        # LOAD_DATASET assumes GET_FILE validated the path.
        # If it's missing/invalid here, it's a fatal pipeline state -> ABORTED.
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="GET_FILE",
                    last_error={"code": "MISSING_DATASET_PATH", "detail": "dataset.path missing at LOAD_DATASET."},
                    node_message=(
                        "Fatal: dataset path is missing at LOAD_DATASET.\n"
                        "Suggested recovery: go back to GET_FILE to collect a valid CSV path."
                    ),
                ),
                "dataset": {**dataset_in, "load_error": "MISSING_DATASET_PATH"},
            }

        dataset_path = dataset_path.strip()

        if not dataset_path.lower().endswith(".csv"):
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="GET_FILE",
                    last_error={"code": "INVALID_DATASET_FORMAT", "detail": f"Path {dataset_path!r} not .csv"},
                    node_message=(
                        f"Fatal: dataset path is not a .csv file: `{dataset_path}`.\n"
                        "Suggested recovery: go back to GET_FILE to collect a valid CSV path."
                    ),
                ),
                "dataset": {**dataset_in, "path": dataset_path, "load_error": "INVALID_DATASET_FORMAT"},
            }

        # Register dataset (repo handles existence/readability/permissions)
        try:
            dataset_id = data_repo.register_csv_dataset(
                conversation_id=conversation_id,
                dataset_path=dataset_path,
            )
        except Exception as e:
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="GET_FILE",
                    last_error={"code": "REGISTER_DATASET_ERROR", "detail": str(e)},
                    node_message=(
                        "Fatal: failed to register dataset in repository.\n"
                        "Suggested recovery: go back to GET_FILE to re-collect a valid path."
                    ),
                ),
                "dataset": {**dataset_in, "path": dataset_path, "load_error": "REGISTER_DATASET_ERROR"},
            }

        # Load dataset (CSV parsing issues, encoding problems, etc.)
        try:
            df = data_repo.get_csv_data(dataset_id)
        except Exception as e:
            return {
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="GET_FILE",
                    last_error={"code": "DATA_LOAD_ERROR", "detail": str(e)},
                    node_message=(
                        "Fatal: failed to load/parse the CSV.\n"
                        "Suggested recovery: go back to GET_FILE to re-collect a valid path."
                    ),
                ),
                "dataset": {**dataset_in, "id": dataset_id, "path": dataset_path, "load_error": "DATA_LOAD_ERROR"},
            }

        # Minimal schema + summary
        n_rows, n_cols = df.shape
        raw_schema: JSONDict = {
            "columns": [{"name": col, "dtype": str(dtype)} for col, dtype in df.dtypes.items()]
        }
        summary: JSONDict = {"n_rows": int(n_rows), "n_cols": int(n_cols)}

        cols_preview = [c["name"] for c in raw_schema.get("columns", [])[:10]]  # type: ignore
        preview_str = ", ".join(cols_preview) + (" ..." if n_cols > 10 else "")

        return {
            **state,
            "control": mk_control(
                status="DONE",
                post_action="PRESENT",
                post_failure_suggested_stage=None,
                last_error=None,
                node_message=(
                    "✅ Dataset loaded.\n"
                    f"- Rows: {n_rows}\n"
                    f"- Columns: {n_cols}\n"
                    f"- First columns: {preview_str}\n"
                ),
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
