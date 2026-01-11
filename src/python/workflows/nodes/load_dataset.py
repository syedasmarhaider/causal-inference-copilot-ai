from __future__ import annotations

from typing import Callable, cast
from uuid import UUID

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMService
from python.workflows.utils.node_llm_message import build_node_message_with_llm
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ACTION, NEED_STAGE, ControlState, Stage, Status
from python.workflows.state.dataset_state import DatasetState
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI, JSONDict


def _require_control(state: ConversationState) -> ControlState:
    return cast(ControlState, state["control"])  # type: ignore


def _as_dataset(state: ConversationState) -> DatasetState:
    return cast(DatasetState, state.get("dataset", {}))  # type: ignore


LOAD_DATASET_PROMPT = (
    "You are the LOAD_DATASET node of a causal inference copilot.\n"
    "You receive a compact internal state snapshot as JSON.\n\n"
    "Write EXACTLY ONE message to the user.\n"
    "- Be short, concrete, and actionable.\n"
    "- Do NOT reveal internal JSON or implementation details.\n"
    "- If intent == 'LOADED_OK': confirm the dataset is loaded and show: rows, columns, and a short column-name preview.\n"
    "- If intent == 'MISSING_PATH': explain that the dataset path is missing and ask the user to provide a CSV path.\n"
    "- If intent == 'NOT_CSV': explain the path must end with .csv and ask for a correct CSV path.\n"
    "- If intent == 'REGISTER_FAILED': explain you could not register the dataset and ask for another path.\n"
    "- If intent == 'PARSE_FAILED': explain the file could not be parsed as CSV and ask for another CSV path.\n\n"
    "When asking for a path, include 2-3 examples:\n"
    "- /path/to/data.csv\n"
    "- ./data/my.csv\n"
    "- C:\\\\data\\\\file.csv\n"
)


def make_load_dataset_node(
    data_repo: DataRepo,
    llm: LLMService,
    *,
    model_name: str = DEFAULT_MODEL_GEMNI,
) -> Callable[[ConversationState], ConversationState]:
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
            last_error: JSONDict | None,
            pending_stage: Stage | None = None,
        ) -> ControlState:
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
                    # node_message filled later via LLM (when presenting)
                    "node_message": "",
                },
            )
            base["pending_stage"] = pending_stage
            return base

        def finalize_with_llm(
            *,
            base_state: ConversationState,
            intent: str,
            fallback: str,
        ) -> ConversationState:
            """
            If we're about to PRESENT, let this node generate its own user-facing message.
            """
            msg = build_node_message_with_llm(
                llm,
                state=base_state,
                system_prompt=LOAD_DATASET_PROMPT,
                intent=intent,
                model_name=model_name,
                temperature=0.4,
                history_window=10,
                fallback=fallback,
            )
            c0 = cast(ControlState, base_state["control"]) # pyright: ignore[reportUnnecessaryCast]
            c1: ControlState = {**c0, "node_message": msg}
            return {**base_state, "control": c1}

        dataset_path = dataset_in.get("path")

        # LOAD_DATASET assumes GET_FILE validated the path.
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            base_state = { # pyright: ignore[reportUnknownVariableType]
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="GET_FILE",
                    last_error={"code": "MISSING_DATASET_PATH", "detail": "dataset.path missing at LOAD_DATASET."},
                ),
                "dataset": {**dataset_in, "load_error": "MISSING_DATASET_PATH"},
            }
            return finalize_with_llm(
                base_state=base_state, # pyright: ignore[reportArgumentType]
                intent="MISSING_PATH",
                fallback=(
                    "I don't have a CSV path to load yet. Please paste the path to your dataset CSV.\n"
                    "Examples: /path/to/data.csv, ./data/my.csv, C:\\data\\file.csv"
                ),
            )

        dataset_path = dataset_path.strip()

        if not dataset_path.lower().endswith(".csv"):
            base_state = { # pyright: ignore[reportUnknownVariableType]
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="GET_FILE",
                    last_error={"code": "INVALID_DATASET_FORMAT", "detail": f"Path {dataset_path!r} not .csv"},
                ),
                "dataset": {**dataset_in, "path": dataset_path, "load_error": "INVALID_DATASET_FORMAT"},
            }
            return finalize_with_llm(
                base_state=base_state, # pyright: ignore[reportArgumentType]
                intent="NOT_CSV",
                fallback=(
                    f"That path doesn’t look like a CSV file: {dataset_path!r}. "
                    "Please provide a path ending with .csv."
                ),
            )

        # Register dataset
        try:
            dataset_id = data_repo.register_csv_dataset(
                conversation_id=conversation_id,
                dataset_path=dataset_path,
            )
        except Exception as e:
            base_state = { # pyright: ignore[reportUnknownVariableType]
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="GET_FILE",
                    last_error={"code": "REGISTER_DATASET_ERROR", "detail": str(e)},
                ),
                "dataset": {**dataset_in, "path": dataset_path, "load_error": "REGISTER_DATASET_ERROR"},
            }
            return finalize_with_llm(
                base_state=base_state, # pyright: ignore[reportArgumentType]
                intent="REGISTER_FAILED",
                fallback=(
                    "I couldn't register that dataset path. Please try another valid CSV path.\n"
                    "Examples: /path/to/data.csv, ./data/my.csv, C:\\data\\file.csv"
                ),
            )

        # Load dataset
        try:
            df = data_repo.get_csv_data(dataset_id)
        except Exception as e:
            base_state = { # pyright: ignore[reportUnknownVariableType]
                **state,
                "control": mk_control(
                    status="ABORTED",
                    post_action="PRESENT",
                    post_failure_suggested_stage="GET_FILE",
                    last_error={"code": "DATA_LOAD_ERROR", "detail": str(e)},
                ),
                "dataset": {**dataset_in, "id": dataset_id, "path": dataset_path, "load_error": "DATA_LOAD_ERROR"},
            }
            return finalize_with_llm(
                base_state=base_state, # pyright: ignore[reportArgumentType]
                intent="PARSE_FAILED",
                fallback=(
                    "I couldn't parse that file as a CSV (encoding/delimiter/format issue). "
                    "Please provide another CSV path (or re-export the file as CSV)."
                ),
            )

        # Minimal schema + summary (deterministic)
        n_rows, n_cols = df.shape
        raw_schema: JSONDict = {
            "columns": [{"name": col, "dtype": str(dtype)} for col, dtype in df.dtypes.items()]
        }
        summary: JSONDict = {"n_rows": int(n_rows), "n_cols": int(n_cols)}

        dataset_out: DatasetState = {
            **dataset_in,
            "id": dataset_id,
            "path": dataset_path,
            "load_error": None,
            "raw_schema": raw_schema,
            "summary": summary,
        }

        base_state_ok: ConversationState = {
            **state,
            "dataset": dataset_out,
            "control": mk_control(
                status="DONE",
                post_action="PRESENT",
                post_failure_suggested_stage=None,
                last_error=None,
                pending_stage=None,
            ),
        }

        # Fallback uses deterministic preview in case LLM fails.
        cols_preview = [c["name"] for c in cast(list[dict[str, object]], raw_schema.get("columns", []))[:10]]
        preview_str = ", ".join([str(x) for x in cols_preview]) + (" ..." if int(n_cols) > 10 else "")

        return finalize_with_llm(
            base_state=base_state_ok,
            intent="LOADED_OK",
            fallback=(
                "✅ Dataset loaded.\n"
                f"- Rows: {int(n_rows)}\n"
                f"- Columns: {int(n_cols)}\n"
                f"- First columns: {preview_str}"
            ),
        )

    return load_dataset
