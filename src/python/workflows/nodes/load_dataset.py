from __future__ import annotations

import inspect
import logging
from typing import Any, Sequence, cast
from uuid import UUID

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.workflows.state.control_state import ACTION, ControlState, Stage, Status
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState
from python.workflows.state.dataset_state import DatasetState
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI

log = logging.getLogger(__name__)


def _require_control(state: ConversationState) -> ControlState:
    return cast(ControlState, state["control"])  # type: ignore


def _as_dataset(state: ConversationState) -> DatasetState:
    ds = state.get("dataset", {})
    if not isinstance(ds, dict): # pyright: ignore[reportUnnecessaryIsInstance]
        return cast(DatasetState, {})  # type: ignore
    return cast(DatasetState, ds)  # type: ignore


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
) -> CallableNodeFunc:
    """
    LOAD_DATASET node (updated for new ControlState):
      - Writes control.current_stage/current_stage_status/action_required/post_failure_suggested_stage/node_message
      - Writes dataset.id/path/raw_schema/summary/load_error
      - Uses build_node_message_with_llm() to generate user-facing node_message
    """

    def mk_control(
        *,
        current_stage: Stage,
        current_stage_status: Status,
        action_required: ACTION,
        node_message: str = "",
    ) -> ControlState:
        return cast(
            ControlState,
            {
                "current_stage": current_stage,
                "current_stage_status": current_stage_status,
                "action_required": action_required,
                "node_message": node_message,
            },
        )

    def finalize_with_llm(
        *,
        base_state: ConversationState,
        intent: str,
        data: str | None = None,
    ) -> ConversationState:
        config = LLMConfig(
            model=model_name,
            temperature=0.0,
        )
        
        history: Sequence[ChatMessage] = [
        ChatMessage(role="system", content=LOAD_DATASET_PROMPT),
        ChatMessage(role="user", content=f"""
           Here is the current internal state snapshot (JSON):
                {base_state}
           Your intent is: {intent}
           Please write the user-facing message accordingly.
           Here are some additional data you can use: {data if data is not None else None}
        """)
      ]

        resp = llm.generate(config=config, history=history)
        msg = resp.content

        c0 =  base_state["control"] 
        c1: ControlState = {**c0, "node_message": msg}
        return {**base_state, "control": c1}

    def load_dataset(state: ConversationState) -> ConversationState:
        control_in = _require_control(state) # pyright: ignore[reportUnusedVariable]
        dataset_in = _as_dataset(state)

        dataset_path = dataset_in.get("path")

        # If path missing => user must provide it => go back to GET_FILE
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            base_state: ConversationState = {
                **state,
                "control": mk_control(
                    current_stage="GET_FILE",
                    current_stage_status="PENDING",
                    action_required="NEEDS_INPUT",
                ),
                "dataset": {**dataset_in, "load_error": "MISSING_DATASET_PATH"},
            }
            return finalize_with_llm(
                base_state=base_state,
                intent="MISSING_PATH",
            )

        dataset_path = dataset_path.strip()

        if not dataset_path.lower().endswith(".csv"):
            base_state = {
                **state,
                "control": mk_control(
                    current_stage="GET_FILE",
                    current_stage_status="PENDING",
                    action_required="NEEDS_INPUT",
                ),
                "dataset": {**dataset_in, "path": dataset_path, "load_error": "INVALID_DATASET_FORMAT"},
            }
            return finalize_with_llm(
                base_state=base_state,
                intent="NOT_CSV",
            )

        # Register dataset
        try:
            dataset_id: UUID = _register_csv_dataset_uuid(data_repo, state=state, dataset_path=dataset_path)
        except Exception as e:
            base_state = {
                **state,
                "control": mk_control(
                    current_stage="LOAD_DATASET",
                    current_stage_status="ABORTED",
                    action_required="NEEDS_INPUT",
                ),
                "dataset": {**dataset_in, "path": dataset_path, "load_error": f"REGISTER_DATASET_ERROR: {e}"},
            }
            return finalize_with_llm(
                base_state=base_state,
                intent="REGISTER_FAILED",
            )

        # Load dataset
        try:
            df = data_repo.get_csv_data(dataset_id)
        except Exception as e:
            base_state = {
                **state,
                "control": mk_control(
                    current_stage="LOAD_DATASET",
                    current_stage_status="ABORTED",
                    action_required="NEEDS_INPUT",
                ),
                "dataset": {
                    **dataset_in,
                    "id": dataset_id,
                    "path": dataset_path,
                    "load_error": f"DATA_LOAD_ERROR: {e}",
                },
            }
            return finalize_with_llm(
                base_state=base_state,
                intent="PARSE_FAILED",
            )

        # Deterministic schema + summary
        n_rows, n_cols = df.shape
        raw_schema = {"columns": [{"name": col, "dtype": str(dtype)} for col, dtype in df.dtypes.items()]} # pyright: ignore[reportUnknownVariableType]
        summary = {"n_rows": int(n_rows), "n_cols": int(n_cols)}

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
                current_stage="LOAD_DATASET",
                current_stage_status="DONE",
                action_required="NONE",
            ),
        }

        cols_preview = [c["name"] for c in cast(list[dict[str, Any]], raw_schema.get("columns", []))[:10]] # pyright: ignore[reportUnknownMemberType]
        preview_str = ", ".join([str(x) for x in cols_preview]) + (" ..." if int(n_cols) > 10 else "")
        data = f"The dataset has {n_rows} rows and {n_cols} columns. Column names include: {preview_str}."

        return finalize_with_llm(
            base_state=base_state_ok,
            intent="LOADED_OK",
            data=data,
        )

    return load_dataset


# -----------------------------------------------------------------------------
# Helpers: robust, but *typed* UUID for get_csv_data(...)
# -----------------------------------------------------------------------------

def _register_csv_dataset_uuid(data_repo: DataRepo, *, state: ConversationState, dataset_path: str) -> UUID:
    """
    Calls DataRepo.register_csv_dataset with whichever signature it supports
    and coerces return into UUID for downstream typing.

    Supported parameter names:
      - dataset_path / path
      - conversation_id (optional)
    """
    fn = getattr(data_repo, "register_csv_dataset")
    sig = inspect.signature(fn)
    params = sig.parameters

    # Prefer keyword calls to avoid positional mistakes
    if "conversation_id" in params and ("dataset_path" in params or "path" in params):
        cid = _find_conversation_id(state)
        if "dataset_path" in params:
            res = fn(conversation_id=cid, dataset_path=dataset_path)  # type: ignore[misc]
        else:
            res = fn(conversation_id=cid, path=dataset_path)  # type: ignore[misc]
        return _coerce_uuid(res)

    if "dataset_path" in params:
        res = fn(dataset_path=dataset_path)  # type: ignore[misc]
        return _coerce_uuid(res)

    if "path" in params:
        res = fn(path=dataset_path)  # type: ignore[misc]
        return _coerce_uuid(res)

    # last resort: positional (still coerce to UUID)
    res = fn(dataset_path)  # type: ignore[misc]
    return _coerce_uuid(res)


def _coerce_uuid(x: Any) -> UUID:
    if isinstance(x, UUID):
        return x
    if isinstance(x, str):
        return UUID(x)
    raise TypeError(f"register_csv_dataset returned non-UUID/non-str id: {type(x).__name__}")


def _find_conversation_id(state: ConversationState) -> UUID:
    """
    Only needed if your repo signature still requires conversation_id.
    Looks in common places. Fail loudly if not found.
    """
    cid = state.get("conversation_id")
    if isinstance(cid, UUID):
        return cid
    if isinstance(cid, str):
        return UUID(cid)

    ctrl = state.get("control")
    if isinstance(ctrl, dict): # pyright: ignore[reportUnnecessaryIsInstance]
        cid2 = ctrl.get("conversation_id")
        if isinstance(cid2, UUID):
            return cid2
        if isinstance(cid2, str):
            return UUID(cid2)

    raise ValueError("conversation_id required by DataRepo.register_csv_dataset but not found in state.")
