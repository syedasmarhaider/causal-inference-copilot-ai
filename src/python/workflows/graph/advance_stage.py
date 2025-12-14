# src/python/workflows/graph/advance_stage.py
from __future__ import annotations

from typing import cast
from uuid import uuid4

from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState, Stage

_STAGE_FLOW: dict[Stage, Stage] = {
    "LOAD_DATASET": "PROPOSE_METADATA",
    "PROPOSE_METADATA": "CONFIRM_METADATA",
    "CONFIRM_METADATA": "SELECT_ESTIMATOR",
    "SELECT_ESTIMATOR": "FIT_MODEL",
    "FIT_MODEL": "PLAN_EFFECTS",
    "PLAN_EFFECTS": "RUN_EFFECTS",
    "RUN_EFFECTS": "DONE",
    "DONE": "DONE",
}


def _bootstrap_control() -> ControlState:
    # For local testing; in prod you might prefer UI to set conversation_id.
    return {
        "conversation_id": uuid4(),
        "status": "PENDING",
        "stage": "LOAD_DATASET",
        "outcome": "NOT_RUN_YET",
        "need": "DATASET_PATH",
        "interrupt_type": None,
        "last_error": None,
        "node_message": "Provide a CSV path to begin.",
    }


def advance_stage_node(state: ConversationState) -> ConversationState:
    if "control" not in state:
        return {**state, "control": _bootstrap_control()}

    control = cast(ControlState, state["control"]) # pyright: ignore[reportUnnecessaryCast]
    stage = control["stage"]
    outcome = control["outcome"]

    # Default: do nothing; conditional edges will route to current stage.
    next_control = control

    # Stage advancement rule:
    # - only advance if the last stage-node reported DONE
    if outcome == "DONE":
        nxt = _STAGE_FLOW[stage]
        next_control = { # pyright: ignore[reportUnknownVariableType]
            **control,
            "stage": nxt,
            "outcome": "NOT_RUN_YET",  # important: make next node execute
            "need": "NONE",
            "status": "OK",
            "interrupt_type": None,
        }

    # If user canceled / aborted, force DONE (optional)
    if outcome == "ABORTED":
        next_control = {**control, "stage": "DONE", "need": "NONE"} # pyright: ignore[reportUnknownVariableType]

    # If we're waiting for input, keep stage stable; router does not bounce around.
    if outcome in ("NEEDS_INPUT", "RETRYABLE_ERROR", "FAILED"):
        # stay on the same stage; next invocation will run same stage again
        # (or you can implement retry policy here based on attempt counters)
        next_control = control

    return {**state, "control": next_control} # pyright: ignore[reportUnknownVariableType, reportReturnType]
