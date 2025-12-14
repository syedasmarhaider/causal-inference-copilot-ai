from __future__ import annotations

from python.workflows.state.control_state import ControlState


def ochestrate(control: ControlState) -> ControlState:
    stage = control["stage"]
    outcome = control["outcome"]
    need = control["need"]

    next_stage = stage
    if stage == "GET_DATASET_PATH":
        if outcome == "DONE":
            next_stage = "LOAD_DATASET"

    if stage == "LOAD_DATASET":
        if outcome == "DONE":
            next_stage = "PROPOSE_METADATA"

    elif stage == "PROPOSE_METADATA":
        if outcome in ("DONE", "RETRYABLE_ERROR"):
            next_stage = "CONFIRM_METADATA"

    elif stage == "CONFIRM_METADATA":
        if outcome == "DONE" and need == "NONE":
            next_stage = "SELECT_ESTIMATOR"

    elif stage == "SELECT_ESTIMATOR":
        if outcome == "DONE":
            next_stage = "FIT_MODEL"

    elif stage == "FIT_MODEL":
        if outcome == "DONE":
            next_stage = "PLAN_EFFECTS"

    elif stage == "PLAN_EFFECTS":
        if outcome == "DONE":
            next_stage = "RUN_EFFECTS"

    elif stage == "RUN_EFFECTS":
        if outcome == "DONE":
            next_stage = "DONE"

    if next_stage != stage:
        # reset for the next node execution
        return {**control, "stage": next_stage, "outcome": "NOT_RUN_YET"}

    return control
