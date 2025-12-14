from __future__ import annotations

from python.workflows.state.conversation_state import ConversationState


def route_by_stage(state: ConversationState) -> str:
    control = state.get("control", {})  # type: ignore[assignment]
    stage = control.get("stage", "LOAD_DATASET")

    # Keep it simple: stage string == node key
    if stage in {
        "LOAD_DATASET",
        "PROPOSE_METADATA",
        "CONFIRM_METADATA",
        "SELECT_ESTIMATOR",
        "FIT_MODEL",
        "PLAN_EFFECTS",
        "RUN_EFFECTS",
        "DONE",
    }:
        return stage

    return "LOAD_DATASET"
