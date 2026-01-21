from __future__ import annotations

from typing import Literal, TypedDict


Stage = Literal[
    "LOAD_DATASET",
    "PROPOSE_AND_CONFIRM_METADATA",
    "COMPILE_PROTOCOL",
    "DONE",
]

Status = Literal[
    "PENDING",
    "DONE",
    "ABORTED",
]

ACTION = Literal[
    "NONE",
    "NEEDS_INPUT",
]

NEED_STAGE = Stage


class ControlState(TypedDict):
    current_stage: Stage
    current_stage_status: Status
    action_required: ACTION
    node_message: str | None


def get_string_control_state(control: ControlState) -> str:
    return (
        f"Stage: {control['current_stage']} | "
        f"Status: {control['current_stage_status']} | "
        f"Action Required: {control['action_required']}"
    )