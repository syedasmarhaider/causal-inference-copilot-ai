from __future__ import annotations

from typing import Literal, TypedDict


Stage = Literal[
    "GET_FILE",
    "LOAD_DATASET",
    "PROPOSE_AND_CONFIRM_METADATA",
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

