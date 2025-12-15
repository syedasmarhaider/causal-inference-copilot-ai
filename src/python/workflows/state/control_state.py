from __future__ import annotations

from typing import Literal, NotRequired, TypedDict
from uuid import UUID

from python.workflows.utils.types import JSONDict


Stage = Literal[
    "GET_FILE",
    "LOAD_DATASET",
    "PROPOSE_METADATA",
    "CONFIRM_METADATA",
    "SELECT_ESTIMATOR",
    "FIT_MODEL",
    "PLAN_EFFECTS",
    "RUN_EFFECTS",
    "DONE",
]

Status = Literal[
    "PENDING",
    "DONE", 
    "RETRYABLE_ERROR",
    "ABORTED",
]

Need = Literal[
    "NONE",
    "PRESENT",
    "NEEDS_INPUT",
    "PRESENT_AND_USER_INPUT",
]

class ControlState(TypedDict):
    conversation_id: UUID
    stage: Stage
    status: Status
    need: Need
    last_error: JSONDict | None
    node_message: str
    pending_stage: NotRequired[Stage | None]