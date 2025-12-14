# src/python/workflows/state/control_state.py
from __future__ import annotations

from typing import Literal, TypedDict
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

Status = Literal["OK", "ERROR", "PENDING"]

Outcome = Literal[
    "NOT_RUN_YET",
    "IN_PROGRESS",
    "DONE",
    "FAILED",
    "NEEDS_INPUT",
    "RETRYABLE_ERROR",
    "ABORTED",
]

Need = Literal[
    "NONE",
    "DATASET_PATH",
    "TREATMENT_OUTCOME",
    "CONFIRM_METADATA",
    "ESTIMATOR_CHOICE",
    "FIT_CONFIRMATION",
    "EFFECT_PLAN_CONFIRMATION",
]

class ControlState(TypedDict):
    conversation_id: UUID
    status: Status
    stage: Stage
    outcome: Outcome
    need: Need
    interrupt_type: str | None
    last_error: JSONDict | None
    node_message: str
