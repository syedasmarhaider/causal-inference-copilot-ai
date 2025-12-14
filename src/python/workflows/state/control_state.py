# src/python/workflows/state/control_state.py
from __future__ import annotations

from typing import Any, Literal, TypeAlias, TypedDict
from uuid import UUID

JSONDict: TypeAlias = dict[str, Any]

Stage: TypeAlias = Literal[
    "LOAD_DATASET",
    "PROPOSE_METADATA",
    "CONFIRM_METADATA",
    "SELECT_ESTIMATOR",
    "FIT_MODEL",
    "PLAN_EFFECTS",
    "RUN_EFFECTS",
    "DONE",
]

Status: TypeAlias = Literal["OK", "ERROR", "PENDING"]

Outcome: TypeAlias = Literal[
    "NOT_RUN_YET",
    "IN_PROGRESS",
    "DONE",
    "FAILED",
    "NEEDS_INPUT",
    "RETRYABLE_ERROR",
    "ABORTED",
]

Need: TypeAlias = Literal[
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
