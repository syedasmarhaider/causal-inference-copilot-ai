from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, TypedDict



Stage = Literal[
    "LOAD_DATASET",
    "PROPOSE_AND_CONFIRM_METADATA",
    "COMPILE_PROTOCOL",
    "VALIDATE_PROTOCOL_STATIC",
    "DONE",
]

CONTROL_STATE_NEXT_STAGE: Final[Mapping[Stage, Stage]] = {
    "LOAD_DATASET": "PROPOSE_AND_CONFIRM_METADATA",
    "PROPOSE_AND_CONFIRM_METADATA": "COMPILE_PROTOCOL",
    "COMPILE_PROTOCOL": "VALIDATE_PROTOCOL_STATIC",
    "VALIDATE_PROTOCOL_STATIC": "DONE",
}

CONTROL_STATE_STAGE_DOC: Final[Mapping[Stage, str]] = {
    "LOAD_DATASET": "Load CSV from dataset.path. Writes dataset.summary/raw_schema (and maybe dataset.id).",
    "PROPOSE_AND_CONFIRM_METADATA": "Propose+confirm metadata: treatment/outcome/controls/covariates/etc.",
    "COMPILE_PROTOCOL": "Compile protocol state.",
    "VALIDATE_PROTOCOL_STATIC": "Validate the protocol statically.",
    "DONE": "Workflow complete.",
}

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