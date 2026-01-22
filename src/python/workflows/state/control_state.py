from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, TypedDict

# TODO: MAKE IT dynamic with ids and node definitions etc

Stage = Literal[
    "LOAD_DATASET",
    "PROTOCOL_DISCUSSION",
    "COMPILE_PROTOCOL",
    "DONE",
]

CONTROL_STATE_NEXT_STAGE: Final[Mapping[Stage, Stage]] = {
    "LOAD_DATASET": "PROTOCOL_DISCUSSION",
    "PROTOCOL_DISCUSSION": "COMPILE_PROTOCOL",
    "COMPILE_PROTOCOL": "DONE",
}

CONTROL_STATE_STAGE_DOC: Final[Mapping[Stage, str]] = {
    "LOAD_DATASET": (
        "Load the dataset (CSV) for this conversation. "
        "Validate accessibility and basic format, then populate dataset.id, dataset.raw_schema (column names/types if available), "
        "and dataset.summary (row count, missingness hints, basic stats). "
        "If loading fails, set dataset.load_error and ask the user to fix the file/path."
    ),
    "PROTOCOL_DISCUSSION": (
        "Interactive protocol intake (target-trial mindset). "
        "Maintain a single PROTOCOL_DISCUSSION Q/A document: fill or refine answers using chat history, "
        "ask only the minimum follow-up questions when something is missing/UNCLEAR, and request user confirmation when complete. "
        "Outputs a stable, human-readable study description that can be compiled into a strict ProtocolState."
    ),
    "COMPILE_PROTOCOL": (
        "Compile the confirmed PROTOCOL_DISCUSSION into a machine-readable ProtocolState. "
        "Enforce strict schema + enum constraints, reject invented or ungrounded details, and return either a valid ProtocolState JSON "
        "or a detailed FEEDBACK message explaining exactly what is missing or inconsistent. "
        "On failure, route back to PROTOCOL_DISCUSSION to fix the specific gaps."
    ),
    "DONE": (
        "Workflow complete. A valid ProtocolState has been produced and stored. "
        "Downstream steps (e.g., static validation, DAG/identification, estimator selection) may proceed from this locked protocol."
    ),
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