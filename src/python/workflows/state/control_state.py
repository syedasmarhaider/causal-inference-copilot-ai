from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, TypedDict

Stage = Literal[
    "LOAD_DATASET",
    "PROTOCOL_DISCUSSION",
    "COMPILE_PROTOCOL",
    "VALIDATE_PROTOCOL_STATIC",
    "INFERENCE_READY",
    "DONE",
]

CONTROL_STATE_NEXT_STAGE: Final[Mapping[Stage, Stage]] = {
    "LOAD_DATASET": "PROTOCOL_DISCUSSION",
    "PROTOCOL_DISCUSSION": "COMPILE_PROTOCOL",
    "COMPILE_PROTOCOL": "VALIDATE_PROTOCOL_STATIC",
    "VALIDATE_PROTOCOL_STATIC": "INFERENCE_READY",
    "INFERENCE_READY": "DONE",
}

CONTROL_STATE_STAGE_DOC: Final[Mapping[Stage, str]] = {
    "LOAD_DATASET": (
        "Load the dataset artifact for this conversation and populate DatasetState. "
        "Verify the file is readable and tabular, then set dataset.id/path, raw schema (columns + dtypes if available), "
        "and a lightweight summary (row/column counts, basic missingness, and minimal descriptive stats). "
        "On failure, set dataset.load_error and return a user-actionable error message."
    ),
    "PROTOCOL_DISCUSSION": (
        "Run an interactive study intake to define a target-trial-style causal protocol in plain language. "
        "Maintain a single evolving discussion record, ask only the minimum follow-ups needed to remove ambiguity, "
        "and obtain explicit user confirmation once all required components are specified. "
        "Output is a stable human-readable protocol description suitable for compilation."
    ),
    "COMPILE_PROTOCOL": (
        "Convert the confirmed discussion record into a strict ProtocolState. "
        "Enforce schema requirements and enums; do not invent columns, windows, or semantics not grounded in the discussion. "
        "Output either (a) a valid ProtocolState object or (b) a precise feedback message describing what must change. "
        "On (b), route back to PROTOCOL_DISCUSSION to resolve the specific issues."
    ),
    "VALIDATE_PROTOCOL_STATIC": (
        "Run deterministic, leakage-agnostic validation of ProtocolState against the loaded dataset. "
        "Check column existence, basic type/shape constraints, feasibility gates (minimum N, arm sizes/imbalance), "
        "and missingness/outcome sanity thresholds. "
        "Write ProtocolStaticValidationState.report with PASS/WARN/FAIL plus metrics and fix hints. "
        "If FAIL, require protocol edits (or dataset fixes) before continuing."
    ),
    "INFERENCE_READY": (
        "Build InferenceReadyState from (DatasetState + ProtocolState + ProtocolStaticValidationState). "
        "This stage must only proceed if protocol_static_validation.report.status is PASS or WARN. "
        "Apply cohort exclusions, canonicalize treatment/outcome encodings (aliases -> canonical), and compute "
        "EconML-ready column sets (T_col, Y_cols, W_cols, X_cols) plus feature_sets (W, X, XW). "
        "Populate prepared_columns metadata (dtype, missing_rate, n_unique, encoding/imputation decisions), "
        "exclusions_summary audit trail, and PreparationMetrics (row counts, treated/control/event counts when applicable). "
        "Optionally materialize a prepared dataset artifact and attach PreparedDatasetArtifact to inference_ready.prepared. "
        "On hard failure, set inference_ready.status=FAILED with error and abort progression."
    ),
    "DONE": (
        "Workflow completed successfully. A valid ProtocolState exists, static validation has passed/warned, "
        "and an InferenceReadyState has been produced (or a failure has been recorded). "
        "Downstream gates (e.g., leakage/temporal legality, DAG/identification, estimator selection) can proceed from this state."
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

NeedStage = Stage


class ControlState(TypedDict):
    """
    Minimal workflow controller for deterministic stage routing.

    - current_stage: the active workflow stage
    - current_stage_status: status of the current stage
    - action_required: whether the system needs user input to proceed
    - node_message: optional user-facing message produced by the last node
    """
    current_stage: Stage
    current_stage_status: Status
    action_required: ACTION
    node_message: str | None


def get_string_control_state(control: ControlState) -> str:
    node_msg = control.get("node_message")
    base = (
        f"Stage: {control['current_stage']} | "
        f"Status: {control['current_stage_status']} | "
        f"Action Required: {control['action_required']}"
    )
    return f"{base} | Message: {node_msg}" if node_msg else base
