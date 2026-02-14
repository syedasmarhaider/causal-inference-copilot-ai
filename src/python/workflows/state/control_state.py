from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, TypedDict
from collections.abc import Mapping
from typing import Final, Literal, TypedDict

Stage = Literal[
    "LOAD_DATASET",
    "PROTOCOL_DISCUSSION",
    "COMPILE_PROTOCOL",
    "VALIDATE_PROTOCOL_STATIC",
    "VALIDATE_PROTOCOL_STATIC_DISCUSSION",
    "INFERENCE_READY",
    "MODEL_SELECTION",
    "MODEL_SELECTION_DISCUSSION",
    "MODEL_PARAMS_FIT_DISCUSSION", 
    "DONE",
]

CONTROL_STATE_NEXT_STAGE: Final[Mapping[Stage, Stage]] = {
    "LOAD_DATASET": "PROTOCOL_DISCUSSION",
    "PROTOCOL_DISCUSSION": "COMPILE_PROTOCOL",
    "COMPILE_PROTOCOL": "VALIDATE_PROTOCOL_STATIC",
    "VALIDATE_PROTOCOL_STATIC": "VALIDATE_PROTOCOL_STATIC_DISCUSSION",
    "VALIDATE_PROTOCOL_STATIC_DISCUSSION": "INFERENCE_READY",
    "INFERENCE_READY": "MODEL_SELECTION",
    "MODEL_SELECTION": "MODEL_SELECTION_DISCUSSION",
    "MODEL_SELECTION_DISCUSSION": "MODEL_PARAMS_FIT_DISCUSSION", 
    "MODEL_PARAMS_FIT_DISCUSSION": "DONE", 
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
    "VALIDATE_PROTOCOL_STATIC_DISCUSSION": (
        "Discuss the static validation report with the user. "
        "Ensure the user understands any issues flagged in the validation report and has an opportunity to address them."   
        "If the report status is WARN, confirm whether the user wants to proceed anyway or edit the protocol/dataset first. "
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
    "MODEL_SELECTION": (
        "Select the top-3 candidate EconML estimator classes (by exact fully-qualified name) using a deterministic "
        "3-call LLM pipeline: (1) draft candidates from InferenceReadyState + DatasetState.summary + ProtocolState "
        "using embedded EconML library notes; (2) strict refutation/finalization with JSON-only output and validation "
        "against an allow-list; (3) grounded rationale summary. "
        "No user input is required in this stage. "
        "Persist results in ModelSelectionState: draft_text, final_json_raw, final_json, selected_top3, "
        "selection_notes/rejected/unknowns, rationale_text, plus allow-list validation metadata. "
        "On any hard failure (missing inputs, JSON parse/validation failure after retry), abort progression."
    ),
    "MODEL_SELECTION_DISCUSSION": (
        "Run an interactive discussion so the user can pick a single estimator to proceed with. "
        "Inputs are the ModelSelectionState (top-3 + rationale/unknowns) and the allowed estimator allow-list. "
        "This stage is allowed to ask the minimum follow-ups needed. "
        "Termination condition: ModelSelectionDiscussionState.selected_model_fqcn is set to an allowed estimator. "
        "If selected, mark DONE and proceed to DONE stage; otherwise remain PENDING with action_required=NEEDS_INPUT."
    ),
    "MODEL_PARAMS_FIT_DISCUSSION": (
        "Run an interactive discussion to set/confirm the estimator FIT parameters (options.* only). "
        "Use the estimator adapter's get_input_requirements(cmd='FIT', ir=InferenceReadyState) as the "
        "authoritative source of allowed knobs, choices, and defaults. "
        "Apply user changes as a JSON patch to model.model_params_fit.params, re-apply defaults, validate "
        "against requirements, and require explicit confirmation to set model.model_params_fit.confirmed=true. "
        "If confirmed, proceed to DONE; otherwise remain PENDING with action_required=NEEDS_INPUT."
    ),

    "DONE": (
        "Workflow completed successfully. A valid ProtocolState exists, static validation has passed/warned, "
        "an InferenceReadyState has been produced (or a failure has been recorded), ModelSelectionState contains a "
        "validated top-3 EconML estimator shortlist + rationale, and ModelSelectionDiscussionState may contain a "
        "final user-selected estimator to run next."
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
