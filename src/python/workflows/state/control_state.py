from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, TypedDict

Stage = Literal[
    "LOAD_DATASET",
    "PROTOCOL_DISCUSSION",
    "COMPILE_PROTOCOL",
    "INFERENCE_READY",
    "VALIDATE_INFERENCE_READY",
    "VALIDATE_INFERENCE_READY_DISCUSSION",
    "MODEL_SELECTION",
    "MODEL_SELECTION_DISCUSSION",
    "MODEL_PARAMS_FIT_DISCUSSION",
    "MODEL_FIT",
    "DONE",
]

Status = Literal["PENDING", "DONE", "ABORTED"]

ACTION = Literal["NONE", "NEEDS_INPUT"]

NeedStage = Stage

CONTROL_STATE_NEXT_STAGE: Final[Mapping[Stage, Stage]] = {
    "LOAD_DATASET": "PROTOCOL_DISCUSSION",
    "PROTOCOL_DISCUSSION": "COMPILE_PROTOCOL",
    "COMPILE_PROTOCOL": "INFERENCE_READY",
    "INFERENCE_READY": "VALIDATE_INFERENCE_READY",
    "VALIDATE_INFERENCE_READY": "VALIDATE_INFERENCE_READY_DISCUSSION",
    "VALIDATE_INFERENCE_READY_DISCUSSION": "MODEL_SELECTION",
    "MODEL_SELECTION": "MODEL_SELECTION_DISCUSSION",
    "MODEL_SELECTION_DISCUSSION": "MODEL_PARAMS_FIT_DISCUSSION",
    "MODEL_PARAMS_FIT_DISCUSSION": "MODEL_FIT",
    "MODEL_FIT": "DONE",
}

CONTROL_STATE_STAGE_DOC: Final[Mapping[Stage, str]] = {
    "LOAD_DATASET": (
        "Load the dataset artifact for this conversation and populate DatasetState. "
        "Verify the file is readable and tabular. Populate dataset identifiers, raw schema, "
        "and a lightweight summary (row/column counts, missingness, minimal stats). "
        "On failure, set dataset.load_error and return a user-actionable error message."
    ),
    "PROTOCOL_DISCUSSION": (
        "Interactive intake to define a target-trial-style causal protocol in plain language. "
        "Ask minimal follow-ups, keep a single evolving discussion record, and obtain explicit confirmation."
    ),
    "COMPILE_PROTOCOL": (
        "Convert the confirmed discussion record into a strict ProtocolState. "
        "Enforce schema/enums; do not invent columns/windows/semantics. "
        "If compilation fails, route back to PROTOCOL_DISCUSSION with precise fix instructions."
    ),
    "INFERENCE_READY": (
        "Build InferenceReadyState from (DatasetState + ProtocolState). "
        "Apply cohort exclusions, canonicalize treatment/outcome encodings, and compute EconML-ready columns "
        "(T_col, Y_cols, W_cols, X_cols) plus feature sets (X/W/XW). "
        "Optionally materialize and attach a prepared dataset artifact (dataset_id + schema_fingerprint + summary). "
        "On hard failure: set inference_ready.status='FAILED' with details and abort."
    ),
    "VALIDATE_INFERENCE_READY": (
        "Run deterministic validation of InferenceReadyState + prepared dataset artifact. "
        "Validate: non-empty column sets, column existence in prepared data, missingness/NaN constraints, "
        "arm sizes/overlap sanity gates, outcome/treatment encoding sanity, and any adapter prerequisites. "
        "Write InferenceReadyValidationState.report with PASS/WARN/FAIL + metrics + fix hints."
    ),
    "VALIDATE_INFERENCE_READY_DISCUSSION": (
        "Discuss the inference-ready validation report with the user. "
        "If status is WARN: confirm proceed vs abort/edit. "
        "If status is FAIL: you MUST keep the workflow in discussion/edit mode (never proceed). "
        "If status is PASS: do not require user input; proceed automatically."
    ),
    "MODEL_SELECTION": (
        "Select top-3 candidate EconML estimators (exact FQCNs) using a deterministic 3-call LLM pipeline "
        "grounded in InferenceReadyState + prepared dataset summary + ProtocolState, validated against an allow-list. "
        "Persist ModelSelectionState: allowed list/map, draft_text, final_json_raw/final_json, selected_top3, "
        "selection_notes/rejected/unknowns, rationale_text."
    ),
    "MODEL_SELECTION_DISCUSSION": (
        "Interactive stage for user to pick one estimator. "
        "Termination: model_state.selected_model_fqcn is set to an allowed estimator; otherwise stay PENDING."
    ),
    "MODEL_PARAMS_FIT_DISCUSSION": (
        "Interactive stage to set/confirm FIT parameters (options.* only) using adapter "
        "get_input_requirements(cmd='FIT', ir=InferenceReadyState) as the authoritative knob/choices/defaults source. "
        "Apply JSON patch into model_state.model_params_fit.params, re-apply defaults, validate, and require explicit "
        "confirmation to set model_state.model_params_fit.confirmed=true."
    ),
    "MODEL_FIT": (
        "Fit the selected estimator on the prepared dataset (no LLM). "
        "Requires model_state.selected_model_fqcn and model_state.model_params_fit.confirmed=true. "
        "Generate model_id, persist model via adapter, then proceed."
    ),
    "DONE": (
        "Workflow completed successfully (protocol compiled, inference-ready produced + validated, "
        "top-3 shortlist generated, final model optionally fit)."
    ),
}

class ControlState(TypedDict):
    """
    Minimal workflow controller for deterministic stage routing.

    - current_stage: active workflow stage
    - current_stage_status: PENDING/DONE/ABORTED
    - action_required: NONE/NEEDS_INPUT
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
