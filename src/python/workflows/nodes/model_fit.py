from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import UUID, uuid4

from python.workflows.state.conversation_state import (
    CallableNodeFunc,
    ConversationState,
    ConversationStateHelpers,
)
from python.workflows.state.model_state import ModelState
from python.workflows.tools.inference.causal_inference_factory import CausalInferenceFactory
from python.workflows.tools.inference.models.causal_command import CausalCommand

log = logging.getLogger(__name__)


def make_model_fit_node(*, causal_factory: CausalInferenceFactory) -> CallableNodeFunc:
    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        return _run(
            user_id=user_id,
            conversation_id=conversation_id,
            state=state,
            causal_factory=causal_factory,
        )

    return node


def _run(
    *,
    user_id: UUID,
    conversation_id: UUID,
    state: ConversationState,
    causal_factory: CausalInferenceFactory,
) -> ConversationState:
    ir = state.get("inference_ready")
    if not ir:
        return _abort(state, "InferenceReadyState missing. Run INFERENCE_READY before MODEL_FIT.")
    if ir.get("error"):
        return _abort(state, f"InferenceReadyState has error: {ir.get('error')}")

    prepared = ir.get("prepared")
    if not prepared:
        return _abort(state, "InferenceReadyState.prepared missing. Re-run INFERENCE_READY to prepare dataset.")

    dataset_id: UUID | None = prepared.get("dataset_id")
    if not dataset_id:
        return _abort(state, "InferenceReadyState.prepared.dataset_id missing. Re-run INFERENCE_READY.")

    model_state: ModelState | None = state.get("model_state")
    if model_state is None or not (model_state.get("selected_model_fqcn") or "").strip():
        return _abort(state, "No selected model found. Run MODEL_SELECTION_DISCUSSION first.")

    estimator_fqcn: str = (model_state.get("selected_model_fqcn") or "").strip()

    mpf = model_state.get("model_params_fit")
    if not isinstance(mpf, dict):
        return _abort(state, "ModelParamsFitState missing. Run MODEL_PARAMS_FIT_DISCUSSION first.")
    if mpf.get("confirmed") is not True:
        return _abort(state, "Fit params are not confirmed. Complete MODEL_PARAMS_FIT_DISCUSSION first.")


    causal_inference = causal_factory.resolve(estimator_fqcn)
    if causal_inference is None:
        return _abort(state, f"Unsupported estimator: {estimator_fqcn}")


    defaults = {}

    user_params: Dict[str, Any] | None = mpf.get("params")
    if user_params is None:
        user_params = {}
        
    options: Dict[str, Any] = {**defaults, **dict(user_params)}

    # Generate model_id for this fit attempt; persist ONLY on success (retry-friendly).
    model_id = uuid4()

    command = CausalCommand(
        cmd="FIT",
        estimator_fqcn=estimator_fqcn,
        dataset_id=dataset_id,
        inputs={
            "T_col": ir.get("T_col"),
            "Y_cols": ir.get("Y_cols"),
            "W_cols": ir.get("W_cols"),
            "X_cols": ir.get("X_cols"),
            "feature_sets": ir.get("feature_sets"),
        },
        options=options,
        meta={
            "run_id": str(conversation_id),
            "model_id": str(model_id),
            "schema_fingerprint": prepared.get("schema_fingerprint"),
        },
    )

    try:
        result = causal_inference.execute(
            command,
            user_id=user_id,
            conversation_id=conversation_id,
            model_id=model_id,
            ir=ir,
        )

        # Persist model_id after successful fit
        mpf["model_id"] = str(model_id)
        model_state["model_params_fit"] = mpf
        state["model_state"] = model_state

        status = getattr(result, "status", None) or getattr(result, "ok", None) or "OK"
        msg = f"Model fit complete. estimator={estimator_fqcn} | model_id={model_id} | status={status}"
        ConversationStateHelpers.append_ai_message(state=state, content=msg)
        return ConversationStateHelpers.set_done(state=state, action="NONE", msg=msg)

    except Exception as e:
        log.exception("MODEL_FIT: execute(FIT) failed: %s", e)
        err = f"Model fit failed. estimator={estimator_fqcn} | model_id={model_id} | error={e}"
        ConversationStateHelpers.append_ai_message(state=state, content=err)
        return ConversationStateHelpers.set_abort(state=state, action="NONE", msg=err)


def _abort(state: ConversationState, msg: str) -> ConversationState:
    ConversationStateHelpers.append_ai_message(state=state, content=msg)
    return ConversationStateHelpers.set_abort(state=state, action="NONE", msg=msg)
