from __future__ import annotations

import logging
from typing import Any, Dict, List
from uuid import UUID, uuid4

from python.workflows.state.conversation_state import (
    CallableNodeFunc,
    ConversationState,
    ConversationStateHelpers,
)
from python.workflows.state.model_state import ModelState
from python.workflows.tools.inference.causal_inference_factory import CausalInferenceFactory
from python.workflows.tools.inference.models.causal_command import CausalCommand, Issue

log = logging.getLogger(__name__)

RAISE_ON_MODEL_FIT_ABORT: bool = True  # raise on any early-abort (missing state, not confirmed, unsupported, etc.)
RAISE_ON_MODEL_FIT_ERROR: bool = True  # raise on execute(FIT) exception


def make_model_fit_node(*, causal_factory: CausalInferenceFactory) -> CallableNodeFunc:
    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        return _run(
            user_id=user_id,
            conversation_id=conversation_id,
            state=state,
            causal_factory=causal_factory,
        )

    return node


def _safe_repr(x: Any, *, max_len: int = 4000) -> str:
    try:
        s = repr(x)
    except Exception:
        s = f"<unreprable {type(x).__name__}>"
    if len(s) > max_len:
        return s[:max_len] + "…<truncated>"
    return s


def _log_issue_details(issues: List[Issue]) -> None:
    log.warning("MODEL_FIT: issues_count=%d", len(issues))
    for i, issue in enumerate(issues):
        # Keep your existing structured fields, but also log full repr for debugging.
        try:
            log.warning(
                "MODEL_FIT: issue[%d]: code=%s message=%s path=%s fix_hint=%s required=%s",
                i,
                getattr(issue, "code", None),
                getattr(issue, "message", None),
                getattr(issue, "path", None),
                getattr(issue, "fix_hint", None),
                getattr(issue, "required", None),
            )
        except Exception:
            log.exception("MODEL_FIT: issue[%d] logging failed; issue_repr=%s", i, _safe_repr(issue))


def _maybe_raise_abort(msg: str) -> None:
    if RAISE_ON_MODEL_FIT_ABORT:
        raise RuntimeError(msg)


def _maybe_raise_error(exc: Exception, msg: str) -> None:
    if RAISE_ON_MODEL_FIT_ERROR:
        raise RuntimeError(msg) from exc


def _run(
    *,
    user_id: UUID,
    conversation_id: UUID,
    state: ConversationState,
    causal_factory: CausalInferenceFactory,
) -> ConversationState:
    log.info(
        "MODEL_FIT: start user_id=%s conversation_id=%s stage=MODEL_FIT",
        str(user_id),
        str(conversation_id),
    )

    ir = state.get("inference_ready")
    if ir is None:
        msg = "InferenceReadyState missing. Run INFERENCE_READY before MODEL_FIT."
        log.error("MODEL_FIT: abort: %s | user_id=%s conversation_id=%s", msg, str(user_id), str(conversation_id))
        _maybe_raise_abort(msg)
        return _abort(state, msg)

    if ir.get("error"):
        msg = f"InferenceReadyState has error: {ir.get('error')}"
        log.error(
            "MODEL_FIT: abort: %s | user_id=%s conversation_id=%s | ir_error=%s | ir_repr=%s",
            msg,
            str(user_id),
            str(conversation_id),
            _safe_repr(ir.get("error")),
            _safe_repr(ir),
        )
        _maybe_raise_abort(msg)
        return _abort(state, msg)

    prepared = ir.get("prepared_dataset")
    if not prepared:
        msg = "InferenceReadyState.prepared missing. Re-run INFERENCE_READY to prepare dataset."
        log.error(
            "MODEL_FIT: abort: %s | user_id=%s conversation_id=%s | ir_keys=%s | ir_repr=%s",
            msg,
            str(user_id),
            str(conversation_id),
            sorted(list(ir.keys())),
            _safe_repr(ir),
        )
        _maybe_raise_abort(msg)
        return _abort(state, msg)

    dataset_id: UUID | None = prepared.get("id")
    if not dataset_id:
        msg = "InferenceReadyState.prepared.dataset_id missing. Re-run INFERENCE_READY."
        log.error(
            "MODEL_FIT: abort: %s | user_id=%s conversation_id=%s | prepared_keys=%s | prepared_repr=%s",
            msg,
            str(user_id),
            str(conversation_id),
            sorted(list(prepared.keys())),
            _safe_repr(prepared),
        )
        _maybe_raise_abort(msg)
        return _abort(state, msg)

    model_state: ModelState | None = state.get("model_state")
    if model_state is None or not (model_state.get("selected_model_fqcn") or "").strip():
        msg = "No selected model found. Run MODEL_SELECTION_DISCUSSION first."
        log.error(
            "MODEL_FIT: abort: %s | user_id=%s conversation_id=%s | model_state_repr=%s",
            msg,
            str(user_id),
            str(conversation_id),
            _safe_repr(model_state),
        )
        _maybe_raise_abort(msg)
        return _abort(state, msg)

    estimator_fqcn: str = (model_state.get("selected_model_fqcn") or "").strip()

    mpf = model_state.get("model_params_fit")
    if not isinstance(mpf, dict):
        msg = "ModelParamsFitState missing. Run MODEL_PARAMS_FIT_DISCUSSION first."
        log.error(
            "MODEL_FIT: abort: %s | user_id=%s conversation_id=%s | estimator=%s | model_params_fit_type=%s | model_state_repr=%s",
            msg,
            str(user_id),
            str(conversation_id),
            estimator_fqcn,
            type(mpf).__name__,
            _safe_repr(model_state),
        )
        _maybe_raise_abort(msg)
        return _abort(state, msg)

    if mpf.get("confirmed") is not True:
        msg = "Fit params are not confirmed. Complete MODEL_PARAMS_FIT_DISCUSSION first."
        log.error(
            "MODEL_FIT: abort: %s | user_id=%s conversation_id=%s | estimator=%s | confirmed=%s | mpf_repr=%s",
            msg,
            str(user_id),
            str(conversation_id),
            estimator_fqcn,
            _safe_repr(mpf.get("confirmed")),
            _safe_repr(mpf),
        )
        _maybe_raise_abort(msg)
        return _abort(state, msg)

    causal_inference = causal_factory.resolve(estimator_fqcn)
    if causal_inference is None:
        msg = f"Unsupported estimator: {estimator_fqcn}"
        log.error(
            "MODEL_FIT: abort: %s | user_id=%s conversation_id=%s | estimator=%s | factory=%s",
            msg,
            str(user_id),
            str(conversation_id),
            estimator_fqcn,
            _safe_repr(causal_factory),
        )
        _maybe_raise_abort(msg)
        return _abort(state, msg)

    defaults: Dict[str, Any] = {}

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

    # Log *everything* we can about the fit context (bounded repr to avoid exploding logs).
    log.info(
        "MODEL_FIT: prepared fit command | user_id=%s conversation_id=%s estimator=%s dataset_id=%s model_id=%s",
        str(user_id),
        str(conversation_id),
        estimator_fqcn,
        str(dataset_id),
        str(model_id),
    )
    log.debug("MODEL_FIT: ir_summary T_col=%s Y_cols=%s W_cols=%s X_cols=%s feature_sets=%s",
              _safe_repr(ir.get("T_col")),
              _safe_repr(ir.get("Y_cols")),
              _safe_repr(ir.get("W_cols")),
              _safe_repr(ir.get("X_cols")),
              _safe_repr(ir.get("feature_sets")))
    log.debug("MODEL_FIT: options=%s", _safe_repr(options))
    log.debug("MODEL_FIT: command=%s", _safe_repr(command))

    try:
        result = causal_inference.execute(
            command,
            user_id=user_id,
            conversation_id=conversation_id,
            model_id=model_id,
            ir=ir,
        )

        log.info(
            "MODEL_FIT: execute returned | user_id=%s conversation_id=%s estimator=%s model_id=%s result_type=%s result_repr=%s",
            str(user_id),
            str(conversation_id),
            estimator_fqcn,
            str(model_id),
            type(result).__name__,
            _safe_repr(result),
        )

        issues: List[Issue] = getattr(result, "issues", [])
        if issues:
            _log_issue_details(issues)

        # Keep your existing per-issue log line too (unchanged semantics).
        for issue in issues:
            logging.warning(
                "Issue: code=%s message=%s path=%s fix_hint=%s required=%s",
                issue.code,
                issue.message,
                issue.path,
                issue.fix_hint,
                issue.required,
            )

        # Persist model_id after successful fit
        mpf["model_id"] = str(model_id)
        model_state["model_params_fit"] = mpf
        state["model_state"] = model_state

        status = getattr(result, "status", None) or getattr(result, "ok", None) or "OK"
        msg = f"Model fit complete. estimator={estimator_fqcn} | model_id={model_id} | status={status}"
        ConversationStateHelpers.append_ai_message(state=state, content=msg)

        log.info(
            "MODEL_FIT: success | user_id=%s conversation_id=%s estimator=%s model_id=%s status=%s",
            str(user_id),
            str(conversation_id),
            estimator_fqcn,
            str(model_id),
            _safe_repr(status),
        )

        return ConversationStateHelpers.set_done(state=state, action="NONE", msg=msg)

    except Exception as e:
        # Exhaustive error logging
        log.exception(
            "MODEL_FIT: execute(FIT) failed | user_id=%s conversation_id=%s estimator=%s dataset_id=%s model_id=%s exc_type=%s exc=%s",
            str(user_id),
            str(conversation_id),
            estimator_fqcn,
            str(dataset_id),
            str(model_id),
            type(e).__name__,
            str(e),
        )
        log.error("MODEL_FIT: command=%s", _safe_repr(command))
        log.error("MODEL_FIT: options=%s", _safe_repr(options))
        log.error("MODEL_FIT: ir_repr=%s", _safe_repr(ir))
        log.error("MODEL_FIT: prepared_repr=%s", _safe_repr(prepared))
        log.error("MODEL_FIT: model_state_repr=%s", _safe_repr(model_state))
        log.error("MODEL_FIT: mpf_repr=%s", _safe_repr(mpf))

        err = f"Model fit failed. estimator={estimator_fqcn} | model_id={model_id} | error={e}"
        ConversationStateHelpers.append_ai_message(state=state, content=err)

        # In test mode, fail fast after logging (one flag flip).
        _maybe_raise_error(e, err)

        return ConversationStateHelpers.set_abort(state=state, action="NONE", msg=err)


def _abort(state: ConversationState, msg: str) -> ConversationState:
    ConversationStateHelpers.append_ai_message(state=state, content=msg)
    return ConversationStateHelpers.set_abort(state=state, action="NONE", msg=msg)
