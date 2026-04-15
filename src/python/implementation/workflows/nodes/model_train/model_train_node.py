from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast
from uuid import UUID, uuid4

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.model_train.model_train_deps import ModelTrainDeps
from python.implementation.workflows.nodes.model_train.model_train_prompts import (
    get_model_failure_summary_prompt,
    get_model_train_node_info,
)
from python.implementation.workflows.nodes.model_train.model_train_state import (
    ModelTrainPayloadModel,
    ModelTrainState,
)
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.causal.inference.causal_command import (
    CommandFailure,
    FitCommand,
    FitInputs,
    FitSuccess,
)
from python.implementation.workflows.tools.causal.inference.causal_model_factory_tool import (
    CausalModelFactoryTool,
)
from python.implementation.workflows.tools.causal.inference.econml.models_meta import (
    get_model_training_label,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.utils.utils import safe_err

log = get_app_logger(__name__, component="model_train_node", log_type="node")
_MAX_TRAINING_ATTEMPTS = 2


class ModelTrainNode(Node):
    NAME: ClassVar[str] = ModelTrainState.NAME

    def __init__(
        self,
        *,
        llm: LLMService,
        data_repo: DataRepo,
        tools_factory: ToolFactory,
    ) -> None:
        self._llm = llm
        self._data_repo = data_repo
        factory_raw = tools_factory.get_tool(CausalModelFactoryTool.NAME)
        self._model_factory = cast(CausalModelFactoryTool, factory_raw)
        profiling_raw = tools_factory.get_tool(DatasetProfilingTool.NAME)
        self._profiling_tool = cast(DatasetProfilingTool, profiling_raw)

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_model_train_node_info()

    def run(
        self,
        *,
        request: NodeRequest,
    ) -> NodeExecutionResult:
        if not isinstance(request.node_state, ModelTrainState):
            raise TypeError(
                f"{self.name}: expected ModelTrainState, got "
                f"{type(request.node_state).__name__}"
            )

        payload = request.node_state.payload.model_copy(deep=True)
        deps = ModelTrainDeps.from_request(request)

        if deps.dataset_id is None:
            return self._needs_data_result(
                request=request,
                user_message=(
                    "I need the compiled working dataset before I can train the selected model."
                ),
            )

        if (
            deps.causal_spec is None
            or deps.transformation_plan is None
            or deps.selected_model is None
        ):
            return self._needs_input_result(
                request=request,
                payload=ModelTrainPayloadModel(),
                user_message=(
                    "I need the confirmed compiled setup and a confirmed selected model "
                    "before I can start training."
                ),
            )

        training_signature = _training_signature(deps=deps)
        if payload.training_signature != training_signature:
            payload = payload.reset_for_signature(training_signature=training_signature)

        if payload.trained_model_id is not None and payload.error_message is None:
            return self._done_result(
                request=request,
                payload=payload,
                user_message=payload.assistant_message
                or "The selected model has already been trained for the current setup.",
            )

        if payload.error_message is not None:
            return self._aborted_result(
                request=request,
                payload=payload,
                user_message=payload.assistant_message
                or "Model training failed for the current setup.",
            )

        try:
            df = self._data_repo.get_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=deps.dataset_id,
            )
        except Exception as exc:
            log.exception("MODEL_TRAIN failed to load compiled dataset", error=exc)
            return self._needs_data_result(
                request=request,
                user_message=(
                    "I could not load the compiled dataset needed for model training. "
                    "Please retry after the dataset is available."
                ),
            )

        if df.empty:
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=(
                    "The compiled dataset is empty, so model training cannot proceed. "
                    "Please review the compilation and filtering steps first."
                ),
                error_message="compiled dataset is empty",
            )

        dataset_summary = deps.dataset_summary
        if dataset_summary is None:
            dataset_summary = self._profiling_tool.extract_dataset_summary(
                df,
                max_categories=200,
                sample_distinct=200,
                compute_quantiles=False,
                strict=True,
            )

        try:
            inference_ready_spec = InferenceReadyCausalSpec(
                causal_spec=deps.causal_spec,
                transformation_plan=deps.transformation_plan,
                data_summary=dataset_summary,
            )
        except Exception as exc:
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=(
                    "The compiled dataset, causal specification, and transformation plan are "
                    "not consistent enough for training yet. Please revise the upstream setup."
                ),
                error_message=f"inference-ready spec invalid: {safe_err(exc)}",
            )

        model = self._model_factory.resolve(deps.selected_model)
        if model is None:
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=(
                    "The selected causal model is not available in the current model catalog. "
                    "Please reselect the model and try again."
                ),
                error_message=f"unsupported model: {deps.selected_model}",
            )

        last_error_message: str | None = None
        last_user_message: str | None = None
        last_warnings: list[str] = []

        for attempt in range(1, _MAX_TRAINING_ATTEMPTS + 1):
            command = FitCommand(
                model_name=deps.selected_model,
                df=df,
                run_id=uuid4(),
                inference_ready_spec=inference_ready_spec,
                inputs=FitInputs(),
            )

            try:
                result = model.execute(
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    command=command,
                )
            except Exception as exc:
                log.exception(
                    "MODEL_TRAIN fit execution crashed",
                    error=exc,
                    attempt=attempt,
                    max_attempts=_MAX_TRAINING_ATTEMPTS,
                )
                last_error_message = f"fit execution failed: {safe_err(exc)}"
                last_user_message = _summarize_model_failure_for_user(
                    llm=self._llm,
                    operation="model training",
                    model_name=deps.selected_model,
                    error_message=safe_err(exc),
                    error_details={"exception": repr(exc)},
                    warnings=[],
                    fallback_message=(
                        "Model training did not complete because the estimator failed during "
                        "fitting. Please review the selected model and the compiled dataset."
                    ),
                )
                last_warnings = []
                if attempt < _MAX_TRAINING_ATTEMPTS:
                    continue
                return self._failed_result(
                    request=request,
                    payload=payload,
                    user_message=_retry_failure_message(
                        base_message=last_user_message,
                        attempts=attempt,
                    ),
                    error_message=last_error_message,
                )

            if isinstance(result, FitSuccess):
                warnings_list = list(result.warnings or [])
                request.orchestrator_state.set(
                    request.node_state.name(),
                    {
                        "trained_model_id": result.fitted_model_id,
                        "training_warnings": warnings_list,
                    },
                )
                success_payload = payload.model_copy(
                    update={
                        "trained_model_id": result.fitted_model_id,
                        "training_warnings": warnings_list,
                        "assistant_message": _success_message(
                            model_name=deps.selected_model,
                            fitted_model_id=result.fitted_model_id,
                            warnings=warnings_list,
                            attempts=attempt,
                        ),
                        "error_message": None,
                    }
                )
                return self._done_result(
                    request=request,
                    payload=success_payload,
                    user_message=success_payload.assistant_message
                    or str(result.fitted_model_id),
                )

            if isinstance(result, CommandFailure):
                last_warnings = list(result.warnings or [])
                last_error_message = result.error.message
                last_user_message = _summarize_model_failure_for_user(
                    llm=self._llm,
                    operation="model training",
                    model_name=deps.selected_model,
                    error_message=result.error.message,
                    error_details=result.error.details,
                    warnings=last_warnings,
                    fallback_message=_failure_message(
                        model_name=deps.selected_model,
                        error_message=result.error.message,
                        warnings=last_warnings,
                    ),
                )
                if attempt < _MAX_TRAINING_ATTEMPTS:
                    continue
                return self._failed_result(
                    request=request,
                    payload=payload.model_copy(update={"training_warnings": last_warnings}),
                    user_message=_retry_failure_message(
                        base_message=last_user_message,
                        attempts=attempt,
                    ),
                    error_message=last_error_message,
                )

            last_error_message = f"unexpected fit result type: {type(result).__name__}"
            last_user_message = (
                "Model training returned an unexpected result and could not be completed."
            )
            last_warnings = []
            if attempt < _MAX_TRAINING_ATTEMPTS:
                continue
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=_retry_failure_message(
                    base_message=last_user_message,
                    attempts=attempt,
                ),
                error_message=last_error_message,
            )

        return self._failed_result(
            request=request,
            payload=payload.model_copy(update={"training_warnings": last_warnings}),
            user_message=_retry_failure_message(
                base_message=(
                    "Model training could not be completed because the estimator failed "
                    "during fitting."
                ),
                attempts=_MAX_TRAINING_ATTEMPTS,
            ),
            error_message=last_error_message or "training failed after retry",
        )

    def _needs_input_result(
        self,
        *,
        request: NodeRequest,
        payload: ModelTrainPayloadModel,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=ModelTrainState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_INPUT",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _needs_data_result(
        self,
        *,
        request: NodeRequest,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=ModelTrainState.init_empty(),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_DATA",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _done_result(
        self,
        *,
        request: NodeRequest,
        payload: ModelTrainPayloadModel,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=ModelTrainState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="DONE",
            action="NONE",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _aborted_result(
        self,
        *,
        request: NodeRequest,
        payload: ModelTrainPayloadModel,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=ModelTrainState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="ABORTED",
            action="NONE",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _failed_result(
        self,
        *,
        request: NodeRequest,
        payload: ModelTrainPayloadModel,
        user_message: str,
        error_message: str,
    ) -> NodeExecutionResult:
        failed_payload = payload.model_copy(
            update={
                "trained_model_id": None,
                "assistant_message": user_message.strip(),
                "error_message": error_message.strip(),
            }
        )
        return self._aborted_result(
            request=request,
            payload=failed_payload,
            user_message=failed_payload.assistant_message or user_message,
        )


def _training_signature(*, deps: ModelTrainDeps) -> str:
    signature_payload = {
        "dataset_id": str(deps.dataset_id),
        "dataset_summary": (
            None
            if deps.dataset_summary is None
            else deps.dataset_summary.model_dump(mode="json", exclude_none=True)
        ),
        "causal_spec": (
            None
            if deps.causal_spec is None
            else deps.causal_spec.model_dump(mode="json", exclude_none=True)
        ),
        "transformation_plan": (
            None
            if deps.transformation_plan is None
            else deps.transformation_plan.model_dump(mode="json", exclude_none=True)
        ),
        "selected_model": deps.selected_model,
    }
    signature_json = json.dumps(
        signature_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(signature_json.encode("utf-8")).hexdigest()


def _success_message(
    *,
    model_name: str,
    fitted_model_id: UUID,
    warnings: Sequence[str],
    attempts: int,
) -> str:
    label = get_model_training_label(model_name)
    retry_text = ""
    if attempts > 1:
        retry_text = f" after {attempts} attempts"
    if not warnings:
        return (
            f"Training completed successfully with {label}{retry_text}. "
            f"The fitted model is saved under id {fitted_model_id} and is ready for effect estimation."
        )
    warning_text = " ".join(str(item).strip() for item in warnings if str(item).strip())
    return (
        f"Training completed successfully with {label}{retry_text}. "
        f"The fitted model is saved under id {fitted_model_id}. "
        f"Warnings reported during training: {warning_text}"
    )


def _failure_message(
    *,
    model_name: str,
    error_message: str,
    warnings: Sequence[str],
) -> str:
    label = get_model_training_label(model_name)
    base = (
        f"Training failed for {label}. "
        f"The estimator reported: {error_message.strip() or 'unknown error'}."
    )
    if not warnings:
        return base
    warning_text = " ".join(str(item).strip() for item in warnings if str(item).strip())
    return f"{base} Additional warnings: {warning_text}"


def _retry_failure_message(*, base_message: str, attempts: int) -> str:
    return f"{base_message.strip()} Training was attempted {attempts} times before stopping."


def _summarize_model_failure_for_user(
    *,
    llm: LLMService | None,
    operation: str,
    model_name: str,
    error_message: str,
    error_details: Mapping[str, Any] | None,
    warnings: Sequence[str],
    fallback_message: str,
) -> str:
    if llm is None:
        return fallback_message

    payload = {
        "operation": operation,
        "model_name": model_name,
        "error_message": error_message,
        "error_details": dict(error_details or {}),
        "warnings": [str(item).strip() for item in warnings if str(item).strip()],
    }

    try:
        return self_or_llm_generate(llm=llm, payload=payload).strip()
    except Exception:
        return fallback_message


def self_or_llm_generate(*, llm: LLMService, payload: dict[str, Any]) -> str:
    return llm.generate(
        system_prompt=get_model_failure_summary_prompt(),
        user_prompt=json.dumps(payload, ensure_ascii=False),
        config=LLMConfig(model="basic", temperature=0.1),
        history=None,
    ).content


__all__ = ["ModelTrainNode"]
