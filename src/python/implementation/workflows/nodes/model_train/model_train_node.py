from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast
from uuid import UUID, uuid4

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.model_train.model_train_deps import ModelTrainDeps
from python.implementation.workflows.nodes.model_train.model_train_prompts import (
    get_model_train_node_info,
)
from python.implementation.workflows.nodes.model_train.model_train_state import (
    ModelTrainPayloadModel,
    ModelTrainState,
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
from python.implementation.workflows.utils.utils import safe_err

log = get_logger(__name__)
_MAX_TRAINING_ATTEMPTS = 2


class ModelTrainNode(Node):
    NAME: ClassVar[str] = ModelTrainState.NAME

    def __init__(
        self,
        *,
        llm: LLMService,
        data_repo: DataRepo,
        tool_factory: ToolFactory
    ) -> None:
        self._llm = llm
        self._data_repo = data_repo
        factory_raw = tool_factory.get_tool(CausalModelFactoryTool.NAME)
        self._model_factory = cast(CausalModelFactoryTool, factory_raw)

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_model_train_node_info()

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        previous_state_dependencies: Mapping[str, State],
        messages_history: Sequence[ChatMessage] | None,
    ) -> State:
        _ = messages_history
        if not isinstance(state, ModelTrainState):
            raise TypeError(f"{self.name}: expected ModelTrainState, got {type(state).__name__}")

        deps = ModelTrainDeps.from_loaded(previous_state_dependencies)
        payload = _bind_payload(state=state, deps=deps)

        if payload.trained_model_id is not None and payload.error_message is None:
            return ModelTrainState(payload)

        try:
            df = self._data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=deps.dataset_id,
            )
        except Exception as exc:
            log.exception("MODEL_TRAIN failed to load cleaned dataset", error=exc)
            return ModelTrainState(
                _failed_payload(
                    payload=payload,
                    message=(
                        "I could not load the cleaned dataset needed for model training. "
                        "Please retry after the cleaned dataset is available."
                    ),
                    error_message=f"dataset load failed: {safe_err(exc)}",
                )
            )

        if df.empty:
            return ModelTrainState(
                _failed_payload(
                    payload=payload,
                    message=(
                        "The cleaned dataset is empty, so model training cannot proceed. "
                        "Please review the filtering and cleaning steps first."
                    ),
                    error_message="cleaned dataset is empty",
                )
            )

        model = self._model_factory.resolve(deps.selected_model)
        if model is None:
            return ModelTrainState(
                _failed_payload(
                    payload=payload,
                    message=(
                        "The selected causal model is not available in the current model catalog. "
                        "Please reselect the model and try again."
                    ),
                    error_message=f"unsupported model: {deps.selected_model}",
                )
            )

        last_error_message: str | None = None
        last_user_message: str | None = None
        last_warnings: list[str] = []

        for attempt in range(1, _MAX_TRAINING_ATTEMPTS + 1):
            command = FitCommand(
                model_name=deps.selected_model,
                df=df,
                run_id=uuid4(),
                inference_ready_spec=deps.inference_ready_spec,
                inputs=FitInputs(),
            )

            try:
                result = model.execute(
                    user_id=user_id,
                    conversation_id=conversation_id,
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
                last_user_message = (
                    "Model training did not complete because the estimator failed during fitting. "
                    "Please review the selected model and the cleaned dataset."
                )
                last_warnings = []
                if attempt < _MAX_TRAINING_ATTEMPTS:
                    continue
                return ModelTrainState(
                    _failed_payload(
                        payload=payload,
                        message=_retry_failure_message(
                            base_message=last_user_message,
                            attempts=attempt,
                        ),
                        error_message=last_error_message,
                    )
                )

            if isinstance(result, FitSuccess):
                warnings_list = list(result.warnings or [])
                return ModelTrainState(
                    payload.model_copy(
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
                )

            if isinstance(result, CommandFailure):
                last_warnings = list(result.warnings or [])
                last_error_message = result.error.message
                last_user_message = _failure_message(
                    model_name=deps.selected_model,
                    error_message=result.error.message,
                    warnings=last_warnings,
                )
                if attempt < _MAX_TRAINING_ATTEMPTS:
                    continue
                return ModelTrainState(
                    _failed_payload(
                        payload=payload,
                        message=_retry_failure_message(
                            base_message=last_user_message,
                            attempts=attempt,
                        ),
                        error_message=last_error_message,
                    )
                )

            last_error_message = f"unexpected fit result type: {type(result).__name__}"
            last_user_message = (
                "Model training returned an unexpected result and could not be completed."
            )
            last_warnings = []
            if attempt < _MAX_TRAINING_ATTEMPTS:
                continue
            return ModelTrainState(
                _failed_payload(
                    payload=payload,
                    message=_retry_failure_message(
                        base_message=last_user_message,
                        attempts=attempt,
                    ),
                    error_message=last_error_message,
                )
            )

        return ModelTrainState(
            _failed_payload(
                payload=payload,
                message=_retry_failure_message(
                    base_message=(
                        "Model training could not be completed because the estimator failed during fitting."
                    ),
                    attempts=_MAX_TRAINING_ATTEMPTS,
                ),
                error_message=last_error_message or "training failed after retry",
            )
        )


def _bind_payload(
    *,
    state: ModelTrainState,
    deps: ModelTrainDeps,
) -> ModelTrainPayloadModel:
    payload = state.payload.model_copy(deep=True)
    current_plan = deps.inference_ready_spec.transformation_plan
    order_covariates = deps.inference_ready_spec.get_covariates_order() or None
    order_effect_modifiers = deps.inference_ready_spec.get_effect_modifiers_order() or None

    reset_required = (
        payload.dataset_id != deps.dataset_id
        or payload.selected_model != deps.selected_model
        or payload.column_transformation_plan != current_plan
        or payload.order_covariates != order_covariates
        or payload.order_effect_modifiers != order_effect_modifiers
    )

    updates: dict[str, Any] = {
        "dataset_id": deps.dataset_id,
        "selected_model": deps.selected_model,
        "column_transformation_plan": current_plan,
        "order_covariates": order_covariates,
        "order_effect_modifiers": order_effect_modifiers,
    }
    if reset_required:
        updates.update(
            {
                "trained_model_id": None,
                "training_warnings": [],
                "assistant_message": None,
                "error_message": None,
            }
        )
    return payload.model_copy(update=updates)


def _failed_payload(
    *,
    payload: ModelTrainPayloadModel,
    message: str,
    error_message: str,
) -> ModelTrainPayloadModel:
    return payload.model_copy(
        update={
            "trained_model_id": None,
            "training_warnings": [],
            "assistant_message": message.strip(),
            "error_message": error_message.strip(),
        }
    )


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
