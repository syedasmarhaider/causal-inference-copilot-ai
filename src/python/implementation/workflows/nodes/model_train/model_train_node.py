from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast
from uuid import UUID, uuid4

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.model_train.model_train_deps import ModelTrainDeps
from python.implementation.workflows.nodes.model_train.model_train_prompts import (
    get_model_failure_summary_prompt,
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

    def __init__(self, *, llm: LLMService, data_repo: DataRepo, tool_factory: ToolFactory) -> None:
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
                last_user_message = _summarize_model_failure_for_user(
                    llm=self._llm,
                    operation="model training",
                    model_name=deps.selected_model,
                    error_message=safe_err(exc),
                    error_details={"exception": repr(exc)},
                    warnings=[],
                    fallback_message=(
                        "Model training did not complete because the estimator failed during fitting. "
                        "Please review the selected model and the cleaned dataset."
                    ),
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
    current_signature = _training_signature(deps=deps)

    reset_required = (
        payload.dataset_id != deps.dataset_id or payload.training_signature != current_signature
    )

    updates: dict[str, Any] = {
        "dataset_id": deps.dataset_id,
        "training_signature": current_signature,
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


def _training_signature(*, deps: ModelTrainDeps) -> str:
    signature_payload = {
        "dataset_id": str(deps.dataset_id),
        "selected_model": deps.selected_model,
        "inference_ready_spec": deps.inference_ready_spec.model_dump(mode="json"),
    }
    signature_json = json.dumps(
        signature_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(signature_json.encode("utf-8")).hexdigest()


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
        return llm.generate(
            system_prompt=get_model_failure_summary_prompt(),
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.1),
            history=None,
        ).content.strip()
    except Exception:
        return fallback_message
