from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, ClassVar, cast
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

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
from python.implementation.workflows.tools.causal.inference.cate_cache import (
    build_all_row_cate_dataframe,
    failed_all_row_cate_summary,
    skipped_all_row_cate_summary,
    summarize_all_row_cate_dataframe,
)
from python.implementation.workflows.tools.causal.inference.causal_command import (
    CATECommand,
    CATEInputs,
    CATESuccess,
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
_NEGATIVE_CONTROL_OUTCOME_UNAVAILABLE_WARNING = (
    "No valid negative-control outcome was provided or identified. "
    "CATE negative-control refutation will not be performed."
)
_PRIMARY_CATE_COLUMN = "primary_cate"
_PRIMARY_CATE_LOWER_COLUMN = "primary_cate_lower"
_PRIMARY_CATE_UPPER_COLUMN = "primary_cate_upper"
_NEGATIVE_CONTROL_CATE_COLUMN = "negative_control_cate"
_NEGATIVE_CONTROL_CATE_LOWER_COLUMN = "negative_control_cate_lower"
_NEGATIVE_CONTROL_CATE_UPPER_COLUMN = "negative_control_cate_upper"


@dataclass(frozen=True)
class _NegativeControlRefutationResult:
    warnings: list[str]
    summary: dict[str, Any] | None
    artifact_id: UUID | None = None
    vectors_dataset_id: UUID | None = None


@dataclass(frozen=True)
class _AllRowCATEResult:
    warnings: list[str]
    summary: dict[str, Any] | None
    dataset_id: UUID | None = None
    cate_values: np.ndarray | None = None
    lower_values: np.ndarray | None = None
    upper_values: np.ndarray | None = None
    stderr_values: np.ndarray | None = None
    shap_values: np.ndarray | None = None
    shap_feature_names: list[str] | None = None


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
                all_row_cate = self._materialize_all_row_cate(
                    request=request,
                    deps=deps,
                    dataframe=df,
                    inference_ready_spec=inference_ready_spec,
                    model=model,
                    primary_fit_result=result,
                )
                refutation = self._run_negative_control_cate_refutation(
                    request=request,
                    deps=deps,
                    dataframe=df,
                    inference_ready_spec=inference_ready_spec,
                    model=model,
                    primary_fit_result=result,
                    primary_cate_result=(
                        (
                            all_row_cate.cate_values,
                            all_row_cate.lower_values,
                            all_row_cate.upper_values,
                        )
                        if all_row_cate.cate_values is not None
                        else None
                    ),
                )
                warnings_list = (
                    list(result.warnings or [])
                    + list(all_row_cate.warnings)
                    + list(refutation.warnings)
                )
                training_spec = _build_training_spec(
                    deps=deps,
                    result=result,
                    attempts=attempt,
                    all_row_cate_summary=all_row_cate.summary,
                    all_row_cate_dataset_id=all_row_cate.dataset_id,
                    negative_control_refutation_summary=refutation.summary,
                    negative_control_refutation_artifact_id=refutation.artifact_id,
                    negative_control_refutation_vectors_dataset_id=refutation.vectors_dataset_id,
                )
                request.orchestrator_state.set(
                    request.node_state.name(),
                    {
                        "trained_model_id": result.fitted_model_id,
                        "training_warnings": warnings_list,
                        "training_spec": training_spec,
                        "all_row_cate_dataset_id": all_row_cate.dataset_id,
                        "all_row_cate_summary": all_row_cate.summary,
                        "negative_control_refutation_artifact_id": refutation.artifact_id,
                        "negative_control_refutation_vectors_dataset_id": (
                            refutation.vectors_dataset_id
                        ),
                        "negative_control_refutation_summary": refutation.summary,
                        "training_error_message": None,
                    },
                )
                success_payload = payload.model_copy(
                    update={
                        "trained_model_id": result.fitted_model_id,
                        "training_warnings": warnings_list,
                        "training_spec": training_spec,
                        "all_row_cate_dataset_id": all_row_cate.dataset_id,
                        "all_row_cate_summary": all_row_cate.summary,
                        "negative_control_refutation_artifact_id": refutation.artifact_id,
                        "negative_control_refutation_vectors_dataset_id": (
                            refutation.vectors_dataset_id
                        ),
                        "negative_control_refutation_summary": refutation.summary,
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
                    user_message=success_payload.assistant_message or str(result.fitted_model_id),
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
                "training_spec": None,
                "all_row_cate_dataset_id": None,
                "all_row_cate_summary": None,
                "negative_control_refutation_artifact_id": None,
                "negative_control_refutation_vectors_dataset_id": None,
                "negative_control_refutation_summary": None,
                "assistant_message": user_message.strip(),
                "error_message": error_message.strip(),
            }
        )
        request.orchestrator_state.set(
            request.node_state.name(),
            {
                "trained_model_id": None,
                "training_warnings": list(failed_payload.training_warnings),
                "training_spec": None,
                "all_row_cate_dataset_id": None,
                "all_row_cate_summary": None,
                "negative_control_refutation_artifact_id": None,
                "negative_control_refutation_vectors_dataset_id": None,
                "negative_control_refutation_summary": None,
                "training_error_message": failed_payload.error_message,
            },
        )
        return self._aborted_result(
            request=request,
            payload=failed_payload,
            user_message=failed_payload.assistant_message or user_message,
        )

    def _materialize_all_row_cate(
        self,
        *,
        request: NodeRequest,
        deps: ModelTrainDeps,
        dataframe: pd.DataFrame,
        inference_ready_spec: InferenceReadyCausalSpec,
        model: Any,
        primary_fit_result: FitSuccess,
    ) -> _AllRowCATEResult:
        effect_modifier_columns = inference_ready_spec.get_effect_modifiers_order()
        if not effect_modifier_columns:
            warning = (
                "All-row CATE cache skipped: no effect modifiers are available for CATE "
                "estimation."
            )
            return _AllRowCATEResult(
                warnings=[warning],
                summary=skipped_all_row_cate_summary(
                    reason="no_effect_modifiers",
                    warning=warning,
                ),
            )

        try:
            x_rows = dataframe.loc[:, effect_modifier_columns].reset_index(drop=True).copy()
        except Exception as exc:
            warning = (
                "All-row CATE cache skipped: could not prepare all-row effect-modifier "
                f"matrix: {safe_err(exc)}"
            )
            return _AllRowCATEResult(
                warnings=[warning],
                summary=failed_all_row_cate_summary(
                    reason="effect_modifier_matrix_unavailable",
                    warning=warning,
                    details={"exception": repr(exc)},
                ),
            )

        command = CATECommand(
            model_name=deps.selected_model,
            df=dataframe,
            run_id=uuid4(),
            inference_ready_spec=inference_ready_spec,
            fitted_model_id=primary_fit_result.fitted_model_id,
            inputs=CATEInputs(x_rows=x_rows),
        )
        try:
            cate_result = model.execute(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                command=command,
            )
        except Exception as exc:
            warning = f"All-row CATE cache failed during CATE computation: {safe_err(exc)}"
            return _AllRowCATEResult(
                warnings=[warning],
                summary=failed_all_row_cate_summary(
                    reason="cate_exception",
                    warning=warning,
                    details={"exception": repr(exc)},
                ),
            )

        if isinstance(cate_result, CommandFailure):
            warning = (
                f"All-row CATE cache failed during CATE computation: {cate_result.error.message}"
            )
            return _AllRowCATEResult(
                warnings=[warning],
                summary=failed_all_row_cate_summary(
                    reason="cate_failed",
                    warning=warning,
                    details={
                        "error_code": cate_result.error.code,
                        "error_details": cate_result.error.details,
                    },
                ),
            )

        if not isinstance(cate_result, CATESuccess):
            warning = (
                "All-row CATE cache failed: CATE returned unexpected result type "
                f"{type(cate_result).__name__}."
            )
            return _AllRowCATEResult(
                warnings=[warning],
                summary=failed_all_row_cate_summary(
                    reason="cate_unexpected_result",
                    warning=warning,
                ),
            )

        cate_values, lower_values, upper_values = _extract_cate_effect_arrays(cate_result.effects)
        if cate_values is None or cate_values.size != len(x_rows):
            warning = "All-row CATE cache failed: CATE vector length did not match row count."
            return _AllRowCATEResult(
                warnings=[warning],
                summary=failed_all_row_cate_summary(
                    reason="cate_vector_length_mismatch",
                    warning=warning,
                    details={
                        "expected_rows": int(len(x_rows)),
                        "actual_rows": None if cate_values is None else int(cate_values.size),
                    },
                ),
            )
        stderr_values = _extract_optional_effect_array(
            cate_result.effects.get("cate_stderr"),
            expected_length=len(x_rows),
        )
        shap_values, shap_feature_names = _extract_shap_effect_matrix(
            cate_result.effects,
            expected_rows=len(x_rows),
        )

        dataset_id = uuid4()
        cate_df = build_all_row_cate_dataframe(
            dataframe=dataframe,
            cate_values=cate_values,
            lower_values=lower_values,
            upper_values=upper_values,
            stderr_values=stderr_values,
            shap_values=shap_values,
            shap_feature_names=shap_feature_names,
            for_treatment=cate_result.effects.get("for_treatment"),
        )
        try:
            self._data_repo.save_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=dataset_id,
                df=cate_df,
                overwrite=True,
                include_index=False,
            )
        except Exception as exc:
            warning = f"All-row CATE cache failed while saving dataset: {safe_err(exc)}"
            return _AllRowCATEResult(
                warnings=[warning],
                summary=failed_all_row_cate_summary(
                    reason="artifact_save_failed",
                    warning=warning,
                    details={"dataset_id": str(dataset_id), "exception": repr(exc)},
                ),
                cate_values=cate_values,
                lower_values=lower_values,
                upper_values=upper_values,
            )

        summary = summarize_all_row_cate_dataframe(
            dataframe=cate_df,
            dataset_id=dataset_id,
            effect_modifier_columns=effect_modifier_columns,
            for_treatment=cate_result.effects.get("for_treatment"),
        )
        return _AllRowCATEResult(
            warnings=list(cate_result.warnings or []),
            summary=summary,
            dataset_id=dataset_id,
            cate_values=cate_values,
            lower_values=lower_values,
            upper_values=upper_values,
            stderr_values=stderr_values,
            shap_values=shap_values,
            shap_feature_names=shap_feature_names,
        )

    def _run_negative_control_cate_refutation(
        self,
        *,
        request: NodeRequest,
        deps: ModelTrainDeps,
        dataframe: pd.DataFrame,
        inference_ready_spec: InferenceReadyCausalSpec,
        model: Any,
        primary_fit_result: FitSuccess,
        primary_cate_result: tuple[np.ndarray, np.ndarray | None, np.ndarray | None] | None = None,
    ) -> _NegativeControlRefutationResult:
        negative_control_outcome = deps.causal_spec.negative_control_outcome
        if negative_control_outcome is None:
            return _NegativeControlRefutationResult(
                warnings=[_NEGATIVE_CONTROL_OUTCOME_UNAVAILABLE_WARNING],
                summary=_failed_refutation_summary(
                    status="SKIPPED",
                    reason="negative_control_outcome_unavailable",
                    warning=_NEGATIVE_CONTROL_OUTCOME_UNAVAILABLE_WARNING,
                    deps=deps,
                    primary_model_id=primary_fit_result.fitted_model_id,
                    negative_control_model_id=None,
                ),
            )

        effect_modifier_columns = inference_ready_spec.get_effect_modifiers_order()
        if not effect_modifier_columns:
            warning = (
                "CATE negative-control refutation skipped: no effect modifiers are available "
                "for CATE estimation."
            )
            return _NegativeControlRefutationResult(
                warnings=[warning],
                summary=_failed_refutation_summary(
                    status="SKIPPED",
                    reason="no_effect_modifiers",
                    warning=warning,
                    deps=deps,
                    primary_model_id=primary_fit_result.fitted_model_id,
                    negative_control_model_id=None,
                ),
            )

        try:
            x_rows = dataframe.loc[:, effect_modifier_columns].reset_index(drop=True).copy()
        except Exception as exc:
            warning = (
                "CATE negative-control refutation skipped: could not prepare all-row "
                f"effect-modifier matrix: {safe_err(exc)}"
            )
            return _NegativeControlRefutationResult(
                warnings=[warning],
                summary=_failed_refutation_summary(
                    status="SKIPPED",
                    reason="effect_modifier_matrix_unavailable",
                    warning=warning,
                    deps=deps,
                    primary_model_id=primary_fit_result.fitted_model_id,
                    negative_control_model_id=None,
                ),
            )

        if primary_cate_result is None:
            primary_cate_result = self._execute_refutation_cate(
                request=request,
                deps=deps,
                dataframe=dataframe,
                inference_ready_spec=inference_ready_spec,
                model=model,
                fitted_model_id=primary_fit_result.fitted_model_id,
                summary_primary_model_id=primary_fit_result.fitted_model_id,
                summary_negative_control_model_id=None,
                x_rows=x_rows,
            )
        if isinstance(primary_cate_result, _NegativeControlRefutationResult):
            return primary_cate_result

        negative_control_spec = deps.causal_spec.model_copy(
            update={
                "outcome_spec": negative_control_outcome,
                "negative_control_outcome": None,
            },
            deep=True,
        )
        try:
            negative_control_inference_ready_spec = InferenceReadyCausalSpec(
                causal_spec=negative_control_spec,
                transformation_plan=deps.transformation_plan,
                data_summary=deps.dataset_summary,
            )
        except Exception as exc:
            warning = (
                "CATE negative-control refutation skipped: negative-control inference spec "
                f"could not be prepared: {safe_err(exc)}"
            )
            return _NegativeControlRefutationResult(
                warnings=[warning],
                summary=_failed_refutation_summary(
                    status="SKIPPED",
                    reason="negative_control_spec_invalid",
                    warning=warning,
                    deps=deps,
                    primary_model_id=primary_fit_result.fitted_model_id,
                    negative_control_model_id=None,
                ),
            )

        negative_control_fit_command = FitCommand(
            model_name=deps.selected_model,
            df=dataframe,
            run_id=uuid4(),
            inference_ready_spec=negative_control_inference_ready_spec,
            inputs=FitInputs(),
        )
        try:
            negative_control_fit_result = model.execute(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                command=negative_control_fit_command,
            )
        except Exception as exc:
            warning = (
                "CATE negative-control refutation failed during negative-control model fit: "
                f"{safe_err(exc)}"
            )
            return _NegativeControlRefutationResult(
                warnings=[warning],
                summary=_failed_refutation_summary(
                    status="FAILED",
                    reason="negative_control_fit_exception",
                    warning=warning,
                    deps=deps,
                    primary_model_id=primary_fit_result.fitted_model_id,
                    negative_control_model_id=None,
                ),
            )

        if isinstance(negative_control_fit_result, CommandFailure):
            warning = (
                "CATE negative-control refutation failed during negative-control model fit: "
                f"{negative_control_fit_result.error.message}"
            )
            return _NegativeControlRefutationResult(
                warnings=[warning],
                summary=_failed_refutation_summary(
                    status="FAILED",
                    reason="negative_control_fit_failed",
                    warning=warning,
                    deps=deps,
                    primary_model_id=primary_fit_result.fitted_model_id,
                    negative_control_model_id=None,
                    details={
                        "error_code": negative_control_fit_result.error.code,
                        "error_details": negative_control_fit_result.error.details,
                    },
                ),
            )

        if not isinstance(negative_control_fit_result, FitSuccess):
            warning = (
                "CATE negative-control refutation failed: negative-control model fit returned "
                f"unexpected result type {type(negative_control_fit_result).__name__}."
            )
            return _NegativeControlRefutationResult(
                warnings=[warning],
                summary=_failed_refutation_summary(
                    status="FAILED",
                    reason="negative_control_fit_unexpected_result",
                    warning=warning,
                    deps=deps,
                    primary_model_id=primary_fit_result.fitted_model_id,
                    negative_control_model_id=None,
                ),
            )

        negative_control_cate_result = self._execute_refutation_cate(
            request=request,
            deps=deps,
            dataframe=dataframe,
            inference_ready_spec=negative_control_inference_ready_spec,
            model=model,
            fitted_model_id=negative_control_fit_result.fitted_model_id,
            summary_primary_model_id=primary_fit_result.fitted_model_id,
            summary_negative_control_model_id=negative_control_fit_result.fitted_model_id,
            x_rows=x_rows,
        )
        if isinstance(negative_control_cate_result, _NegativeControlRefutationResult):
            return negative_control_cate_result

        primary_cate, primary_lower, primary_upper = primary_cate_result
        negative_control_cate, negative_control_lower, negative_control_upper = (
            negative_control_cate_result
        )
        artifact_id = uuid4()
        vectors_dataset_id = uuid4()
        summary = {
            "status": "COMPLETED",
            "reason": None,
            "selected_model": deps.selected_model,
            "primary_model_id": str(primary_fit_result.fitted_model_id),
            "negative_control_model_id": str(negative_control_fit_result.fitted_model_id),
            "primary_outcome": deps.causal_spec.outcome_spec.model_dump(mode="json"),
            "negative_control_outcome": negative_control_outcome.model_dump(mode="json"),
            "effect_modifier_columns": effect_modifier_columns,
            "row_count": int(len(x_rows)),
            "artifact_id": str(artifact_id),
            "vectors_dataset_id": str(vectors_dataset_id),
            "comparison": _compare_cate_vectors(
                primary_cate=primary_cate,
                negative_control_cate=negative_control_cate,
            ),
            "warnings": list(negative_control_fit_result.warnings or []),
        }
        vectors_df = _build_refutation_vectors_dataframe(
            dataframe=dataframe,
            id_column=str(deps.causal_spec.id_col).strip(),
            effect_modifier_columns=effect_modifier_columns,
            primary_cate=primary_cate,
            primary_lower=primary_lower,
            primary_upper=primary_upper,
            negative_control_cate=negative_control_cate,
            negative_control_lower=negative_control_lower,
            negative_control_upper=negative_control_upper,
        )

        try:
            self._data_repo.save_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=vectors_dataset_id,
                df=vectors_df,
                overwrite=True,
                include_index=False,
            )
            self._data_repo.save_json_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=artifact_id,
                json_data=json.dumps(summary, ensure_ascii=False, allow_nan=False, default=str),
                overwrite=True,
            )
        except Exception as exc:
            warning = (
                "CATE negative-control refutation failed while saving artifacts: "
                f"{safe_err(exc)}"
            )
            return _NegativeControlRefutationResult(
                warnings=[warning],
                summary=_failed_refutation_summary(
                    status="FAILED",
                    reason="artifact_save_failed",
                    warning=warning,
                    deps=deps,
                    primary_model_id=primary_fit_result.fitted_model_id,
                    negative_control_model_id=negative_control_fit_result.fitted_model_id,
                ),
            )

        return _NegativeControlRefutationResult(
            warnings=[],
            summary=summary,
            artifact_id=artifact_id,
            vectors_dataset_id=vectors_dataset_id,
        )

    def _execute_refutation_cate(
        self,
        *,
        request: NodeRequest,
        deps: ModelTrainDeps,
        dataframe: pd.DataFrame,
        inference_ready_spec: InferenceReadyCausalSpec,
        model: Any,
        fitted_model_id: UUID,
        summary_primary_model_id: UUID,
        summary_negative_control_model_id: UUID | None,
        x_rows: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None] | _NegativeControlRefutationResult:
        command = CATECommand(
            model_name=deps.selected_model,
            df=dataframe,
            run_id=uuid4(),
            inference_ready_spec=inference_ready_spec,
            fitted_model_id=fitted_model_id,
            inputs=CATEInputs(x_rows=x_rows),
        )
        outcome_column = str(inference_ready_spec.causal_spec.outcome_spec.column).strip()
        try:
            cate_result = model.execute(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                command=command,
            )
        except Exception as exc:
            warning = (
                "CATE negative-control refutation failed during all-row CATE computation "
                f"for outcome {outcome_column!r}: {safe_err(exc)}"
            )
            return _NegativeControlRefutationResult(
                warnings=[warning],
                summary=_failed_refutation_summary(
                    status="FAILED",
                    reason="cate_exception",
                    warning=warning,
                    deps=deps,
                    primary_model_id=summary_primary_model_id,
                    negative_control_model_id=summary_negative_control_model_id,
                    details={"outcome_column": outcome_column},
                ),
            )

        if isinstance(cate_result, CommandFailure):
            warning = (
                "CATE negative-control refutation failed during all-row CATE computation "
                f"for outcome {outcome_column!r}: {cate_result.error.message}"
            )
            return _NegativeControlRefutationResult(
                warnings=[warning],
                summary=_failed_refutation_summary(
                    status="FAILED",
                    reason="cate_failed",
                    warning=warning,
                    deps=deps,
                    primary_model_id=summary_primary_model_id,
                    negative_control_model_id=summary_negative_control_model_id,
                    details={
                        "outcome_column": outcome_column,
                        "error_code": cate_result.error.code,
                        "error_details": cate_result.error.details,
                    },
                ),
            )

        if not isinstance(cate_result, CATESuccess):
            warning = (
                "CATE negative-control refutation failed: all-row CATE returned unexpected "
                f"result type {type(cate_result).__name__} for outcome {outcome_column!r}."
            )
            return _NegativeControlRefutationResult(
                warnings=[warning],
                summary=_failed_refutation_summary(
                    status="FAILED",
                    reason="cate_unexpected_result",
                    warning=warning,
                    deps=deps,
                    primary_model_id=summary_primary_model_id,
                    negative_control_model_id=summary_negative_control_model_id,
                    details={"outcome_column": outcome_column},
                ),
            )

        cate_values, lower_values, upper_values = _extract_cate_effect_arrays(cate_result.effects)
        if cate_values is None or cate_values.size != len(x_rows):
            warning = (
                "CATE negative-control refutation failed: all-row CATE vector length did not "
                f"match fit-row count for outcome {outcome_column!r}."
            )
            return _NegativeControlRefutationResult(
                warnings=[warning],
                summary=_failed_refutation_summary(
                    status="FAILED",
                    reason="cate_vector_length_mismatch",
                    warning=warning,
                    deps=deps,
                    primary_model_id=summary_primary_model_id,
                    negative_control_model_id=summary_negative_control_model_id,
                    details={
                        "outcome_column": outcome_column,
                        "expected_rows": int(len(x_rows)),
                        "actual_rows": None if cate_values is None else int(cate_values.size),
                    },
                ),
            )
        return cate_values, lower_values, upper_values


def _training_signature(*, deps: ModelTrainDeps) -> str:
    signature_payload = {
        "dataset_id": str(deps.dataset_id),
        "dataset_summary": deps.dataset_summary.model_dump(mode="json", exclude_none=True),
        "causal_spec": deps.causal_spec.model_dump(mode="json", exclude_none=True),
        "transformation_plan": deps.transformation_plan.model_dump(mode="json", exclude_none=True),
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


def _build_training_spec(
    *,
    deps: ModelTrainDeps,
    result: FitSuccess,
    attempts: int,
    all_row_cate_summary: dict[str, Any] | None,
    all_row_cate_dataset_id: UUID | None,
    negative_control_refutation_summary: dict[str, Any] | None,
    negative_control_refutation_artifact_id: UUID | None,
    negative_control_refutation_vectors_dataset_id: UUID | None,
) -> dict[str, Any]:
    del deps
    meta = result.meta or {}
    training_spec: dict[str, Any] = {
        "fit": {
            "attempts": int(attempts),
            "backend": _json_safe_training_value(meta.get("backend")),
            "columns": _json_safe_training_value(meta.get("columns", {})),
            "used_init_kwargs": _json_safe_training_value(meta.get("used_init_kwargs", {})),
            "artifacts": _json_safe_training_value(result.artifacts or {}),
            "warnings": [str(item) for item in (result.warnings or [])],
            "started_at": _json_safe_training_value(result.started_at),
            "finished_at": _json_safe_training_value(result.finished_at),
        },
    }
    if all_row_cate_summary is not None:
        cate_summary = _json_safe_training_value(all_row_cate_summary)
        if isinstance(cate_summary, dict):
            cate_summary = dict(cate_summary)
            cate_summary["dataset_id"] = (
                str(all_row_cate_dataset_id)
                if all_row_cate_dataset_id is not None
                else cate_summary.get("dataset_id")
            )
        training_spec["all_row_cate"] = cate_summary
    if negative_control_refutation_summary is not None:
        refutation_summary = _json_safe_training_value(negative_control_refutation_summary)
        if isinstance(refutation_summary, dict):
            refutation_summary = dict(refutation_summary)
            refutation_summary["artifact_id"] = (
                str(negative_control_refutation_artifact_id)
                if negative_control_refutation_artifact_id is not None
                else refutation_summary.get("artifact_id")
            )
            refutation_summary["vectors_dataset_id"] = (
                str(negative_control_refutation_vectors_dataset_id)
                if negative_control_refutation_vectors_dataset_id is not None
                else refutation_summary.get("vectors_dataset_id")
            )
        training_spec["negative_control_refutation"] = refutation_summary
    return training_spec


def _failed_refutation_summary(
    *,
    status: str,
    reason: str,
    warning: str,
    deps: ModelTrainDeps,
    primary_model_id: UUID,
    negative_control_model_id: UUID | None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    negative_control_outcome = deps.causal_spec.negative_control_outcome
    return {
        "status": status,
        "reason": reason,
        "selected_model": deps.selected_model,
        "primary_model_id": str(primary_model_id),
        "negative_control_model_id": (
            str(negative_control_model_id) if negative_control_model_id is not None else None
        ),
        "primary_outcome": deps.causal_spec.outcome_spec.model_dump(mode="json"),
        "negative_control_outcome": (
            None
            if negative_control_outcome is None
            else negative_control_outcome.model_dump(mode="json")
        ),
        "artifact_id": None,
        "vectors_dataset_id": None,
        "warning": warning,
        "warnings": [warning],
        "details": _json_safe_training_value(dict(details or {})),
    }


def _extract_cate_effect_arrays(
    effects: Mapping[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    cate_values = _to_1d_float_array(effects.get("cate"))
    lower_values: np.ndarray | None = None
    upper_values: np.ndarray | None = None

    interval = effects.get("cate_interval")
    if isinstance(interval, (list, tuple)) and len(interval) >= 2:
        lower_values = _to_1d_float_array(interval[0])
        upper_values = _to_1d_float_array(interval[1])
    elif isinstance(interval, Mapping):
        lower_values = _to_1d_float_array(interval.get("lower"))
        upper_values = _to_1d_float_array(interval.get("upper"))

    return cate_values, lower_values, upper_values


def _extract_optional_effect_array(value: Any, *, expected_length: int) -> np.ndarray | None:
    values = _to_1d_float_array(value)
    if values is None or values.size != expected_length:
        return None
    return values


def _extract_shap_effect_matrix(
    effects: Mapping[str, Any],
    *,
    expected_rows: int,
) -> tuple[np.ndarray | None, list[str] | None]:
    raw_values = effects.get("shap_values")
    if raw_values is None:
        return None, None
    try:
        values = np.asarray(raw_values, dtype=float)
    except Exception:
        return None, None
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] != expected_rows:
        return None, None

    raw_meta = effects.get("shap_meta")
    if not isinstance(raw_meta, Mapping):
        return None, None
    raw_feature_names = raw_meta.get("feature_names")
    if not isinstance(raw_feature_names, Sequence) or isinstance(
        raw_feature_names,
        (str, bytes, bytearray),
    ):
        return None, None

    feature_names = [str(name) for name in raw_feature_names]
    if len(feature_names) != values.shape[1]:
        return None, None
    return values.astype(float, copy=False), feature_names


def _to_1d_float_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(float, copy=False).ravel()


def _aligned_optional_vector(values: np.ndarray | None, *, length: int) -> np.ndarray:
    if values is None or values.size != length:
        return np.full(length, np.nan, dtype=float)
    return values.astype(float, copy=False)


def _build_refutation_vectors_dataframe(
    *,
    dataframe: pd.DataFrame,
    id_column: str,
    effect_modifier_columns: Sequence[str],
    primary_cate: np.ndarray,
    primary_lower: np.ndarray | None,
    primary_upper: np.ndarray | None,
    negative_control_cate: np.ndarray,
    negative_control_lower: np.ndarray | None,
    negative_control_upper: np.ndarray | None,
) -> pd.DataFrame:
    length = int(primary_cate.size)
    vector_df = pd.DataFrame({"fit_row": np.arange(1, length + 1, dtype=int)})
    if id_column in dataframe.columns:
        vector_df[id_column] = dataframe[id_column].reset_index(drop=True)
    for column in effect_modifier_columns:
        if column in dataframe.columns:
            vector_df[str(column)] = dataframe[str(column)].reset_index(drop=True)
    vector_df[_PRIMARY_CATE_COLUMN] = primary_cate.astype(float, copy=False)
    vector_df[_PRIMARY_CATE_LOWER_COLUMN] = _aligned_optional_vector(
        primary_lower,
        length=length,
    )
    vector_df[_PRIMARY_CATE_UPPER_COLUMN] = _aligned_optional_vector(
        primary_upper,
        length=length,
    )
    vector_df[_NEGATIVE_CONTROL_CATE_COLUMN] = negative_control_cate.astype(
        float,
        copy=False,
    )
    vector_df[_NEGATIVE_CONTROL_CATE_LOWER_COLUMN] = _aligned_optional_vector(
        negative_control_lower,
        length=length,
    )
    vector_df[_NEGATIVE_CONTROL_CATE_UPPER_COLUMN] = _aligned_optional_vector(
        negative_control_upper,
        length=length,
    )
    return vector_df


def _compare_cate_vectors(
    *,
    primary_cate: np.ndarray,
    negative_control_cate: np.ndarray,
) -> dict[str, Any]:
    primary = np.asarray(primary_cate, dtype=float).ravel()
    negative = np.asarray(negative_control_cate, dtype=float).ravel()
    finite_mask = np.isfinite(primary) & np.isfinite(negative)
    primary_finite = primary[finite_mask]
    negative_finite = negative[finite_mask]

    mean_abs_primary = _mean_abs_or_none(primary_finite)
    mean_abs_negative = _mean_abs_or_none(negative_finite)
    ratio = None
    if mean_abs_primary is not None and mean_abs_primary > 0 and mean_abs_negative is not None:
        ratio = float(mean_abs_negative / mean_abs_primary)

    rmse = None
    same_sign_fraction = None
    if primary_finite.size > 0:
        rmse = float(np.sqrt(np.mean((primary_finite - negative_finite) ** 2)))
        same_sign_fraction = float(np.mean(np.sign(primary_finite) == np.sign(negative_finite)))

    return {
        "n_rows": int(primary.size),
        "n_finite_pairs": int(primary_finite.size),
        "primary_cate_summary": _summarize_numeric_array(primary),
        "negative_control_cate_summary": _summarize_numeric_array(negative),
        "mean_abs_primary_cate": mean_abs_primary,
        "mean_abs_negative_control_cate": mean_abs_negative,
        "mean_abs_negative_control_to_primary_ratio": ratio,
        "pearson_correlation": _safe_correlation(primary_finite, negative_finite),
        "spearman_correlation": _safe_correlation(
            pd.Series(primary_finite).rank(method="average").to_numpy(dtype=float),
            pd.Series(negative_finite).rank(method="average").to_numpy(dtype=float),
        ),
        "rmse": rmse,
        "same_sign_fraction": same_sign_fraction,
    }


def _summarize_numeric_array(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "max": float(np.max(finite)),
    }


def _mean_abs_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(np.abs(values)))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or right.size < 2:
        return None
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _json_safe_training_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {str(key): _json_safe_training_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_training_value(item) for item in value]

    get_params = getattr(value, "get_params", None)
    if callable(get_params):
        serialized: dict[str, Any] = {
            "type": _type_name(value),
            "repr": repr(value),
        }
        with suppress(Exception):
            serialized["params"] = _json_safe_training_value(get_params(deep=False))
        return serialized

    return {
        "type": _type_name(value),
        "repr": repr(value),
    }


def _type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


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
