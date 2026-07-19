from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, cast
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from python.domain.models.models import ArtifactRef
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.causal_validate.causal_validate_deps import (
    CausalValidateDeps,
)
from python.implementation.workflows.nodes.causal_validate.causal_validate_prompts import (
    CAUSAL_VALIDATE_INITIAL_SUMMARY_SYSTEM_PROMPT,
    CAUSAL_VALIDATE_INITIAL_SUMMARY_USER_PROMPT_TEMPLATE,
    CAUSAL_VALIDATE_QUERY_SUMMARY_SYSTEM_PROMPT,
    CAUSAL_VALIDATE_QUERY_SUMMARY_USER_PROMPT_TEMPLATE,
    CAUSAL_VALIDATE_ROUTE_SYSTEM_PROMPT,
    CAUSAL_VALIDATE_ROUTE_USER_PROMPT_TEMPLATE,
    get_causal_validate_node_info,
)
from python.implementation.workflows.nodes.causal_validate.causal_validate_state import (
    CausalValidatePayloadModel,
    CausalValidateState,
)
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.causal.inference.causal_command import (
    CommandFailure,
    FitCommand,
    FitInputs,
    ValidateCommand,
    ValidateSuccess,
)
from python.implementation.workflows.tools.causal.inference.causal_model import CausalModel
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.tools.plot_tool.plot_tool import PlotTool
from python.implementation.workflows.utils.utils import safe_err

log = get_app_logger(__name__, component="causal_validate_node", log_type="node")

_WORKING_TABLE_PREFIX = "df_"
_WORKING_TABLE_HASH_HEX_LEN = 16
_DATA_MANIPULATION_RETRY_ATTEMPTS = 3
_ARTIFACT_KIND_VALIDATION_ROWS = "causal_validation_rows"
_ARTIFACT_KIND_DR_TEST_SUMMARY = "causal_validation_dr_test_summary"
_ARTIFACT_KIND_CHART_SPEC = "chart_spec"
_CAUSAL_MODEL_FACTORY_TOOL_NAME = "CAUSAL_MODEL_FACTORY"
_VALIDATION_RESULT_COLUMNS = (
    "effect_row",
    "outer_fold",
    "cate_oof",
    "cate_oof_lower",
    "cate_oof_upper",
    "dr_outcome_oof",
)


class _ValidationRouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal[
        "answer_from_context",
        "query_patient_validation",
        "query_dr_test_summary",
        "generate_validation_graph",
        "clarify",
    ]
    assistant_message: str | None = None
    request_summary: str | None = None
    query_target: Literal["patient_validation", "dr_test_summary"] | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> _ValidationRouteDecision:
        if self.action in ("answer_from_context", "clarify"):
            if not self.assistant_message:
                raise ValueError(f"{self.action} requires assistant_message")
            return self

        if not self.request_summary:
            raise ValueError(f"{self.action} requires request_summary")
        if self.action == "query_patient_validation":
            self.query_target = "patient_validation"
        elif self.action == "query_dr_test_summary":
            self.query_target = "dr_test_summary"
        elif self.query_target is None:
            raise ValueError("generate_validation_graph requires query_target")
        return self


class _CausalModelResolver(Protocol):
    def resolve(self, estimator_fqcn: str) -> CausalModel | None: ...


@dataclass(frozen=True)
class _ResolvedValidationContext:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel
    selected_model: str
    trained_model_id: UUID
    inference_ready_spec: InferenceReadyCausalSpec


class CausalValidateNode(Node):
    """Run and cache model validation, then query only the cached validation outputs."""

    NAME: ClassVar[str] = CausalValidateState.NAME

    def __init__(
        self,
        *,
        llm: LLMService,
        data_repo: DataRepo,
        tools_factory: ToolFactory,
    ) -> None:
        self._llm = llm
        self._data_repo = data_repo
        self._model_factory = cast(
            _CausalModelResolver,
            tools_factory.get_tool(_CAUSAL_MODEL_FACTORY_TOOL_NAME),
        )
        self._data_manipulation_tool = cast(
            DataManipulationTool,
            tools_factory.get_tool(DataManipulationTool.NAME),
        )
        self._plot_tool = cast(
            PlotTool,
            tools_factory.get_tool(PlotTool.NAME),
        )
        self._profiling_tool = cast(
            DatasetProfilingTool,
            tools_factory.get_tool(DatasetProfilingTool.NAME),
        )

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_causal_validate_node_info()

    def run(self, *, request: NodeRequest) -> NodeExecutionResult:
        if not isinstance(request.node_state, CausalValidateState):
            raise TypeError(
                f"{self.name}: expected CausalValidateState, got "
                f"{type(request.node_state).__name__}"
            )

        payload = request.node_state.payload.model_copy(deep=True)
        try:
            deps = CausalValidateDeps.from_request(request)
        except Exception as exc:
            log.info("CAUSAL_VALIDATE dependencies unavailable", error=safe_err(exc))
            return self._needs_data_result(
                request=request,
                user_message=(
                    "Causal validation needs a compiled dataset, a confirmed causal "
                    "specification and model, and a completed training run."
                ),
            )

        try:
            dataframe = self._data_repo.get_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=deps.dataset_id,
                limit=None,
            )
        except Exception as exc:
            log.exception("CAUSAL_VALIDATE failed to load compiled dataset", error=exc)
            return self._needs_data_result(
                request=request,
                user_message=(
                    "I could not load the compiled dataset needed for causal validation. "
                    "Please retry after the dataset is available."
                ),
            )

        if dataframe.empty:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message="The compiled dataset is empty, so causal validation cannot run.",
                error_message="compiled dataset is empty",
            )

        try:
            inference_ready_spec = InferenceReadyCausalSpec(
                causal_spec=deps.causal_spec,
                transformation_plan=deps.transformation_plan,
                data_summary=deps.dataset_summary,
            )
        except Exception as exc:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "The compiled dataset, causal specification, and transformation plan "
                    "are not consistent enough for causal validation."
                ),
                error_message=f"inference-ready spec invalid: {safe_err(exc)}",
            )

        model = self._model_factory.resolve(deps.selected_model)
        if model is None:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "The confirmed causal model is not available in the current model catalog."
                ),
                error_message=f"unsupported model: {deps.selected_model}",
            )

        resolved = _ResolvedValidationContext(
            dataset_id=deps.dataset_id,
            dataset_summary=deps.dataset_summary,
            selected_model=deps.selected_model,
            trained_model_id=deps.trained_model_id,
            inference_ready_spec=inference_ready_spec,
        )
        source_signature = _source_signature(resolved=resolved)
        if payload.source_signature != source_signature:
            payload = payload.reset_for_signature(source_signature=source_signature)

        history = (
            list(request.read_only_messages_history[-6:])
            if request.read_only_messages_history
            else []
        )
        if not _has_complete_cache(payload):
            return self._run_initial_validation(
                request=request,
                dataframe=dataframe,
                resolved=resolved,
                payload=payload,
                model=model,
                history=history,
            )

        latest_user_message = _latest_user_message(history)
        if latest_user_message is None:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=payload.assistant_message
                or "Validation is cached. Ask about held-out CATEs, folds, or DR diagnostics.",
                artifact_refs=payload.message_artifact_refs,
            )

        return self._handle_follow_up(
            request=request,
            resolved=resolved,
            payload=payload,
            history=history,
            latest_user_message=latest_user_message,
        )

    def _run_initial_validation(
        self,
        *,
        request: NodeRequest,
        dataframe: pd.DataFrame,
        resolved: _ResolvedValidationContext,
        payload: CausalValidatePayloadModel,
        model: CausalModel,
        history: Sequence[ChatMessage],
    ) -> NodeExecutionResult:
        fit_command = FitCommand(
            model_name=resolved.selected_model,
            df=dataframe,
            run_id=uuid4(),
            inference_ready_spec=resolved.inference_ready_spec,
            inputs=FitInputs(),
        )
        command = ValidateCommand(fit_command=fit_command)

        try:
            result = model.execute(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                command=command,
            )
        except Exception as exc:
            log.exception("CAUSAL_VALIDATE validation execution crashed", error=exc)
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "The causal validation run failed while computing held-out estimates. "
                    "Please review the model and validation configuration before retrying."
                ),
                error_message=f"validate execution failed: {safe_err(exc)}",
            )

        if isinstance(result, CommandFailure):
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "The causal model could not complete outer-CV validation. "
                    f"{result.error.message}"
                ),
                error_message=result.error.message,
            )

        if not isinstance(result, ValidateSuccess):
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message="Causal validation returned an unexpected result type.",
                error_message=f"unexpected validate result type: {type(result).__name__}",
            )

        try:
            cached_validation_df = _build_cached_validation_dataframe(
                source_dataframe=dataframe,
                validation_dataframe=result.validation_dataframe,
            )
            dr_test_summary_df = result.dr_test_summary.reset_index(drop=True).copy()
            validation_dataset_id = uuid4()
            dr_test_summary_dataset_id = uuid4()
            self._data_repo.save_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=validation_dataset_id,
                df=cached_validation_df,
                overwrite=True,
                include_index=False,
            )
            self._data_repo.save_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=dr_test_summary_dataset_id,
                df=dr_test_summary_df,
                overwrite=True,
                include_index=False,
            )
        except Exception as exc:
            log.exception("CAUSAL_VALIDATE failed to materialize cache", error=exc)
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message="Validation completed, but its cached results could not be saved.",
                error_message=f"validation cache failed: {safe_err(exc)}",
            )

        validation_summary = _build_validation_summary(
            resolved=resolved,
            result=result,
            validation_dataframe=cached_validation_df,
            dr_test_summary=dr_test_summary_df,
            validation_dataset_id=validation_dataset_id,
            dr_test_summary_dataset_id=dr_test_summary_dataset_id,
        )
        artifact_refs = [
            _build_data_artifact_ref(
                artifact_id=validation_dataset_id,
                artifact_kind=_ARTIFACT_KIND_VALIDATION_ROWS,
            ),
            _build_data_artifact_ref(
                artifact_id=dr_test_summary_dataset_id,
                artifact_kind=_ARTIFACT_KIND_DR_TEST_SUMMARY,
            ),
        ]
        assistant_message = _summarize_initial_validation(
            llm=self._llm,
            validation_summary=validation_summary,
            history=history,
        )
        next_payload = payload.model_copy(
            update={
                "validation_dataset_id": validation_dataset_id,
                "dr_test_summary_dataset_id": dr_test_summary_dataset_id,
                "validation_summary": validation_summary,
                "latest_query_result_raw_json_str": None,
                "latest_request_summary": None,
                "error_message": None,
            }
        )
        return self._needs_input_result(
            request=request,
            payload=next_payload,
            user_message=assistant_message,
            artifact_refs=artifact_refs,
        )

    def _handle_follow_up(
        self,
        *,
        request: NodeRequest,
        resolved: _ResolvedValidationContext,
        payload: CausalValidatePayloadModel,
        history: Sequence[ChatMessage],
        latest_user_message: str,
    ) -> NodeExecutionResult:
        cached_context = {
            "validation_summary": payload.validation_summary,
            "latest_request_summary": payload.latest_request_summary,
            "latest_query_result": _loads_or_none(payload.latest_query_result_raw_json_str),
            "identifier_column": str(resolved.inference_ready_spec.causal_spec.id_col).strip(),
            "effect_modifiers": resolved.inference_ready_spec.get_effect_modifiers_order(),
            "selected_model": resolved.selected_model,
        }
        try:
            decision = self._llm.generate_json(
                schema=_ValidationRouteDecision,
                system_prompt=CAUSAL_VALIDATE_ROUTE_SYSTEM_PROMPT,
                user_prompt=CAUSAL_VALIDATE_ROUTE_USER_PROMPT_TEMPLATE.format(
                    cached_context_json=_dumps(cached_context),
                    messages_json=_dumps(_messages_payload(history)),
                ),
                config=LLMConfig(model="basic", temperature=0.2),
                history=history,
                max_attempts=3,
            )
        except Exception as exc:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "I could not interpret that validation request. Please ask directly "
                    "about held-out CATEs, outer folds, or DRTester diagnostics."
                ),
                error_message=f"validation route generation failed: {safe_err(exc)}",
            )

        if decision.action in ("answer_from_context", "clarify"):
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=cast(str, decision.assistant_message),
            )

        query_target = cast(Literal["patient_validation", "dr_test_summary"], decision.query_target)
        request_summary = cast(str, decision.request_summary)
        try:
            source_df = self._load_cached_dataframe(
                request=request,
                payload=payload,
                query_target=query_target,
            )
            source_summary = self._profiling_tool.extract_dataset_summary(
                source_df,
                max_categories=200,
                sample_distinct=200,
                compute_quantiles=False,
                strict=True,
            )
            query_result_df = self._run_data_manipulation_tool(
                dataframe=source_df,
                conversation_id=request.conversation_id,
                summary_json=self._profiling_tool.dataset_summary_to_json(source_summary),
                instructions=_build_query_instructions(
                    request_summary=request_summary,
                    query_target=query_target,
                    identifier_column=str(resolved.inference_ready_spec.causal_spec.id_col).strip(),
                ),
            )
        except Exception as exc:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "I could not query the cached causal-validation result. Please simplify "
                    "the requested cohort, metric, or fold comparison."
                ),
                error_message=f"cached validation query failed: {safe_err(exc)}",
            )

        if query_result_df.empty:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message="No cached validation rows matched that request.",
            )

        query_payload = {
            "request_summary": request_summary,
            "query_target": query_target,
            "row_count": int(len(query_result_df)),
            "columns": [str(column) for column in query_result_df.columns],
            "records": _dataframe_records(query_result_df, max_rows=100),
        }
        assistant_message = _summarize_query(
            llm=self._llm,
            request_summary=request_summary,
            validation_summary=payload.validation_summary or {},
            query_payload=query_payload,
            history=history,
        )
        artifact_refs: list[ArtifactRef] = []
        error_message: str | None = None
        if decision.action == "generate_validation_graph":
            try:
                artifact_refs = self._generate_plot_artifacts(
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    dataframe=query_result_df,
                    user_intent=_build_graph_intent(
                        user_request=latest_user_message,
                        request_summary=request_summary,
                        query_target=query_target,
                    ),
                )
            except Exception as exc:
                error_message = f"validation graph generation failed: {safe_err(exc)}"
                assistant_message = (
                    f"{assistant_message} I queried the cached validation results, but I "
                    "could not render the requested graph."
                )

        next_payload = payload.model_copy(
            update={
                "latest_query_result_raw_json_str": _dumps(query_payload),
                "latest_request_summary": request_summary,
            }
        )
        return self._needs_input_result(
            request=request,
            payload=next_payload,
            user_message=assistant_message,
            artifact_refs=artifact_refs,
            error_message=error_message,
        )

    def _load_cached_dataframe(
        self,
        *,
        request: NodeRequest,
        payload: CausalValidatePayloadModel,
        query_target: Literal["patient_validation", "dr_test_summary"],
    ) -> pd.DataFrame:
        dataset_id = (
            payload.validation_dataset_id
            if query_target == "patient_validation"
            else payload.dr_test_summary_dataset_id
        )
        if dataset_id is None:
            raise ValueError(f"cached {query_target} dataset id is missing")
        return self._data_repo.get_csv_data(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            dataset_id=dataset_id,
            limit=None,
        )

    def _run_data_manipulation_tool(
        self,
        *,
        dataframe: pd.DataFrame,
        conversation_id: UUID,
        summary_json: str,
        instructions: str,
    ) -> pd.DataFrame:
        manipulate = self._data_manipulation_tool.manipulate
        params = inspect.signature(manipulate).parameters
        kwargs: dict[str, Any] = {
            "dataframe": dataframe,
            "data_summary": summary_json,
            "instructions": instructions,
        }
        if "table_name" in params:
            kwargs["table_name"] = _conversation_id_to_table_name(conversation_id)
        elif "conversation_id" in params:
            kwargs["conversation_id"] = str(conversation_id)
        else:
            raise TypeError(
                "data manipulation tool must accept either 'table_name' or 'conversation_id'"
            )
        if "retry_attempts" in params:
            kwargs["retry_attempts"] = _DATA_MANIPULATION_RETRY_ATTEMPTS
        return manipulate(**kwargs)

    def _generate_plot_artifacts(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataframe: pd.DataFrame,
        user_intent: str,
    ) -> list[ArtifactRef]:
        summary = self._profiling_tool.extract_dataset_summary(
            dataframe,
            max_categories=200,
            sample_distinct=200,
            compute_quantiles=False,
            strict=True,
        )
        specs = self._plot_tool.generate_specs(
            dataframe=dataframe,
            data_summary=summary,
            user_intent=user_intent,
        )
        artifact_refs: list[ArtifactRef] = []
        for spec in specs:
            artifact_id = uuid4()
            self._data_repo.save_json_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=artifact_id,
                json_data=json.dumps(spec, ensure_ascii=False, allow_nan=False),
                overwrite=True,
            )
            artifact_refs.append(_build_graph_artifact_ref(artifact_id=artifact_id))
        return artifact_refs

    def _needs_input_result(
        self,
        *,
        request: NodeRequest,
        payload: CausalValidatePayloadModel,
        user_message: str,
        artifact_refs: Sequence[ArtifactRef] | None = None,
        error_message: str | None = None,
    ) -> NodeExecutionResult:
        updated_payload = payload.model_copy(
            update={
                "assistant_message": user_message,
                "message_artifact_refs": list(artifact_refs or []),
                "error_message": error_message,
            }
        )
        return NodeExecutionResult(
            new_node_state=CausalValidateState(updated_payload),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_INPUT",
            response_messages=[
                ChatMessage(
                    role="assistant",
                    content=user_message,
                    artifact_refs=list(artifact_refs or []) or None,
                )
            ],
        )

    def _needs_data_result(
        self,
        *,
        request: NodeRequest,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=CausalValidateState.init_empty(),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_DATA",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )


def _has_complete_cache(payload: CausalValidatePayloadModel) -> bool:
    return (
        payload.validation_dataset_id is not None
        and payload.dr_test_summary_dataset_id is not None
        and payload.validation_summary is not None
    )


def _source_signature(*, resolved: _ResolvedValidationContext) -> str:
    signature_payload = {
        "dataset_id": str(resolved.dataset_id),
        "dataset_summary": resolved.dataset_summary.model_dump(mode="json", exclude_none=True),
        "causal_spec": resolved.inference_ready_spec.causal_spec.model_dump(
            mode="json", exclude_none=True
        ),
        "transformation_plan": resolved.inference_ready_spec.transformation_plan.model_dump(
            mode="json", exclude_none=True
        ),
        "selected_model": resolved.selected_model,
        "trained_model_id": str(resolved.trained_model_id),
    }
    serialized = json.dumps(
        signature_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_cached_validation_dataframe(
    *,
    source_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    missing_columns = [
        column
        for column in _VALIDATION_RESULT_COLUMNS
        if column not in validation_dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "validation result is missing required columns: " + ", ".join(missing_columns)
        )

    validation = validation_dataframe.loc[:, list(_VALIDATION_RESULT_COLUMNS)].copy()
    effect_rows_numeric = pd.to_numeric(validation["effect_row"], errors="coerce")
    if effect_rows_numeric.isna().any():
        raise ValueError("validation effect_row contains non-numeric values")
    effect_rows = effect_rows_numeric.astype(int)
    if not np.array_equal(effect_rows.to_numpy(dtype=float), effect_rows_numeric.to_numpy()):
        raise ValueError("validation effect_row contains non-integer values")
    expected_rows = set(range(1, len(source_dataframe) + 1))
    actual_rows = set(effect_rows.tolist())
    if len(validation) != len(source_dataframe) or actual_rows != expected_rows:
        raise ValueError("validation result does not cover every compiled row exactly once")
    if effect_rows.duplicated().any():
        raise ValueError("validation effect_row contains duplicate rows")

    validation["effect_row"] = effect_rows
    validation = validation.set_index("effect_row", drop=False).reindex(
        range(1, len(source_dataframe) + 1)
    )
    source = source_dataframe.reset_index(drop=True).copy()
    overlapping_columns = [
        column for column in _VALIDATION_RESULT_COLUMNS if column in source.columns
    ]
    if overlapping_columns:
        source = source.drop(columns=overlapping_columns)
    return pd.concat(
        [source, validation.reset_index(drop=True)],
        axis=1,
    )


def _build_validation_summary(
    *,
    resolved: _ResolvedValidationContext,
    result: ValidateSuccess,
    validation_dataframe: pd.DataFrame,
    dr_test_summary: pd.DataFrame,
    validation_dataset_id: UUID,
    dr_test_summary_dataset_id: UUID,
) -> dict[str, Any]:
    folds = pd.to_numeric(validation_dataframe["outer_fold"], errors="coerce")
    lower = pd.to_numeric(validation_dataframe["cate_oof_lower"], errors="coerce")
    upper = pd.to_numeric(validation_dataframe["cate_oof_upper"], errors="coerce")
    return {
        "selected_model": resolved.selected_model,
        "trained_model_id": str(resolved.trained_model_id),
        "source_dataset_id": str(resolved.dataset_id),
        "validation_dataset_id": str(validation_dataset_id),
        "dr_test_summary_dataset_id": str(dr_test_summary_dataset_id),
        "row_count": int(len(validation_dataframe)),
        "outer_folds": sorted(int(value) for value in folds.dropna().unique()),
        "validation_columns": [str(column) for column in validation_dataframe.columns],
        "dr_test_summary_columns": [str(column) for column in dr_test_summary.columns],
        "identifier_column": str(resolved.inference_ready_spec.causal_spec.id_col).strip(),
        "effect_modifiers": resolved.inference_ready_spec.get_effect_modifiers_order(),
        "experiment_type": str(resolved.inference_ready_spec.causal_spec.experiment_type),
        "outcome_kind": str(resolved.inference_ready_spec.causal_spec.outcome_spec.kind),
        "cate_oof": _numeric_summary(validation_dataframe["cate_oof"]),
        "dr_outcome_oof": _numeric_summary(validation_dataframe["dr_outcome_oof"]),
        "cate_interval_rows": int((lower.notna() & upper.notna()).sum()),
        "dr_test_summary_row_count": int(len(dr_test_summary)),
        "dr_test_summary_preview": _dataframe_records(dr_test_summary, max_rows=100),
        "warnings": [str(warning) for warning in result.warnings],
        "meta": _json_safe_value(result.meta),
    }


def _numeric_summary(series: pd.Series[Any]) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _build_query_instructions(
    *,
    request_summary: str,
    query_target: Literal["patient_validation", "dr_test_summary"],
    identifier_column: str,
) -> str:
    if query_target == "patient_validation":
        return (
            "Run a read-only analytical query over the cached row-level causal-validation "
            "dataframe. Do not fit a model and do not recompute ATE, CATE, or validation. "
            "The dataframe contains original compiled columns plus `effect_row`, "
            "`outer_fold`, `cate_oof`, `cate_oof_lower`, `cate_oof_upper`, and "
            f"`dr_outcome_oof`; the identifier is `{identifier_column}`. `cate_oof` is a "
            "held-out conditional treatment-effect estimate. `dr_outcome_oof` is a doubly "
            "robust outcome score, not a treatment effect. Return only the rows, groups, "
            f"and metrics needed to answer: {request_summary}"
        )
    return (
        "Run a read-only analytical query over the cached DRTester summary dataframe. "
        "Do not fit a model and do not recompute validation. Preserve metric names from "
        "the table and do not invent pass/fail thresholds. Return only the rows and "
        f"metrics needed to answer: {request_summary}"
    )


def _build_graph_intent(
    *,
    user_request: str,
    request_summary: str,
    query_target: Literal["patient_validation", "dr_test_summary"],
) -> str:
    if query_target == "patient_validation":
        context = (
            "The input is a queried subset or aggregation of cached out-of-fold causal "
            "validation rows. `cate_oof` is the held-out CATE, its lower and upper columns "
            "are interval bounds when present, and `dr_outcome_oof` is not a causal effect."
        )
    else:
        context = (
            "The input is a queried DRTester diagnostic summary. Plot only metrics present "
            "in the dataframe and do not add pass/fail thresholds."
        )
    return (
        "Create a clear Vega-Lite validation chart. "
        f"{context} Validation request: {request_summary}. "
        f"Latest user wording: {user_request}"
    )


def _summarize_initial_validation(
    *,
    llm: LLMService,
    validation_summary: Mapping[str, Any],
    history: Sequence[ChatMessage],
) -> str:
    try:
        return llm.generate(
            system_prompt=CAUSAL_VALIDATE_INITIAL_SUMMARY_SYSTEM_PROMPT,
            user_prompt=CAUSAL_VALIDATE_INITIAL_SUMMARY_USER_PROMPT_TEMPLATE.format(
                validation_context_json=_dumps(validation_summary)
            ),
            config=LLMConfig(model="basic", temperature=0.2),
            history=history,
        ).content.strip()
    except Exception:
        return (
            f"Outer-CV causal validation completed for {validation_summary.get('row_count', 0)} "
            "rows. I cached the held-out row-level estimates and the DRTester summary for "
            "follow-up queries and graphs."
        )


def _summarize_query(
    *,
    llm: LLMService,
    request_summary: str,
    validation_summary: Mapping[str, Any],
    query_payload: Mapping[str, Any],
    history: Sequence[ChatMessage],
) -> str:
    try:
        return llm.generate(
            system_prompt=CAUSAL_VALIDATE_QUERY_SUMMARY_SYSTEM_PROMPT,
            user_prompt=CAUSAL_VALIDATE_QUERY_SUMMARY_USER_PROMPT_TEMPLATE.format(
                request_summary=request_summary,
                validation_context_json=_dumps(validation_summary),
                query_result_json=_dumps(query_payload),
            ),
            config=LLMConfig(model="basic", temperature=0.2),
            history=history,
        ).content.strip()
    except Exception:
        return (
            f"The cached validation query returned {query_payload.get('row_count', 0)} "
            "row(s). Review the returned validation metrics or ask for a narrower summary."
        )


def _latest_user_message(messages: Sequence[ChatMessage]) -> str | None:
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return message.content.strip()
    return None


def _messages_payload(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _loads_or_none(value: str | None) -> Any:
    if value is None or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _dumps(value: Any) -> str:
    return json.dumps(
        _json_safe_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    )


def _dataframe_records(dataframe: pd.DataFrame, *, max_rows: int) -> list[dict[str, Any]]:
    return [
        {str(key): _json_safe_value(value) for key, value in row.items()}
        for row in dataframe.head(max_rows).to_dict(orient="records")
    ]


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe_value(value.item())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _conversation_id_to_table_name(conversation_id: UUID) -> str:
    digest = hashlib.sha256(str(conversation_id).encode("ascii")).hexdigest()
    return f"{_WORKING_TABLE_PREFIX}{digest[:_WORKING_TABLE_HASH_HEX_LEN]}"


def _build_data_artifact_ref(*, artifact_id: UUID, artifact_kind: str) -> ArtifactRef:
    return {
        "id": artifact_id,
        "kind": "data",
        "format": "csv",
        "artifact_meta": {"kind": artifact_kind},
    }


def _build_graph_artifact_ref(*, artifact_id: UUID) -> ArtifactRef:
    return {
        "id": artifact_id,
        "kind": "graph",
        "format": "json",
        "artifact_meta": {"kind": _ARTIFACT_KIND_CHART_SPEC},
    }


__all__ = ["CausalValidateNode"]
