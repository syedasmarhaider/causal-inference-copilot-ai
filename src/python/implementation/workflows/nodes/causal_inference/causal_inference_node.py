from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast
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
from python.implementation.workflows.nodes.causal_inference.causal_inference_deps import (
    CausalInferenceDeps,
)
from python.implementation.workflows.nodes.causal_inference.causal_inference_prompts import (
    CATE_SUMMARY_SYSTEM_PROMPT,
    CATE_SUMMARY_USER_PROMPT_TEMPLATE,
    CAUSAL_INFERENCE_ATE_SUMMARY_SYSTEM_PROMPT,
    CAUSAL_INFERENCE_ATE_SUMMARY_USER_PROMPT_TEMPLATE,
    CAUSAL_INFERENCE_ROUTE_SYSTEM_PROMPT,
    CAUSAL_INFERENCE_ROUTE_USER_PROMPT_TEMPLATE,
    INVALID_CATE_PLAN_SYSTEM_PROMPT,
    INVALID_CATE_PLAN_USER_PROMPT_TEMPLATE,
    get_causal_inference_node_info,
    get_model_failure_summary_prompt,
)
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import (
    CausalInferencePayloadModel,
    CausalInferenceState,
)
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.causal.inference.cate_cache import (
    CATE_COLUMN,
    CATE_LOWER_COLUMN,
    CATE_STDERR_COLUMN,
    CATE_UPPER_COLUMN,
)
from python.implementation.workflows.tools.causal.inference.causal_command import (
    ATECommand,
    ATEInputsModel,
    ATESuccess,
    CommandFailure,
)
from python.implementation.workflows.tools.causal.inference.causal_model import CausalModel
from python.implementation.workflows.tools.causal.inference.causal_model_factory_tool import (
    CausalModelFactoryTool,
)
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

log = get_app_logger(__name__, component="causal_inference_node", log_type="node")

_WORKING_TABLE_PREFIX = "df_"
_WORKING_TABLE_HASH_HEX_LEN = 16
_ARTIFACT_KIND_CHART_SPEC = "chart_spec"
_GROUP_KEY_COLUMN = "group_key"
_CATE_COLUMN = CATE_COLUMN
_CATE_LOWER_COLUMN = CATE_LOWER_COLUMN
_CATE_UPPER_COLUMN = CATE_UPPER_COLUMN
_CATE_STDERR_COLUMN = CATE_STDERR_COLUMN
_CACHED_CATE_EFFECT_COLUMNS = frozenset(
    {
        _CATE_COLUMN,
        _CATE_LOWER_COLUMN,
        _CATE_UPPER_COLUMN,
        _CATE_STDERR_COLUMN,
    }
)
_DATA_MANIPULATION_RETRY_ATTEMPTS = 3


class _InferenceRouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal[
        "answer_from_context",
        "compute_cate",
        "generate_ate_graph",
        "generate_cate_graph",
        "clarify",
    ]
    assistant_message: str | None = None
    cate_request_summary: str | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> _InferenceRouteDecision:
        if self.action in ("answer_from_context", "clarify") and not self.assistant_message:
            raise ValueError(f"{self.action} requires assistant_message")
        if self.action in ("compute_cate", "generate_cate_graph") and not self.cate_request_summary:
            raise ValueError(f"{self.action} requires cate_request_summary")
        return self


@dataclass(frozen=True)
class _ResolvedInferenceContext:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel
    selected_model: str
    trained_model_id: UUID
    inference_ready_spec: InferenceReadyCausalSpec
    all_row_cate_dataset_id: UUID | None = None
    all_row_cate_summary: dict[str, Any] | None = None
    negative_control_refutation_summary: dict[str, Any] | None = None
    negative_control_refutation_artifact_id: UUID | None = None
    negative_control_refutation_vectors_dataset_id: UUID | None = None


class CausalInferenceNode(Node):
    NAME: ClassVar[str] = CausalInferenceState.NAME

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
            CausalModelFactoryTool,
            tools_factory.get_tool(CausalModelFactoryTool.NAME),
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
        return get_causal_inference_node_info()

    def run(
        self,
        *,
        request: NodeRequest,
    ) -> NodeExecutionResult:
        if not isinstance(request.node_state, CausalInferenceState):
            raise TypeError(
                f"{self.name}: expected CausalInferenceState, got "
                f"{type(request.node_state).__name__}"
            )

        payload = request.node_state.payload.model_copy(deep=True)
        deps = CausalInferenceDeps.from_request(request)

        try:
            dataframe = self._data_repo.get_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=deps.dataset_id,
                limit=None,
            )
        except Exception as exc:
            log.exception("CAUSAL_INFERENCE failed to load dataset", error=exc)
            return self._needs_data_result(
                request=request,
                user_message=(
                    "I could not load the compiled dataset needed for causal inference. "
                    "Please retry after the dataset is available."
                ),
            )

        if dataframe.empty:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "The compiled dataset is empty, so causal effect estimation cannot proceed."
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
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "The compiled dataset, causal specification, and transformation plan are "
                    "not consistent enough for causal inference yet. Please revise the "
                    "upstream setup."
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

        resolved = _ResolvedInferenceContext(
            dataset_id=deps.dataset_id,
            dataset_summary=dataset_summary,
            selected_model=deps.selected_model,
            trained_model_id=deps.trained_model_id,
            inference_ready_spec=inference_ready_spec,
            all_row_cate_dataset_id=deps.all_row_cate_dataset_id,
            all_row_cate_summary=deps.all_row_cate_summary,
            negative_control_refutation_summary=deps.negative_control_refutation_summary,
            negative_control_refutation_artifact_id=(deps.negative_control_refutation_artifact_id),
            negative_control_refutation_vectors_dataset_id=(
                deps.negative_control_refutation_vectors_dataset_id
            ),
        )
        source_signature = _source_signature(resolved=resolved)
        if payload.source_signature != source_signature:
            payload = payload.reset_for_signature(source_signature=source_signature)

        history = (
            list(request.read_only_messages_history[-6:])
            if request.read_only_messages_history
            else []
        )

        if payload.ate_result_raw_json_str is None:
            return self._compute_initial_ate(
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
                or "Ask a follow-up question about the causal effect estimate or request a subgroup analysis.",
                artifact_refs=payload.message_artifact_refs,
            )

        return self._handle_follow_up(
            request=request,
            dataframe=dataframe,
            resolved=resolved,
            payload=payload,
            model=model,
            history=history,
            latest_user_message=latest_user_message,
        )

    def _compute_initial_ate(
        self,
        *,
        request: NodeRequest,
        dataframe: pd.DataFrame,
        resolved: _ResolvedInferenceContext,
        payload: CausalInferencePayloadModel,
        model: CausalModel,
        history: Sequence[ChatMessage],
    ) -> NodeExecutionResult:
        command = ATECommand(
            model_name=resolved.selected_model,
            df=dataframe,
            run_id=uuid4(),
            inference_ready_spec=resolved.inference_ready_spec,
            fitted_model_id=resolved.trained_model_id,
            inputs=ATEInputsModel(),
        )

        try:
            result = model.execute(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                command=command,
            )
        except Exception as exc:
            log.exception("CAUSAL_INFERENCE ATE execution crashed", error=exc)
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=_summarize_model_failure_for_user(
                    llm=self._llm,
                    operation="overall treatment-effect estimation",
                    model_name=resolved.selected_model,
                    error_message=safe_err(exc),
                    error_details={"exception": repr(exc)},
                    warnings=[],
                    fallback_message=(
                        "The global treatment effect could not be computed because the estimator failed."
                    ),
                ),
                error_message=f"ate execution failed: {safe_err(exc)}",
            )

        if isinstance(result, CommandFailure):
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=_summarize_model_failure_for_user(
                    llm=self._llm,
                    operation="overall treatment-effect estimation",
                    model_name=resolved.selected_model,
                    error_message=result.error.message,
                    error_details=result.error.details,
                    warnings=result.warnings,
                    fallback_message=(
                        "I could not compute the overall treatment effect from the trained model."
                    ),
                ),
                error_message=result.error.message,
            )

        if not isinstance(result, ATESuccess):
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "The treatment-effect computation returned an unexpected result type."
                ),
                error_message=f"unexpected ate result type: {type(result).__name__}",
            )

        ate_payload = _normalize_ate_result(result)
        ate_payload_json = _dumps(ate_payload)
        assistant_message = _summarize_ate(
            llm=self._llm,
            selected_model=resolved.selected_model,
            causal_spec=resolved.inference_ready_spec.causal_spec,
            ate_payload_json=ate_payload_json,
            history=history,
        )
        next_payload = payload.model_copy(
            update={
                "ate_result_raw_json_str": ate_payload_json,
                "error_message": None,
            }
        )
        return self._needs_input_result(
            request=request,
            payload=next_payload,
            user_message=assistant_message,
        )

    def _handle_follow_up(
        self,
        *,
        request: NodeRequest,
        dataframe: pd.DataFrame,
        resolved: _ResolvedInferenceContext,
        payload: CausalInferencePayloadModel,
        model: CausalModel,
        history: Sequence[ChatMessage],
        latest_user_message: str,
    ) -> NodeExecutionResult:
        cached_context = {
            "ate_result": _loads_or_none(payload.ate_result_raw_json_str),
            "latest_cate_result": _loads_or_none(payload.latest_cate_result_raw_json_str),
            "latest_cate_request_summary": payload.latest_cate_request_summary,
            "queryable_columns": _dataset_summary_column_names(resolved.dataset_summary),
            "identifier_column": str(resolved.inference_ready_spec.causal_spec.id_col).strip(),
            "effect_modifiers": resolved.inference_ready_spec.get_effect_modifiers_order(),
            "selected_model": resolved.selected_model,
            "all_row_cate_summary": resolved.all_row_cate_summary,
            "negative_control_refutation_summary": (resolved.negative_control_refutation_summary),
        }

        try:
            decision = self._llm.generate_json(
                schema=_InferenceRouteDecision,
                system_prompt=CAUSAL_INFERENCE_ROUTE_SYSTEM_PROMPT,
                user_prompt=CAUSAL_INFERENCE_ROUTE_USER_PROMPT_TEMPLATE.format(
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
                    "I could not interpret that follow-up request from the cached inference "
                    "context. Please restate the question more directly."
                ),
                error_message=f"inference route generation failed: {safe_err(exc)}",
            )

        if decision.action in ("answer_from_context", "clarify"):
            cached_cate_request_summary = payload.latest_cate_request_summary
            if _requests_effect_graph(
                user_request=latest_user_message,
                request_summary=cached_cate_request_summary or latest_user_message,
            ):
                return self._compute_or_reuse_cate(
                    request=request,
                    dataframe=dataframe,
                    resolved=resolved,
                    payload=payload,
                    model=model,
                    history=history,
                    user_request=latest_user_message,
                    request_summary=cached_cate_request_summary or latest_user_message,
                    produce_graph=True,
                )
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=cast(str, decision.assistant_message),
            )

        if decision.action == "generate_ate_graph":
            ate_payload = _loads_or_none(payload.ate_result_raw_json_str)
            if not isinstance(ate_payload, dict):
                return self._needs_input_result(
                    request=request,
                    payload=payload,
                    user_message="The cached ATE result is missing or invalid.",
                    error_message="cached ate result missing",
                )
            try:
                artifact_refs = self._generate_plot_artifacts(
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    dataframe=_build_ate_plot_dataframe(ate_payload),
                    user_intent=_build_ate_graph_user_intent(latest_user_message),
                )
            except Exception as exc:
                return self._needs_input_result(
                    request=request,
                    payload=payload,
                    user_message=(
                        "I could not render the overall treatment-effect graph right now."
                    ),
                    error_message=f"ate graph generation failed: {safe_err(exc)}",
                )
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message="Here is the causal effect graph for the overall treatment effect.",
                artifact_refs=artifact_refs,
            )

        if decision.action in ("compute_cate", "generate_cate_graph"):
            produce_graph = decision.action == "generate_cate_graph" or _requests_effect_graph(
                user_request=latest_user_message,
                request_summary=cast(str, decision.cate_request_summary),
            )
            return self._compute_or_reuse_cate(
                request=request,
                dataframe=dataframe,
                resolved=resolved,
                payload=payload,
                model=model,
                history=history,
                user_request=latest_user_message,
                request_summary=cast(str, decision.cate_request_summary),
                produce_graph=produce_graph,
            )

        return self._needs_input_result(
            request=request,
            payload=payload,
            user_message="The inference router returned an unsupported action.",
            error_message=f"unsupported route action: {decision.action}",
        )

    def _compute_or_reuse_cate(
        self,
        *,
        request: NodeRequest,
        dataframe: pd.DataFrame,
        resolved: _ResolvedInferenceContext,
        payload: CausalInferencePayloadModel,
        model: CausalModel,
        history: Sequence[ChatMessage],
        user_request: str,
        request_summary: str,
        produce_graph: bool,
    ) -> NodeExecutionResult:
        cached_cate_payload = _loads_or_none(payload.latest_cate_result_raw_json_str)
        if (
            not produce_graph
            and _should_reuse_latest_cate(payload=payload, request_summary=request_summary)
            and isinstance(cached_cate_payload, dict)
        ):
            assistant_message = _summarize_cate(
                llm=self._llm,
                selected_model=resolved.selected_model,
                causal_spec=resolved.inference_ready_spec.causal_spec,
                cate_payload=cached_cate_payload,
                negative_control_refutation_summary=(resolved.negative_control_refutation_summary),
                history=history,
            )
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=assistant_message,
            )

        effect_modifier_columns = resolved.inference_ready_spec.get_effect_modifiers_order()
        if not effect_modifier_columns:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "CATE is not available because the confirmed protocol has no effect modifiers."
                ),
            )

        identifier_column = str(resolved.inference_ready_spec.causal_spec.id_col).strip()
        if resolved.all_row_cate_dataset_id is None:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "Cached row-level CATE is not available for this trained model. Please "
                    "retrain the model so I can answer CATE subgroup and patient-benefit "
                    "questions from the stored all-row CATE dataset."
                ),
                error_message="all_row_cate_dataset_id missing",
            )

        try:
            cached_cate_df = self._data_repo.get_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=resolved.all_row_cate_dataset_id,
                limit=None,
            )
        except Exception as exc:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "I could not load the cached row-level CATE dataset. Please retrain the "
                    "model so the cached CATE dataset can be regenerated."
                ),
                error_message=f"cached cate dataset load failed: {safe_err(exc)}",
            )

        if cached_cate_df.empty or _CATE_COLUMN not in cached_cate_df.columns:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "The cached row-level CATE dataset is missing or invalid. Please retrain "
                    "the model so I can regenerate it."
                ),
                error_message="cached cate dataset empty or missing cate column",
            )

        cached_summary = self._profiling_tool.extract_dataset_summary(
            cached_cate_df,
            max_categories=200,
            sample_distinct=200,
            compute_quantiles=False,
            strict=True,
        )
        queryable_columns = [str(column) for column in cached_cate_df.columns]
        requested_filter_columns = _extract_explicit_column_mentions(
            texts=[user_request, request_summary],
            available_columns=queryable_columns,
        )
        effect_modifier_set = {str(column).strip() for column in effect_modifier_columns}
        non_effect_modifier_filter_columns = [
            column
            for column in requested_filter_columns
            if str(column).strip() not in effect_modifier_set
            and str(column).strip() not in _CACHED_CATE_EFFECT_COLUMNS
            and not str(column).strip().startswith("shap_")
        ]

        try:
            query_result_df = self._run_data_manipulation_tool(
                dataframe=cached_cate_df,
                conversation_id=request.conversation_id,
                summary_json=self._profiling_tool.dataset_summary_to_json(cached_summary),
                instructions=_build_cached_cate_query_instructions(
                    request_summary=request_summary,
                    effect_modifier_columns=effect_modifier_columns,
                    identifier_column=identifier_column,
                    all_row_cate_summary=resolved.all_row_cate_summary,
                ),
            )
        except Exception as exc:
            return self._invalid_cate_plan_result(
                request=request,
                payload=payload,
                dataset_summary=cached_summary,
                queryable_columns=queryable_columns,
                effect_modifier_columns=effect_modifier_columns,
                user_request=user_request,
                issue_text=f"Cached CATE query failed: {safe_err(exc)}",
                history=history,
            )

        if query_result_df.empty:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "No cached CATE rows matched that request. Please broaden the subgroup "
                    "or ranking definition and try again."
                ),
            )

        cate_payload = _build_cached_cate_query_payload(
            request_summary=request_summary,
            resolved=resolved,
            identifier_column=identifier_column,
            requested_filter_columns=requested_filter_columns,
            non_effect_modifier_filter_columns=non_effect_modifier_filter_columns,
            query_result_df=query_result_df,
        )
        assistant_message = _summarize_cate(
            llm=self._llm,
            selected_model=resolved.selected_model,
            causal_spec=resolved.inference_ready_spec.causal_spec,
            cate_payload=cate_payload,
            negative_control_refutation_summary=resolved.negative_control_refutation_summary,
            history=history,
        )

        next_payload = payload.model_copy(
            update={
                "latest_cate_result_raw_json_str": _dumps(cate_payload),
                "latest_cate_request_summary": request_summary,
            }
        )

        artifact_refs: list[ArtifactRef] = []
        error_message: str | None = None
        if produce_graph:
            try:
                artifact_refs = self._generate_plot_artifacts(
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    dataframe=query_result_df,
                    user_intent=_build_cate_graph_user_intent(
                        user_request=user_request,
                        request_summary=request_summary,
                    ),
                )
            except Exception as exc:
                error_message = f"cate graph generation failed: {safe_err(exc)}"
                assistant_message = (
                    f"{assistant_message} I queried the cached CATE results, but I could not "
                    "render the requested graph right now."
                )

        return self._needs_input_result(
            request=request,
            payload=next_payload,
            user_message=assistant_message,
            artifact_refs=artifact_refs,
            error_message=error_message,
        )

    def _invalid_cate_plan_result(
        self,
        *,
        request: NodeRequest,
        payload: CausalInferencePayloadModel,
        dataset_summary: DatasetSummaryModel,
        queryable_columns: Sequence[str],
        effect_modifier_columns: Sequence[str],
        user_request: str,
        issue_text: str,
        history: Sequence[ChatMessage],
    ) -> NodeExecutionResult:
        try:
            assistant_message = self._llm.generate(
                system_prompt=INVALID_CATE_PLAN_SYSTEM_PROMPT,
                user_prompt=INVALID_CATE_PLAN_USER_PROMPT_TEMPLATE.format(
                    dataset_summary_json=dataset_summary.model_dump_json(),
                    queryable_columns_json=_dumps(list(queryable_columns)),
                    effect_modifier_columns_json=_dumps(list(effect_modifier_columns)),
                    user_request=user_request,
                    issue_text=issue_text,
                ),
                config=LLMConfig(model="basic", temperature=0.2),
                history=history,
            ).content.strip()
        except Exception:
            assistant_message = (
                "I could not prepare that subgroup analysis yet. You can define the cohort "
                "with any compiled column, but the final cohort output must contain only "
                "group_key plus the confirmed effect modifiers used for effect estimation."
            )

        return self._needs_input_result(
            request=request,
            payload=payload,
            user_message=assistant_message,
            error_message=issue_text,
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
            artifact_refs.append(
                _build_data_artifact_ref(
                    artifact_id=artifact_id,
                    artifact_format="json",
                    artifact_kind=_ARTIFACT_KIND_CHART_SPEC,
                )
            )
        return artifact_refs

    def _needs_input_result(
        self,
        *,
        request: NodeRequest,
        payload: CausalInferencePayloadModel,
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
            new_node_state=CausalInferenceState(updated_payload),
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
            new_node_state=CausalInferenceState.init_empty(),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_DATA",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )


def _source_signature(*, resolved: _ResolvedInferenceContext) -> str:
    signature_payload = {
        "dataset_id": str(resolved.dataset_id),
        "dataset_summary": resolved.dataset_summary.model_dump(mode="json", exclude_none=True),
        "causal_spec": resolved.inference_ready_spec.causal_spec.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "transformation_plan": resolved.inference_ready_spec.transformation_plan.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "selected_model": resolved.selected_model,
        "trained_model_id": str(resolved.trained_model_id),
        "all_row_cate_dataset_id": (
            None
            if resolved.all_row_cate_dataset_id is None
            else str(resolved.all_row_cate_dataset_id)
        ),
        "all_row_cate_summary": resolved.all_row_cate_summary,
        "negative_control_refutation_artifact_id": (
            None
            if resolved.negative_control_refutation_artifact_id is None
            else str(resolved.negative_control_refutation_artifact_id)
        ),
        "negative_control_refutation_vectors_dataset_id": (
            None
            if resolved.negative_control_refutation_vectors_dataset_id is None
            else str(resolved.negative_control_refutation_vectors_dataset_id)
        ),
        "negative_control_refutation_summary": (resolved.negative_control_refutation_summary),
    }
    signature_json = json.dumps(
        signature_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(signature_json.encode("utf-8")).hexdigest()


def _loads_or_none(value: str | None) -> Any:
    if value is None or not value.strip():
        return None
    return json.loads(value)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _messages_payload(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def _latest_user_message(messages: Sequence[ChatMessage]) -> str | None:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return None


def _normalize_ate_result(result: ATESuccess) -> dict[str, Any]:
    item = result.ate[0] if result.ate else {}
    estimate = _scalar_from_any(item.get("ate"))
    lower, upper = _interval_from_any(item.get("ate_interval"))
    sensitivity = _normalize_ate_sensitivity(item)
    return {
        "contrast": dict(result.contrast),
        "estimate": estimate,
        "interval": (
            {"lower": lower, "upper": upper} if lower is not None and upper is not None else None
        ),
        "sensitivity": sensitivity,
        "warnings": list(result.warnings or []),
        "meta": dict(result.meta or {}),
    }


def _normalize_ate_sensitivity(item: Mapping[str, Any]) -> dict[str, Any] | None:
    summary = item.get("sensitivity_summary")
    robustness_value = item.get("robustness_value")
    sensitivity_interval = item.get("sensitivity_interval")

    lower, upper = _interval_from_any(sensitivity_interval)
    normalized: dict[str, Any] = {}

    if summary is not None:
        normalized["summary"] = str(summary)

    if robustness_value is not None:
        scalar = _scalar_from_any(robustness_value)
        normalized["robustness_value"] = scalar if scalar is not None else robustness_value

    if lower is not None and upper is not None:
        normalized["interval"] = {"lower": lower, "upper": upper}
    elif sensitivity_interval is not None:
        normalized["interval"] = sensitivity_interval

    return normalized or None


def _scalar_from_any(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.generic)):
        return float(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size == 0:
            return None
        return float(arr[0])
    return None


def _interval_from_any(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    if isinstance(value, dict):
        return _scalar_from_any(value.get("lower")), _scalar_from_any(value.get("upper"))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _scalar_from_any(value[0]), _scalar_from_any(value[1])
    if isinstance(value, np.ndarray) and value.size >= 2:
        arr = np.asarray(value, dtype=float).ravel()
        return float(arr[0]), float(arr[1])
    return None, None


def _summarize_numeric_array(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _dataframe_records(dataframe: pd.DataFrame, *, max_rows: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in dataframe.head(max_rows).to_dict(orient="records"):
        records.append({str(key): _json_safe_scalar(value) for key, value in row.items()})
    return records


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe_scalar(value.item())
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, str):
        return value
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe_scalar(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_scalar(item) for item in value]
    return str(value)


def _summarize_ate(
    *,
    llm: LLMService,
    selected_model: str,
    causal_spec: Any,
    ate_payload_json: str,
    history: Sequence[ChatMessage],
) -> str:
    context = {
        "selected_model": selected_model,
        "causal_spec": causal_spec.model_dump(mode="json"),
    }
    try:
        return llm.generate(
            system_prompt=CAUSAL_INFERENCE_ATE_SUMMARY_SYSTEM_PROMPT,
            user_prompt=CAUSAL_INFERENCE_ATE_SUMMARY_USER_PROMPT_TEMPLATE.format(
                context_json=_dumps(context),
                ate_result_json=ate_payload_json,
            ),
            config=LLMConfig(model="basic", temperature=0.2),
            history=history,
        ).content.strip()
    except Exception:
        ate_payload = _loads_or_none(ate_payload_json) or {}
        estimate = ate_payload.get("estimate")
        return (
            "I computed the overall treatment effect successfully. "
            f"The estimated effect is {estimate} on the model outcome scale."
        )


def _summarize_model_failure_for_user(
    *,
    llm: LLMService,
    operation: str,
    model_name: str,
    error_message: str,
    error_details: Mapping[str, Any] | None,
    warnings: Sequence[str],
    fallback_message: str,
) -> str:
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
            user_prompt=_dumps(payload),
            config=LLMConfig(model="basic", temperature=0.1),
            history=None,
        ).content.strip()
    except Exception:
        return fallback_message


def _summarize_cate(
    *,
    llm: LLMService,
    selected_model: str,
    causal_spec: Any,
    cate_payload: dict[str, Any],
    negative_control_refutation_summary: dict[str, Any] | None,
    history: Sequence[ChatMessage],
) -> str:
    context = {
        "selected_model": selected_model,
        "causal_spec": causal_spec.model_dump(mode="json"),
        "negative_control_refutation_summary": negative_control_refutation_summary,
    }
    try:
        summary = llm.generate(
            system_prompt=CATE_SUMMARY_SYSTEM_PROMPT,
            user_prompt=CATE_SUMMARY_USER_PROMPT_TEMPLATE.format(
                context_json=_dumps(context),
                cate_payload_json=_dumps(cate_payload),
            ),
            config=LLMConfig(model="basic", temperature=0.2),
            history=history,
        ).content.strip()
    except Exception:
        cohorts = cate_payload.get("cohorts") or []
        summary = (
            f"I computed subgroup effect estimates for {len(cohorts)} cohort(s). "
            "Please review the effect graph or ask a follow-up question about the heterogeneity."
        )
    return _append_cate_filter_disclaimer(summary=summary, cate_payload=cate_payload)


def _should_reuse_latest_cate(
    *,
    payload: CausalInferencePayloadModel,
    request_summary: str,
) -> bool:
    return (
        payload.latest_cate_result_raw_json_str is not None
        and payload.latest_cate_request_summary is not None
        and payload.latest_cate_request_summary.casefold() == request_summary.casefold()
    )


def _requests_effect_graph(*, user_request: str, request_summary: str) -> bool:
    text = f"{user_request} {request_summary}".casefold()
    graph_markers = (
        "graph",
        "chart",
        "plot",
        "visual",
        "figure",
        "boxplot",
        "box plot",
        "forest plot",
        "distribution",
        "histogram",
        "density",
    )
    effect_markers = (
        "cate",
        "ite",
        "individual treatment effect",
        "individual effect",
        "subgroup",
        "heterogeneity",
        "treatment effect",
        "effect estimate",
    )
    return any(marker in text for marker in graph_markers) and any(
        marker in text for marker in effect_markers
    )


def _dataset_summary_column_names(summary: DatasetSummaryModel) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for profile in summary.profiles:
        column = str(profile.name).strip()
        if not column or column in seen:
            continue
        seen.add(column)
        columns.append(column)
    return columns


def _extract_explicit_column_mentions(
    *,
    texts: Sequence[str],
    available_columns: Sequence[str],
) -> list[str]:
    normalized_columns: list[str] = []
    seen_columns: set[str] = set()
    for column in available_columns:
        normalized = str(column).strip()
        if not normalized or normalized.casefold() in seen_columns:
            continue
        seen_columns.add(normalized.casefold())
        normalized_columns.append(normalized)

    normalized_texts = [str(text) for text in texts if str(text).strip()]
    if not normalized_columns or not normalized_texts:
        return []

    matches: list[tuple[int, int, int, int, str]] = []
    for column_index, column in enumerate(normalized_columns):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        for text_index, text in enumerate(normalized_texts):
            for match in pattern.finditer(text):
                matches.append(
                    (
                        text_index,
                        match.start(),
                        -(match.end() - match.start()),
                        column_index,
                        column,
                    )
                )

    selected_columns: list[str] = []
    seen_selected: set[str] = set()
    occupied_ranges: dict[int, list[tuple[int, int]]] = {}
    for text_index, start, negative_length, _column_index, column in sorted(matches):
        end = start - negative_length
        ranges = occupied_ranges.setdefault(text_index, [])
        if any(not (end <= range_start or start >= range_end) for range_start, range_end in ranges):
            continue
        if column.casefold() in seen_selected:
            continue
        ranges.append((start, end))
        seen_selected.add(column.casefold())
        selected_columns.append(column)

    return selected_columns


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        normalized.append(text)
    return normalized


def _build_cate_filter_disclaimer(*, cate_payload: Mapping[str, Any]) -> str:
    non_effect_modifier_filter_columns = _normalize_string_list(
        cate_payload.get("non_effect_modifier_filter_columns")
    )
    if not non_effect_modifier_filter_columns:
        return ""

    effect_modifier_columns = _normalize_string_list(cate_payload.get("effect_modifier_columns"))
    quoted_filter_columns = ", ".join(non_effect_modifier_filter_columns)
    quoted_effect_modifier_columns = ", ".join(effect_modifier_columns) or "none"
    return (
        f"Note: the subgroup was filtered using {quoted_filter_columns}, but the effect "
        "estimate was still calculated using only the confirmed effect modifiers: "
        f"{quoted_effect_modifier_columns}."
    )


def _append_cate_filter_disclaimer(*, summary: str, cate_payload: Mapping[str, Any]) -> str:
    disclaimer = _build_cate_filter_disclaimer(cate_payload=cate_payload)
    if not disclaimer:
        return summary
    if disclaimer.casefold() in summary.casefold():
        return summary
    return f"{summary} {disclaimer}".strip()


def _build_cached_cate_query_instructions(
    *,
    request_summary: str,
    effect_modifier_columns: Sequence[str],
    identifier_column: str,
    all_row_cate_summary: Mapping[str, Any] | None,
) -> str:
    quoted_effect_modifiers = ", ".join(str(column) for column in effect_modifier_columns)
    summary_text = _dumps(dict(all_row_cate_summary or {}))
    return (
        "Prepare a read-only analytical result set from the cached all-row CATE dataframe. "
        "The dataframe already contains the original compiled protocol-scope columns plus "
        f"`{_CATE_COLUMN}`, `{_CATE_LOWER_COLUMN}`, `{_CATE_UPPER_COLUMN}`, "
        f"`{_CATE_STDERR_COLUMN}`, and any `shap_` effect-modifier columns that were "
        f"available from EconML. Do not recompute CATE. `{_CATE_COLUMN}` is the "
        "row-level treatment-effect contrast on the model outcome scale. "
        f"The identifier column is `{identifier_column}`. "
        "The CATE values were estimated using only these confirmed effect modifiers: "
        f"{quoted_effect_modifiers}. "
        f"Use DuckDB SQL over the provided table to answer: {request_summary}. "
        f"For highest-benefit, best-responder, or top-CATE requests, rank by `{_CATE_COLUMN}` "
        "descending unless the user explicitly asks for lowest or harmful effects. "
        "For patient-type/profile questions, prefer compact grouped summaries over raw "
        "individual lists, and include CATE metrics such as mean_cate, median_cate, "
        "max_cate, min_cate, or row_count. "
        "For individual-patient questions, include the identifier when available. "
        "Return only columns that support the answer and include at least one CATE-derived "
        "metric or row-level CATE column. Do not invent source columns. "
        f"Cached CATE summary JSON: {summary_text}"
    )


def _build_cached_cate_query_payload(
    *,
    request_summary: str,
    resolved: _ResolvedInferenceContext,
    identifier_column: str,
    requested_filter_columns: Sequence[str],
    non_effect_modifier_filter_columns: Sequence[str],
    query_result_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "request_summary": request_summary,
        "analysis_kind": "cached_cate_query",
        "outcome_kind": str(resolved.inference_ready_spec.causal_spec.outcome_spec.kind),
        "experiment_type": str(resolved.inference_ready_spec.causal_spec.experiment_type),
        "identifier_column": identifier_column,
        "requested_filter_columns": list(requested_filter_columns),
        "non_effect_modifier_filter_columns": list(non_effect_modifier_filter_columns),
        "effect_modifier_columns": resolved.inference_ready_spec.get_effect_modifiers_order(),
        "all_row_cate_summary": resolved.all_row_cate_summary,
        "query_result": {
            "row_count": int(len(query_result_df)),
            "columns": [str(column) for column in query_result_df.columns],
            "records": _dataframe_records(query_result_df, max_rows=100),
        },
        "cohorts": _cohort_summaries_from_query_result(query_result_df),
        "errors": [],
    }


def _cohort_summaries_from_query_result(query_result_df: pd.DataFrame) -> list[dict[str, Any]]:
    if (
        _GROUP_KEY_COLUMN not in query_result_df.columns
        or _CATE_COLUMN not in query_result_df.columns
    ):
        return []
    summaries: list[dict[str, Any]] = []
    for group_key, group_df in query_result_df.groupby(_GROUP_KEY_COLUMN, sort=False, dropna=False):
        cate_values = pd.to_numeric(group_df[_CATE_COLUMN], errors="coerce").to_numpy(dtype=float)
        summaries.append(
            {
                "group_key": str(group_key),
                "row_count": int(len(group_df)),
                "estimate_summary": _summarize_numeric_array(cate_values),
            }
        )
    return summaries


def _build_ate_plot_dataframe(ate_payload: Mapping[str, Any]) -> pd.DataFrame:
    interval = ate_payload.get("interval") or {}
    return pd.DataFrame(
        [
            {
                "label": "ATE",
                "estimate": ate_payload.get("estimate"),
                "lower": interval.get("lower"),
                "upper": interval.get("upper"),
            }
        ]
    )


def _build_ate_graph_user_intent(user_request: str) -> str:
    latest_request = user_request.strip() or "Show the overall treatment effect."
    return (
        "Create a causal effect graph for the overall average treatment effect. "
        "This dataframe is already an effect-summary table, not raw patient data. "
        "Use `estimate` as the point estimate and `lower`/`upper` as the confidence interval. "
        f"Latest user request: {latest_request}"
    )


def _build_cate_graph_user_intent(
    *,
    user_request: str,
    request_summary: str,
) -> str:
    latest_request = user_request.strip() or "Show the subgroup treatment-effect graph."
    return (
        "Create a causal graph for conditional treatment effects. "
        "Each row in the dataframe is an individual-level CATE estimate. "
        f"`{_GROUP_KEY_COLUMN}` identifies requested cohorts when multiple groups are present. "
        f"`{_CATE_COLUMN}` is the estimated effect and `{_CATE_LOWER_COLUMN}`/`{_CATE_UPPER_COLUMN}` "
        f"are interval bounds when available. `{_CATE_STDERR_COLUMN}` is the row-level "
        "standard error when available. "
        "Use a real Vega-Lite causal-effect visualization, never a markdown or ASCII chart. "
        "For box plot requests, use a Vega-Lite boxplot mark with group_key on the categorical axis "
        "and cate on the quantitative axis. For ITE or individual-effect requests, plot cate by "
        "the identifier column when available and color or facet by group_key when groups exist. "
        "Otherwise use an appropriate distribution for a single cohort, a cohort comparison when "
        "multiple group_key values exist, or a trend against a continuous effect modifier when clinically requested. "
        f"Subgroup intent: {request_summary}. Latest user request: {latest_request}"
    )


def _conversation_id_to_table_name(conversation_id: UUID) -> str:
    digest = hashlib.sha256(str(conversation_id).encode("ascii")).hexdigest()
    return f"{_WORKING_TABLE_PREFIX}{digest[:_WORKING_TABLE_HASH_HEX_LEN]}"


def _build_data_artifact_ref(
    *,
    artifact_id: UUID,
    artifact_format: str,
    artifact_kind: str,
) -> ArtifactRef:
    return {
        "id": artifact_id,
        "kind": "data",
        "format": cast(Any, artifact_format),
        "artifact_meta": {"kind": artifact_kind},
    }


__all__ = ["CausalInferenceNode"]
