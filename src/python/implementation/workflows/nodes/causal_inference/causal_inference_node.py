from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, cast
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from python.domain.models.models import ArtifactRef
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.ochestrator_state import ReadOnlyOchestratorState
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_logger
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
from python.implementation.workflows.tools.causal.inference.causal_command import (
    ATECommand,
    ATEInputsModel,
    ATESuccess,
    CATECommand,
    CATEInputs,
    CATESuccess,
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

log = get_logger(__name__)

_WORKING_TABLE_PREFIX = "df_"
_WORKING_TABLE_HASH_HEX_LEN = 16
_ARTIFACT_KIND_CHART_SPEC = "chart_spec"
_GROUP_KEY_COLUMN = "group_key"
_CATE_COLUMN = "cate"
_CATE_LOWER_COLUMN = "cate_lower"
_CATE_UPPER_COLUMN = "cate_upper"
_DATA_MANIPULATION_RETRY_ATTEMPTS = 3


class _InferenceRouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal[
        "answer_from_context",
        "compute_cate",
        "generate_ate_graph",
        "generate_cate_graph",
        "handoff_dataset_graph",
        "clarify",
    ]
    assistant_message: str | None = None
    cate_request_summary: str | None = None
    dataset_graph_request: str | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> _InferenceRouteDecision:
        if self.action in ("answer_from_context", "clarify") and not self.assistant_message:
            raise ValueError(f"{self.action} requires assistant_message")
        if self.action in ("compute_cate", "generate_cate_graph") and not self.cate_request_summary:
            raise ValueError(f"{self.action} requires cate_request_summary")
        if self.action == "handoff_dataset_graph":
            if not self.assistant_message:
                raise ValueError("handoff_dataset_graph requires assistant_message")
            if not self.dataset_graph_request:
                raise ValueError("handoff_dataset_graph requires dataset_graph_request")
        return self


class CausalInferenceNode(Node):
    NAME: ClassVar[str] = CausalInferenceState.NAME

    def __init__(
        self,
        *,
        llm: LLMService,
        data_repo: DataRepo,
        tool_factory: ToolFactory,
    ) -> None:
        self._llm = llm
        self._data_repo = data_repo
        self._model_factory = cast(
            CausalModelFactoryTool,
            tool_factory.get_tool(CausalModelFactoryTool.NAME),
        )
        self._data_manipulation_tool = cast(
            DataManipulationTool,
            tool_factory.get_tool(DataManipulationTool.NAME),
        )
        self._plot_tool = cast(
            PlotTool,
            tool_factory.get_tool(PlotTool.NAME),
        )
        self._profiling_tool = cast(
            DatasetProfilingTool,
            tool_factory.get_tool(DatasetProfilingTool.NAME),
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
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        readonly_orchestrator_state: ReadOnlyOchestratorState,
        messages_history: Sequence[ChatMessage] | None,
    ) -> State:
        if not isinstance(state, CausalInferenceState):
            raise TypeError(
                f"{self.name}: expected CausalInferenceState, got {type(state).__name__}"
            )

        deps = CausalInferenceDeps.from_loaded(readonly_orchestrator_state)
        payload = _bind_payload(state=state)
        history = list(messages_history[-6:]) if messages_history else []

        try:
            dataframe = self._data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=deps.dataset_id,
                limit=None,
            )
        except Exception as exc:
            log.exception("CAUSAL_INFERENCE failed to load dataset", error=exc)
            return CausalInferenceState(
                _failed_payload(
                    payload=payload,
                    assistant_message=(
                        "I could not load the cleaned dataset needed for causal inference. "
                        "Please retry after the cleaned dataset is available."
                    ),
                    error_message=f"dataset load failed: {safe_err(exc)}",
                )
            )

        if dataframe.empty:
            return CausalInferenceState(
                _failed_payload(
                    payload=payload,
                    assistant_message=(
                        "The cleaned dataset is empty, so causal effect estimation cannot proceed."
                    ),
                    error_message="cleaned dataset is empty",
                )
            )

        model = self._model_factory.resolve(deps.selected_model)
        if model is None:
            return CausalInferenceState(
                _failed_payload(
                    payload=payload,
                    assistant_message=(
                        "The confirmed causal model is not available in the current model catalog."
                    ),
                    error_message=f"unsupported model: {deps.selected_model}",
                )
            )

        if payload.ate_result_raw_json_str is None:
            return self._compute_initial_ate(
                user_id=user_id,
                conversation_id=conversation_id,
                dataframe=dataframe,
                deps=deps,
                payload=payload,
                model=model,
                history=history,
            )

        return self._handle_follow_up(
            user_id=user_id,
            conversation_id=conversation_id,
            dataframe=dataframe,
            deps=deps,
            payload=payload,
            model=model,
            history=history,
        )

    def _compute_initial_ate(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataframe: pd.DataFrame,
        deps: CausalInferenceDeps,
        payload: CausalInferencePayloadModel,
        model: CausalModel,
        history: Sequence[ChatMessage],
    ) -> CausalInferenceState:
        command = ATECommand(
            model_name=deps.selected_model,
            df=dataframe,
            run_id=uuid4(),
            inference_ready_spec=deps.inference_ready_spec,
            fitted_model_id=deps.trained_model_id,
            inputs=ATEInputsModel(),
        )

        try:
            result = model.execute(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
            )
        except Exception as exc:
            log.exception("CAUSAL_INFERENCE ATE execution crashed", error=exc)
            return CausalInferenceState(
                _failed_payload(
                    payload=payload,
                    assistant_message=_summarize_model_failure_for_user(
                        llm=self._llm,
                        operation="overall treatment-effect estimation",
                        model_name=deps.selected_model,
                        error_message=safe_err(exc),
                        error_details={"exception": repr(exc)},
                        warnings=[],
                        fallback_message=(
                            "The global treatment effect could not be computed because the estimator failed."
                        ),
                    ),
                    error_message=f"ate execution failed: {safe_err(exc)}",
                )
            )

        if isinstance(result, CommandFailure):
            return CausalInferenceState(
                _failed_payload(
                    payload=payload,
                    assistant_message=_summarize_model_failure_for_user(
                        llm=self._llm,
                        operation="overall treatment-effect estimation",
                        model_name=deps.selected_model,
                        error_message=result.error.message,
                        error_details=result.error.details,
                        warnings=result.warnings,
                        fallback_message=(
                            "I could not compute the overall treatment effect from the trained model."
                        ),
                    ),
                    error_message=result.error.message,
                )
            )

        if not isinstance(result, ATESuccess):
            return CausalInferenceState(
                _failed_payload(
                    payload=payload,
                    assistant_message=(
                        "The treatment-effect computation returned an unexpected result type."
                    ),
                    error_message=f"unexpected ate result type: {type(result).__name__}",
                )
            )

        ate_payload = _normalize_ate_result(result)
        ate_payload_json = _dumps(ate_payload)
        assistant_message = _summarize_ate(
            llm=self._llm,
            deps=deps,
            ate_payload_json=ate_payload_json,
            history=history,
        )
        return CausalInferenceState(
            payload.model_copy(
                update={
                    "ate_result_raw_json_str": ate_payload_json,
                    "assistant_message": assistant_message,
                    "system_message": None,
                    "message_artifact_refs": [],
                    "error_message": None,
                }
            )
        )

    def _handle_follow_up(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataframe: pd.DataFrame,
        deps: CausalInferenceDeps,
        payload: CausalInferencePayloadModel,
        model: CausalModel,
        history: Sequence[ChatMessage],
    ) -> CausalInferenceState:
        cached_context = {
            "ate_result": _loads_or_none(payload.ate_result_raw_json_str),
            "latest_cate_result": _loads_or_none(payload.latest_cate_result_raw_json_str),
            "latest_cate_request_summary": payload.latest_cate_request_summary,
            "effect_modifiers": deps.inference_ready_spec.get_effect_modifiers_order(),
            "selected_model": deps.selected_model,
        }
        latest_user_message = _latest_user_message(history)

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

        if decision.action in ("answer_from_context", "clarify"):
            return CausalInferenceState(
                payload.model_copy(
                    update={
                        "assistant_message": decision.assistant_message,
                        "system_message": None,
                        "message_artifact_refs": [],
                        "error_message": None,
                    }
                )
            )

        if decision.action == "handoff_dataset_graph":
            system_message = _build_dataset_graph_handoff_message(
                user_intent=decision.dataset_graph_request or latest_user_message,
            )
            return CausalInferenceState(
                payload.model_copy(
                    update={
                        "assistant_message": decision.assistant_message,
                        "system_message": system_message,
                        "message_artifact_refs": [],
                        "error_message": None,
                    }
                )
            )

        if decision.action == "generate_ate_graph":
            ate_payload = _loads_or_none(payload.ate_result_raw_json_str)
            if not isinstance(ate_payload, dict):
                return CausalInferenceState(
                    _failed_payload(
                        payload=payload,
                        assistant_message="The cached ATE result is missing or invalid.",
                        error_message="cached ate result missing",
                    )
                )
            try:
                artifact_refs = self._generate_plot_artifacts(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataframe=_build_ate_plot_dataframe(ate_payload),
                    user_intent=_build_ate_graph_user_intent(latest_user_message),
                )
            except Exception as exc:
                return CausalInferenceState(
                    _failed_payload(
                        payload=payload,
                        assistant_message=(
                            "I could not render the overall treatment-effect graph right now."
                        ),
                        error_message=f"ate graph generation failed: {safe_err(exc)}",
                    )
                )
            return CausalInferenceState(
                payload.model_copy(
                    update={
                        "assistant_message": (
                            "Here is the causal effect graph for the overall treatment effect."
                        ),
                        "system_message": None,
                        "message_artifact_refs": artifact_refs,
                        "error_message": None,
                    }
                )
            )

        if decision.action in ("compute_cate", "generate_cate_graph"):
            return self._compute_or_reuse_cate(
                user_id=user_id,
                conversation_id=conversation_id,
                dataframe=dataframe,
                deps=deps,
                payload=payload,
                model=model,
                history=history,
                user_request=latest_user_message,
                request_summary=cast(str, decision.cate_request_summary),
                produce_graph=decision.action == "generate_cate_graph",
            )

        return CausalInferenceState(
            _failed_payload(
                payload=payload,
                assistant_message="The inference router returned an unsupported action.",
                error_message=f"unsupported route action: {decision.action}",
            )
        )

    def _compute_or_reuse_cate(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataframe: pd.DataFrame,
        deps: CausalInferenceDeps,
        payload: CausalInferencePayloadModel,
        model: CausalModel,
        history: Sequence[ChatMessage],
        user_request: str,
        request_summary: str,
        produce_graph: bool,
    ) -> CausalInferenceState:
        cached_cate_payload = _loads_or_none(payload.latest_cate_result_raw_json_str)
        if (
            not produce_graph
            and _should_reuse_latest_cate(payload=payload, request_summary=request_summary)
            and isinstance(cached_cate_payload, dict)
        ):
            assistant_message = _summarize_cate(
                llm=self._llm,
                deps=deps,
                cate_payload=cached_cate_payload,
                history=history,
            )
            return CausalInferenceState(
                payload.model_copy(
                    update={
                        "assistant_message": assistant_message,
                        "system_message": None,
                        "message_artifact_refs": [],
                        "error_message": None,
                    }
                )
            )

        effect_modifier_columns = deps.inference_ready_spec.get_effect_modifiers_order()
        if not effect_modifier_columns:
            return CausalInferenceState(
                payload.model_copy(
                    update={
                        "assistant_message": (
                            "CATE is not available because the confirmed protocol has no effect modifiers."
                        ),
                        "system_message": None,
                        "message_artifact_refs": [],
                        "error_message": None,
                    }
                )
            )

        effect_modifier_frame = dataframe.loc[:, effect_modifier_columns].copy()
        effect_modifier_summary = _filter_dataset_summary_to_effect_modifiers(
            summary=deps.dataset_summary,
            effect_modifiers=effect_modifier_columns,
        )

        try:
            selection_df = self._run_data_manipulation_tool(
                dataframe=effect_modifier_frame,
                conversation_id=conversation_id,
                summary_json=self._profiling_tool.dataset_summary_to_json(effect_modifier_summary),
                instructions=_build_cate_selection_instructions(
                    request_summary=request_summary,
                    effect_modifier_columns=effect_modifier_columns,
                ),
            )
        except Exception as exc:
            return self._invalid_cate_plan_state(
                payload=payload,
                effect_modifier_summary=effect_modifier_summary,
                effect_modifier_columns=effect_modifier_columns,
                user_request=request_summary,
                issue_text=f"Subgroup cohort selection failed: {safe_err(exc)}",
                history=history,
            )

        issue_text = _validate_cate_selection_dataframe(
            selection_df=selection_df,
            effect_modifiers=effect_modifier_columns,
            request_summary=request_summary,
        )
        if issue_text is not None:
            return self._invalid_cate_plan_state(
                payload=payload,
                effect_modifier_summary=effect_modifier_summary,
                effect_modifier_columns=effect_modifier_columns,
                user_request=request_summary,
                issue_text=issue_text,
                history=history,
            )

        cate_payload, cate_plot_df = self._execute_cate_selection(
            user_id=user_id,
            conversation_id=conversation_id,
            dataframe=dataframe,
            deps=deps,
            model=model,
            selection_df=selection_df,
            request_summary=request_summary,
            effect_modifier_columns=effect_modifier_columns,
        )
        if isinstance(cate_payload, dict) and cate_plot_df is None and cate_payload.get("errors"):
            return CausalInferenceState(
                _failed_payload(
                    payload=payload,
                    assistant_message=_summarize_model_failure_for_user(
                        llm=self._llm,
                        operation="subgroup effect estimation",
                        model_name=deps.selected_model,
                        error_message="CATE computation failed for all requested cohorts.",
                        error_details={"cohort_errors": cate_payload.get("errors")},
                        warnings=[],
                        fallback_message=(
                            "I could not compute the requested subgroup effects from the trained model."
                        ),
                    ),
                    error_message=_format_cohort_error_details(cate_payload["errors"]),
                )
            )
        if cate_payload is None or cate_plot_df is None or cate_plot_df.empty:
            return CausalInferenceState(
                payload.model_copy(
                    update={
                        "assistant_message": (
                            "No subgroup rows matched the requested CATE analysis. "
                            "Please broaden the subgroup definition and try again."
                        ),
                        "system_message": None,
                        "message_artifact_refs": [],
                        "error_message": None,
                    }
                )
            )

        assistant_message = _summarize_cate(
            llm=self._llm,
            deps=deps,
            cate_payload=cate_payload,
            history=history,
        )

        artifact_refs: list[ArtifactRef] = []
        system_message: str | None = None
        if produce_graph:
            try:
                artifact_refs = self._generate_plot_artifacts(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataframe=cate_plot_df,
                    user_intent=_build_cate_graph_user_intent(
                        user_request=user_request,
                        request_summary=request_summary,
                    ),
                )
            except Exception as exc:
                system_message = f"cate graph generation failed: {safe_err(exc)}"
                assistant_message = (
                    f"{assistant_message} I computed the subgroup effects, but I could not render "
                    "the requested graph right now."
                )

        return CausalInferenceState(
            payload.model_copy(
                update={
                    "latest_cate_result_raw_json_str": _dumps(cate_payload),
                    "latest_cate_request_summary": request_summary,
                    "assistant_message": assistant_message,
                    "system_message": system_message,
                    "message_artifact_refs": artifact_refs,
                    "error_message": None,
                }
            )
        )

    def _invalid_cate_plan_state(
        self,
        *,
        payload: CausalInferencePayloadModel,
        effect_modifier_summary: DatasetSummaryModel,
        effect_modifier_columns: Sequence[str],
        user_request: str,
        issue_text: str,
        history: Sequence[ChatMessage],
    ) -> CausalInferenceState:
        try:
            assistant_message = self._llm.generate(
                system_prompt=INVALID_CATE_PLAN_SYSTEM_PROMPT,
                user_prompt=INVALID_CATE_PLAN_USER_PROMPT_TEMPLATE.format(
                    effect_modifier_summary_json=effect_modifier_summary.model_dump_json(),
                    effect_modifier_columns_json=_dumps(list(effect_modifier_columns)),
                    user_request=user_request,
                    issue_text=issue_text,
                ),
                config=LLMConfig(model="basic", temperature=0.2),
                history=history,
            ).content.strip()
        except Exception:
            assistant_message = (
                "I could not prepare that subgroup analysis yet. "
                "Please restate the subgroup using only confirmed effect modifiers."
            )

        return CausalInferenceState(
            payload.model_copy(
                update={
                    "assistant_message": assistant_message,
                    "system_message": issue_text,
                    "message_artifact_refs": [],
                    "error_message": None,
                }
            )
        )

    def _execute_cate_selection(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataframe: pd.DataFrame,
        deps: CausalInferenceDeps,
        model: CausalModel,
        selection_df: pd.DataFrame,
        request_summary: str,
        effect_modifier_columns: Sequence[str],
    ) -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
        plot_frames: list[pd.DataFrame] = []
        cohort_summaries: list[dict[str, Any]] = []
        cohort_errors: list[dict[str, Any]] = []

        grouped = selection_df.groupby(_GROUP_KEY_COLUMN, sort=False, dropna=False)
        for group_key, group_df in grouped:
            normalized_group_key = str(group_key).strip()
            x_rows = group_df.loc[:, list(effect_modifier_columns)].reset_index(drop=True).copy()

            command = CATECommand(
                model_name=deps.selected_model,
                df=dataframe,
                run_id=uuid4(),
                inference_ready_spec=deps.inference_ready_spec,
                fitted_model_id=deps.trained_model_id,
                inputs=CATEInputs(x_rows=x_rows),
            )

            try:
                result = model.execute(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    command=command,
                )
            except Exception as exc:
                cohort_errors.append(
                    {
                        "group_key": normalized_group_key,
                        "error": f"cate execution failed: {safe_err(exc)}",
                    }
                )
                continue

            if isinstance(result, CommandFailure):
                cohort_errors.append(
                    {
                        "group_key": normalized_group_key,
                        "error": result.error.message,
                    }
                )
                continue

            if not isinstance(result, CATESuccess):
                cohort_errors.append(
                    {
                        "group_key": normalized_group_key,
                        "error": f"unexpected cate result type: {type(result).__name__}",
                    }
                )
                continue

            cate_values, lower_values, upper_values = _extract_cate_effect_arrays(result.effects)
            if cate_values is None or cate_values.size == 0:
                cohort_errors.append(
                    {
                        "group_key": normalized_group_key,
                        "error": "cate result did not contain usable effect values",
                    }
                )
                continue

            if cate_values.size != len(x_rows):
                cohort_errors.append(
                    {
                        "group_key": normalized_group_key,
                        "error": (
                            "cate result size did not match the number of requested subgroup rows"
                        ),
                    }
                )
                continue

            cohort_plot_df = x_rows.copy()
            cohort_plot_df[_GROUP_KEY_COLUMN] = normalized_group_key
            cohort_plot_df[_CATE_COLUMN] = cate_values.astype(float, copy=False)
            cohort_plot_df[_CATE_LOWER_COLUMN] = _aligned_interval_column(
                interval_values=lower_values,
                length=len(x_rows),
            )
            cohort_plot_df[_CATE_UPPER_COLUMN] = _aligned_interval_column(
                interval_values=upper_values,
                length=len(x_rows),
            )
            plot_frames.append(cohort_plot_df)

            cohort_summaries.append(
                {
                    "group_key": normalized_group_key,
                    "row_count": int(len(x_rows)),
                    "estimate_summary": _summarize_numeric_array(cate_values),
                    "interval_summary": _summarize_interval_arrays(lower_values, upper_values),
                }
            )

        if not plot_frames:
            if cohort_errors:
                return {
                    "request_summary": request_summary,
                    "effect_modifier_columns": list(effect_modifier_columns),
                    "errors": cohort_errors,
                }, None
            return None, None

        plot_df = pd.concat(plot_frames, ignore_index=True)
        cate_payload = {
            "request_summary": request_summary,
            "outcome_kind": str(deps.inference_ready_spec.causal_spec.outcome_spec.kind),
            "experiment_type": str(deps.inference_ready_spec.causal_spec.experiment_type),
            "effect_modifier_columns": list(effect_modifier_columns),
            "cohorts": cohort_summaries,
            "errors": cohort_errors,
        }
        return cate_payload, plot_df

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
                json_data=json.dumps(spec, ensure_ascii=False),
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


def _bind_payload(
    *,
    state: CausalInferenceState,
) -> CausalInferencePayloadModel:
    payload = state.payload.model_copy(deep=True)
    return payload.model_copy(
        update={
            "assistant_message": None,
            "system_message": None,
            "message_artifact_refs": [],
            "error_message": None,
        }
    )


def _failed_payload(
    *,
    payload: CausalInferencePayloadModel,
    assistant_message: str,
    error_message: str,
) -> CausalInferencePayloadModel:
    return payload.model_copy(
        update={
            "assistant_message": assistant_message.strip(),
            "system_message": error_message.strip(),
            "message_artifact_refs": [],
            "error_message": None,
        }
    )


def _loads_or_none(value: str | None) -> Any:
    if value is None or not value.strip():
        return None
    return json.loads(value)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _messages_payload(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def _latest_user_message(messages: Sequence[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return ""


def _normalize_ate_result(result: ATESuccess) -> dict[str, Any]:
    item = result.ate[0] if result.ate else {}
    estimate = _scalar_from_any(item.get("ate"))
    lower, upper = _interval_from_any(item.get("ate_interval"))
    return {
        "contrast": dict(result.contrast),
        "estimate": estimate,
        "interval": (
            {"lower": lower, "upper": upper} if lower is not None and upper is not None else None
        ),
        "warnings": list(result.warnings or []),
        "meta": dict(result.meta or {}),
    }


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
    elif isinstance(interval, dict):
        lower_values = _to_1d_float_array(interval.get("lower"))
        upper_values = _to_1d_float_array(interval.get("upper"))

    return cate_values, lower_values, upper_values


def _to_1d_float_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=float)
    else:
        return None
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(float, copy=False).ravel()


def _aligned_interval_column(
    *,
    interval_values: np.ndarray | None,
    length: int,
) -> np.ndarray:
    if interval_values is None or interval_values.size != length:
        return np.full(length, np.nan, dtype=float)
    return interval_values.astype(float, copy=False)


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


def _summarize_interval_arrays(
    lower: np.ndarray | None,
    upper: np.ndarray | None,
) -> dict[str, Any]:
    if lower is None or upper is None or lower.shape != upper.shape:
        return {"available": False}
    mask = np.isfinite(lower) & np.isfinite(upper)
    if not np.any(mask):
        return {"available": False}
    lower_f = lower[mask]
    upper_f = upper[mask]
    return {
        "available": True,
        "mean_lower": float(np.mean(lower_f)),
        "mean_upper": float(np.mean(upper_f)),
        "frac_crosses_zero": float(np.mean((lower_f <= 0.0) & (upper_f >= 0.0))),
    }


def _summarize_ate(
    *,
    llm: LLMService,
    deps: CausalInferenceDeps,
    ate_payload_json: str,
    history: Sequence[ChatMessage],
) -> str:
    context = {
        "selected_model": deps.selected_model,
        "causal_spec": deps.inference_ready_spec.causal_spec.model_dump(mode="json"),
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


def _format_cohort_error_details(errors: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for item in errors[:5]:
        group_key = str(item.get("group_key", "")).strip() or "unknown"
        error_text = str(item.get("error", "")).strip() or "unknown error"
        parts.append(f"{group_key}: {error_text}")
    return " | ".join(parts) if parts else "cate computation failed"


def _summarize_cate(
    *,
    llm: LLMService,
    deps: CausalInferenceDeps,
    cate_payload: dict[str, Any],
    history: Sequence[ChatMessage],
) -> str:
    context = {
        "selected_model": deps.selected_model,
        "causal_spec": deps.inference_ready_spec.causal_spec.model_dump(mode="json"),
    }
    try:
        return llm.generate(
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
        return (
            f"I computed subgroup effect estimates for {len(cohorts)} cohort(s). "
            "Please review the effect graph or ask a follow-up question about the heterogeneity."
        )


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


def _filter_dataset_summary_to_effect_modifiers(
    *,
    summary: DatasetSummaryModel,
    effect_modifiers: Sequence[str],
) -> DatasetSummaryModel:
    wanted = {str(column) for column in effect_modifiers}
    kept = [profile for profile in summary.profiles if str(profile.name) in wanted]
    return DatasetSummaryModel.model_validate(
        {
            "n_rows": int(summary.n_rows),
            "profiles": [profile.model_dump(mode="python") for profile in kept],
        }
    )


def _build_dataset_graph_handoff_message(*, user_intent: str) -> str:
    return _dumps(
        {
            "handoff_target": "DATASET",
            "handoff_kind": "graph_request",
            "graph_scope": "data",
            "user_intent": user_intent,
            "source_state": CausalInferenceState.NAME,
        }
    )


def _build_cate_selection_instructions(
    *,
    request_summary: str,
    effect_modifier_columns: Sequence[str],
) -> str:
    quoted_columns = ", ".join(effect_modifier_columns)
    return (
        "Prepare a read-only analytical result set for CATE cohort selection. "
        "Use only the provided effect modifier columns. "
        "Return one row per matched individual and do not aggregate. "
        f"The final result set must contain exactly these columns: {_GROUP_KEY_COLUMN}, {quoted_columns}. "
        f"`{_GROUP_KEY_COLUMN}` must be a non-empty text label describing the requested cohort. "
        "If the request implies a comparison, return all requested cohorts in the same result set "
        f"with distinct `{_GROUP_KEY_COLUMN}` values. "
        "If the request implies a single subgroup, still return a single cohort with a constant "
        f"`{_GROUP_KEY_COLUMN}` value. "
        "Do not return treatment, outcome, covariates, IDs, or invented columns. "
        f"Clinical subgroup request: {request_summary}"
    )


def _validate_cate_selection_dataframe(
    *,
    selection_df: pd.DataFrame,
    effect_modifiers: Sequence[str],
    request_summary: str,
) -> str | None:
    if selection_df.empty:
        return "No cohort rows matched the requested subgroup definition."

    columns = [str(column) for column in selection_df.columns]
    expected_columns = {_GROUP_KEY_COLUMN, *[str(column) for column in effect_modifiers]}
    missing_columns = sorted(expected_columns - set(columns))
    if missing_columns:
        return f"The cohort-selection result is missing required columns: {missing_columns}."

    extra_columns = sorted(set(columns) - expected_columns)
    if extra_columns:
        return (
            "The cohort-selection result contains unsupported columns outside the confirmed "
            f"effect modifiers: {extra_columns}."
        )

    group_series = selection_df[_GROUP_KEY_COLUMN].astype(str).str.strip()
    if group_series.eq("").any():
        return "Every selected subgroup row must have a non-empty group_key label."

    if _looks_like_comparison_request(request_summary) and group_series.nunique(dropna=True) < 2:
        return "The request implies a subgroup comparison, but only one cohort was returned."

    return None


def _looks_like_comparison_request(text: str) -> bool:
    lowered = text.casefold()
    return any(
        marker in lowered
        for marker in (
            "compare",
            "comparison",
            "versus",
            " vs ",
            "difference between",
            "between ",
        )
    )


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
        "are interval bounds when available. "
        "Use an appropriate causal-effect visualization for the request: a distribution for a "
        "single cohort, a cohort comparison when multiple group_key values exist, or a trend "
        "against a continuous effect modifier when clinically requested. "
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
