from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from collections.abc import Sequence
from typing import Any, ClassVar, cast
from uuid import UUID

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from python.domain.models.models import (
    ArtifactRef,
    ChatMessage,
    get_chat_messages_role_and_message_json,
)
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMConfig, LLMService
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.data_statistics.data_statistics_deps import (
    DataStatisticsDeps,
)
from python.implementation.workflows.nodes.data_statistics.data_statistics_prompts import (
    data_statistics_final_response_system_prompt,
    data_statistics_intent_classification_system_prompt,
    data_statistics_node_info,
    data_statistics_off_topic_system_prompt,
    data_statistics_summary_answer_system_prompt,
)
from python.implementation.workflows.nodes.data_statistics.data_statistics_state import (
    DataStatisticsState,
)
from python.implementation.workflows.tools.advanced_analytics.advanced_analytics_tool import (
    AdvancedAnalyticsTool,
)
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.tools.plot_tool.plot_tool import PlotTool
from python.implementation.workflows.utils.utils import JSONDict, safe_err

log = get_app_logger(__name__, component="data_statistics_node", log_type="node")

_ANALYTICAL_QUERY_RETRY_ATTEMPTS = 3
_WORKING_TABLE_PREFIX = "df_"
_WORKING_TABLE_HASH_HEX_LEN = 16
_ARTIFACT_KIND_ANALYTICAL_RESULT = "analytical_result"
_ARTIFACT_KIND_CHART_SPEC = "chart_spec"
_INITIAL_SUMMARY_MAX_COLUMNS = 8
_OFF_TOPIC_FALLBACK_MESSAGE = (
    "This statistics stage is read-only. I can answer dataset-summary questions, run "
    "read-only analytical queries, perform statistical analyses, and generate charts. "
    "I cannot change the dataset or move into downstream causal or model stages from here."
)


class DataStatisticsIntentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent_summary_question: bool = False
    intent_summary_question_brief: str = ""
    intent_readonly_query: bool = False
    intent_readonly_query_brief: str = ""
    intent_statistical_analysis: bool = False
    intent_statistical_analysis_brief: str = ""
    intent_chart: bool = False
    intent_chart_brief: str = ""
    intent_out_of_scope: bool = False

    @field_validator(
        "intent_summary_question_brief",
        "intent_readonly_query_brief",
        "intent_statistical_analysis_brief",
        "intent_chart_brief",
        mode="before",
    )
    @classmethod
    def _normalize_brief(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @model_validator(mode="after")
    def _validate(self) -> DataStatisticsIntentModel:
        in_scope_count = int(self.intent_summary_question) + int(self.intent_readonly_query)
        in_scope_count += int(self.intent_statistical_analysis) + int(self.intent_chart)

        if self.intent_out_of_scope and in_scope_count > 0:
            raise ValueError("intent_out_of_scope cannot be combined with in-scope intents")
        if self.intent_summary_question and not self.intent_summary_question_brief:
            raise ValueError("intent_summary_question_brief is required")
        if self.intent_readonly_query and not self.intent_readonly_query_brief:
            raise ValueError("intent_readonly_query_brief is required")
        if self.intent_statistical_analysis and not self.intent_statistical_analysis_brief:
            raise ValueError("intent_statistical_analysis_brief is required")
        if self.intent_chart and not self.intent_chart_brief:
            raise ValueError("intent_chart_brief is required")
        return self

    def has_any_intent(self) -> bool:
        return (
            self.intent_summary_question
            or self.intent_readonly_query
            or self.intent_statistical_analysis
            or self.intent_chart
        )


class DataStatisticsNode(Node):
    NAME: ClassVar[str] = DataStatisticsState.NAME

    def __init__(
        self,
        *,
        data_repo: DataRepo,
        llm: LLMService,
        tools_factory: ToolFactory,
    ) -> None:
        self._data_repo = data_repo
        self._llm = llm
        self._data_manipulation_tool = cast(
            DataManipulationTool, tools_factory.get_tool(DataManipulationTool.NAME)
        )
        self._plot_tool = cast(PlotTool, tools_factory.get_tool(PlotTool.NAME))
        self._profiling_tool = cast(
            DatasetProfilingTool, tools_factory.get_tool(DatasetProfilingTool.NAME)
        )
        self._advanced_analytics_tool = cast(
            AdvancedAnalyticsTool, tools_factory.get_tool(AdvancedAnalyticsTool.NAME)
        )

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return data_statistics_node_info()

    def run(
        self,
        *,
        request: NodeRequest,
    ) -> NodeExecutionResult:
        if not isinstance(request.node_state, DataStatisticsState):
            raise TypeError(
                f"{self.name}: expected DataStatisticsState, got "
                f"{type(request.node_state).__name__}"
            )

        
        deps = DataStatisticsDeps.from_request(request)
       

        current_df = self._data_repo.get_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=deps.dataset_id,
                limit=1_000_000,
            )
     

        current_summary = deps.dataset_summary
        current_summary_json = self._profiling_tool.dataset_summary_to_json(current_summary)

        latest_user_message = _latest_user_message(request.read_only_messages_history)
        if not latest_user_message:
            return self._needs_input_result(
                request=request,
                user_message=self._build_ready_message(summary=current_summary),
            )

        history_text = _recent_history_text(request.read_only_messages_history)
        try:
            intent = self._classify_intent(
                latest_user_message=latest_user_message,
                chat_history=history_text,
                dataset_summary=current_summary_json,
            )
        except Exception as exc:
            log.exception("failed to classify data statistics intent", error=safe_err(exc))
            return self._needs_input_result(
                request=request,
                user_message=(
                    "I could not classify that statistics request. Please ask again more directly."
                ),
            )

        if intent.intent_out_of_scope or not intent.has_any_intent():
            off_topic_reply = self._build_off_topic_response(
                user_message=latest_user_message,
                chat_history=history_text,
            )
            return self._needs_input_result(
                request=request,
                user_message=off_topic_reply,
            )

        summary_answer: str | None = None
        query_result: JSONDict | None = None
        statistics_result: JSONDict | None = None
        chart_result: JSONDict | None = None
        analytical_artifact_refs: list[ArtifactRef] = []
        chart_artifact_refs: list[ArtifactRef] = []

        working_df = current_df
        working_summary = current_summary
        working_summary_json = current_summary_json
        query_followup_requested = (
            intent.intent_statistical_analysis or intent.intent_chart
        )
        query_failed = False

        if intent.intent_summary_question:
            try:
                summary_answer = self._answer_summary_question(
                    intent_brief=intent.intent_summary_question_brief or latest_user_message,
                    dataset_summary=current_summary_json,
                    chat_history=history_text,
                )
            except Exception as exc:
                log.exception("failed to answer data statistics summary question", error=safe_err(exc))
                summary_answer = (
                    "I could not answer that precisely from the current dataset summary alone."
                )

        if intent.intent_readonly_query:
            try:
                (
                    query_result,
                    analytical_artifact_refs,
                    working_df,
                    working_summary,
                    working_summary_json,
                ) = self._run_readonly_query(
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    dataframe=working_df,
                    summary_model=working_summary,
                    summary_json=working_summary_json,
                    instructions=intent.intent_readonly_query_brief or latest_user_message,
                    prepare_followup_data=query_followup_requested,
                )
            except Exception as exc:
                log.exception("failed to run data statistics readonly query", error=safe_err(exc))
                query_failed = True
                query_result = {
                    "status": "error",
                    "instruction": intent.intent_readonly_query_brief or latest_user_message,
                    "detail": "I could not complete the requested read-only query.",
                }

        if intent.intent_statistical_analysis:
            if query_failed and intent.intent_readonly_query:
                statistics_result = {
                    "status": "skipped",
                    "detail": (
                        "Statistical analysis was not run because the requested read-only "
                        "query could not be completed."
                    ),
                }
            else:
                try:
                    statistics_result = self._run_statistical_analysis(
                        dataframe=working_df,
                        summary_model=working_summary,
                        user_request=(
                            intent.intent_statistical_analysis_brief or latest_user_message
                        ),
                    )
                except Exception as exc:
                    log.exception("failed to run data statistics analysis", error=safe_err(exc))
                    statistics_result = {
                        "status": "error",
                        "detail": "I could not complete the requested statistical analysis.",
                    }

        if intent.intent_chart:
            if query_failed and intent.intent_readonly_query:
                chart_result = {
                    "status": "skipped",
                    "detail": (
                        "Chart generation was not run because the requested read-only query "
                        "could not be completed."
                    ),
                }
            else:
                try:
                    chart_result, chart_artifact_refs = self._run_chart_intent(
                        user_id=request.user_id,
                        conversation_id=request.conversation_id,
                        dataframe=working_df,
                        summary_model=working_summary,
                        instructions=intent.intent_chart_brief or latest_user_message,
                    )
                except Exception as exc:
                    log.exception("failed to generate data statistics charts", error=safe_err(exc))
                    chart_result = {
                        "status": "error",
                        "detail": "I could not generate the requested chart output.",
                    }

        try:
            final_message = self._build_final_message(
                summary_answer=summary_answer,
                query_result=query_result,
                statistics_result=statistics_result,
                chart_result=chart_result,
                dataset_context={
                    "original_user_message": latest_user_message,
                    "handled_intents": {
                        "summary_question": intent.intent_summary_question,
                        "readonly_query": intent.intent_readonly_query,
                        "statistical_analysis": intent.intent_statistical_analysis,
                        "chart": intent.intent_chart,
                    },
                    "analysis_dataset_rows": int(len(working_df)),
                    "analysis_dataset_columns": [
                        str(column) for column in working_df.columns
                    ],
                    "used_query_result_for_followups": bool(
                        intent.intent_readonly_query
                        and not query_failed
                        and query_followup_requested
                    ),
                },
            )
        except Exception as exc:
            log.exception("failed to build final data statistics response", error=safe_err(exc))
            final_message = self._build_final_message_fallback(
                summary_answer=summary_answer,
                query_result=query_result,
                statistics_result=statistics_result,
                chart_result=chart_result,
            )

        return self._needs_input_result(
            request=request,
            user_message=final_message,
            artifact_refs=[*analytical_artifact_refs, *chart_artifact_refs],
        )

    def _classify_intent(
        self,
        *,
        latest_user_message: str,
        chat_history: str | None,
        dataset_summary: str,
    ) -> DataStatisticsIntentModel:
        payload: JSONDict = {
            "latest_user_message": latest_user_message,
            "chat_history": chat_history,
            "dataset_summary": dataset_summary,
        }
        return self._llm.generate_json(
            schema=DataStatisticsIntentModel,
            system_prompt=data_statistics_intent_classification_system_prompt(),
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.0, top_p=1.0),
            history=None,
            max_attempts=2,
        )

    def _answer_summary_question(
        self,
        *,
        intent_brief: str,
        dataset_summary: str,
        chat_history: str | None,
    ) -> str:
        payload: JSONDict = {
            "user_intent_brief": intent_brief,
            "dataset_summary": dataset_summary,
            "chat_history": chat_history,
        }
        response = self._llm.generate(
            system_prompt=data_statistics_summary_answer_system_prompt(),
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.2),
            history=None,
        )
        answer = response.content.strip()
        if not answer:
            return "I could not determine a precise answer from the summary alone."
        return answer

    def _run_readonly_query(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataframe: pd.DataFrame,
        summary_model: Any,
        summary_json: str,
        instructions: str,
        prepare_followup_data: bool,
    ) -> tuple[JSONDict, list[ArtifactRef], pd.DataFrame, Any, str]:
        result_df = self._run_data_manipulation_tool(
            dataframe=dataframe,
            conversation_id=conversation_id,
            summary_json=summary_json,
            instructions=instructions,
        )

        analytical_result_id = uuid.uuid4()
        self._data_repo.save_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=analytical_result_id,
            df=result_df,
            overwrite=True,
            include_index=False,
        )
        analytical_artifact_ref = _build_artifact_ref(
            artifact_id=analytical_result_id,
            artifact_type="data",
            artifact_format="csv",
            artifact_kind=_ARTIFACT_KIND_ANALYTICAL_RESULT,
        )

        if prepare_followup_data:
            analytical_summary = self._profiling_tool.extract_dataset_summary(
                result_df,
                max_categories=200,
                sample_distinct=200,
                compute_quantiles=False,
                strict=True,
            )
            next_dataframe = result_df
            next_summary = analytical_summary
            next_summary_json = self._profiling_tool.dataset_summary_to_json(
                analytical_summary
            )
        else:
            next_dataframe = dataframe
            next_summary = summary_model
            next_summary_json = summary_json

        return (
            {
                "status": "analytical_query",
                "instruction": instructions,
                "analytical_result_id": str(analytical_result_id),
                "result": _dataframe_preview(result_df),
            },
            [analytical_artifact_ref],
            next_dataframe,
            next_summary,
            next_summary_json,
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
                "data manipulation tool must accept either 'table_name' or "
                "'conversation_id'"
            )
        if "retry_attempts" in params:
            kwargs["retry_attempts"] = _ANALYTICAL_QUERY_RETRY_ATTEMPTS

        return manipulate(**kwargs)

    def _run_statistical_analysis(
        self,
        *,
        dataframe: pd.DataFrame,
        summary_model: Any,
        user_request: str,
    ) -> JSONDict:
        result = self._advanced_analytics_tool.analyze(
            dataframe=dataframe,
            data_summary=summary_model,
            user_request=user_request,
        )
        return {
            "status": "statistics_complete",
            "analysis_type": result.analysis_type,
            "summary": result.summary,
            "tables": result.tables,
            "metrics": result.metrics,
        }

    def _run_chart_intent(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataframe: pd.DataFrame,
        summary_model: Any,
        instructions: str,
    ) -> tuple[JSONDict, list[ArtifactRef]]:
        specs = self._plot_tool.generate_specs(
            dataframe=dataframe,
            data_summary=summary_model,
            user_intent=instructions,
        )
        saved_refs: list[ArtifactRef] = []
        for spec in specs:
            saved_id = uuid.uuid4()
            self._data_repo.save_json_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=saved_id,
                json_data=json.dumps(spec, ensure_ascii=False),
                overwrite=True,
            )
            saved_refs.append(
                _build_artifact_ref(
                    artifact_id=saved_id,
                    artifact_type="graph",
                    artifact_format="json",
                    artifact_kind=_ARTIFACT_KIND_CHART_SPEC,
                )
            )

        return (
            {
                "status": "charts_saved",
                "instruction": instructions,
                "saved_chart_spec_ids": [
                    str(artifact_id)
                    for saved_ref in saved_refs
                    if (artifact_id := saved_ref.get("id")) is not None
                ],
                "saved_chart_count": len(saved_refs),
            },
            saved_refs,
        )

    def _build_ready_message(self, *, summary: Any) -> str:
        profiles = list(summary.profiles)
        shown_profiles = profiles[:_INITIAL_SUMMARY_MAX_COLUMNS]
        preview = ", ".join(
            f"{profile.name} ({profile.inferred_kind.lower()})"
            for profile in shown_profiles
        )
        extra_columns = len(profiles) - len(shown_profiles)
        if extra_columns > 0:
            preview += f", +{extra_columns} more"
        return (
            f"Data statistics is ready — {summary.n_rows} rows, {len(profiles)} columns: "
            f"{preview}. Ask for read-only summaries, queries, statistical analyses, or charts."
        )

    def _build_off_topic_response(
        self,
        *,
        user_message: str,
        chat_history: str | None,
    ) -> str:
        payload: JSONDict = {
            "user_message": user_message,
            "chat_history": chat_history,
        }
        try:
            response = self._llm.generate(
                system_prompt=data_statistics_off_topic_system_prompt(),
                user_prompt=json.dumps(payload, ensure_ascii=False),
                config=LLMConfig(model="basic", temperature=0.3),
                history=None,
            )
            answer = response.content.strip()
            if answer:
                return answer
        except Exception as exc:
            log.exception("failed to generate off-topic response", error=safe_err(exc))
        return _OFF_TOPIC_FALLBACK_MESSAGE

    def _build_final_message(
        self,
        *,
        summary_answer: str | None,
        query_result: JSONDict | None,
        statistics_result: JSONDict | None,
        chart_result: JSONDict | None,
        dataset_context: JSONDict,
    ) -> str:
        payload: JSONDict = {
            "summary_answer": summary_answer,
            "query_result": query_result,
            "statistics_result": statistics_result,
            "chart_result": chart_result,
            "dataset_context": dataset_context,
        }
        response = self._llm.generate(
            system_prompt=data_statistics_final_response_system_prompt(),
            user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
            config=LLMConfig(model="basic", temperature=0.3),
            history=None,
        )
        return response.content

    def _build_final_message_fallback(
        self,
        *,
        summary_answer: str | None,
        query_result: JSONDict | None,
        statistics_result: JSONDict | None,
        chart_result: JSONDict | None,
    ) -> str:
        parts: list[str] = []

        if summary_answer:
            parts.append(summary_answer.strip())

        if query_result is not None:
            status = str(query_result.get("status", "")).strip()
            if status == "analytical_query":
                parts.append("Ran the requested read-only analytical query.")
            elif status == "error":
                parts.append(
                    str(
                        query_result.get(
                            "detail", "The requested read-only analytical query failed."
                        )
                    )
                )

        if statistics_result is not None:
            status = str(statistics_result.get("status", "")).strip()
            if status == "statistics_complete":
                parts.append(str(statistics_result.get("summary", "")).strip())
            elif status in {"error", "skipped"}:
                parts.append(
                    str(
                        statistics_result.get(
                            "detail", "The requested statistical analysis could not be completed."
                        )
                    )
                )

        if chart_result is not None:
            status = str(chart_result.get("status", "")).strip()
            if status == "charts_saved":
                count = int(chart_result.get("saved_chart_count", 0) or 0)
                noun = "chart" if count == 1 else "charts"
                parts.append(f"Generated {count} {noun}." if count > 0 else "Generated chart output.")
            elif status in {"error", "skipped"}:
                parts.append(
                    str(
                        chart_result.get(
                            "detail", "The requested chart output could not be completed."
                        )
                    )
                )

        return " ".join(part for part in parts if part) or "Completed the data statistics request."

    def _needs_input_result(
        self,
        *,
        request: NodeRequest,
        user_message: str,
        artifact_refs: Sequence[ArtifactRef] | None = None,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=DataStatisticsState.init_empty(),
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
            new_node_state=DataStatisticsState.init_empty(),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_DATA",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )


def _latest_user_message(messages_history: Sequence[ChatMessage] | None) -> str | None:
    if not messages_history:
        return None

    for message in reversed(messages_history):
        if message.role != "user":
            continue
        content = message.content.strip()
        if content:
            return content

    return None


def _recent_history_text(messages_history: Sequence[ChatMessage] | None) -> str | None:
    if not messages_history or len(messages_history) <= 1:
        return None

    recent_messages = list(messages_history[-5:-1])
    if not recent_messages:
        return None

    return get_chat_messages_role_and_message_json(recent_messages)


def _dataframe_preview(dataframe: pd.DataFrame, *, row_limit: int = 10) -> JSONDict:
    preview_df = dataframe.head(row_limit).copy()
    for column in preview_df.columns:
        if pd.api.types.is_datetime64_any_dtype(preview_df[column]):
            preview_df[column] = preview_df[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    preview_df = preview_df.where(pd.notnull(preview_df), None)
    return {
        "row_count": int(len(dataframe)),
        "columns": [str(column) for column in dataframe.columns],
        "preview_rows": preview_df.to_dict(orient="records"),
    }


def _conversation_id_to_table_name(conversation_id: UUID) -> str:
    digest = hashlib.sha256(str(conversation_id).encode("ascii")).hexdigest()
    return f"{_WORKING_TABLE_PREFIX}{digest[:_WORKING_TABLE_HASH_HEX_LEN]}"


def _build_artifact_ref(
    *,
    artifact_id: UUID,
    artifact_type: str,
    artifact_format: str,
    artifact_kind: str,
) -> ArtifactRef:
    return {
        "id": artifact_id,
        "kind": cast(Any, artifact_type),
        "format": cast(Any, artifact_format),
        "artifact_meta": {"kind": artifact_kind},
    }
