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
    data_statistics_advanced_analytics_request_prompt,
    data_statistics_final_response_system_prompt,
    data_statistics_intent_classification_system_prompt,
    data_statistics_node_info,
    data_statistics_off_topic_system_prompt,
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
    "This is a data statistics stage. I can run analytical queries, "
    "formal statistical tests, and generate charts. "
    "I cannot change the dataset or move into downstream causal or model stages from here."
)


class DataStatisticsIntentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent_analytics: bool = False
    intent_analytics_brief: str = ""
    intent_chart: bool = False
    intent_chart_brief: str = ""
    intent_advanced_analytics: bool = False
    intent_advanced_analytics_brief: str = ""
    intent_out_of_scope: bool = False

    @field_validator(
        "intent_analytics_brief",
        "intent_chart_brief",
        "intent_advanced_analytics_brief",
        mode="before",
    )
    @classmethod
    def _normalize_brief(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @model_validator(mode="after")
    def _validate(self) -> DataStatisticsIntentModel:
        in_scope_count = (
            int(self.intent_analytics)
            + int(self.intent_chart)
            + int(self.intent_advanced_analytics)
        )

        if self.intent_out_of_scope and in_scope_count > 0:
            raise ValueError("intent_out_of_scope cannot be combined with in-scope intents")
        if self.intent_analytics and not self.intent_analytics_brief:
            raise ValueError("intent_analytics_brief is required")
        if self.intent_chart and not self.intent_chart_brief:
            raise ValueError("intent_chart_brief is required")
        if self.intent_advanced_analytics and not self.intent_advanced_analytics_brief:
            raise ValueError("intent_advanced_analytics_brief is required")
        return self

    def has_any_intent(self) -> bool:
        return self.intent_analytics or self.intent_chart or self.intent_advanced_analytics


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

        try:
            deps = DataStatisticsDeps.from_request(request)
        except Exception as exc:
            log.info("missing data statistics dependencies", error=safe_err(exc))
            return self._needs_data_result(
                request=request,
                user_message=(
                    "I do not have a working dataset available for data statistics yet. "
                    "Please upload or select a dataset first."
                ),
            )

        try:
            current_df = self._data_repo.get_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=deps.dataset_id,
                limit=1_000_000,
            )
        except Exception as exc:
            log.exception(
                "failed to load data statistics dataset",
                dataset_id=str(deps.dataset_id),
                error=safe_err(exc),
            )
            return self._needs_data_result(
                request=request,
                user_message=(
                    "I could not load the working dataset for data statistics. Please "
                    "re-upload or reselect the dataset and try again."
                ),
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

        analytics_result: JSONDict | None = None
        advanced_analytics_result: JSONDict | None = None
        chart_result: JSONDict | None = None
        analytical_artifact_refs: list[ArtifactRef] = []
        chart_artifact_refs: list[ArtifactRef] = []

        working_df = current_df
        working_summary = current_summary
        working_summary_json = current_summary_json
        analytics_failed = False

        # ── intent_analytics → DataManipulationTool (DuckDB) ──
        if intent.intent_analytics:
            try:
                (
                    analytics_result,
                    analytical_artifact_refs,
                    working_df,
                    working_summary,
                    working_summary_json,
                ) = self._run_analytics_query(
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    dataframe=working_df,
                    summary_model=working_summary,
                    summary_json=working_summary_json,
                    instructions=intent.intent_analytics_brief or latest_user_message,
                    prepare_followup_data=intent.intent_chart,
                )
            except Exception as exc:
                log.exception("analytics query failed", error=safe_err(exc))
                analytics_failed = True
                analytics_result = {
                    "status": "error",
                    "instruction": intent.intent_analytics_brief or latest_user_message,
                    "detail": (
                        "I encountered an error while attempting to run the analytical "
                        "query. Please try rephrasing your request."
                    ),
                }

        # ── intent_chart → DataManipulationTool (DuckDB) + PlotTool ──
        if intent.intent_chart:
            if analytics_failed and intent.intent_analytics:
                chart_result = {
                    "status": "skipped",
                    "detail": (
                        "Chart generation was skipped because the preceding analytical "
                        "query could not be completed."
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
                    log.exception("chart generation failed", error=safe_err(exc))
                    chart_result = {
                        "status": "error",
                        "detail": "I could not generate the requested charts.",
                    }

        # ── intent_advanced_analytics → AdvancedAnalyticsTool + PlotTool ──
        if intent.intent_advanced_analytics:
            try:
                advanced_instructions = self._build_advanced_analytics_request(
                    instructions=intent.intent_advanced_analytics_brief or latest_user_message,
                    latest_user_message=latest_user_message,
                    chat_history=history_text,
                )
                adv_result, adv_chart_refs = self._run_advanced_analytics_intent(
                    user_id=request.user_id,
                    conversation_id=request.conversation_id,
                    dataframe=current_df,
                    summary_model=current_summary,
                    instructions=advanced_instructions,
                )
                advanced_analytics_result = adv_result
                chart_artifact_refs.extend(adv_chart_refs)
            except Exception as exc:
                log.exception("advanced analytics failed", error=safe_err(exc))
                detail = str(exc).strip() if isinstance(exc, ValueError) else ""
                advanced_analytics_result = {
                    "status": "error",
                    "detail": detail or "I could not complete the requested statistical analysis.",
                }

        try:
            final_message = self._build_final_message(
                analytics_result=analytics_result,
                advanced_analytics_result=advanced_analytics_result,
                chart_result=chart_result,
                dataset_context={
                    "original_user_message": latest_user_message,
                    "handled_intents": {
                        "analytics": intent.intent_analytics,
                        "chart": intent.intent_chart,
                        "advanced_analytics": intent.intent_advanced_analytics,
                    },
                    "analysis_dataset_rows": int(len(working_df)),
                    "analysis_dataset_columns": [str(column) for column in working_df.columns],
                },
            )
        except Exception as exc:
            log.exception("failed to build final data statistics response", error=safe_err(exc))
            final_message = self._build_final_message_fallback(
                analytics_result=analytics_result,
                advanced_analytics_result=advanced_analytics_result,
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

    # ── intent_analytics → DataManipulationTool (DuckDB) ──

    def _run_analytics_query(
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
            next_summary_json = self._profiling_tool.dataset_summary_to_json(analytical_summary)
        else:
            next_dataframe = dataframe
            next_summary = summary_model
            next_summary_json = summary_json

        return (
            {
                "status": "analytics_complete",
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
                "data manipulation tool must accept either 'table_name' or 'conversation_id'"
            )
        if "retry_attempts" in params:
            kwargs["retry_attempts"] = _ANALYTICAL_QUERY_RETRY_ATTEMPTS

        return manipulate(**kwargs)

    # ── intent_chart → DataManipulationTool (DuckDB) + PlotTool ──

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
                json_data=json.dumps(spec, ensure_ascii=False, allow_nan=False),
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

    # ── intent_advanced_analytics → AdvancedAnalyticsTool + PlotTool ──

    def _build_advanced_analytics_request(
        self,
        *,
        instructions: str,
        latest_user_message: str,
        chat_history: str | None,
    ) -> str:
        template = data_statistics_advanced_analytics_request_prompt()
        return template.format(
            resolved_request=instructions.strip() or latest_user_message.strip(),
            latest_user_message=latest_user_message.strip(),
            chat_history=(
                chat_history.strip() if chat_history and chat_history.strip() else "(none)"
            ),
        )

    def _run_advanced_analytics_intent(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataframe: pd.DataFrame,
        summary_model: Any,
        instructions: str,
    ) -> tuple[JSONDict, list[ArtifactRef]]:
        result = self._advanced_analytics_tool.analyze(
            dataframe=dataframe,
            data_summary=summary_model,
            user_request=instructions,
        )

        adv_result: JSONDict = {
            "status": "advanced_analytics_complete",
            "analysis_type": result.analysis_type,
            "summary": result.summary,
            "tables": result.tables,
            "metrics": result.metrics,
        }

        # Generate charts from the advanced analytics results
        chart_refs: list[ArtifactRef] = []
        try:
            chart_instructions = (
                f"Generate charts to visualize the results of the following "
                f"{result.analysis_type} analysis: {instructions}"
            )
            specs = self._plot_tool.generate_specs(
                dataframe=dataframe,
                data_summary=summary_model,
                user_intent=chart_instructions,
            )
            for spec in specs:
                saved_id = uuid.uuid4()
                self._data_repo.save_json_data(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataset_id=saved_id,
                    json_data=json.dumps(spec, ensure_ascii=False, allow_nan=False),
                    overwrite=True,
                )
                chart_refs.append(
                    _build_artifact_ref(
                        artifact_id=saved_id,
                        artifact_type="graph",
                        artifact_format="json",
                        artifact_kind=_ARTIFACT_KIND_CHART_SPEC,
                    )
                )
            adv_result["charts_generated"] = len(chart_refs)
        except Exception as exc:
            log.warning(
                "chart generation for advanced analytics skipped",
                error=safe_err(exc),
            )
            adv_result["charts_generated"] = 0

        return adv_result, chart_refs

    # ── Response builders ──

    def _build_ready_message(self, *, summary: Any) -> str:
        profiles = list(summary.profiles)
        shown_profiles = profiles[:_INITIAL_SUMMARY_MAX_COLUMNS]
        preview = ", ".join(
            f"{profile.name} ({profile.inferred_kind.lower()})" for profile in shown_profiles
        )
        extra_columns = len(profiles) - len(shown_profiles)
        if extra_columns > 0:
            preview += f", +{extra_columns} more"
        return (
            f"Data statistics is ready — {summary.n_rows} rows, {len(profiles)} columns: "
            f"{preview}. Ask for analytical queries, statistical tests, or charts."
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
        analytics_result: JSONDict | None,
        advanced_analytics_result: JSONDict | None,
        chart_result: JSONDict | None,
        dataset_context: JSONDict,
    ) -> str:
        payload: JSONDict = {
            "analytics_result": analytics_result,
            "advanced_analytics_result": advanced_analytics_result,
            "chart_result": chart_result,
            "dataset_context": dataset_context,
        }
        response = self._llm.generate(
            system_prompt=data_statistics_final_response_system_prompt(),
            user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
            config=LLMConfig(model="basic", temperature=0.3),
            history=None,
        )
        content = response.content.strip()
        if not content:
            raise ValueError("LLM returned empty final response")
        return content

    def _build_final_message_fallback(
        self,
        *,
        analytics_result: JSONDict | None,
        advanced_analytics_result: JSONDict | None,
        chart_result: JSONDict | None,
    ) -> str:
        parts: list[str] = []

        if analytics_result is not None:
            status = str(analytics_result.get("status", "")).strip()
            if status == "analytics_complete":
                parts.append("Ran the requested analytical query.")
            elif status == "error":
                parts.append(str(analytics_result.get("detail", "The analytical query failed.")))

        if advanced_analytics_result is not None:
            status = str(advanced_analytics_result.get("status", "")).strip()
            if status == "advanced_analytics_complete":
                parts.append(str(advanced_analytics_result.get("summary", "")).strip())
            elif status in {"error", "skipped"}:
                parts.append(
                    str(
                        advanced_analytics_result.get(
                            "detail", "The statistical analysis could not be completed."
                        )
                    )
                )

        if chart_result is not None:
            status = str(chart_result.get("status", "")).strip()
            if status == "charts_saved":
                count = int(chart_result.get("saved_chart_count", 0) or 0)
                noun = "chart" if count == 1 else "charts"
                parts.append(
                    f"Generated {count} {noun}." if count > 0 else "Generated chart output."
                )
            elif status in {"error", "skipped"}:
                parts.append(
                    str(chart_result.get("detail", "The requested charts could not be generated."))
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
