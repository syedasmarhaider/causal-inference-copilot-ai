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

from python.domain.models.models import ArtifactRef, get_chat_messages_role_and_message_json
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import (
    ChatMessage,
    LLMConfig,
    LLMService,
)
from python.domain.workflows.node import Node
from python.domain.workflows.ochestrator_state import ReadOnlyOchestratorState
from python.domain.workflows.node_state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.data_dashboard.data_dashboard_state import (
    DataDashboardPayloadModel,
    DataDashboardState,
)
from python.implementation.workflows.nodes.dataset.dataset_prompts import (
    dataset_final_response_system_prompt,
    dataset_intent_classification_system_prompt,
    dataset_missing_data_system_prompt,
    dataset_summary_answer_system_prompt,
    prev_state_revert_message,
)
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
    DatasetSummaryModel,
)
from python.implementation.workflows.tools.plot_tool.plot_tool import PlotTool
from python.implementation.workflows.utils.utils import JSONDict, safe_err

log = get_app_logger(__name__, component="data_dashboard_node", log_type="node")

_DATA_MANIPULATION_RETRY_ATTEMPTS = 3
_WORKING_TABLE_PREFIX = "df_"
_WORKING_TABLE_HASH_HEX_LEN = 16
_ARTIFACT_KIND_ANALYTICAL_RESULT = "analytical_result"
_ARTIFACT_KIND_CHART_SPEC = "chart_spec"
_INITIAL_SUMMARY_MAX_COLUMNS = 8

_READY_MESSAGE = (
    "Dashboard dataset is ready. You can ask questions about the data, request statistical "
    "queries, apply transformations, or generate charts."
)
_OFF_TOPIC_MESSAGE = (
    "I can only help with data exploration here — inspecting columns, answering dataset "
    "questions, running statistical or analytical queries, transforming data, and generating "
    "charts. Please rephrase your request around the current dataset."
)
_OFF_TOPIC_CLARIFICATION_MESSAGE = (
    "I mean this dashboard only works on the current dataset. I can inspect columns, answer "
    "data questions, run analytical queries, clean or reshape the data, and generate charts. "
    "Please rephrase your request."
)


class DashboardIntentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent_data_question: bool = False
    intent_data_question_brief: str = ""
    intent_manupulation_question: bool = False
    intent_manupulation_question_brief: str = ""
    intent_manupulation_is_analytical_query: bool = False
    intent_chart: bool = False
    intent_chart_brief: str = ""

    @field_validator(
        "intent_data_question_brief",
        "intent_manupulation_question_brief",
        "intent_chart_brief",
        mode="before",
    )
    @classmethod
    def _normalize_brief(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @model_validator(mode="after")
    def _validate(self) -> DashboardIntentModel:
        if self.intent_manupulation_is_analytical_query and not self.intent_manupulation_question:
            raise ValueError(
                "intent_manupulation_is_analytical_query requires intent_manupulation_question"
            )
        if self.intent_data_question and not self.intent_data_question_brief:
            raise ValueError("intent_data_question_brief is required")
        if self.intent_manupulation_question and not self.intent_manupulation_question_brief:
            raise ValueError("intent_manupulation_question_brief is required")
        if self.intent_chart and not self.intent_chart_brief:
            raise ValueError("intent_chart_brief is required")
        return self

    def has_any_intent(self) -> bool:
        return self.intent_data_question or self.intent_manupulation_question or self.intent_chart


class DataDashboardNode(Node):
    NAME: ClassVar[str] = DataDashboardState.NAME
    _data_manipulation_tool: DataManipulationTool
    _plot_tool: PlotTool
    _profiling_tool: DatasetProfilingTool

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

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return (
            "Data dashboard node. Supports data exploration, statistical queries, "
            "dataset transformations, chart generation, and reverting to previous dataset versions."
        )

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        readonly_orchestrator_state: ReadOnlyOchestratorState,
        messages_history: Sequence[ChatMessage] | None,
    ) -> State:
        if not isinstance(state, DataDashboardState):
            raise TypeError(f"{self.name}: expected DataDashboardState, got {type(state).__name__}")

        dataset_iterations = list(state.payload.dataset_iterations)
        latest_summary = (
            state.payload.latest_summary.model_copy(deep=True)
            if state.payload.latest_summary is not None
            else None
        )

        # --- revert ---
        if _is_revert_request(messages_history):
            return self._handle_revert(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_iterations=dataset_iterations,
                current_latest_summary=latest_summary,
            )

        # --- load dataset ---
        current_df: pd.DataFrame
        current_summary: DatasetSummaryModel
        current_summary_json: str
        loaded_this_turn = False

        if dataset_iterations:
            latest_iteration = dataset_iterations[-1]
            try:
                current_df = self._data_repo.get_csv_data(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataset_id=latest_iteration,
                    limit=1_000_000,
                )
            except Exception as exc:
                log.exception(
                    "failed to load dashboard dataset iteration",
                    dataset_id=str(latest_iteration),
                    error=safe_err(exc),
                )
                return self._build_state(
                    dataset_iterations=dataset_iterations,
                    latest_summary=latest_summary,
                    user_message=(
                        "I could not load the current working dataset. "
                        "Please re-upload the CSV or try again."
                    ),
                )
            current_summary = latest_summary or self._profiling_tool.extract_dataset_summary(
                current_df,
                max_categories=200,
                sample_distinct=200,
                compute_quantiles=False,
                strict=True,
            )
            current_summary_json = self._profiling_tool.dataset_summary_to_json(current_summary)
        else:
            try:
                current_df = self._data_repo.get_csv_data(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataset_id=DataDashboardState.INIT_DATA_ID,
                    limit=1_000_000,
                )
            except Exception:
                return self._build_state(
                    user_message=self._build_missing_data_message(
                        messages_history=messages_history
                    ),
                )
            current_summary = self._profiling_tool.extract_dataset_summary(
                current_df,
                max_categories=200,
                sample_distinct=200,
                compute_quantiles=False,
                strict=True,
            )
            current_summary_json = self._profiling_tool.dataset_summary_to_json(current_summary)
            dataset_iterations.append(DataDashboardState.INIT_DATA_ID)
            latest_summary = current_summary
            loaded_this_turn = True

        persisted_latest_summary = current_summary

        # --- no user message yet ---
        latest_user_message = _latest_user_message(messages_history)
        if not latest_user_message:
            return self._build_state(
                dataset_iterations=dataset_iterations,
                latest_summary=persisted_latest_summary,
                user_message=(
                    self._build_loaded_dataset_message(summary=current_summary)
                    if loaded_this_turn
                    else _READY_MESSAGE
                ),
            )

        # --- recent history for context ---
        last_4_history = (
            messages_history[-5:-1] if messages_history and len(messages_history) > 1 else None
        )
        last_4_history_text: str | None = (
            get_chat_messages_role_and_message_json(last_4_history) if last_4_history else None
        )

        # --- classify intent ---
        try:
            intent = self._classify_intent(
                latest_user_message=latest_user_message,
                chat_history=last_4_history_text,
                dataset_summary=current_summary_json,
            )
        except Exception as exc:
            log.exception("failed to classify dashboard intent", error=safe_err(exc))
            return self._build_state(
                dataset_iterations=dataset_iterations,
                latest_summary=persisted_latest_summary,
                user_message=(
                    "Dataset is loaded but I could not classify your request. "
                    "Please ask again more directly."
                ),
            )

        if not intent.has_any_intent():
            if loaded_this_turn:
                return self._build_state(
                    dataset_iterations=dataset_iterations,
                    latest_summary=persisted_latest_summary,
                    user_message=self._build_loaded_dataset_message(summary=current_summary),
                )
            if _is_dashboard_scope_message(_latest_assistant_message(messages_history)):
                return self._build_state(
                    dataset_iterations=dataset_iterations,
                    latest_summary=persisted_latest_summary,
                    user_message=_OFF_TOPIC_CLARIFICATION_MESSAGE,
                )
            return self._build_state(
                dataset_iterations=dataset_iterations,
                latest_summary=persisted_latest_summary,
                user_message=_OFF_TOPIC_MESSAGE,
            )

        # --- execute intents ---
        summary_answer: str | None = None
        manipulation_result: JSONDict | None = None
        chart_result: JSONDict | None = None
        manipulation_artifact_refs: list[ArtifactRef] = []
        chart_artifact_refs: list[ArtifactRef] = []

        working_df = current_df
        working_summary = current_summary
        working_summary_json = current_summary_json

        if intent.intent_data_question:
            summary_answer = self._answer_summary_question(
                intent_brief=intent.intent_data_question_brief or latest_user_message,
                dataset_summary=current_summary_json,
                chat_history=last_4_history_text,
            )

        if intent.intent_manupulation_question:
            try:
                (
                    manipulation_result,
                    manipulation_artifact_refs,
                    dataset_iterations,
                    working_df,
                    working_summary,
                    working_summary_json,
                    persisted_latest_summary,
                ) = self._run_manipulation_intent(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataset_iterations=dataset_iterations,
                    dataframe=working_df,
                    summary_model=working_summary,
                    summary_json=working_summary_json,
                    profiling_tool=self._profiling_tool,
                    instructions=intent.intent_manupulation_question_brief or latest_user_message,
                    analytical_query=intent.intent_manupulation_is_analytical_query,
                    prepare_chart_data=intent.intent_chart,
                )
            except Exception as exc:
                log.exception("failed to run dashboard manipulation intent", error=safe_err(exc))
                return self._build_state(
                    dataset_iterations=dataset_iterations,
                    latest_summary=persisted_latest_summary,
                    user_message=(
                        "I could not complete that data query or transformation. "
                        "Please try rephrasing the request."
                    ),
                )

        if intent.intent_chart:
            try:
                chart_result, chart_artifact_refs = self._run_chart_intent(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataframe=working_df,
                    summary_model=working_summary,
                    instructions=intent.intent_chart_brief or latest_user_message,
                )
            except Exception as exc:
                log.exception("failed to generate dashboard charts", error=safe_err(exc))
                return self._build_state(
                    dataset_iterations=dataset_iterations,
                    latest_summary=persisted_latest_summary,
                    user_message=(
                        "I could not generate the requested chart. "
                        "Please try rephrasing the chart request."
                    ),
                )

        try:
            final_message = self._build_final_message(
                summary_answer=summary_answer,
                manipulation_result=manipulation_result,
                chart_result=chart_result,
                dataset_context={
                    "dataset_loaded_this_turn": loaded_this_turn,
                    "original_user_message": latest_user_message,
                    "handled_intents": {
                        "data_question": intent.intent_data_question,
                        "manipulation_question": intent.intent_manupulation_question,
                        "chart": intent.intent_chart,
                    },
                    "active_dataset_rows": int(len(working_df)),
                    "active_dataset_columns": [str(c) for c in working_df.columns],
                },
            )
        except Exception as exc:
            log.exception("failed to build dashboard final response", error=safe_err(exc))
            final_message = self._build_final_message_fallback(
                summary_answer=summary_answer,
                manipulation_result=manipulation_result,
                chart_result=chart_result,
            )

        return self._build_state(
            dataset_iterations=dataset_iterations,
            latest_summary=persisted_latest_summary,
            user_message=final_message,
            response_message_artifact_refs=[*manipulation_artifact_refs, *chart_artifact_refs],
        )

    # -------------------------------------------------------------------------
    # State builder
    # -------------------------------------------------------------------------

    def _build_state(
        self,
        *,
        user_message: str,
        dataset_iterations: Sequence[UUID] | None = None,
        latest_summary: DatasetSummaryModel | None = None,
        response_message_artifact_refs: Sequence[ArtifactRef] | None = None,
    ) -> DataDashboardState:
        return DataDashboardState(
            DataDashboardPayloadModel(
                dataset_iterations=list(dataset_iterations or []),
                latest_summary=(
                    latest_summary.model_copy(deep=True) if latest_summary is not None else None
                ),
                user_message=user_message,
            ),
            response_message_artifact_refs=list(response_message_artifact_refs or []),
        )

    # -------------------------------------------------------------------------
    # Revert
    # -------------------------------------------------------------------------

    def _handle_revert(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataset_iterations: list[UUID],
        current_latest_summary: DatasetSummaryModel | None,
    ) -> DataDashboardState:
        if not dataset_iterations:
            return self._build_state(
                user_message="No working dataset to revert. Please upload a CSV dataset.",
            )
        if len(dataset_iterations) == 1:
            return self._build_state(
                dataset_iterations=dataset_iterations,
                latest_summary=current_latest_summary,
                user_message="No previous dataset version to revert to.",
            )
        reverted_iterations = dataset_iterations[:-1]
        reverted_summary = self._load_latest_summary(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=reverted_iterations[-1],
        )
        return self._build_state(
            dataset_iterations=reverted_iterations,
            latest_summary=reverted_summary,
            user_message="Reverted to the previous working dataset version.",
        )

    def _load_latest_summary(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
    ) -> DatasetSummaryModel | None:
        try:
            df = self._data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=dataset_id,
                limit=1_000_000,
            )
        except Exception as exc:
            log.exception(
                "failed to load dataset for summary refresh",
                dataset_id=str(dataset_id),
                error=safe_err(exc),
            )
            return None
        return self._profiling_tool.extract_dataset_summary(
            df,
            max_categories=200,
            sample_distinct=200,
            compute_quantiles=False,
            strict=True,
        )

    # -------------------------------------------------------------------------
    # LLM helpers
    # -------------------------------------------------------------------------

    def _build_missing_data_message(
        self,
        *,
        messages_history: Sequence[ChatMessage] | None,
    ) -> str:
        response = self._llm.generate(
            system_prompt=None,
            user_prompt=dataset_missing_data_system_prompt(),
            config=LLMConfig(model="mini", temperature=0.4),
            history=messages_history,
        )
        return response.content.strip()

    def _build_loaded_dataset_message(self, *, summary: DatasetSummaryModel) -> str:
        profiles = list(summary.profiles)
        column_count = len(profiles)
        shown = profiles[:_INITIAL_SUMMARY_MAX_COLUMNS]
        preview = ", ".join(
            f"{p.name} ({p.inferred_kind.lower()})" for p in shown
        )
        if column_count > _INITIAL_SUMMARY_MAX_COLUMNS:
            preview = f"{preview}, +{column_count - _INITIAL_SUMMARY_MAX_COLUMNS} more columns"
        return (
            f"Dashboard dataset loaded. Profiled {summary.n_rows} rows and {column_count} "
            f"columns: {preview}. You can now explore the data, run statistical queries, "
            "transform it, or generate charts."
        )

    def _classify_intent(
        self,
        *,
        latest_user_message: str,
        chat_history: str | None,
        dataset_summary: str,
    ) -> DashboardIntentModel:
        payload: JSONDict = {
            "latest_user_message": latest_user_message,
            "chat_history": chat_history,
            "dataset_summary": dataset_summary,
        }
        return self._llm.generate_json(
            schema=DashboardIntentModel,
            system_prompt=dataset_intent_classification_system_prompt(),
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
        try:
            response = self._llm.generate(
                system_prompt=dataset_summary_answer_system_prompt(),
                user_prompt=json.dumps(payload, ensure_ascii=False),
                config=LLMConfig(model="basic", temperature=0.2),
                history=None,
            )
        except Exception as exc:
            log.exception("failed to answer dashboard summary question", error=safe_err(exc))
            return "I could not answer that from the dataset summary alone."
        answer = response.content.strip() if response.content else ""
        return answer or "I could not answer that from the dataset summary alone."

    def _build_final_message(
        self,
        *,
        summary_answer: str | None,
        manipulation_result: JSONDict | None,
        chart_result: JSONDict | None,
        dataset_context: JSONDict,
    ) -> str:
        payload: JSONDict = {
            "summary_answer": summary_answer,
            "manipulation_result": manipulation_result,
            "chart_result": chart_result,
            "dataset_context": dataset_context,
        }
        response = self._llm.generate(
            system_prompt=dataset_final_response_system_prompt(),
            user_prompt=json.dumps(payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.3),
            history=None,
        )
        return response.content

    def _build_final_message_fallback(
        self,
        *,
        summary_answer: str | None,
        manipulation_result: JSONDict | None,
        chart_result: JSONDict | None,
    ) -> str:
        parts: list[str] = []
        if summary_answer:
            parts.append(summary_answer.strip())
        if manipulation_result is not None:
            status = str(manipulation_result.get("status", "")).strip()
            if status == "dataset_updated":
                parts.append("Saved an updated working dataset version.")
            elif status == "analytical_query":
                parts.append("Ran the requested analytical query.")
        if chart_result is not None:
            count = int(chart_result.get("saved_chart_count", 0) or 0)
            noun = "chart" if count == 1 else "charts"
            parts.append(f"Generated {count} {noun}." if count > 0 else "Generated the chart output.")
        return " ".join(parts) if parts else "Completed the dashboard request."

    # -------------------------------------------------------------------------
    # Manipulation intent
    # -------------------------------------------------------------------------

    def _run_manipulation_intent(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataset_iterations: list[UUID],
        dataframe: pd.DataFrame,
        summary_model: DatasetSummaryModel,
        summary_json: str,
        profiling_tool: DatasetProfilingTool,
        instructions: str,
        analytical_query: bool,
        prepare_chart_data: bool,
    ) -> tuple[
        JSONDict,
        list[ArtifactRef],
        list[UUID],
        pd.DataFrame,
        DatasetSummaryModel,
        str,
        DatasetSummaryModel,
    ]:
        result_df = self._run_data_manipulation_tool(
            dataframe=dataframe,
            conversation_id=conversation_id,
            summary_json=summary_json,
            instructions=instructions,
        )

        if analytical_query:
            analytical_result_id = uuid.uuid4()
            self._data_repo.save_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=analytical_result_id,
                df=result_df,
                overwrite=True,
                include_index=False,
            )
            analytical_artifact_ref = _build_data_artifact_ref(
                artifact_id=analytical_result_id,
                artifact_format="csv",
                artifact_kind=_ARTIFACT_KIND_ANALYTICAL_RESULT,
            )
            if prepare_chart_data:
                analytical_summary = profiling_tool.extract_dataset_summary(
                    result_df,
                    max_categories=200,
                    sample_distinct=200,
                    compute_quantiles=False,
                    strict=True,
                )
                analytical_summary_json = profiling_tool.dataset_summary_to_json(analytical_summary)
                next_df, next_summary, next_summary_json = result_df, analytical_summary, analytical_summary_json
            else:
                next_df, next_summary, next_summary_json = dataframe, summary_model, summary_json
            return (
                {
                    "status": "analytical_query",
                    "instruction": instructions,
                    "analytical_result_id": str(analytical_result_id),
                    "result": _dataframe_preview(result_df),
                },
                [analytical_artifact_ref],
                dataset_iterations,
                next_df,
                next_summary,
                next_summary_json,
                summary_model,
            )

        new_dataset_id = uuid.uuid4()
        self._data_repo.save_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=new_dataset_id,
            df=result_df,
            overwrite=True,
            include_index=False,
        )
        new_summary = profiling_tool.extract_dataset_summary(
            result_df,
            max_categories=200,
            sample_distinct=200,
            compute_quantiles=False,
            strict=True,
        )
        dataset_iterations.append(new_dataset_id)
        new_summary_json = profiling_tool.dataset_summary_to_json(new_summary)
        return (
            {
                "status": "dataset_updated",
                "instruction": instructions,
                "new_dataset_id": str(new_dataset_id),
                "result": _dataframe_preview(result_df),
            },
            [],
            dataset_iterations,
            result_df,
            new_summary,
            new_summary_json,
            new_summary,
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

    # -------------------------------------------------------------------------
    # Chart intent
    # -------------------------------------------------------------------------

    def _run_chart_intent(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataframe: pd.DataFrame,
        summary_model: DatasetSummaryModel,
        instructions: str,
    ) -> tuple[JSONDict, list[ArtifactRef]]:
        specs = self._plot_tool.generate_specs(
            dataframe=dataframe,
            data_summary=summary_model,
            user_intent=instructions,
        )
        saved_ids: list[ArtifactRef] = []
        for spec in specs:
            saved_id = uuid.uuid4()
            self._data_repo.save_json_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=saved_id,
                json_data=json.dumps(spec, ensure_ascii=False),
                overwrite=True,
            )
            saved_ids.append(
                _build_data_artifact_ref(
                    artifact_id=saved_id,
                    artifact_format="json",
                    artifact_kind=_ARTIFACT_KIND_CHART_SPEC,
                )
            )
        return (
            {
                "status": "charts_saved",
                "instruction": instructions,
                "saved_chart_spec_ids": [
                    str(ref["id"]) for ref in saved_ids if "id" in ref
                ],
                "saved_chart_count": len(saved_ids),
            },
            saved_ids,
        )


# -------------------------------------------------------------------------
# Module-level helpers
# -------------------------------------------------------------------------

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


def _latest_assistant_message(messages_history: Sequence[ChatMessage] | None) -> str | None:
    if not messages_history:
        return None
    for message in reversed(messages_history):
        if message.role != "assistant":
            continue
        content = message.content.strip()
        if content:
            return content
    return None


def _is_revert_request(messages_history: Sequence[ChatMessage] | None) -> bool:
    if not messages_history:
        return False
    last = messages_history[-1]
    if last.role != "user":
        return False
    return prev_state_revert_message == last.content.lower().strip()


def _is_dashboard_scope_message(value: str | None) -> bool:
    if not value:
        return False
    normalized = " ".join(value.strip().casefold().split())
    return "dashboard only works on the current dataset" in normalized or (
        "i can only help with data exploration" in normalized
    )


def _dataframe_preview(dataframe: pd.DataFrame, *, row_limit: int = 10) -> JSONDict:
    preview_df = dataframe.head(row_limit).copy()
    for col in preview_df.columns:
        if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
            preview_df[col] = preview_df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    preview_df = preview_df.where(pd.notnull(preview_df), None)
    return {
        "row_count": int(len(dataframe)),
        "columns": [str(c) for c in dataframe.columns],
        "preview_rows": preview_df.to_dict(orient="records"),
    }


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


__all__ = ["DashboardIntentModel", "DataDashboardNode"]

