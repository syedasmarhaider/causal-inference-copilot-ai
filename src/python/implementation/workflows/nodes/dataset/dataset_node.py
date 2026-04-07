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
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.dataset.dataset_prompts import (
    dataset_final_response_system_prompt,
    dataset_intent_classification_system_prompt,
    dataset_missing_data_system_prompt,
    dataset_node_info,
    dataset_protocol_cleaning_instructions_system_prompt,
    dataset_summary_answer_system_prompt,
    prev_state_revert_message,
)
from python.implementation.workflows.nodes.dataset.dataset_state import (
    DatasetIterationModel,
    DatasetPayloadModel,
    DatasetState,
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

log = get_app_logger(__name__, component="dataset_node", log_type="node")

_DATA_MANIPULATION_RETRY_ATTEMPTS = 3
_WORKING_TABLE_PREFIX = "df_"
_WORKING_TABLE_HASH_HEX_LEN = 16
_ARTIFACT_KIND_WORKING_DATASET = "working_dataset"
_ARTIFACT_KIND_ANALYTICAL_RESULT = "analytical_result"
_ARTIFACT_KIND_CHART_SPEC = "chart_spec"
_INITIAL_SUMMARY_MAX_COLUMNS = 8
_READY_DATASET_MESSAGE = (
    "Dataset is ready. Ask about the data, request analytical or statistical queries, ask "
    "for transformations, or ask for charts."
)
_OFF_TOPIC_MESSAGE = (
    "That request is outside the dataset stage. Here I can inspect the current data, answer "
    "dataset questions, run analytical queries, clean or reshape the dataset, and generate "
    "charts. I cannot do model training or causal estimation from here. If you want to move "
    "forward with causal analysis, tell me the treatment, outcome, study type, and time zero "
    "so I can switch to protocol discussion."
)
_OFF_TOPIC_CLARIFICATION_MESSAGE = (
    "I mean this step only works on the current dataset itself. I can inspect columns, answer "
    "dataset questions, run read-only analytical queries, clean or reshape the data, and "
    "generate charts. I cannot train a model or run causal inference from here. If you want "
    "to move forward with causal analysis, tell me the treatment, outcome, study type, and "
    "time zero so I can switch to protocol discussion."
)


class DatasetIntentModel(BaseModel):
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
    def _validate(self) -> DatasetIntentModel:
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


class DatasetNode(Node):
    NAME: ClassVar[str] = DatasetState.NAME
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
        return dataset_node_info()

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        readonly_orchestrator_state: ReadOnlyOchestratorState,
        messages_history: Sequence[ChatMessage] | None,
    ) -> State:
        if not isinstance(state, DatasetState):
            raise TypeError(f"{self.name}: expected DatasetState, got {type(state).__name__}")

        dataset_iterations = [
            item.model_copy(deep=True) for item in state.payload.dataset_iterations
        ]
        latest_summary = (
            state.payload.latest_summary.model_copy(deep=True)
            if state.payload.latest_summary is not None
            else None
        )

        if _is_revert_request(messages_history):
            return self._handle_revert(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_iterations=dataset_iterations,
                current_latest_summary=latest_summary,
            )

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
                    dataset_id=latest_iteration.dataset_id,
                    limit=1_000_000,
                )
            except Exception as exc:
                log.exception(
                    "failed to load latest dataset iteration",
                    dataset_id=str(latest_iteration.dataset_id),
                    error=safe_err(exc),
                )
                return self._build_state(
                    dataset_iterations=dataset_iterations,
                    latest_summary=latest_summary,
                    user_message=(
                        "I could not load the current working dataset. Please re-upload the CSV "
                        "or try again."
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
                    dataset_id=DatasetState.INIT_DATA_ID,
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
            dataset_iterations.append(
                DatasetIterationModel(
                    dataset_id=DatasetState.INIT_DATA_ID,
                )
            )
            latest_summary = current_summary
            loaded_this_turn = True

        persisted_latest_summary = current_summary

        protocol_discussion = _get_protocol_discussion(readonly_orchestrator_state)
        if protocol_discussion is not None:
            try:
                cleaning_instructions = self._build_protocol_cleaning_instructions(
                    protocol_discussion=protocol_discussion,
                    dataset_summary=current_summary_json,
                    recent_chat_history=_last_n_messages_text(messages_history, limit=3),
                )
                working_df = self._run_data_manipulation_tool(
                    dataframe=current_df,
                    summary_json=current_summary_json,
                    conversation_id=conversation_id,
                    instructions=cleaning_instructions,
                )

                new_dataset_id = uuid.uuid4()
                self._data_repo.save_csv_data(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataset_id=new_dataset_id,
                    df=working_df,
                    overwrite=True,
                    include_index=False,
                )
                dataset_iterations.append(DatasetIterationModel(dataset_id=new_dataset_id))
                persisted_latest_summary = self._profiling_tool.extract_dataset_summary(
                    working_df,
                    max_categories=200,
                    sample_distinct=200,
                    compute_quantiles=False,
                    strict=True,
                )
            except Exception as exc:
                log.exception(
                    "failed to apply protocol-driven dataset cleaning",
                    error=safe_err(exc),
                )
                return self._build_state(
                    dataset_iterations=dataset_iterations,
                    latest_summary=persisted_latest_summary,
                    user_message=(
                        "I could not apply protocol-driven cleaning to the current dataset. "
                        "Please review the confirmed protocol discussion and try again."
                    ),
                )
            return self._build_state(
                dataset_iterations=dataset_iterations,
                latest_summary=persisted_latest_summary,
                user_message=self._build_protocol_cleaning_completed_message(
                    rows=int(len(working_df)),
                    columns=[str(column) for column in working_df.columns],
                ),
            )

        latest_user_message = _latest_user_message(messages_history)
        if not latest_user_message:
            return self._build_state(
                dataset_iterations=dataset_iterations,
                latest_summary=persisted_latest_summary,
                user_message=(
                    self._build_loaded_dataset_message(summary=current_summary)
                    if loaded_this_turn
                    else self._build_ready_message()
                ),
            )

        last_4_messages_history = (
            messages_history[-5:-1] if messages_history and len(messages_history) > 1 else None
        )
        last_4_messages_history_text: str | None = None
        if last_4_messages_history:
            last_4_messages_history_text = get_chat_messages_role_and_message_json(
                last_4_messages_history
            )

        try:
            intent = self._classify_intent(
                latest_user_message=latest_user_message,
                chat_history=last_4_messages_history_text,
                dataset_summary=current_summary_json,
            )
        except Exception as exc:
            log.exception("failed to classify dataset intent", error=safe_err(exc))
            return self._build_state(
                dataset_iterations=dataset_iterations,
                latest_summary=persisted_latest_summary,
                user_message=(
                    "Dataset is loaded, but I could not classify your request. Please ask again "
                    "more directly."
                ),
            )

        if not intent.has_any_intent():
            if loaded_this_turn:
                return self._build_state(
                    dataset_iterations=dataset_iterations,
                    latest_summary=persisted_latest_summary,
                    user_message=self._build_loaded_dataset_message(summary=current_summary),
                )
            if _is_dataset_scope_message(_latest_assistant_message(messages_history)):
                return self._build_state(
                    dataset_iterations=dataset_iterations,
                    latest_summary=persisted_latest_summary,
                    user_message=self._build_off_topic_clarification_message(),
                )
            return self._build_state(
                dataset_iterations=dataset_iterations,
                latest_summary=persisted_latest_summary,
                user_message=self._build_off_topic_message(),
            )

        summary_answer: str | None = None
        manipulation_result: JSONDict | None = None
        chart_result: JSONDict | None = None
        manipulation_artifact_refs: list[ArtifactRef] = []
        chart_artifact_refs: list[ArtifactRef] = []

        if intent.intent_data_question:
            summary_answer = self._answer_summary_question(
                intent_brief=intent.intent_data_question_brief or latest_user_message,
                dataset_summary=current_summary_json,
                chat_history=last_4_messages_history_text,
            )

        working_df = current_df
        working_summary = current_summary
        working_summary_json = current_summary_json

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
                log.exception("failed to run dataset manipulation intent", error=safe_err(exc))
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
                log.exception("failed to generate dataset charts", error=safe_err(exc))
                return self._build_state(
                    dataset_iterations=dataset_iterations,
                    latest_summary=persisted_latest_summary,
                    user_message=(
                        "I could not generate the requested chart output. "
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
                    "active_dataset_columns": [str(column) for column in working_df.columns],
                },
            )
        except Exception as exc:
            log.exception("failed to build final dataset response", error=safe_err(exc))
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

    def _build_state(
        self,
        *,
        user_message: str,
        dataset_iterations: Sequence[DatasetIterationModel] | None = None,
        latest_summary: DatasetSummaryModel | None = None,
        response_message_artifact_refs: Sequence[ArtifactRef] | None = None,
    ) -> DatasetState:
        normalized_iterations = [
            iteration.model_copy(deep=True) for iteration in (dataset_iterations or [])
        ]
        state = DatasetState(
            DatasetPayloadModel(
                dataset_iterations=normalized_iterations,
                latest_summary=(
                    latest_summary.model_copy(deep=True) if latest_summary is not None else None
                ),
                user_message=user_message,
            ),
            response_message_artifact_refs=list(response_message_artifact_refs or []),
        )
        return state

    def _handle_revert(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataset_iterations: list[DatasetIterationModel],
        current_latest_summary: DatasetSummaryModel | None,
    ) -> DatasetState:
        if not dataset_iterations:
            return self._build_state(
                user_message="There is no working dataset to revert. Please upload a CSV dataset.",
            )
        if len(dataset_iterations) == 1:
            return self._build_state(
                dataset_iterations=dataset_iterations,
                latest_summary=current_latest_summary,
                user_message="There is no previous dataset version to revert to.",
            )
        reverted_iterations = dataset_iterations[:-1]
        reverted_summary = self._load_latest_summary(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=reverted_iterations[-1].dataset_id,
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
            dataframe = self._data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=dataset_id,
                limit=1_000_000,
            )
        except Exception as exc:
            log.exception(
                "failed to load dataset while refreshing latest summary",
                dataset_id=str(dataset_id),
                error=safe_err(exc),
            )
            return None
        return self._profiling_tool.extract_dataset_summary(
            dataframe,
            max_categories=200,
            sample_distinct=200,
            compute_quantiles=False,
            strict=True,
        )

    def _build_ready_message(self) -> str:
        return _READY_DATASET_MESSAGE

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

    def _build_off_topic_message(self) -> str:
        return _OFF_TOPIC_MESSAGE

    def _build_off_topic_clarification_message(self) -> str:
        return _OFF_TOPIC_CLARIFICATION_MESSAGE

    def _build_loaded_dataset_message(
        self,
        *,
        summary: DatasetSummaryModel,
    ) -> str:
        profiles = list(summary.profiles)
        column_count = len(profiles)
        shown_profiles = profiles[:_INITIAL_SUMMARY_MAX_COLUMNS]
        column_preview = ", ".join(
            f"{profile.name} ({profile.inferred_kind.lower()})" for profile in shown_profiles
        )
        if column_count > _INITIAL_SUMMARY_MAX_COLUMNS:
            column_preview = (
                f"{column_preview}, +{column_count - _INITIAL_SUMMARY_MAX_COLUMNS} more columns"
            )
        return (
            f"Dataset loaded successfully. I profiled {summary.n_rows} rows and {column_count} "
            f"columns. Columns: {column_preview}. I can now inspect the data, answer dataset "
            "questions, reshape it, and generate charts. If you want to start causal analysis "
            "next, tell me the treatment, outcome, study type, and time zero."
        )

    def _build_protocol_cleaning_completed_message(
        self,
        *,
        rows: int,
        columns: Sequence[str],
    ) -> str:
        column_count = len(columns)
        shown_columns = list(columns[:_INITIAL_SUMMARY_MAX_COLUMNS])
        column_preview = ", ".join(shown_columns)
        if column_count > _INITIAL_SUMMARY_MAX_COLUMNS:
            column_preview = (
                f"{column_preview}, +{column_count - _INITIAL_SUMMARY_MAX_COLUMNS} more columns"
            )
        return (
            "Applied the confirmed protocol cleaning request and saved a working dataset "
            f"for validation. The cleaned dataset now has {rows} rows and {column_count} columns"
            + (f": {column_preview}." if column_preview else ".")
        )

    def _build_protocol_cleaning_instructions(
        self,
        *,
        protocol_discussion: str,
        dataset_summary: str,
        recent_chat_history: str | None,
    ) -> str:
        response = self._llm.generate(
            system_prompt=dataset_protocol_cleaning_instructions_system_prompt(),
            user_prompt=json.dumps(
                {
                    "protocol_discussion": protocol_discussion,
                    "dataset_summary": dataset_summary,
                    "recent_chat_history": recent_chat_history,
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="basic", temperature=0.0, top_p=1.0),
            history=None,
        )
        instructions = response.content.strip()
        if not instructions:
            raise ValueError("Protocol cleaning instructions cannot be empty")
        return instructions

    def _classify_intent(
        self,
        *,
        latest_user_message: str,
        chat_history: str | None,
        dataset_summary: str,
    ) -> DatasetIntentModel:
        payload: JSONDict = {
            "latest_user_message": latest_user_message,
            "chat_history": chat_history,
            "dataset_summary": dataset_summary,
        }
        return self._llm.generate_json(
            schema=DatasetIntentModel,
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
            log.exception("failed to answer summary question", error=safe_err(exc))
            return "I could not answer that from the dataset summary alone."

        answer = response.content.strip() if response.content else ""
        return answer or "I could not answer that from the dataset summary alone."

    def _run_manipulation_intent(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataset_iterations: list[DatasetIterationModel],
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
        list[DatasetIterationModel],
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
                next_dataframe = result_df
                next_summary = analytical_summary
                next_summary_json = analytical_summary_json
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
                dataset_iterations,
                next_dataframe,
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
        dataset_iterations.append(
            DatasetIterationModel(
                dataset_id=new_dataset_id,
            )
        )
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
                    str(artifact_id)
                    for saved_id in saved_ids
                    if (artifact_id := saved_id.get("id")) is not None
                ],
                "saved_chart_count": len(saved_ids),
            },
            saved_ids,
        )

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
            if count > 0:
                noun = "chart" if count == 1 else "charts"
                parts.append(f"Generated {count} {noun}.")
            else:
                parts.append("Generated the requested chart output.")

        if not parts:
            return "Completed the dataset request."

        return " ".join(parts)


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


def _get_protocol_discussion(
    readonly_orchestrator_state: ReadOnlyOchestratorState,
) -> str | None:
    protocol_discussion = readonly_orchestrator_state.get("protocol_discussion")
    if not isinstance(protocol_discussion, str):
        return None
    normalized_protocol_discussion = protocol_discussion.strip()
    if not normalized_protocol_discussion:
        return None
    return normalized_protocol_discussion


def _last_n_messages_text(
    messages_history: Sequence[ChatMessage] | None,
    *,
    limit: int,
) -> str | None:
    if not messages_history or limit <= 0:
        return None

    recent_messages = list(messages_history[-limit:])
    if not recent_messages:
        return None

    return get_chat_messages_role_and_message_json(recent_messages)


def _normalize_message_text(value: str) -> str:
    return " ".join(value.strip().casefold().split()).strip("?.! ")


def _is_dataset_scope_message(value: str | None) -> bool:
    if not value:
        return False
    normalized = _normalize_message_text(value)
    return "dataset stage" in normalized and (
        "outside the dataset stage" in normalized
        or "i cannot help with that" in normalized
        or "i mean this step only works on the current dataset itself" in normalized
    )


def _is_revert_request(messages_history: Sequence[ChatMessage] | None) -> bool:
    if not messages_history:
        return False
    last_message = messages_history[-1]
    if last_message.role != "user":
        return False
    return prev_state_revert_message == last_message.content.lower()


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


__all__ = ["DatasetIntentModel", "DatasetNode"]
