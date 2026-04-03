from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast
from uuid import UUID

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from python.domain.models.models import Artifact_Id
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import (
    ChatMessage,
    LLMConfig,
    LLMService,
    get_chat_messages_role_and_message_json,
)
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.dataset.dataset_prompts import (
    dataset_final_response_system_prompt,
    dataset_intent_classification_system_prompt,
    dataset_missing_data_system_prompt,
    dataset_node_info,
    dataset_summary_answer_system_prompt,
    prev_state_revert_message
)
from python.implementation.workflows.nodes.dataset.dataset_state import (
    DatasetIterationModel,
    DatasetPayloadModel,
    DatasetState,
)
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.plot_tool.plot_tool import (
    PlotTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
    DatasetSummaryModel,
)
from python.implementation.workflows.utils.utils import JSONDict, safe_err

log = get_app_logger(__name__, component="dataset_node", log_type="node")



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
        text = str(value).strip()
        return text

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
        return (
            self.intent_data_question
            or self.intent_manupulation_question
            or self.intent_chart
        )


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
        self._data_manipulation_tool = cast(DataManipulationTool, tools_factory.get_tool(DataManipulationTool.NAME))
        self._plot_tool = cast(PlotTool, tools_factory.get_tool(PlotTool.NAME))
        self._profiling_tool = cast(DatasetProfilingTool, tools_factory.get_tool(DatasetProfilingTool.NAME))

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
        previous_state_dependencies: Mapping[str, State],
        messages_history: Sequence[ChatMessage] | None,
        state: State,
    ) -> State:
        del previous_state_dependencies
        
        if not isinstance(state, DatasetState):
            raise TypeError(f"{self.name}: expected DatasetState, got {type(state).__name__}")

        dataset_iterations = [item.model_copy(deep=True) for item in state.payload.dataset_iterations]
        if _is_revert_request(messages_history):
            return self._handle_revert(dataset_iterations=dataset_iterations)


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
                return DatasetState(
                    DatasetPayloadModel(
                        dataset_iterations=dataset_iterations,
                        user_message=(
                            "I could not load the current working dataset. Please re-upload the CSV "
                            "or try again."
                        ),
                    )
                )

            current_summary = latest_iteration.summary or self._profiling_tool.extract_dataset_summary(
                current_df,
                max_categories=200,
                sample_distinct=200,
                compute_quantiles=False,
                strict=True,
            )
            dataset_iterations[-1] = latest_iteration.model_copy(update={"summary": current_summary})
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
                return DatasetState(
                    DatasetPayloadModel(
                        user_message=self._build_missing_data_message(messages_history=messages_history),
                    )
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
                    summary=current_summary,
                )
            )
            
            loaded_this_turn = True

        latest_user_message = _latest_user_message(messages_history)
        if not latest_user_message:
            return DatasetState(
                DatasetPayloadModel(
                    dataset_iterations=dataset_iterations,
                    user_message=(
                        "Dataset is ready. Ask about the data, request a transformation, or ask for charts."
                    ),
                )
            )
            
        last_4_messages_history = messages_history[-5:-1] if messages_history and len(messages_history) > 1 else None
        last_4_messages_history_text: str | None = None
        if last_4_messages_history:
             last_4_messages_history_text = get_chat_messages_role_and_message_json(last_4_messages_history)   

        try:
            intent = self._classify_intent(
                latest_user_message=latest_user_message,
                chat_history=last_4_messages_history_text,
                dataset_summary=current_summary_json,
            )
        except Exception as exc:
            log.exception("failed to classify dataset intent", error=safe_err(exc))
            return DatasetState(
                DatasetPayloadModel(
                    dataset_iterations=dataset_iterations,
                    user_message=(
                        "Dataset is loaded, but I could not classify your request. Please ask again more directly."
                    ),
                )
            )

        if not intent.has_any_intent():
            return DatasetState(
                DatasetPayloadModel(
                    dataset_iterations=dataset_iterations,
                    user_message=self._build_off_topic_message(
                        latest_user_message=latest_user_message,
                    ),
                )
            )

        summary_answer: str | None = None
        manipulation_result: JSONDict | None = None
        chart_result: JSONDict | None = None

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
            (
                manipulation_result,
                dataset_iterations,
                working_df,
                working_summary,
                working_summary_json,
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
            )

        if intent.intent_chart:
            chart_result, dataset_iterations = self._run_chart_intent(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_iterations=dataset_iterations,
                dataframe=working_df,
                summary_json=working_summary_json,
                instructions=intent.intent_chart_brief or latest_user_message,
            )

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
        return DatasetState(
            DatasetPayloadModel(
                dataset_iterations=dataset_iterations,
                user_message=final_message,
            )
        )

    def _handle_revert(self, *, dataset_iterations: list[DatasetIterationModel]) -> DatasetState:
        if not dataset_iterations:
            return DatasetState(
                DatasetPayloadModel(
                    user_message="There is no working dataset to revert. Please upload a CSV dataset.",
                )
            )
        if len(dataset_iterations) == 1:
            return DatasetState(
                DatasetPayloadModel(
                    dataset_iterations=dataset_iterations,
                    user_message="There is no previous dataset version to revert to.",
                )
            )
        return DatasetState(
            DatasetPayloadModel(
                dataset_iterations=dataset_iterations[:-1],
                user_message="Reverted to the previous working dataset version.",
            )
        )

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
            

    def _build_off_topic_message(self, *, latest_user_message: str) -> str:
        return (
            "I cannot help with that from the dataset stage. Here I only handle dataset "
            "understanding, data manipulation, and chart generation. If you want downstream "
            "workflow steps, continue through the pipeline after the data is ready."
        )
        

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
    ) -> tuple[JSONDict, list[DatasetIterationModel], pd.DataFrame, DatasetSummaryModel, str]:
        result_df = self._data_manipulation_tool.manipulate(
            dataframe=dataframe,
            conversation_id=str(conversation_id),
            data_summary=summary_json,
            instructions=instructions,
        )

        if analytical_query:
            return (
                {
                    "status": "analytical_query",
                    "instruction": instructions,
                    "result": _dataframe_preview(result_df),
                },
                dataset_iterations,
                dataframe,
                summary_model,
                summary_json,
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
                summary=new_summary,
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
            dataset_iterations,
            result_df,
            new_summary,
            new_summary_json,
        )

    def _run_chart_intent(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataset_iterations: list[DatasetIterationModel],
        dataframe: pd.DataFrame,
        summary_json: str,
        instructions: str,
    ) -> tuple[JSONDict, list[DatasetIterationModel]]:
        specs = self._plot_tool.generate_specs(
            dataframe=dataframe,
            data_summary=summary_json,
            user_intent=instructions,
        )
        saved_ids: list[Artifact_Id] = []
        for spec in specs:
            saved_id = uuid.uuid4()
            self._data_repo.save_json_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=saved_id,
                json_data=json.dumps(spec, ensure_ascii=False),
                overwrite=True,
            )
            saved_ids.append(Artifact_Id(
                id=saved_id,
                type="json"
            ))
            
        latest_iteration = dataset_iterations[-1]
        dataset_iterations[-1] = latest_iteration.model_copy(
            update={"saved_vega_lite_specs_file_ids": saved_ids}
        )
        return (
            {
                "status": "charts_saved",
                "instruction": instructions,
                "saved_chart_spec_ids": [str(saved_id) for saved_id in saved_ids],
                "saved_chart_count": len(saved_ids),
            },
            dataset_iterations,
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

__all__ = ["DatasetIntentModel", "DatasetNode"]
