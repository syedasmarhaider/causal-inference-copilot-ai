from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pandas as pd

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.domain.workflows.tool import Tool
from python.implementation.workflows.nodes.dataset.dataset_node import (
    DatasetIntentModel,
    DatasetNode,
)
from python.implementation.workflows.nodes.dataset.dataset_state import DatasetState
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)


@dataclass
class _FakeDataRepo:
    dataframe: pd.DataFrame

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        del user_id, conversation_id, dataset_id, limit
        return self.dataframe.copy()

    def save_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        df: pd.DataFrame,
        *,
        overwrite: bool = True,
        include_index: bool = False,
    ) -> None:
        del user_id, conversation_id, dataset_id, df, overwrite, include_index
        raise NotImplementedError

    def get_json_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
    ) -> str:
        del user_id, conversation_id, dataset_id
        raise NotImplementedError

    def save_json_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        json_data: str,
        *,
        overwrite: bool = True,
    ) -> None:
        del user_id, conversation_id, dataset_id, json_data, overwrite
        raise NotImplementedError


@dataclass
class _FakeLLM:
    intent: DatasetIntentModel
    generate_calls: list[dict[str, object]] = field(default_factory=list)
    generate_json_calls: list[dict[str, object]] = field(default_factory=list)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
    ) -> LLMResponse:
        self.generate_calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "config": config,
                "history": history,
            }
        )
        raise AssertionError("generate should not be called in off-topic dataset test")

    def generate_json(
        self,
        *,
        schema: type[DatasetIntentModel],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> DatasetIntentModel:
        self.generate_json_calls.append(
            {
                "schema": schema,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "config": config,
                "history": history,
                "max_attempts": max_attempts,
            }
        )
        return self.intent


@dataclass
class _FakeToolFactory:
    profiling_tool: DatasetProfilingTool

    def get_tool_names(self) -> list[str]:
        return [self.profiling_tool.get_tool_name()]

    def get_tool_info(self, name: str) -> str:
        if name != self.profiling_tool.get_tool_name():
            raise KeyError(name)
        return self.profiling_tool.get_tool_info()

    def get_tools_info(self) -> dict[str, str]:
        return {self.profiling_tool.get_tool_name(): self.profiling_tool.get_tool_info()}

    def has_tool(self, name: str) -> bool:
        return name == self.profiling_tool.get_tool_name()

    def get_tool(self, name: str) -> Tool:
        if name != self.profiling_tool.get_tool_name():
            raise KeyError(name)
        return self.profiling_tool


@dataclass
class _UnusedDataManipulationTool:
    def manipulate(
        self,
        *,
        dataframe: pd.DataFrame,
        conversation_id: str,
        data_summary: str,
        instructions: str | None = None,
    ) -> pd.DataFrame:
        del conversation_id, data_summary, instructions
        return dataframe.copy()


@dataclass
class _UnusedPlotTool:
    def generate_specs(
        self,
        *,
        dataframe: pd.DataFrame,
        data_summary: str,
        user_intent: str,
    ) -> list[dict[str, object]]:
        del dataframe, data_summary, user_intent
        return []


def test_dataset_intent_model_allows_all_false_with_empty_briefs() -> None:
    intent = DatasetIntentModel(
        intent_data_question=False,
        intent_data_question_brief="",
        intent_manupulation_question=False,
        intent_manupulation_question_brief="",
        intent_manupulation_is_analytical_query=False,
        intent_chart=False,
        intent_chart_brief="",
    )

    assert intent.has_any_intent() is False


def test_dataset_node_returns_off_topic_message_for_model_training_request() -> None:
    node = DatasetNode(
        data_repo=_FakeDataRepo(
            dataframe=pd.DataFrame(
                [
                    {"age": 65, "outcome": 1},
                    {"age": 70, "outcome": 0},
                ]
            )
        ),
        llm=_FakeLLM(
            intent=DatasetIntentModel(
                intent_data_question=False,
                intent_data_question_brief="",
                intent_manupulation_question=False,
                intent_manupulation_question_brief="",
                intent_manupulation_is_analytical_query=False,
                intent_chart=False,
                intent_chart_brief="",
            )
        ),
        data_manipulation_tool=_UnusedDataManipulationTool(),
        plot_tool=_UnusedPlotTool(),
    )

    state = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        tool_factory=_FakeToolFactory(profiling_tool=DatasetProfilingTool()),
        previous_state_dependencies={},
        messages_history=[ChatMessage(role="user", content="train a model on this dataset")],
        state=DatasetState.init_empty(),
    )

    assert isinstance(state, DatasetState)
    assert state.status == "PENDING"
    assert state.message.action == "NONE"
    assert state.message.txt_message.lower().find("dataset stage") >= 0
    assert state.message.txt_message.lower().find("chart generation") >= 0
