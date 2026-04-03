from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.implementation.workflows.nodes.dataset.dataset_node import (
    DatasetIntentModel,
    DatasetNode,
)
from python.implementation.workflows.nodes.dataset.dataset_prompts import (
    prev_state_revert_message,
)
from python.implementation.workflows.nodes.dataset.dataset_state import (
    DatasetIterationModel,
    DatasetPayloadModel,
    DatasetState,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)


@dataclass
class _FakeDataRepo:
    dataframe: pd.DataFrame | None = None
    get_csv_error: Exception | None = None
    saved_csv_calls: list[dict[str, object]] = field(default_factory=list)
    saved_json_calls: list[dict[str, object]] = field(default_factory=list)

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        del user_id, conversation_id, dataset_id, limit
        if self.get_csv_error is not None:
            raise self.get_csv_error
        assert self.dataframe is not None
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
        self.saved_csv_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "dataset_id": dataset_id,
                "df": df.copy(),
                "overwrite": overwrite,
                "include_index": include_index,
            }
        )

    def save_json_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        json_data: str,
        *,
        overwrite: bool = True,
    ) -> None:
        self.saved_json_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "dataset_id": dataset_id,
                "json_data": json_data,
                "overwrite": overwrite,
            }
        )


@dataclass
class _FakeLLM:
    json_outputs: list[object] = field(default_factory=list)
    generate_outputs: list[object] = field(default_factory=list)
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
        if not self.generate_outputs:
            raise AssertionError("unexpected generate call")
        next_output = self.generate_outputs.pop(0)
        if isinstance(next_output, Exception):
            raise next_output
        if isinstance(next_output, LLMResponse):
            return next_output
        return LLMResponse(content=str(next_output))

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
        if not self.json_outputs:
            raise AssertionError("unexpected generate_json call")
        next_output = self.json_outputs.pop(0)
        if isinstance(next_output, Exception):
            raise next_output
        assert isinstance(next_output, DatasetIntentModel)
        return next_output


@dataclass
class _FakeDataManipulationTool:
    result_dataframe: pd.DataFrame
    calls: list[dict[str, object]] = field(default_factory=list)

    def manipulate(
        self,
        *,
        dataframe: pd.DataFrame,
        conversation_id: str,
        data_summary: str,
        instructions: str | None = None,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "dataframe": dataframe.copy(),
                "conversation_id": conversation_id,
                "data_summary": data_summary,
                "instructions": instructions,
            }
        )
        return self.result_dataframe.copy()


@dataclass
class _FakePlotTool:
    specs: list[dict[str, object]]
    calls: list[dict[str, object]] = field(default_factory=list)

    def generate_specs(
        self,
        *,
        dataframe: pd.DataFrame,
        data_summary: str,
        user_intent: str,
    ) -> list[dict[str, object]]:
        self.calls.append(
            {
                "dataframe": dataframe.copy(),
                "data_summary": data_summary,
                "user_intent": user_intent,
            }
        )
        return [dict(spec) for spec in self.specs]


@dataclass
class _FakeToolFactory:
    profiling_tool: object
    data_manipulation_tool: object
    plot_tool: object
    calls: list[str] = field(default_factory=list)

    def get_tool(self, name: str) -> object:
        self.calls.append(name)
        if name == "DATA_PROFILING":
            return self.profiling_tool
        if name == "DATA_MANIPULATION":
            return self.data_manipulation_tool
        if name == "PLOT_TOOL":
            return self.plot_tool
        raise KeyError(name)

    def get_tool_names(self) -> list[str]:
        return ["DATA_PROFILING", "DATA_MANIPULATION", "PLOT_TOOL"]

    def get_tool_info(self, name: str) -> str:
        del name
        return "info"

    def get_tools_info(self) -> dict[str, str]:
        return {name: "info" for name in self.get_tool_names()}

    def has_tool(self, name: str) -> bool:
        return name in set(self.get_tool_names())


def _make_node_and_tools(
    *,
    dataframe: pd.DataFrame | None = None,
    get_csv_error: Exception | None = None,
    json_outputs: list[object] | None = None,
    generate_outputs: list[object] | None = None,
    manipulation_df: pd.DataFrame | None = None,
    plot_specs: list[dict[str, object]] | None = None,
) -> tuple[DatasetNode, _FakeDataRepo, _FakeLLM, _FakeDataManipulationTool, _FakePlotTool, _FakeToolFactory]:
    data_repo = _FakeDataRepo(dataframe=dataframe, get_csv_error=get_csv_error)
    llm = _FakeLLM(
        json_outputs=list(json_outputs or []),
        generate_outputs=list(generate_outputs or []),
    )
    manipulation_tool = _FakeDataManipulationTool(
        result_dataframe=(manipulation_df.copy() if manipulation_df is not None else pd.DataFrame([{"x": 1}]))
    )
    plot_tool = _FakePlotTool(specs=list(plot_specs or []))
    tool_factory = _FakeToolFactory(
        profiling_tool=DatasetProfilingTool(),
        data_manipulation_tool=manipulation_tool,
        plot_tool=plot_tool,
    )
    node = DatasetNode(data_repo=data_repo, llm=llm, tools_factory=tool_factory)
    return node, data_repo, llm, manipulation_tool, plot_tool, tool_factory


def _base_dataset_state(*, iterations: list[DatasetIterationModel] | None = None) -> DatasetState:
    return DatasetState(DatasetPayloadModel(dataset_iterations=list(iterations or [])))


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


@pytest.mark.parametrize(
    ("payload", "error_pattern"),
    [
        (
            {
                "intent_data_question": True,
                "intent_data_question_brief": "",
                "intent_manupulation_question": False,
                "intent_manupulation_question_brief": "",
                "intent_manupulation_is_analytical_query": False,
                "intent_chart": False,
                "intent_chart_brief": "",
            },
            r"intent_data_question_brief is required",
        ),
        (
            {
                "intent_data_question": False,
                "intent_data_question_brief": "",
                "intent_manupulation_question": False,
                "intent_manupulation_question_brief": "",
                "intent_manupulation_is_analytical_query": True,
                "intent_chart": False,
                "intent_chart_brief": "",
            },
            r"intent_manupulation_is_analytical_query requires intent_manupulation_question",
        ),
        (
            {
                "intent_data_question": False,
                "intent_data_question_brief": "",
                "intent_manupulation_question": False,
                "intent_manupulation_question_brief": "",
                "intent_manupulation_is_analytical_query": False,
                "intent_chart": True,
                "intent_chart_brief": "",
            },
            r"intent_chart_brief is required",
        ),
    ],
)
def test_dataset_intent_model_validates_required_briefs(
    payload: dict[str, Any],
    error_pattern: str,
) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        DatasetIntentModel.model_validate(payload)


def test_dataset_node_constructor_resolves_required_tools() -> None:
    node, _, _, _, _, tool_factory = _make_node_and_tools(
        dataframe=pd.DataFrame([{"age": 65, "outcome": 1}]),
    )

    assert isinstance(node, DatasetNode)
    assert tool_factory.calls == ["DATA_MANIPULATION", "PLOT_TOOL", "DATA_PROFILING"]


def test_dataset_node_returns_missing_data_message_when_dataset_is_unavailable() -> None:
    node, _, llm, _, _, _ = _make_node_and_tools(
        get_csv_error=FileNotFoundError("missing"),
        generate_outputs=["Please upload a CSV first."],
    )

    state = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={},
        messages_history=[ChatMessage(role="user", content="what can you do?")],
        state=DatasetState.init_empty(),
    )

    assert state.message.action == "NEEDS_DATA"
    assert state.message.txt_message == "Please upload a CSV first."
    assert len(llm.generate_calls) == 1


def test_dataset_node_returns_ready_message_when_dataset_loaded_and_no_user_message() -> None:
    node, _, _, _, _, _ = _make_node_and_tools(
        dataframe=pd.DataFrame([{"age": 65, "outcome": 1}]),
    )

    state = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={},
        messages_history=None,
        state=DatasetState.init_empty(),
    )

    assert state.message.action == "NONE"
    assert "ask about the data" in state.message.txt_message.lower()
    assert state.latest_iteration is not None
    assert state.latest_iteration.dataset_id == DatasetState.INIT_DATA_ID


def test_dataset_node_returns_off_topic_message_for_model_training_request() -> None:
    node, _, llm, _, _, _ = _make_node_and_tools(
        dataframe=pd.DataFrame([{"age": 65, "outcome": 1}, {"age": 70, "outcome": 0}]),
        json_outputs=[
            DatasetIntentModel(
                intent_data_question=False,
                intent_data_question_brief="",
                intent_manupulation_question=False,
                intent_manupulation_question_brief="",
                intent_manupulation_is_analytical_query=False,
                intent_chart=False,
                intent_chart_brief="",
            )
        ],
    )

    state = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={},
        messages_history=[ChatMessage(role="user", content="train a model on this dataset")],
        state=DatasetState.init_empty(),
    )

    assert state.status == "PENDING"
    assert state.message.action == "NONE"
    assert "dataset stage" in state.message.txt_message.lower()
    assert "chart generation" in state.message.txt_message.lower()
    assert len(llm.generate_json_calls) == 1


def test_dataset_node_reverts_to_previous_dataset_iteration() -> None:
    first_id = uuid4()
    second_id = uuid4()
    node, _, _, _, _, _ = _make_node_and_tools(
        dataframe=pd.DataFrame([{"x": 1}]),
    )
    state = _base_dataset_state(
        iterations=[
            DatasetIterationModel(dataset_id=first_id),
            DatasetIterationModel(dataset_id=second_id),
        ]
    )

    reverted_state = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={},
        messages_history=[ChatMessage(role="user", content=prev_state_revert_message)],
        state=state,
    )

    assert reverted_state.latest_iteration is not None
    assert reverted_state.latest_iteration.dataset_id == first_id
    assert "reverted" in reverted_state.message.txt_message.lower()


def test_dataset_node_answers_summary_question_and_uses_final_llm_message() -> None:
    node, data_repo, llm, manipulation_tool, plot_tool, _ = _make_node_and_tools(
        dataframe=pd.DataFrame([{"age": 65, "outcome": 1}, {"age": 70, "outcome": 0}]),
        json_outputs=[
            DatasetIntentModel(
                intent_data_question=True,
                intent_data_question_brief="summarize the outcome column",
                intent_manupulation_question=False,
                intent_manupulation_question_brief="",
                intent_manupulation_is_analytical_query=False,
                intent_chart=False,
                intent_chart_brief="",
            )
        ],
        generate_outputs=["Outcome looks balanced.", "Final dataset response."],
    )

    state = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={},
        messages_history=[ChatMessage(role="user", content="summarize outcome")],
        state=DatasetState.init_empty(),
    )

    assert state.message.txt_message == "Final dataset response."
    assert len(llm.generate_calls) == 2
    assert manipulation_tool.calls == []
    assert plot_tool.calls == []
    assert data_repo.saved_csv_calls == []
    assert data_repo.saved_json_calls == []


def test_dataset_node_runs_analytical_manipulation_without_saving_new_dataset() -> None:
    node, data_repo, llm, manipulation_tool, _, _ = _make_node_and_tools(
        dataframe=pd.DataFrame([{"age": 65, "outcome": 1}, {"age": 70, "outcome": 0}]),
        manipulation_df=pd.DataFrame([{"outcome": 1, "count": 1}, {"outcome": 0, "count": 1}]),
        json_outputs=[
            DatasetIntentModel(
                intent_data_question=False,
                intent_data_question_brief="",
                intent_manupulation_question=True,
                intent_manupulation_question_brief="count outcome values",
                intent_manupulation_is_analytical_query=True,
                intent_chart=False,
                intent_chart_brief="",
            )
        ],
        generate_outputs=["Query complete."],
    )

    state = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={},
        messages_history=[ChatMessage(role="user", content="count outcome values")],
        state=DatasetState.init_empty(),
    )

    assert state.message.txt_message == "Query complete."
    assert len(manipulation_tool.calls) == 1
    assert data_repo.saved_csv_calls == []
    assert state.latest_iteration is not None
    assert state.latest_iteration.dataset_id == DatasetState.INIT_DATA_ID


def test_dataset_node_saves_new_dataset_for_mutating_manipulation() -> None:
    node, data_repo, _, manipulation_tool, _, _ = _make_node_and_tools(
        dataframe=pd.DataFrame([{"age": 65, "outcome": 1}, {"age": 70, "outcome": 0}]),
        manipulation_df=pd.DataFrame([{"age": 65}, {"age": 70}]),
        json_outputs=[
            DatasetIntentModel(
                intent_data_question=False,
                intent_data_question_brief="",
                intent_manupulation_question=True,
                intent_manupulation_question_brief="drop outcome column",
                intent_manupulation_is_analytical_query=False,
                intent_chart=False,
                intent_chart_brief="",
            )
        ],
        generate_outputs=["Updated dataset saved."],
    )

    state = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={},
        messages_history=[ChatMessage(role="user", content="drop outcome column")],
        state=DatasetState.init_empty(),
    )

    assert len(manipulation_tool.calls) == 1
    assert len(data_repo.saved_csv_calls) == 1
    assert data_repo.saved_csv_calls[0]["include_index"] is False
    assert state.latest_iteration is not None
    assert state.latest_iteration.dataset_id != DatasetState.INIT_DATA_ID
    assert list(data_repo.saved_csv_calls[0]["df"].columns) == ["age"]


def test_dataset_node_saves_chart_specs_and_adds_artifact_ids() -> None:
    node, data_repo, _, _, plot_tool, _ = _make_node_and_tools(
        dataframe=pd.DataFrame([{"age": 65, "outcome": 1}, {"age": 70, "outcome": 0}]),
        plot_specs=[
            {"mark": "bar", "data": {"values": [{"age": 65}]}, "encoding": {"x": {"field": "age"}}},
            {"mark": "line", "data": {"values": [{"outcome": 1}]}, "encoding": {"y": {"field": "outcome"}}},
        ],
        json_outputs=[
            DatasetIntentModel(
                intent_data_question=False,
                intent_data_question_brief="",
                intent_manupulation_question=False,
                intent_manupulation_question_brief="",
                intent_manupulation_is_analytical_query=False,
                intent_chart=True,
                intent_chart_brief="plot age and outcome",
            )
        ],
        generate_outputs=["Charts saved."],
    )

    state = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={},
        messages_history=[ChatMessage(role="user", content="plot age and outcome")],
        state=DatasetState.init_empty(),
    )

    assert len(plot_tool.calls) == 1
    assert len(data_repo.saved_json_calls) == 2
    assert state.latest_iteration is not None
    assert len(state.latest_iteration.saved_vega_lite_specs_file_ids) == 2
    assert state.message.artifact_ids is not None
    assert len(state.message.artifact_ids) == 3


def test_dataset_node_returns_classification_failure_message_when_intent_call_raises() -> None:
    node, _, _, _, _, _ = _make_node_and_tools(
        dataframe=pd.DataFrame([{"age": 65, "outcome": 1}]),
        json_outputs=[RuntimeError("classification failed")],
    )

    state = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={},
        messages_history=[ChatMessage(role="user", content="do something")],
        state=DatasetState.init_empty(),
    )

    assert "could not classify" in state.message.txt_message.lower()
