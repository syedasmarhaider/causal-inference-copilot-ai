from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pandas as pd

from python.domain.models.models import ChatMessage
from python.domain.service.llm_service import LLMConfig, LLMResponse
from python.implementation.workflows.nodes.dataset.dataset_node import DatasetNode
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
)
from python.implementation.workflows.tools.plot_tool.plot_tool import PlotTool


@dataclass
class _FakeDataRepo:
    dataframe: pd.DataFrame
    saved_csv_calls: list[dict[str, object]] = field(default_factory=list)

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
        del user_id, conversation_id, overwrite, include_index
        self.saved_csv_calls.append({"dataset_id": dataset_id, "df": df.copy()})

    def save_json_data(self, **kwargs: object) -> None:
        raise AssertionError(f"unexpected save_json_data call: {kwargs}")


@dataclass
class _FakeLLM:
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
        output = self.generate_outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        if isinstance(output, LLMResponse):
            return output
        return LLMResponse(content=str(output))

    def generate_json(self, **kwargs: object) -> object:
        self.generate_json_calls.append(dict(kwargs))
        raise AssertionError("unexpected generate_json call")


@dataclass
class _FakeDataManipulationTool:
    result_dataframe: pd.DataFrame
    calls: list[dict[str, object]] = field(default_factory=list)

    NAME = DataManipulationTool.NAME

    def manipulate(
        self,
        *,
        dataframe: pd.DataFrame,
        table_name: str,
        data_summary: str,
        instructions: str,
        retry_attempts: int | None = None,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "dataframe": dataframe.copy(),
                "table_name": table_name,
                "data_summary": data_summary,
                "instructions": instructions,
                "retry_attempts": retry_attempts,
            }
        )
        return self.result_dataframe.copy()


@dataclass
class _FakePlotTool:
    NAME = PlotTool.NAME

    def generate_specs(self, **kwargs: object) -> list[dict[str, object]]:
        raise AssertionError(f"unexpected generate_specs call: {kwargs}")


@dataclass
class _FakeToolFactory:
    profiling_tool: object
    data_manipulation_tool: object
    plot_tool: object

    def get_tool(self, name: str) -> object:
        if name == DatasetProfilingTool.NAME:
            return self.profiling_tool
        if name == DataManipulationTool.NAME:
            return self.data_manipulation_tool
        if name == PlotTool.NAME:
            return self.plot_tool
        raise KeyError(name)

    def get_tool_names(self) -> list[str]:
        return [DatasetProfilingTool.NAME, DataManipulationTool.NAME, PlotTool.NAME]

    def get_tool_info(self, name: str) -> str:
        return name

    def get_tools_info(self) -> dict[str, str]:
        return {name: name for name in self.get_tool_names()}

    def has_tool(self, name: str) -> bool:
        return name in self.get_tool_names()


@dataclass(frozen=True)
class _FakeReadonlyOrchestratorState:
    payload: dict[str, object]

    def get(self, key: str) -> object | None:
        return self.payload.get(key)


def _make_node(
    *,
    dataframe: pd.DataFrame,
    manipulation_df: pd.DataFrame,
    llm_outputs: list[object] | None = None,
) -> tuple[DatasetNode, _FakeDataRepo, _FakeLLM, _FakeDataManipulationTool]:
    data_repo = _FakeDataRepo(dataframe=dataframe)
    llm = _FakeLLM(generate_outputs=list(llm_outputs or []))
    manipulation_tool = _FakeDataManipulationTool(result_dataframe=manipulation_df)
    tool_factory = _FakeToolFactory(
        profiling_tool=DatasetProfilingTool(),
        data_manipulation_tool=manipulation_tool,
        plot_tool=_FakePlotTool(),
    )
    return (
        DatasetNode(data_repo=data_repo, llm=llm, tools_factory=tool_factory),
        data_repo,
        llm,
        manipulation_tool,
    )


def _dataset_state() -> DatasetState:
    profiling_tool = DatasetProfilingTool()
    dataframe = pd.DataFrame([{"age": 65, "outcome": 1, "extra": "x"}])
    summary = profiling_tool.extract_dataset_summary(
        dataframe,
        max_categories=200,
        sample_distinct=200,
        compute_quantiles=False,
        strict=True,
    )
    return DatasetState(
        DatasetPayloadModel(
            dataset_iterations=[DatasetIterationModel(dataset_id=DatasetState.INIT_DATA_ID)],
            latest_summary=summary,
        )
    )


def test_dataset_node_does_not_auto_clean_when_protocol_exists_but_cleaning_is_not_pending() -> (
    None
):
    node, data_repo, llm, manipulation_tool = _make_node(
        dataframe=pd.DataFrame([{"age": 65, "outcome": 1, "extra": "x"}]),
        manipulation_df=pd.DataFrame([{"age": 65, "outcome_status": 1}]),
    )

    state = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        readonly_orchestrator_state=_FakeReadonlyOrchestratorState(
            {
                "protocol_discussion": "Confirmed protocol discussion.",
                "dataset_cleaning_pending": False,
            }
        ),
        messages_history=None,
        state=_dataset_state(),
    )

    assert state.action() == "NEEDS_INPUT"
    assert "dataset is ready" in state.messages()[0].content.lower()
    assert llm.generate_calls == []
    assert llm.generate_json_calls == []
    assert manipulation_tool.calls == []
    assert data_repo.saved_csv_calls == []


def test_dataset_node_applies_cleaning_when_protocol_exists_and_cleaning_is_pending() -> None:
    node, data_repo, llm, manipulation_tool = _make_node(
        dataframe=pd.DataFrame([{"age": 65, "outcome": 1, "extra": "x"}]),
        manipulation_df=pd.DataFrame([{"age": 65, "outcome_status": 1}]),
        llm_outputs=["Map outcome to outcome_status and drop extra."],
    )

    state = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        readonly_orchestrator_state=_FakeReadonlyOrchestratorState(
            {
                "protocol_discussion": "Confirmed protocol discussion.",
                "dataset_cleaning_pending": True,
            }
        ),
        messages_history=None,
        state=_dataset_state(),
    )

    assert state.action() == "NONE"
    assert "applied the confirmed protocol cleaning request" in state.messages()[0].content.lower()
    assert len(llm.generate_calls) == 1
    assert llm.generate_json_calls == []
    assert len(manipulation_tool.calls) == 1
    assert manipulation_tool.calls[0]["instructions"] == (
        "Map outcome to outcome_status and drop extra."
    )
    assert len(data_repo.saved_csv_calls) == 1
