from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
from pydantic import BaseModel

from python.domain.models.models import ChatMessage
from python.domain.service.llm_service import LLMConfig, LLMResponse
from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.nodes.data_statistics.data_statistics_node import (
    DataStatisticsNode,
)
from python.implementation.workflows.nodes.data_statistics.data_statistics_state import (
    DataStatisticsState,
)
from python.implementation.workflows.tools.advanced_analytics.advanced_analytics_models import (
    AnalyticsResultModel,
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


def _row_level_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"istatus": "control", "outcome": 0.0, "age": 23, "sex": "F"},
            {"istatus": "treated", "outcome": 1.0, "age": 25, "sex": "M"},
            {"istatus": "control", "outcome": 0.0, "age": 28, "sex": "F"},
            {"istatus": "treated", "outcome": 1.0, "age": 31, "sex": "M"},
        ]
    )


@dataclass
class _FakeOrchestratorState:
    values: dict[str, Any]

    def get(self, key: str) -> Any:
        return self.values.get(key)


@dataclass
class _FakeDataRepo:
    dataframe: pd.DataFrame
    saved_csv: list[pd.DataFrame] = field(default_factory=list)
    saved_json: list[str] = field(default_factory=list)

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        start: int = 0,
        limit: int | None = None,
    ) -> pd.DataFrame:
        del user_id, conversation_id, dataset_id, start, limit
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
        del user_id, conversation_id, dataset_id, overwrite, include_index
        self.saved_csv.append(df.copy())

    def save_json_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        json_data: str,
        *,
        overwrite: bool = True,
    ) -> None:
        del user_id, conversation_id, dataset_id, overwrite
        self.saved_json.append(json_data)


@dataclass
class _FakeLLM:
    json_outputs: list[object]
    generate_outputs: list[str] = field(default_factory=lambda: ["done"])
    generate_json_calls: list[dict[str, object]] = field(default_factory=list)
    generate_calls: list[dict[str, object]] = field(default_factory=list)

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
        return LLMResponse(content=self.generate_outputs.pop(0))

    def generate_json(
        self,
        *,
        schema: type[BaseModel],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> Any:
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
        payload = next_output.model_dump() if isinstance(next_output, BaseModel) else next_output
        return schema.model_validate(payload)


@dataclass
class _FakeDataManipulationTool:
    result: pd.DataFrame
    calls: list[dict[str, object]] = field(default_factory=list)

    def manipulate(
        self,
        *,
        dataframe: pd.DataFrame,
        data_summary: str,
        instructions: str,
        table_name: str,
        retry_attempts: int,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "dataframe": dataframe.copy(),
                "data_summary": data_summary,
                "instructions": instructions,
                "table_name": table_name,
                "retry_attempts": retry_attempts,
            }
        )
        return self.result.copy()


@dataclass
class _FakeAdvancedAnalyticsTool:
    calls: list[dict[str, object]] = field(default_factory=list)

    def analyze(
        self,
        *,
        dataframe: pd.DataFrame,
        data_summary: Any,
        user_request: str,
    ) -> AnalyticsResultModel:
        self.calls.append(
            {
                "dataframe": dataframe.copy(),
                "data_summary": data_summary,
                "user_request": user_request,
            }
        )
        return AnalyticsResultModel(
            analysis_type="propensity_score",
            summary="propensity complete",
            tables={},
            metrics={},
        )


@dataclass
class _FakePlotTool:
    def generate_specs(
        self,
        *,
        dataframe: pd.DataFrame,
        data_summary: Any,
        user_intent: str,
    ) -> list[dict[str, Any]]:
        del dataframe, data_summary, user_intent
        return []


@dataclass
class _FakeToolFactory:
    data_manipulation_tool: _FakeDataManipulationTool
    advanced_analytics_tool: _FakeAdvancedAnalyticsTool
    profiling_tool: DatasetProfilingTool
    plot_tool: _FakePlotTool

    def get_tool(self, name: str) -> Any:
        return {
            DataManipulationTool.NAME: self.data_manipulation_tool,
            AdvancedAnalyticsTool.NAME: self.advanced_analytics_tool,
            DatasetProfilingTool.NAME: self.profiling_tool,
            PlotTool.NAME: self.plot_tool,
        }[name]


def _build_node_request(
    *,
    dataframe: pd.DataFrame,
    summary: Any,
    messages: list[ChatMessage],
    dataset_id: UUID,
) -> NodeRequest:
    return NodeRequest(
        user_id=uuid4(),
        conversation_id=uuid4(),
        node_state=DataStatisticsState.init_empty(),
        orchestrator_state=_FakeOrchestratorState(
            {
                "working_dataset_id": dataset_id,
                "latest_dataset_summary": summary,
            }
        ),
        read_only_messages_history=messages,
    )


def test_advanced_analytics_uses_original_row_level_data_after_sql_analytics() -> None:
    dataframe = _row_level_dataframe()
    aggregate = pd.DataFrame({"istatus": ["control", "treated"], "n": [2, 2]})
    profiling_tool = DatasetProfilingTool()
    summary = profiling_tool.extract_dataset_summary(
        dataframe,
        max_categories=10,
        sample_distinct=10,
        compute_quantiles=False,
        strict=True,
    )
    data_manipulation_tool = _FakeDataManipulationTool(result=aggregate)
    advanced_tool = _FakeAdvancedAnalyticsTool()
    node = DataStatisticsNode(
        data_repo=_FakeDataRepo(dataframe=dataframe),
        llm=_FakeLLM(
            json_outputs=[
                {
                    "intent_analytics": True,
                    "intent_analytics_brief": "Count rows by istatus.",
                    "intent_chart": False,
                    "intent_chart_brief": "",
                    "intent_advanced_analytics": True,
                    "intent_advanced_analytics_brief": (
                        "Estimate propensity scores for istatus using age and sex."
                    ),
                    "intent_out_of_scope": False,
                }
            ]
        ),
        tools_factory=_FakeToolFactory(
            data_manipulation_tool=data_manipulation_tool,
            advanced_analytics_tool=advanced_tool,
            profiling_tool=profiling_tool,
            plot_tool=_FakePlotTool(),
        ),
    )

    result = node.run(
        request=_build_node_request(
            dataframe=dataframe,
            summary=summary,
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "Count rows by treatment and estimate propensity scores for istatus "
                        "using age and sex."
                    ),
                )
            ],
            dataset_id=uuid4(),
        )
    )

    assert result.status == "PENDING"
    assert len(data_manipulation_tool.calls) == 1
    assert len(advanced_tool.calls) == 1
    captured = advanced_tool.calls[0]["dataframe"]
    assert isinstance(captured, pd.DataFrame)
    pd.testing.assert_frame_equal(captured.reset_index(drop=True), dataframe.reset_index(drop=True))


def test_advanced_analytics_request_includes_latest_prompt_and_recent_history() -> None:
    dataframe = _row_level_dataframe()
    profiling_tool = DatasetProfilingTool()
    summary = profiling_tool.extract_dataset_summary(
        dataframe,
        max_categories=10,
        sample_distinct=10,
        compute_quantiles=False,
        strict=True,
    )
    advanced_tool = _FakeAdvancedAnalyticsTool()
    node = DataStatisticsNode(
        data_repo=_FakeDataRepo(dataframe=dataframe),
        llm=_FakeLLM(
            json_outputs=[
                {
                    "intent_analytics": False,
                    "intent_analytics_brief": "",
                    "intent_chart": False,
                    "intent_chart_brief": "",
                    "intent_advanced_analytics": True,
                    "intent_advanced_analytics_brief": "Estimate propensity scores.",
                    "intent_out_of_scope": False,
                }
            ]
        ),
        tools_factory=_FakeToolFactory(
            data_manipulation_tool=_FakeDataManipulationTool(result=pd.DataFrame()),
            advanced_analytics_tool=advanced_tool,
            profiling_tool=profiling_tool,
            plot_tool=_FakePlotTool(),
        ),
    )

    _ = node.run(
        request=_build_node_request(
            dataframe=dataframe,
            summary=summary,
            messages=[
                ChatMessage(
                    role="user",
                    content="Treatment is istatus and the outcome is outcome.",
                ),
                ChatMessage(role="assistant", content="Noted."),
                ChatMessage(
                    role="user",
                    content="Use age and sex as covariates for propensity scores.",
                ),
            ],
            dataset_id=uuid4(),
        )
    )

    assert len(advanced_tool.calls) == 1
    user_request = advanced_tool.calls[0]["user_request"]
    assert isinstance(user_request, str)
    assert "Estimate propensity scores." in user_request
    assert "Use age and sex as covariates for propensity scores." in user_request
    assert "Treatment is istatus and the outcome is outcome." in user_request
    assert "identify exactly one treatment column and the covariate columns" in user_request
