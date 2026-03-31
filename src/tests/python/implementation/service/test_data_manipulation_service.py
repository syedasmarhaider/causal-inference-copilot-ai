from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.implementation.workflows.nodes.service.data_manupulation_service.data_manipulation_service import (
    DataManipulationService,
    DataManipulationSQLPlan,
)


@dataclass
class _FakeLLMService:
    plan: DataManipulationSQLPlan
    calls: list[dict[str, object]] = field(default_factory=list)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
    ) -> LLMResponse:
        del system_prompt, user_prompt, config, history
        raise NotImplementedError

    def generate_json(
        self,
        *,
        schema: type[DataManipulationSQLPlan],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> DataManipulationSQLPlan:
        self.calls.append(
            {
                "schema": schema,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "config": config,
                "history": history,
                "max_attempts": max_attempts,
            }
        )
        return self.plan


@dataclass
class _FakeWorkingDataRepo:
    calls: list[dict[str, object]] = field(default_factory=list)

    def execute_sql(self, *, dataframe: pd.DataFrame, request: object) -> object:
        self.calls.append({"dataframe": dataframe.copy(), "request": request})
        return type(
            "Result",
            (),
            {
                "table_name": request.table_name,
                "executed_statements": request.statements,
                "columns": ("a",),
                "row_count": int(len(dataframe)),
                "has_result_set": True,
                "elapsed_ms": 1.0,
                "dataframe": dataframe[["a"]].copy() if "a" in dataframe.columns else dataframe.copy(),
            },
        )()


def test_manipulate_generates_sql_and_executes_working_repo() -> None:
    llm = _FakeLLMService(
        plan=DataManipulationSQLPlan(
            statements=["SELECT a FROM working_data ORDER BY a DESC"],
            table_name="working_data",
        )
    )
    repo = _FakeWorkingDataRepo()
    service = DataManipulationService(llm=llm, working_data_repo=repo)
    input_df = pd.DataFrame([{"a": 1}, {"a": 3}, {"a": 2}])

    result = service.manipulate(
        dataframe=input_df,
        user_intent="sort column a descending",
        data_summary='{"n_rows": 3, "profiles": [{"name": "a", "inferred_kind": "NUMERIC"}]}',
        table_name="working_data",
        chat_history=[ChatMessage(role="user", content="please sort a")],
    )

    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["schema"] is DataManipulationSQLPlan
    assert "sort column a descending" in str(call["user_prompt"])
    assert "dataset_summary" in str(call["user_prompt"])
    assert call["history"] == [ChatMessage(role="user", content="please sort a")]

    assert len(repo.calls) == 1
    repo_call = repo.calls[0]
    assert repo_call["request"].table_name == "working_data"
    assert repo_call["request"].statements == ("SELECT a FROM working_data ORDER BY a DESC",)

    assert result.sql_request.table_name == "working_data"
    assert result.sql_result.has_result_set is True
    assert result.sql_result.row_count == 3


def test_manipulate_limits_history_to_configured_window() -> None:
    llm = _FakeLLMService(
        plan=DataManipulationSQLPlan(
            statements=["SELECT a FROM input_table"],
            table_name="input_table",
        )
    )
    repo = _FakeWorkingDataRepo()
    service = DataManipulationService(
        llm=llm,
        working_data_repo=repo,
        max_history_messages=2,
    )

    history = [
        ChatMessage(role="user", content="msg-1"),
        ChatMessage(role="assistant", content="msg-2"),
        ChatMessage(role="user", content="msg-3"),
    ]

    _ = service.manipulate(
        dataframe=pd.DataFrame([{"a": 10}]),
        user_intent="select one row",
        data_summary='{"n_rows": 1}',
        table_name="input_table",
        chat_history=history,
    )

    llm_history = llm.calls[0]["history"]
    assert llm_history == history[-2:]


def test_manipulate_overrides_llm_table_name_with_requested_table_name() -> None:
    llm = _FakeLLMService(
        plan=DataManipulationSQLPlan(
            statements=["SELECT a FROM wrong_table"],
            table_name="wrong_table",
        )
    )
    repo = _FakeWorkingDataRepo()
    service = DataManipulationService(llm=llm, working_data_repo=repo)

    result = service.manipulate(
        dataframe=pd.DataFrame([{"a": 1}]),
        user_intent="show a",
        data_summary='{"n_rows": 1}',
        table_name="expected_table",
        chat_history=None,
    )

    assert result.sql_request.table_name == "expected_table"
    assert repo.calls[0]["request"].table_name == "expected_table"


@pytest.mark.parametrize(
    ("user_intent", "data_summary", "table_name", "error_pattern"),
    [
        ("   ", '{"n_rows": 1}', "t", r"user_intent must be non-empty"),
        ("select", "   ", "t", r"data_summary must be non-empty"),
        ("select", '{"n_rows": 1}', "   ", r"table_name must be non-empty"),
    ],
)
def test_manipulate_rejects_empty_required_inputs(
    user_intent: str,
    data_summary: str,
    table_name: str,
    error_pattern: str,
) -> None:
    llm = _FakeLLMService(
        plan=DataManipulationSQLPlan(
            statements=["SELECT 1"],
            table_name="t",
        )
    )
    repo = _FakeWorkingDataRepo()
    service = DataManipulationService(llm=llm, working_data_repo=repo)

    with pytest.raises(ValueError, match=error_pattern):
        _ = service.manipulate(
            dataframe=pd.DataFrame([{"a": 1}]),
            user_intent=user_intent,
            data_summary=data_summary,
            table_name=table_name,
            chat_history=None,
        )

