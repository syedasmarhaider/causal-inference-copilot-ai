from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from python.domain.repo.working_data_repo import WorkingDataSQLResult
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationSQLPlan,
    DataManipulationTool,
)

_UUID = "2ebc18a3-3777-44f0-9698-715d4f0ae454"
_SHORT_TABLE = "conv_2ebc18a33777"


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

    def execute_sql(self, *, dataframe: pd.DataFrame, request: object) -> WorkingDataSQLResult:
        self.calls.append({"dataframe": dataframe.copy(), "request": request})
        return WorkingDataSQLResult(
            table_name=request.table_name,
            executed_statements=tuple(request.statements),
            columns=("a",),
            row_count=int(len(dataframe)),
            has_result_set=True,
            elapsed_ms=1.0,
            dataframe=dataframe[["a"]].copy() if "a" in dataframe.columns else dataframe.copy(),
        )


def test_manipulate_executes_sql_with_shortened_uuid_table_name() -> None:
    llm = _FakeLLMService(
        plan=DataManipulationSQLPlan(
            statements=[f"SELECT a FROM {_SHORT_TABLE} ORDER BY a DESC"],
            table_name=_SHORT_TABLE,
        )
    )
    repo = _FakeWorkingDataRepo()
    tool = DataManipulationTool(llm=llm, working_data_repo=repo)

    output_df = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}, {"a": 3}, {"a": 2}]),
        conversation_id=_UUID,
        instructions="sort a descending",
        data_summary='{"n_rows": 3}',
    )

    assert len(llm.calls) == 1
    assert _SHORT_TABLE in str(llm.calls[0]["user_prompt"])
    assert len(repo.calls) == 1
    assert repo.calls[0]["request"].table_name == _SHORT_TABLE
    assert repo.calls[0]["request"].statements == (f"SELECT a FROM {_SHORT_TABLE} ORDER BY a DESC",)
    assert output_df.to_dict(orient="records") == [{"a": 1}, {"a": 3}, {"a": 2}]


def test_manipulate_accepts_quoted_table_references() -> None:
    llm = _FakeLLMService(
        plan=DataManipulationSQLPlan(
            statements=[f'SELECT a FROM "{_SHORT_TABLE}"'],
            table_name=_SHORT_TABLE,
        )
    )
    repo = _FakeWorkingDataRepo()
    tool = DataManipulationTool(llm=llm, working_data_repo=repo)

    _ = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}]),
        conversation_id=_UUID,
        instructions="show a",
        data_summary='{"n_rows": 1}',
    )

    assert repo.calls[0]["request"].statements == (f'SELECT a FROM "{_SHORT_TABLE}"',)


def test_manipulate_raises_when_llm_table_name_does_not_match_expected() -> None:
    llm = _FakeLLMService(
        plan=DataManipulationSQLPlan(
            statements=["SELECT a FROM wrong_table"],
            table_name="wrong_table",
        )
    )
    tool = DataManipulationTool(llm=llm, working_data_repo=_FakeWorkingDataRepo())

    with pytest.raises(ValueError, match=r"sql plan table_name mismatch"):
        _ = tool.manipulate(
            dataframe=pd.DataFrame([{"a": 1}]),
            conversation_id=_UUID,
            instructions="show a",
            data_summary='{"n_rows": 1}',
        )


def test_manipulate_raises_when_statement_does_not_reference_expected_table_name() -> None:
    llm = _FakeLLMService(
        plan=DataManipulationSQLPlan(
            statements=["SELECT a FROM another_table"],
            table_name=_SHORT_TABLE,
        )
    )
    tool = DataManipulationTool(llm=llm, working_data_repo=_FakeWorkingDataRepo())

    with pytest.raises(ValueError, match=r"does not reference expected table_name"):
        _ = tool.manipulate(
            dataframe=pd.DataFrame([{"a": 1}]),
            conversation_id=_UUID,
            instructions="show a",
            data_summary='{"n_rows": 1}',
        )


def test_manipulate_allows_missing_instructions_and_uses_fallback_prompt_intent() -> None:
    llm = _FakeLLMService(
        plan=DataManipulationSQLPlan(
            statements=[f"SELECT a FROM {_SHORT_TABLE}"],
            table_name=_SHORT_TABLE,
        )
    )
    tool = DataManipulationTool(llm=llm, working_data_repo=_FakeWorkingDataRepo())

    _ = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}]),
        conversation_id=_UUID,
        data_summary='{"n_rows": 1}',
    )

    assert "No explicit user instruction provided." in str(llm.calls[0]["user_prompt"])


def test_manipulate_supports_multiple_statements_when_each_references_table() -> None:
    llm = _FakeLLMService(
        plan=DataManipulationSQLPlan(
            statements=[
                f"CREATE TEMP TABLE tmp AS SELECT a FROM {_SHORT_TABLE}",
                f"SELECT a FROM {_SHORT_TABLE}",
            ],
            table_name=_SHORT_TABLE,
        )
    )
    repo = _FakeWorkingDataRepo()
    tool = DataManipulationTool(llm=llm, working_data_repo=repo)

    _ = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}]),
        conversation_id=_UUID,
        instructions="show a",
        data_summary='{"n_rows": 1}',
    )

    assert repo.calls[0]["request"].statements == (
        f"CREATE TEMP TABLE tmp AS SELECT a FROM {_SHORT_TABLE}",
        f"SELECT a FROM {_SHORT_TABLE}",
    )


@pytest.mark.parametrize(
    ("conversation_id", "expected_table_name"),
    [
        ("my convo", "my_convo"),
        ("123-run", "t_123_run"),
        ("@@@", "working_data"),
    ],
)
def test_sanitize_table_name_handles_non_uuid_inputs(
    conversation_id: str,
    expected_table_name: str,
) -> None:
    assert DataManipulationTool._sanitize_table_name(conversation_id) == expected_table_name


def test_statement_references_table_rejects_partial_identifier_matches() -> None:
    assert (
        DataManipulationTool._statement_references_table(
            statement="SELECT * FROM conv_2ebc18a33777_suffix",
            table_name=_SHORT_TABLE,
        )
        is False
    )


@pytest.mark.parametrize(
    ("conversation_id", "data_summary", "error_pattern"),
    [
        ("", '{"n_rows": 1}', r"conversation_id must be non-empty"),
        (_UUID, "", r"data_summary must be non-empty"),
    ],
)
def test_manipulate_validates_required_inputs(
    conversation_id: str,
    data_summary: str,
    error_pattern: str,
) -> None:
    llm = _FakeLLMService(
        plan=DataManipulationSQLPlan(
            statements=[f"SELECT a FROM {_SHORT_TABLE}"],
            table_name=_SHORT_TABLE,
        )
    )
    tool = DataManipulationTool(llm=llm, working_data_repo=_FakeWorkingDataRepo())

    with pytest.raises(ValueError, match=error_pattern):
        _ = tool.manipulate(
            dataframe=pd.DataFrame([{"a": 1}]),
            conversation_id=conversation_id,
            data_summary=data_summary,
        )


def test_data_manipulation_tool_validates_max_attempts() -> None:
    with pytest.raises(ValueError, match=r"max_attempts must be >= 1"):
        DataManipulationTool(
            llm=_FakeLLMService(
                plan=DataManipulationSQLPlan(
                    statements=[f"SELECT a FROM {_SHORT_TABLE}"],
                    table_name=_SHORT_TABLE,
                )
            ),
            working_data_repo=_FakeWorkingDataRepo(),
            max_attempts=0,
        )
