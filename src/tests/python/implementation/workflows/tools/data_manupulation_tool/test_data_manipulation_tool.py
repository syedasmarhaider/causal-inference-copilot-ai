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

_TABLE = "valid_table"


@dataclass
class _FakeLLMService:
    plans: list[object]
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
        max_attempts: int = 1,
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
        if not self.plans:
            raise AssertionError("unexpected generate_json call")
        next_plan = self.plans.pop(0)
        if isinstance(next_plan, Exception):
            raise next_plan
        assert isinstance(next_plan, DataManipulationSQLPlan)
        return next_plan


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


def test_manipulate_executes_sql_with_passed_table_name() -> None:
    llm = _FakeLLMService(
        plans=[
            DataManipulationSQLPlan(
                statements=[f"SELECT a FROM {_TABLE} ORDER BY a DESC"],
                table_name=_TABLE,
            )
        ]
    )
    repo = _FakeWorkingDataRepo()
    tool = DataManipulationTool(llm=llm, working_data_repo=repo)

    output_df = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}, {"a": 3}, {"a": 2}]),
        table_name=_TABLE,
        instructions="sort a descending",
        data_summary='{"n_rows": 3}',
    )

    assert len(llm.calls) == 1
    assert _TABLE in str(llm.calls[0]["user_prompt"])
    assert len(repo.calls) == 1
    assert repo.calls[0]["request"].table_name == _TABLE
    assert repo.calls[0]["request"].statements == (f"SELECT a FROM {_TABLE} ORDER BY a DESC",)
    assert output_df.to_dict(orient="records") == [{"a": 1}, {"a": 3}, {"a": 2}]


def test_manipulate_accepts_quoted_table_references() -> None:
    llm = _FakeLLMService(
        plans=[
            DataManipulationSQLPlan(
                statements=[f'SELECT a FROM "{_TABLE}"'],
                table_name=_TABLE,
            )
        ]
    )
    repo = _FakeWorkingDataRepo()
    tool = DataManipulationTool(llm=llm, working_data_repo=repo)

    _ = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}]),
        table_name=_TABLE,
        instructions="show a",
        data_summary='{"n_rows": 1}',
    )

    assert repo.calls[0]["request"].statements == (f'SELECT a FROM "{_TABLE}"',)


def test_manipulate_retries_on_table_name_mismatch() -> None:
    llm = _FakeLLMService(
        plans=[
            DataManipulationSQLPlan(
                statements=["SELECT a FROM wrong_table"],
                table_name="wrong_table",
            ),
            DataManipulationSQLPlan(
                statements=[f"SELECT a FROM {_TABLE}"],
                table_name=_TABLE,
            ),
        ]
    )
    repo = _FakeWorkingDataRepo()
    tool = DataManipulationTool(llm=llm, working_data_repo=repo, max_attempts=1)

    _ = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}]),
        table_name=_TABLE,
        instructions="show a",
        data_summary='{"n_rows": 1}',
        retry_attempts=2,
    )

    assert len(llm.calls) == 2
    assert llm.calls[0]["max_attempts"] == 1
    assert "Previous invalid plan" in str(llm.calls[1]["user_prompt"])
    assert repo.calls[0]["request"].table_name == _TABLE


def test_manipulate_retries_on_missing_table_reference() -> None:
    llm = _FakeLLMService(
        plans=[
            DataManipulationSQLPlan(
                statements=["SELECT a FROM another_table"],
                table_name=_TABLE,
            ),
            DataManipulationSQLPlan(
                statements=[f"SELECT a FROM {_TABLE}"],
                table_name=_TABLE,
            ),
        ]
    )
    repo = _FakeWorkingDataRepo()
    tool = DataManipulationTool(llm=llm, working_data_repo=repo)

    _ = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}]),
        table_name=_TABLE,
        instructions="show a",
        data_summary='{"n_rows": 1}',
        retry_attempts=2,
    )

    assert len(llm.calls) == 2
    assert repo.calls[0]["request"].statements == (f"SELECT a FROM {_TABLE}",)
    assert "at least once" in str(llm.calls[1]["user_prompt"])


def test_manipulate_raises_runtime_error_after_retry_budget_is_exhausted() -> None:
    llm = _FakeLLMService(
        plans=[
            DataManipulationSQLPlan(
                statements=["SELECT a FROM wrong_table"],
                table_name="wrong_table",
            )
        ]
    )
    tool = DataManipulationTool(llm=llm, working_data_repo=_FakeWorkingDataRepo())

    with pytest.raises(RuntimeError, match=r"Failed data manipulation plan after 1 attempts"):
        _ = tool.manipulate(
            dataframe=pd.DataFrame([{"a": 1}]),
            table_name=_TABLE,
            instructions="show a",
            data_summary='{"n_rows": 1}',
            retry_attempts=1,
        )


def test_manipulate_supports_multiple_statements_with_temp_table_flow() -> None:
    llm = _FakeLLMService(
        plans=[
            DataManipulationSQLPlan(
                statements=[
                    f"CREATE TEMP TABLE tmp AS SELECT a FROM {_TABLE}",
                    "SELECT a FROM tmp",
                ],
                table_name=_TABLE,
            )
        ]
    )
    repo = _FakeWorkingDataRepo()
    tool = DataManipulationTool(llm=llm, working_data_repo=repo)

    _ = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}]),
        table_name=_TABLE,
        instructions="show a",
        data_summary='{"n_rows": 1}',
    )

    assert repo.calls[0]["request"].statements == (
        f"CREATE TEMP TABLE tmp AS SELECT a FROM {_TABLE}",
        "SELECT a FROM tmp",
    )


@pytest.mark.parametrize(
    ("table_name", "error_pattern"),
    [
        ("", r"table_name must be non-empty"),
        ("bad table", r"Invalid table_name"),
        ("123table", r"Invalid table_name"),
    ],
)
def test_manipulate_validates_table_name(
    table_name: str,
    error_pattern: str,
) -> None:
    llm = _FakeLLMService(
        plans=[
            DataManipulationSQLPlan(
                statements=[f"SELECT a FROM {_TABLE}"],
                table_name=_TABLE,
            )
        ]
    )
    tool = DataManipulationTool(llm=llm, working_data_repo=_FakeWorkingDataRepo())

    with pytest.raises(ValueError, match=error_pattern):
        _ = tool.manipulate(
            dataframe=pd.DataFrame([{"a": 1}]),
            table_name=table_name,
            data_summary='{"n_rows": 1}',
            instructions="select a",
        )


def test_statement_references_table_rejects_partial_identifier_matches() -> None:
    assert (
        DataManipulationTool._statement_references_table(
            statement="SELECT * FROM valid_table_suffix",
            table_name=_TABLE,
        )
        is False
    )


@pytest.mark.parametrize(
    ("data_summary", "instructions", "retry_attempts", "error_pattern"),
    [
        ("", "select a", None, r"data_summary must be non-empty"),
        ('{"n_rows": 1}', "", None, r"instructions must be non-empty"),
        ('{"n_rows": 1}', "select a", 0, r"retry_attempts must be >= 1"),
    ],
)
def test_manipulate_validates_required_inputs(
    data_summary: str,
    instructions: str,
    retry_attempts: int | None,
    error_pattern: str,
) -> None:
    llm = _FakeLLMService(
        plans=[
            DataManipulationSQLPlan(
                statements=[f"SELECT a FROM {_TABLE}"],
                table_name=_TABLE,
            )
        ]
    )
    tool = DataManipulationTool(llm=llm, working_data_repo=_FakeWorkingDataRepo())

    with pytest.raises(ValueError, match=error_pattern):
        _ = tool.manipulate(
            dataframe=pd.DataFrame([{"a": 1}]),
            table_name=_TABLE,
            data_summary=data_summary,
            instructions=instructions,
            retry_attempts=retry_attempts,
        )


def test_data_manipulation_tool_validates_constructor_max_attempts() -> None:
    with pytest.raises(ValueError, match=r"max_attempts must be >= 1"):
        DataManipulationTool(
            llm=_FakeLLMService(
                plans=[
                    DataManipulationSQLPlan(
                        statements=[f"SELECT a FROM {_TABLE}"],
                        table_name=_TABLE,
                    )
                ]
            ),
            working_data_repo=_FakeWorkingDataRepo(),
            max_attempts=0,
        )
