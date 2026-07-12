from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from python.domain.repo.analytics_repo import AnalyticsSQLResult
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationSQLPlan,
    DataManipulationTool,
)

_TABLE = "valid_table"


def _plan_payload(
    *,
    table_name: str = _TABLE,
    statements: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "statements": statements if statements is not None else [f"SELECT a FROM {_TABLE}"],
        "table_name": table_name,
    }


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

        last_validation_error: ValidationError | None = None

        for _ in range(max_attempts):
            if not self.plans:
                raise AssertionError("unexpected generate_json call")

            next_plan = self.plans.pop(0)
            if isinstance(next_plan, Exception):
                raise next_plan

            payload = next_plan.model_dump() if isinstance(next_plan, BaseModel) else next_plan
            try:
                return schema.model_validate(payload)
            except ValidationError as exc:
                last_validation_error = exc

        raise RuntimeError(
            f"Failed JSON schema={schema.__name__} after {max_attempts} attempts. "
            f"Last error: {last_validation_error}"
        )


@dataclass
class _FakeAnalyticsRepo:
    calls: list[dict[str, object]] = field(default_factory=list)
    errors: list[Exception] = field(default_factory=list)

    def execute_sql(self, *, dataframe: pd.DataFrame, request: object) -> AnalyticsSQLResult:
        self.calls.append({"dataframe": dataframe.copy(), "request": request})
        if self.errors:
            raise self.errors.pop(0)
        return AnalyticsSQLResult(
            table_name=request.table_name,
            executed_statements=tuple(request.statements),
            columns=("a",),
            row_count=int(len(dataframe)),
            has_result_set=True,
            elapsed_ms=1.0,
            dataframe=dataframe[["a"]].copy() if "a" in dataframe.columns else dataframe.copy(),
        )


def test_plan_schema_factory_binds_expected_table_name_without_shared_state() -> None:
    alpha_schema = DataManipulationSQLPlan.for_table_name("alpha")
    beta_schema = DataManipulationSQLPlan.for_table_name("beta")

    assert alpha_schema is not beta_schema
    assert issubclass(alpha_schema, DataManipulationSQLPlan)
    assert issubclass(beta_schema, DataManipulationSQLPlan)
    assert alpha_schema.EXPECTED_TABLE_NAME == "alpha"
    assert beta_schema.EXPECTED_TABLE_NAME == "beta"

    alpha_plan = alpha_schema.model_validate(
        {"statements": ["SELECT a FROM alpha"], "table_name": "alpha"}
    )
    assert alpha_plan.table_name == "alpha"

    with pytest.raises(ValidationError, match=r"expected='alpha' got='beta'"):
        alpha_schema.model_validate({"statements": ["SELECT a FROM beta"], "table_name": "beta"})

    with pytest.raises(ValidationError, match=r"expected='beta' got='alpha'"):
        beta_schema.model_validate({"statements": ["SELECT a FROM alpha"], "table_name": "alpha"})


def test_plan_schema_requires_source_table_reference() -> None:
    schema = DataManipulationSQLPlan.for_table_name(_TABLE)

    with pytest.raises(ValidationError, match=r"does not reference expected table_name"):
        schema.model_validate(
            {
                "statements": [
                    "CREATE TEMP TABLE tmp AS SELECT a FROM somewhere_else",
                    "SELECT a FROM tmp",
                ],
                "table_name": _TABLE,
            }
        )


def test_manipulate_executes_sql_with_dynamic_schema() -> None:
    llm = _FakeLLMService(
        plans=[_plan_payload(statements=[f"SELECT a FROM {_TABLE} ORDER BY a DESC"])]
    )
    repo = _FakeAnalyticsRepo()
    tool = DataManipulationTool(llm=llm, analytics_repo=repo)

    output_df = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}, {"a": 3}, {"a": 2}]),
        table_name=_TABLE,
        instructions="sort a descending",
        data_summary='{"n_rows": 3}',
    )

    assert len(llm.calls) == 1
    assert _TABLE in str(llm.calls[0]["user_prompt"])
    assert llm.calls[0]["max_attempts"] == 3
    assert llm.calls[0]["schema"] is not DataManipulationSQLPlan
    assert llm.calls[0]["schema"].EXPECTED_TABLE_NAME == _TABLE
    assert len(repo.calls) == 1
    assert repo.calls[0]["request"].table_name == _TABLE
    assert repo.calls[0]["request"].statements == (f"SELECT a FROM {_TABLE} ORDER BY a DESC",)
    assert output_df.to_dict(orient="records") == [{"a": 1}, {"a": 3}, {"a": 2}]


def test_manipulate_accepts_quoted_table_references() -> None:
    llm = _FakeLLMService(plans=[_plan_payload(statements=[f'SELECT a FROM "{_TABLE}"'])])
    repo = _FakeAnalyticsRepo()
    tool = DataManipulationTool(llm=llm, analytics_repo=repo)

    _ = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}]),
        table_name=_TABLE,
        instructions="show a",
        data_summary='{"n_rows": 1}',
    )

    assert repo.calls[0]["request"].statements == (f'SELECT a FROM "{_TABLE}"',)


def test_manipulate_repairs_failed_sql_once_and_executes_repaired_plan() -> None:
    original_intent = "Report the median of a without changing the requested statistic"
    failed_statement = (
        f"SELECT PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY a) AS median_a FROM {_TABLE}"
    )
    repaired_statement = f"SELECT quantile_cont(a, 0.50) AS median_a FROM {_TABLE}"
    execution_error = RuntimeError(
        "Catalog Error: Scalar Function with name percentile_cont does not exist"
    )
    llm = _FakeLLMService(
        plans=[
            _plan_payload(statements=[failed_statement]),
            _plan_payload(statements=[repaired_statement]),
        ]
    )
    repo = _FakeAnalyticsRepo(errors=[execution_error])
    tool = DataManipulationTool(llm=llm, analytics_repo=repo)

    output_df = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}, {"a": 3}, {"a": 2}]),
        table_name=_TABLE,
        instructions=original_intent,
        data_summary='{"n_rows": 3}',
    )

    assert len(llm.calls) == 2
    repair_prompt = str(llm.calls[1]["user_prompt"])
    assert original_intent in repair_prompt
    assert failed_statement in repair_prompt
    assert str(execution_error) in repair_prompt
    assert len(repo.calls) == 2
    assert repo.calls[0]["request"].statements == (failed_statement,)
    assert repo.calls[1]["request"].statements == (repaired_statement,)
    assert output_df.to_dict(orient="records") == [{"a": 1}, {"a": 3}, {"a": 2}]


def test_manipulate_propagates_second_execution_error_without_third_attempt() -> None:
    failed_statement = f"SELECT unavailable_function(a) FROM {_TABLE}"
    repaired_statement = f"SELECT another_unavailable_function(a) FROM {_TABLE}"
    first_error = RuntimeError("Catalog Error: unavailable_function does not exist")
    second_error = RuntimeError("Catalog Error: repaired SQL still cannot execute")
    llm = _FakeLLMService(
        plans=[
            _plan_payload(statements=[failed_statement]),
            _plan_payload(statements=[repaired_statement]),
        ]
    )
    repo = _FakeAnalyticsRepo(errors=[first_error, second_error])
    tool = DataManipulationTool(llm=llm, analytics_repo=repo)

    with pytest.raises(RuntimeError) as exc_info:
        _ = tool.manipulate(
            dataframe=pd.DataFrame([{"a": 1}]),
            table_name=_TABLE,
            instructions="Apply the requested function to a",
            data_summary='{"n_rows": 1}',
        )

    assert exc_info.value is second_error
    assert len(llm.calls) == 2
    assert len(repo.calls) == 2
    assert repo.calls[0]["request"].statements == (failed_statement,)
    assert repo.calls[1]["request"].statements == (repaired_statement,)


def test_manipulate_uses_llm_internal_retry_for_table_name_mismatch() -> None:
    llm = _FakeLLMService(
        plans=[
            _plan_payload(table_name="wrong_table", statements=["SELECT a FROM wrong_table"]),
            _plan_payload(statements=[f"SELECT a FROM {_TABLE}"]),
        ]
    )
    repo = _FakeAnalyticsRepo()
    tool = DataManipulationTool(llm=llm, analytics_repo=repo)

    _ = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}]),
        table_name=_TABLE,
        instructions="show a",
        data_summary='{"n_rows": 1}',
        retry_attempts=2,
    )

    assert len(llm.calls) == 1
    assert llm.calls[0]["max_attempts"] == 2
    assert repo.calls[0]["request"].table_name == _TABLE


def test_manipulate_uses_llm_internal_retry_for_missing_source_table_reference() -> None:
    llm = _FakeLLMService(
        plans=[
            _plan_payload(statements=["SELECT a FROM another_table"]),
            _plan_payload(statements=[f"SELECT a FROM {_TABLE}"]),
        ]
    )
    repo = _FakeAnalyticsRepo()
    tool = DataManipulationTool(llm=llm, analytics_repo=repo)

    _ = tool.manipulate(
        dataframe=pd.DataFrame([{"a": 1}]),
        table_name=_TABLE,
        instructions="show a",
        data_summary='{"n_rows": 1}',
        retry_attempts=2,
    )

    assert len(llm.calls) == 1
    assert llm.calls[0]["max_attempts"] == 2
    assert repo.calls[0]["request"].statements == (f"SELECT a FROM {_TABLE}",)


def test_manipulate_raises_runtime_error_after_retry_budget_is_exhausted() -> None:
    llm = _FakeLLMService(
        plans=[_plan_payload(table_name="wrong_table", statements=["SELECT a FROM wrong_table"])]
    )
    tool = DataManipulationTool(llm=llm, analytics_repo=_FakeAnalyticsRepo())

    with pytest.raises(
        RuntimeError,
        match=r"Failed JSON schema=DataManipulationSQLPlanFor_valid_table after 1 attempts",
    ):
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
            _plan_payload(
                statements=[
                    f"CREATE TEMP TABLE tmp AS SELECT a FROM {_TABLE}",
                    "SELECT a FROM tmp",
                ]
            ),
        ]
    )
    repo = _FakeAnalyticsRepo()
    tool = DataManipulationTool(llm=llm, analytics_repo=repo)

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
def test_manipulate_validates_table_name(table_name: str, error_pattern: str) -> None:
    tool = DataManipulationTool(
        llm=_FakeLLMService(plans=[_plan_payload()]),
        analytics_repo=_FakeAnalyticsRepo(),
    )

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
    tool = DataManipulationTool(
        llm=_FakeLLMService(plans=[_plan_payload()]),
        analytics_repo=_FakeAnalyticsRepo(),
    )

    with pytest.raises(ValueError, match=error_pattern):
        _ = tool.manipulate(
            dataframe=pd.DataFrame([{"a": 1}]),
            table_name=_TABLE,
            data_summary=data_summary,
            instructions=instructions,
            retry_attempts=retry_attempts,
        )
