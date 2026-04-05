from __future__ import annotations

import pandas as pd
import pytest

from python.domain.repo.working_data_repo import WorkingDataSQLRequest
from python.implementation.repo.duckdb_working_analytics_repo import DuckDBWorkingDataRepo


def test_execute_sql_returns_last_result_set_and_metadata() -> None:
    repo = DuckDBWorkingDataRepo()
    df = pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}])
    request = WorkingDataSQLRequest(
        table_name="input_table",
        statements=(
            "SELECT a, b FROM input_table ORDER BY a DESC LIMIT 2",
        ),
    )

    result = repo.execute_sql(dataframe=df, request=request)

    assert result.table_name == "input_table"
    assert result.executed_statements == (
        "SELECT a, b FROM input_table ORDER BY a DESC LIMIT 2",
    )
    assert result.columns == ("a", "b")
    assert result.row_count == 2
    assert result.has_result_set is True
    assert result.elapsed_ms >= 0.0
    assert result.dataframe.to_dict(orient="records") == [{"a": 5, "b": 6}, {"a": 3, "b": 4}]


def test_execute_sql_mutating_statement_returns_count_result_set() -> None:
    repo = DuckDBWorkingDataRepo()
    df = pd.DataFrame([{"a": 1}, {"a": 2}])
    request = WorkingDataSQLRequest(
        table_name="t",
        statements=("DELETE FROM t WHERE a = 999",),
    )

    result = repo.execute_sql(dataframe=df, request=request)

    assert result.has_result_set is True
    assert result.columns == ("Count",)
    assert result.row_count == 1
    assert result.dataframe.to_dict(orient="records") == [{"Count": 0}]


def test_execute_sql_rejects_invalid_table_name() -> None:
    repo = DuckDBWorkingDataRepo()
    df = pd.DataFrame([{"a": 1}])
    request = WorkingDataSQLRequest(
        table_name="bad table",
        statements=("SELECT * FROM \"bad table\"",),
    )

    with pytest.raises(ValueError, match=r"Invalid table_name"):
        repo.execute_sql(dataframe=df, request=request)


def test_execute_sql_rejects_dataframe_with_duplicate_columns() -> None:
    repo = DuckDBWorkingDataRepo()
    df = pd.DataFrame([[1, 2]], columns=["x", "x"])
    request = WorkingDataSQLRequest(
        table_name="t",
        statements=("SELECT * FROM t",),
    )

    with pytest.raises(ValueError, match=r"duplicate column names"):
        repo.execute_sql(dataframe=df, request=request)


def test_execute_sql_rejects_dataframe_without_columns() -> None:
    repo = DuckDBWorkingDataRepo()
    df = pd.DataFrame(index=[0, 1, 2])
    request = WorkingDataSQLRequest(
        table_name="t",
        statements=("SELECT 1",),
    )

    with pytest.raises(ValueError, match=r"at least one column"):
        repo.execute_sql(dataframe=df, request=request)


def test_execute_sql_reloads_table_when_new_dataframe_is_passed() -> None:
    repo = DuckDBWorkingDataRepo()
    request = WorkingDataSQLRequest(table_name="t", statements=("SELECT SUM(a) AS total FROM t",))

    result_1 = repo.execute_sql(
        dataframe=pd.DataFrame([{"a": 1}, {"a": 2}]),
        request=request,
    )
    result_2 = repo.execute_sql(
        dataframe=pd.DataFrame([{"a": 10}, {"a": 20}]),
        request=request,
    )

    assert result_1.dataframe.to_dict(orient="records") == [{"total": 3.0}]
    assert result_2.dataframe.to_dict(orient="records") == [{"total": 30.0}]
