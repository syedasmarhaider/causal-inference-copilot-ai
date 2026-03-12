from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import ClassVar, List

import duckdb
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.models.models import NonEmptyStr
from python.domain.workflows.tool import Tool


class SQLStatements(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    statements: List[NonEmptyStr] = Field(min_length=1)
    table_name: NonEmptyStr
    analytic_only: bool

    @model_validator(mode="after")
    def _validate_statements(self) -> "SQLStatements":
        if not self.statements:
            raise ValueError("statements must contain at least one SQL statement")
        return self


@dataclass(frozen=True)
class DuckDBSQLExecutionResult:
    table_name: str
    executed_statements: List[str]
    columns: List[str]
    row_count: int
    has_result_set: bool
    elapsed_ms: float
    dataframe: pd.DataFrame


@dataclass(frozen=True)
class DuckDBInMemorySQLTool(Tool):
    NAME: ClassVar[str] = "DUCKDB_IN_MEMORY_SQL"
    _REGISTERED_DF_NAME: ClassVar[str] = "__input_dataframe"
    _IDENT_RE: ClassVar[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def get_tool_name(self) -> str:
        return self.NAME

    def get_tool_info(self) -> str:
        return (
            "Runs SQL statements against an in-memory DuckDB database. "
            "The provided pandas DataFrame is automatically materialized as a temp table "
            "using the requested table_name. Statements are executed in order, and the "
            "final statement result is returned as a pandas DataFrame."
        )

    def execute(
        self,
        *,
        dataframe: pd.DataFrame,
        sql_request: SQLStatements,
    ) -> DuckDBSQLExecutionResult:
        self._validate_dataframe(dataframe)
        self._validate_identifier(str(sql_request.table_name))

        statements = [str(statement).strip() for statement in sql_request.statements]
        table_name = str(sql_request.table_name)

        con = duckdb.connect(":memory:")
        try:
            self._register_dataframe_as_table(
                con=con,
                dataframe=dataframe,
                table_name=table_name,
            )

            started = time.perf_counter()

            last_has_result_set = False
            last_dataframe = pd.DataFrame()
            last_columns: List[str] = []

            for statement in statements:
                con.execute(statement)

                description = con.description
                has_result_set = bool(description)

                if has_result_set:
                    result_df = con.fetchdf()
                    result_columns = [str(col) for col in result_df.columns]
                else:
                    result_df = pd.DataFrame()
                    result_columns = []

                last_has_result_set = has_result_set
                last_dataframe = result_df
                last_columns = result_columns

            elapsed_ms = (time.perf_counter() - started) * 1000.0

            return DuckDBSQLExecutionResult(
                table_name=table_name,
                executed_statements=statements,
                columns=last_columns,
                row_count=int(len(last_dataframe)),
                has_result_set=last_has_result_set,
                elapsed_ms=elapsed_ms,
                dataframe=last_dataframe,
            )
        finally:
            con.close()

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame) -> None:
        if len(dataframe.columns) == 0:
            raise ValueError("dataframe must have at least one column")

        duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
        if duplicate_columns:
            duplicate_columns = [str(col) for col in duplicate_columns]
            raise ValueError(
                f"dataframe has duplicate column names: {duplicate_columns}"
            )

    @classmethod
    def _validate_identifier(cls, value: str) -> None:
        if not cls._IDENT_RE.fullmatch(value):
            raise ValueError(
                f"Invalid table_name '{value}'. "
                "Allowed pattern: [A-Za-z_][A-Za-z0-9_]*"
            )

    def _register_dataframe_as_table(
        self,
        *,
        con: duckdb.DuckDBPyConnection,
        dataframe: pd.DataFrame,
        table_name: str,
    ) -> None:
        con.register(self._REGISTERED_DF_NAME, dataframe)
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE {self._quote_ident(table_name)} AS "
            f"SELECT * FROM {self._quote_ident(self._REGISTERED_DF_NAME)}"
        )

    @staticmethod
    def _quote_ident(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'