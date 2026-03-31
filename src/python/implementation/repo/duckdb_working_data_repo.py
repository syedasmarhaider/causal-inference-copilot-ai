from __future__ import annotations

import re
import time
from dataclasses import dataclass

import duckdb
import pandas as pd

from python.domain.repo.working_data_repo import (
    WorkingDataSQLRequest,
    WorkingDataSQLResult,
    WorkingDatatRepo,
)

_REGISTERED_DF_NAME = "__working_input_dataframe"
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ANALYTIC_ONLY_ALLOWED_START = re.compile(r"^\s*(select|with|values)\b", re.IGNORECASE)


@dataclass(frozen=True)
class DuckDBWorkingDatatRepo(WorkingDatatRepo):
    def execute_sql(
        self,
        *,
        dataframe: pd.DataFrame,
        request: WorkingDataSQLRequest,
    ) -> WorkingDataSQLResult:
        self._validate_dataframe(dataframe)
        self._validate_identifier(str(request.table_name))

        statements = tuple(str(statement).strip() for statement in request.statements)
        table_name = str(request.table_name)

        if request.analytic_only:
            for statement in statements:
                if not self._is_analytic_only_statement(statement):
                    raise ValueError(
                        "analytic_only request cannot include mutating/non-analytic SQL statements"
                    )

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
            last_columns: tuple[str, ...] = ()

            for statement in statements:
                con.execute(statement)

                description = con.description
                has_result_set = bool(description)
                if has_result_set:
                    result_df = con.fetchdf()
                    result_columns = tuple(str(col) for col in result_df.columns)
                else:
                    result_df = pd.DataFrame()
                    result_columns = ()

                last_has_result_set = has_result_set
                last_dataframe = result_df
                last_columns = result_columns

            elapsed_ms = (time.perf_counter() - started) * 1000.0

            return WorkingDataSQLResult(
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

    @staticmethod
    def _is_analytic_only_statement(statement: str) -> bool:
        return bool(_ANALYTIC_ONLY_ALLOWED_START.match(statement))

    @staticmethod
    def _validate_identifier(value: str) -> None:
        if not _IDENT_RE.fullmatch(value):
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
        con.register(_REGISTERED_DF_NAME, dataframe)
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE {self._quote_ident(table_name)} AS "
            f"SELECT * FROM {self._quote_ident(_REGISTERED_DF_NAME)}"
        )

    @staticmethod
    def _quote_ident(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

