from __future__ import annotations

import contextlib
import re
import threading
import time
from dataclasses import dataclass, field

import duckdb
import pandas as pd

from python.domain.repo.analytics_repo import AnalyticsRepo, AnalyticsSQLRequest, AnalyticsSQLResult



_REGISTERED_DF_NAME = "__working_input_dataframe"
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class DuckDBAnalyticsRepo(AnalyticsRepo):
    database: str = ":memory:"
    _connection: duckdb.DuckDBPyConnection = field(init=False, repr=False)
    _lock: threading.RLock = field(init=False, repr=False, default_factory=threading.RLock)
    _closed: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_connection", duckdb.connect(self.database))

    def execute_sql(
        self,
        *,
        dataframe: pd.DataFrame,
        request: AnalyticsSQLRequest,
    ) -> AnalyticsSQLResult:
        self._validate_dataframe(dataframe)
        self._validate_identifier(str(request.table_name))

        statements = tuple(str(statement).strip() for statement in request.statements)
        table_name = str(request.table_name)

        with self._lock:
            if self._closed:
                raise RuntimeError("DuckDBWorkingDataRepo is closed")

            self._register_dataframe_as_table(
                con=self._connection,
                dataframe=dataframe,
                table_name=table_name,
            )

            started = time.perf_counter()

            last_has_result_set = False
            last_dataframe = pd.DataFrame()
            last_columns: tuple[str, ...] = ()

            for statement in statements:
                self._connection.execute(statement)

                description = self._connection.description
                has_result_set = bool(description)
                if has_result_set:
                    result_df = self._connection.fetchdf()
                    result_columns = tuple(str(col) for col in result_df.columns)
                else:
                    result_df = pd.DataFrame()
                    result_columns = ()

                last_has_result_set = has_result_set
                last_dataframe = result_df
                last_columns = result_columns

            elapsed_ms = (time.perf_counter() - started) * 1000.0

            return AnalyticsSQLResult(
                table_name=table_name,
                executed_statements=statements,
                columns=last_columns,
                row_count=int(len(last_dataframe)),
                has_result_set=last_has_result_set,
                elapsed_ms=elapsed_ms,
                dataframe=last_dataframe,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            with contextlib.suppress(Exception):
                self._connection.close()
            object.__setattr__(self, "_closed", True)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

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
        try:
            con.execute(
                f"CREATE OR REPLACE TEMP TABLE {self._quote_ident(table_name)} AS "
                f"SELECT * FROM {self._quote_ident(_REGISTERED_DF_NAME)}"
            )
        finally:
            with contextlib.suppress(Exception):
                con.unregister(_REGISTERED_DF_NAME)

    @staticmethod
    def _quote_ident(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'
