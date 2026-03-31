from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from python.domain.models.models import NonEmptyStr


@dataclass(frozen=True)
class WorkingDataSQLRequest:
    statements: Sequence[NonEmptyStr]
    table_name: NonEmptyStr

    def __post_init__(self) -> None:
        if len(self.statements) == 0:
            raise ValueError("statements must contain at least one SQL statement")
        if not str(self.table_name).strip():
            raise ValueError("table_name must be non-empty")
        for statement in self.statements:
            if not str(statement).strip():
                raise ValueError("statements must not contain empty SQL statements")


@dataclass(frozen=True)
class WorkingDataSQLResult:
    table_name: str
    executed_statements: tuple[str, ...]
    columns: tuple[str, ...]
    row_count: int
    has_result_set: bool
    elapsed_ms: float
    dataframe: pd.DataFrame


class WorkingDatatRepo(Protocol):
    """
    Analytical SQL execution over tabular data.

    Implementations may execute in memory (DuckDB) or via external engines.
    """

    def execute_sql(
        self,
        *,
        dataframe: pd.DataFrame,
        request: WorkingDataSQLRequest,
    ) -> WorkingDataSQLResult:
        """
        Execute SQL statements against the provided dataframe and return the
        final statement result plus execution metadata.
        """
        ...
