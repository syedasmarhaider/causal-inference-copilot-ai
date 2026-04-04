from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.models.models import NonEmptyStr
from python.domain.repo.working_data_repo import WorkingDataRepo, WorkingDataSQLRequest
from python.domain.service.llm_service import AvailableModelsKey, LLMConfig, LLMService
from python.domain.workflows.tool import Tool
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool_prompts import (
    DATA_MANIPULATION_SQL_SYSTEM_PROMPT,
    DATA_MANIPULATION_SQL_USER_PROMPT_TEMPLATE,
)

log = get_app_logger(__name__, component="data_manipulation_tool", log_type="tool")

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DataManipulationSQLPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    EXPECTED_TABLE_NAME: ClassVar[str | None] = None

    statements: list[NonEmptyStr] = Field(min_length=1)
    table_name: NonEmptyStr

    @model_validator(mode="after")
    def _validate_statements(self) -> DataManipulationSQLPlan:
        if not self.statements:
            raise ValueError("statements must contain at least one SQL statement")

        expected_table_name = type(self).EXPECTED_TABLE_NAME
        if expected_table_name is None:
            return self

        returned_table_name = str(self.table_name).strip()
        if returned_table_name != expected_table_name:
            raise ValueError(
                "sql plan table_name mismatch: "
                f"expected='{expected_table_name}' got='{returned_table_name}'"
            )

        if not _statements_reference_table(
            statements=self.statements,
            table_name=expected_table_name,
        ):
            raise ValueError(
                "sql plan does not reference expected table_name "
                f"(table_name='{expected_table_name}')"
            )

        return self

    @classmethod
    def for_table_name(cls, table_name: str) -> type[DataManipulationSQLPlan]:
        normalized_table_name = table_name.strip()
        if not normalized_table_name:
            raise ValueError("table_name must be non-empty")

        return type(
            f"{cls.__name__}For_{normalized_table_name}",
            (cls,),
            {
                "__module__": cls.__module__,
                "EXPECTED_TABLE_NAME": normalized_table_name,
            },
        )


@dataclass(frozen=True)
class DataManipulationTool(Tool):
    llm: LLMService
    working_data_repo: WorkingDataRepo
    model: AvailableModelsKey = "basic"

    NAME: ClassVar[str] = "DATA_MANIPULATION"

    def get_tool_name(self) -> str:
        return self.NAME

    def get_tool_info(self) -> str:
        return (
            "Tool for manipulating or analytically querying tabular data via natural language "
            "instructions. The tool generates DuckDB SQL and executes it against the provided "
            "dataframe. It supports both dataset-changing transformations and read-only "
            "analytical result sets, including grouped statistics, chart-ready aggregations, "
            "windowed calculations, reshaping, and other multi-step SQL workflows."
        )

    def manipulate(
        self,
        *,
        dataframe: pd.DataFrame,
        table_name: str,
        data_summary: str,
        instructions: str,
        retry_attempts: int = 3,
    ) -> pd.DataFrame:
        normalized_table_name = table_name.strip()
        normalized_data_summary = data_summary.strip()
        normalized_instructions = instructions.strip() if instructions and instructions.strip() else ""
        effective_retry_attempts = retry_attempts

        if not normalized_table_name:
            raise ValueError("table_name must be non-empty")
        self._validate_table_name(normalized_table_name)
        if not normalized_data_summary:
            raise ValueError("data_summary must be non-empty")
        if not normalized_instructions:
            raise ValueError("instructions must be non-empty")
        if effective_retry_attempts <= 0:
            raise ValueError("retry_attempts must be >= 1")

        base_user_prompt = DATA_MANIPULATION_SQL_USER_PROMPT_TEMPLATE.format(
            table_name=normalized_table_name,
            user_intent=normalized_instructions,
            data_summary=normalized_data_summary,
        )
        plan_schema = DataManipulationSQLPlan.for_table_name(normalized_table_name)

        log.info(
            "generating data manipulation sql plan",
            table_name=normalized_table_name,
            max_attempts=effective_retry_attempts,
        )
        sql_plan = self.llm.generate_json(
            schema=plan_schema,
            system_prompt=DATA_MANIPULATION_SQL_SYSTEM_PROMPT,
            user_prompt=base_user_prompt,
            config=LLMConfig(model=self.model, temperature=0.0, top_p=1.0),
            history=None,
            max_attempts=effective_retry_attempts,
        )

        statements = tuple(str(statement).strip() for statement in sql_plan.statements)
        sql_request = WorkingDataSQLRequest(
            statements=statements,
            table_name=normalized_table_name,
        )
        sql_result = self.working_data_repo.execute_sql(
            dataframe=dataframe,
            request=sql_request,
        )

        log.info(
            "data manipulation sql executed",
            table_name=sql_request.table_name,
            statements_count=len(sql_request.statements),
            row_count=sql_result.row_count,
            has_result_set=sql_result.has_result_set,
        )
        return sql_result.dataframe

    @staticmethod
    def _validate_table_name(table_name: str) -> None:
        if not _IDENT_RE.fullmatch(table_name):
            raise ValueError(
                f"Invalid table_name '{table_name}'. "
                "Allowed pattern: [A-Za-z_][A-Za-z0-9_]*"
            )

    @staticmethod
    def _statement_references_table(*, statement: str, table_name: str) -> bool:
        return _statement_references_table(statement=statement, table_name=table_name)


def _statement_references_table(*, statement: str, table_name: str) -> bool:
    bare_pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(table_name)}(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    quoted_pattern = re.compile(rf'"{re.escape(table_name)}"', re.IGNORECASE)
    return bool(bare_pattern.search(statement) or quoted_pattern.search(statement))


def _statements_reference_table(*, statements: Sequence[str], table_name: str) -> bool:
    return any(
        _statement_references_table(statement=str(statement), table_name=table_name)
        for statement in statements
    )
