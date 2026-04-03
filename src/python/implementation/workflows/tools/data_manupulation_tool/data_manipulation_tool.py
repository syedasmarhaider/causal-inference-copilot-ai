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

    statements: list[NonEmptyStr] = Field(min_length=1)
    table_name: NonEmptyStr

    @model_validator(mode="after")
    def _validate_statements(self) -> DataManipulationSQLPlan:
        if not self.statements:
            raise ValueError("statements must contain at least one SQL statement")
        return self


@dataclass(frozen=True)
class DataManipulationTool(Tool):
    llm: LLMService
    working_data_repo: WorkingDataRepo
    model: AvailableModelsKey = "basic"
    max_attempts: int = 2

    NAME: ClassVar[str] = "DATA_MANIPULATION"

    def get_tool_name(self) -> str:
        return self.NAME

    def get_tool_info(self) -> str:
        return "Tool for manipulating tabular data via natural language instructions. The tool generates SQL statements to perform the requested manipulations and executes them against the provided dataframe. The output is a new dataframe resulting from the SQL operations."

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be >= 1")

    def manipulate(
        self,
        *,
        dataframe: pd.DataFrame,
        table_name: str,
        data_summary: str,
        instructions: str,
        retry_attempts: int | None = None,
    ) -> pd.DataFrame:
        normalized_table_name = table_name.strip()
        normalized_data_summary = data_summary.strip()
        normalized_instructions = instructions.strip() if instructions and instructions.strip() else ""
        effective_retry_attempts = retry_attempts if retry_attempts is not None else self.max_attempts

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

        current_user_prompt = base_user_prompt
        last_error: Exception | None = None
        last_plan: DataManipulationSQLPlan | None = None

        for attempt in range(1, effective_retry_attempts + 1):
            log.info(
                "generating data manipulation sql plan",
                table_name=normalized_table_name,
                attempt=attempt,
                max_attempts=effective_retry_attempts,
            )
            try:
                sql_plan = self.llm.generate_json(
                    schema=DataManipulationSQLPlan,
                    system_prompt=DATA_MANIPULATION_SQL_SYSTEM_PROMPT,
                    user_prompt=current_user_prompt,
                    config=LLMConfig(model=self.model, temperature=0.0, top_p=1.0),
                    history=None,
                    max_attempts=1,
                )
                last_plan = sql_plan

                statements = tuple(str(statement).strip() for statement in sql_plan.statements)
                self._validate_plan(
                    plan=sql_plan,
                    statements=statements,
                    expected_table_name=normalized_table_name,
                )

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
                    attempt=attempt,
                )
                return sql_result.dataframe
            except Exception as exc:
                last_error = exc
                if attempt >= effective_retry_attempts:
                    break

                log.warning(
                    "retrying invalid data manipulation sql plan",
                    table_name=normalized_table_name,
                    attempt=attempt,
                    max_attempts=effective_retry_attempts,
                    error=str(exc),
                )
                current_user_prompt = self._build_retry_user_prompt(
                    base_user_prompt=base_user_prompt,
                    expected_table_name=normalized_table_name,
                    error=exc,
                    invalid_plan=last_plan,
                )

        assert last_error is not None
        raise RuntimeError(
            f"Failed data manipulation plan after {effective_retry_attempts} attempts. "
            f"Last error: {last_error}"
        ) from last_error

    @staticmethod
    def _validate_table_name(table_name: str) -> None:
        if not _IDENT_RE.fullmatch(table_name):
            raise ValueError(
                f"Invalid table_name '{table_name}'. "
                "Allowed pattern: [A-Za-z_][A-Za-z0-9_]*"
            )

    def _validate_plan(
        self,
        *,
        plan: DataManipulationSQLPlan,
        statements: Sequence[str],
        expected_table_name: str,
    ) -> None:
        returned_table_name = str(plan.table_name).strip()
        if returned_table_name != expected_table_name:
            raise ValueError(
                "sql plan table_name mismatch: "
                f"expected='{expected_table_name}' got='{returned_table_name}'"
            )
        self._assert_plan_references_source_table(
            statements=statements,
            table_name=expected_table_name,
        )

    @staticmethod
    def _build_retry_user_prompt(
        *,
        base_user_prompt: str,
        expected_table_name: str,
        error: Exception,
        invalid_plan: DataManipulationSQLPlan | None,
    ) -> str:
        invalid_plan_text = (
            invalid_plan.model_dump_json(indent=2)
            if invalid_plan is not None
            else "<no valid JSON plan was produced>"
        )
        return (
            f"{base_user_prompt}\n\n"
            "Your previous SQL plan was invalid. Fix it and return a corrected JSON plan.\n\n"
            "Additional hard rules:\n"
            f"- table_name MUST be exactly '{expected_table_name}'\n"
            f"- The SQL plan MUST reference '{expected_table_name}' at least once\n"
            "- Temp tables or CTEs derived from that input table are allowed\n"
            "- Output ONLY strict JSON matching the schema\n\n"
            f"Validation error:\n{error}\n\n"
            f"Previous invalid plan:\n{invalid_plan_text}"
        )

    def _assert_plan_references_source_table(
        self,
        *,
        statements: Sequence[str],
        table_name: str,
    ) -> None:
        if any(
            self._statement_references_table(statement=statement, table_name=table_name)
            for statement in statements
        ):
            return

        raise ValueError(
            "sql plan does not reference expected table_name "
            f"(table_name='{table_name}')"
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
