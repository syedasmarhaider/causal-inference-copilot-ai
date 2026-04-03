from __future__ import annotations

import re
from collections.abc import Sequence

from typing import ClassVar
from uuid import UUID

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.models.models import NonEmptyStr
from python.domain.repo.working_data_repo import WorkingDataSQLRequest, WorkingDatatRepo
from python.domain.service.llm_service import AvailableModelsKey, LLMConfig, LLMService
from python.domain.workflows.tool import Tool
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.node_service.data_manupulation_service.data_manipulation_prompts import DATA_MANIPULATION_SQL_SYSTEM_PROMPT, DATA_MANIPULATION_SQL_USER_PROMPT_TEMPLATE

log = get_app_logger(__name__, component="data_manipulation_tool", log_type="tool")


class DataManipulationSQLPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    statements: list[NonEmptyStr] = Field(min_length=1)
    table_name: NonEmptyStr

    @model_validator(mode="after")
    def _validate_statements(self) -> DataManipulationSQLPlan:
        if not self.statements:
            raise ValueError("statements must contain at least one SQL statement")
        return self


class DataManipulationTool(Tool):
    llm: LLMService
    working_data_repo: WorkingDatatRepo
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
        conversation_id: str,
        data_summary: str,
        instructions: str | None = None,
    ) -> pd.DataFrame:
        normalized_conversation_id = conversation_id.strip()
        normalized_data_summary = data_summary.strip()
        normalized_table_name = self._sanitize_table_name(normalized_conversation_id)
        normalized_instructions = instructions.strip() if instructions and instructions.strip() else ""

        if not normalized_conversation_id:
            raise ValueError("conversation_id must be non-empty")
        if not normalized_data_summary:
            raise ValueError("data_summary must be non-empty")

        user_prompt = DATA_MANIPULATION_SQL_USER_PROMPT_TEMPLATE.format(
            table_name=normalized_table_name,
            user_intent=(
                normalized_instructions
                if normalized_instructions
                else "No explicit user instruction provided."
            ),
            data_summary=normalized_data_summary,
        )

        log.info(
            "generating data manipulation sql plan",
            table_name=normalized_table_name,
        )
        sql_plan = self.llm.generate_json(
            schema=DataManipulationSQLPlan,
            system_prompt=DATA_MANIPULATION_SQL_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            config=LLMConfig(model=self.model, temperature=0.0, top_p=1.0),
            history=None,
            max_attempts=self.max_attempts,
        )

        returned_table_name = str(sql_plan.table_name).strip()
        if returned_table_name != normalized_table_name:
            raise ValueError(
                "sql plan table_name mismatch: "
                f"expected='{normalized_table_name}' got='{returned_table_name}'"
            )

        statements = tuple(str(statement).strip() for statement in sql_plan.statements)
        self._assert_table_presence_in_statements(
            statements=statements,
            table_name=normalized_table_name,
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
        )
        return sql_result.dataframe

    @staticmethod
    def _sanitize_table_name(value: str) -> str:
        try:
            parsed_uuid = UUID(value)
            return f"conv_{parsed_uuid.hex[:12]}"
        except ValueError:
            pass

        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
        if not sanitized:
            return "working_data"
        if not re.match(r"^[A-Za-z_]", sanitized):
            sanitized = f"t_{sanitized}"
        return sanitized

    def _assert_table_presence_in_statements(
        self,
        *,
        statements: Sequence[str],
        table_name: str,
    ) -> None:
        for idx, statement in enumerate(statements, start=1):
            if not self._statement_references_table(statement=statement, table_name=table_name):
                raise ValueError(
                    "sql statement does not reference expected table_name "
                    f"(index={idx}, table_name='{table_name}')"
                )

    @staticmethod
    def _statement_references_table(*, statement: str, table_name: str) -> bool:
        bare_pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(table_name)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        quoted_pattern = re.compile(rf'"{re.escape(table_name)}"', re.IGNORECASE)
        return bool(bare_pattern.search(statement) or quoted_pattern.search(statement))
