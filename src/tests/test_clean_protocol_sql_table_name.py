from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, TypeVar

from pydantic import BaseModel

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_node import (
    CleanProtocolNode,
)
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import (
    CleanProtocolPayloadModel,
)
from python.implementation.workflows.tools.causal.causal_spec import (
    BinaryOutcomeSpecModel,
    BinaryTreatmentSpecModel,
    CausalSpec,
)
from python.implementation.workflows.tools.data_processing.data_processing_tool import (
    SQLStatements,
)

T = TypeVar("T", bound=BaseModel)


@dataclass
class _SummaryStub:
    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "rows": 3,
            "columns": ["treatment", "outcome", "age"],
        }


@dataclass
class _FakeLLM:
    response: SQLStatements

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Optional[Sequence[ChatMessage]],
    ) -> LLMResponse:
        raise AssertionError("generate should not be called in this test")

    def generate_json(
        self,
        *,
        schema: type[T],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Optional[Sequence[ChatMessage]],
        max_attempts: int = 3,
    ) -> T:
        return self.response  # type: ignore[return-value]


def _causal_spec() -> CausalSpec:
    return CausalSpec(
        treatment_spec=BinaryTreatmentSpecModel(
            kind="binary",
            column="treatment",
            treated="1",
            control="0",
        ),
        outcome_spec=BinaryOutcomeSpecModel(
            kind="binary",
            column="outcome",
            event="1",
            non_event="0",
        ),
        covariates=["age"],
        effect_modifiers=[],
        experiment_type="OBSERVATIONAL",
    )


def test_generate_sql_request_forces_canonical_input_table_name() -> None:
    node = CleanProtocolNode(
        data_repo=object(),
        llm=_FakeLLM(
            response=SQLStatements(
                statements=['SELECT * FROM "cohort_df"'],
                table_name="wrong_alias",
                analytic_only=False,
            )
        ),
    )

    request = node._generate_sql_request(
        mode="MODIFY",
        user_request="drop null rows",
        protocol_discussion="Compare treated vs control patients.",
        causal_spec=_causal_spec(),
        source_summary=_SummaryStub(),
        state=CleanProtocolPayloadModel(),
        history=None,
    )

    assert request.table_name == "cohort_df"


def test_generate_question_sql_forces_canonical_input_table_name() -> None:
    node = CleanProtocolNode(
        data_repo=object(),
        llm=_FakeLLM(
            response=SQLStatements(
                statements=['SELECT COUNT(*) AS n FROM "cohort_df"'],
                table_name="wrong_alias",
                analytic_only=True,
            )
        ),
    )

    request = node._generate_question_sql(
        history=None,
        user_question="How many rows are left?",
        protocol_discussion="Compare treated vs control patients.",
        causal_spec=_causal_spec(),
        source_summary=_SummaryStub(),
        state=CleanProtocolPayloadModel(),
    )

    assert request.table_name == "cohort_df"
