from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel, ValidationError

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_node import (
    CleanProtocolNode,
    _IntentGateModel,
    _prepend_dataset_scope_note,
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
    responses: list[SQLStatements]
    calls: int = 0

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
    ) -> LLMResponse:
        raise AssertionError("generate should not be called in this test")

    def generate_json(
        self,
        *,
        schema: type[T],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> T:
        if not self.responses:
            raise AssertionError("No fake LLM responses left")
        self.calls += 1
        return self.responses.pop(0)  # type: ignore[return-value]


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
    llm = _FakeLLM(
        responses=[
            SQLStatements(
                statements=['SELECT * FROM "cohort_df"'],
                table_name="wrong_alias",
                analytic_only=False,
            )
        ]
    )
    node = CleanProtocolNode(
        data_repo=object(),
        llm=llm,
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
    assert llm.calls == 1


def test_generate_question_sql_forces_canonical_input_table_name() -> None:
    llm = _FakeLLM(
        responses=[
            SQLStatements(
                statements=['SELECT COUNT(*) AS n FROM "cohort_df"'],
                table_name="wrong_alias",
                analytic_only=True,
            )
        ]
    )
    node = CleanProtocolNode(
        data_repo=object(),
        llm=llm,
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
    assert llm.calls == 1


def test_generate_sql_request_repairs_implicit_complete_case_filtering() -> None:
    llm = _FakeLLM(
        responses=[
            SQLStatements(
                statements=[
                    'SELECT "treatment", "outcome", "age" FROM "cohort_df" WHERE "age" IS NOT NULL'
                ],
                table_name="cohort_df",
                analytic_only=False,
            ),
            SQLStatements(
                statements=[
                    'SELECT "treatment", "outcome", "age" FROM "cohort_df"'
                ],
                table_name="cohort_df",
                analytic_only=False,
            ),
        ]
    )
    node = CleanProtocolNode(
        data_repo=object(),
        llm=llm,
    )

    request = node._generate_sql_request(
        mode="MODIFY",
        user_request="standardize treatment values only",
        protocol_discussion="Compare treated vs control patients.",
        causal_spec=_causal_spec(),
        source_summary=_SummaryStub(),
        state=CleanProtocolPayloadModel(),
        history=None,
    )

    assert llm.calls == 2
    assert request.statements == ['SELECT "treatment", "outcome", "age" FROM "cohort_df"']


def test_generate_sql_request_allows_explicit_complete_case_filtering() -> None:
    llm = _FakeLLM(
        responses=[
            SQLStatements(
                statements=[
                    'SELECT "treatment", "outcome", "age" FROM "cohort_df" WHERE "age" IS NOT NULL'
                ],
                table_name="cohort_df",
                analytic_only=False,
            )
        ]
    )
    node = CleanProtocolNode(
        data_repo=object(),
        llm=llm,
    )

    request = node._generate_sql_request(
        mode="MODIFY",
        user_request="drop rows with missing age before we proceed",
        protocol_discussion="Compare treated vs control patients.",
        causal_spec=_causal_spec(),
        source_summary=_SummaryStub(),
        state=CleanProtocolPayloadModel(),
        history=None,
    )

    assert llm.calls == 1
    assert request.statements == [
        'SELECT "treatment", "outcome", "age" FROM "cohort_df" WHERE "age" IS NOT NULL'
    ]


def test_intent_gate_requires_dataset_scope_for_modify_and_question() -> None:
    with pytest.raises(ValidationError):
        _IntentGateModel.model_validate(
            {
                "action": "MODIFY",
                "reason": "user wants another filter",
                "reply_to_user": "Applying another filter.",
                "revert_target": None,
            }
        )

    parsed = _IntentGateModel.model_validate(
        {
            "action": "ANSWER_QUESTION",
            "reason": "user asks about original upload",
            "reply_to_user": "Answering from the original uploaded dataset.",
            "dataset_scope": "ORIGINAL_DATASET",
            "revert_target": None,
        }
    )

    assert parsed.dataset_scope == "ORIGINAL_DATASET"


def test_dataset_scope_note_is_prepended_to_user_message() -> None:
    note = _prepend_dataset_scope_note(
        message="I kept the same columns and only changed treatment encoding.",
        dataset_scope="CURRENT_DATASET",
        operation="MODIFY",
    )

    assert note.startswith("Cleaning scope: current cleaned dataset.")
