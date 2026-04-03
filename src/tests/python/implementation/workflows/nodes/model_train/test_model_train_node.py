from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.implementation.workflows.nodes.model_train.model_train_node import (
    UserPlanInput,
    _generate_encoding_plan,
    _validate_plan_against_constraints,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
    NumericColumnProfileModel,
    NumericSummaryModel,
)

T = TypeVar("T", bound=BaseModel)


@dataclass
class _GeneratePlanLLMStub:
    plan_response: TransformPlan
    triage_calls: int = 0
    plan_calls: int = 0
    plan_prompts: list[str] = field(default_factory=list)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
    ) -> LLMResponse:
        raise AssertionError("generate() should not be called in this test")

    def generate_json(
        self,
        *,
        schema: type[T],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> T:
        if schema is UserPlanInput:
            self.triage_calls += 1
            return UserPlanInput(
                needs_user_input=False,
                message="I have enough information to proceed with planning.",
            )  # type: ignore[return-value]

        if schema is TransformPlan:
            self.plan_calls += 1
            self.plan_prompts.append(user_prompt)
            return self.plan_response  # type: ignore[return-value]

        raise AssertionError(f"Unexpected schema requested: {schema}")


def _build_numeric_summary(*column_names: str) -> DatasetSummaryModel:
    return DatasetSummaryModel(
        n_rows=5,
        profiles=[
            NumericColumnProfileModel(
                name=column_name,
                dtype="float64",
                n_rows=5,
                n_missing=1,
                missing_rate=0.2,
                distinct_count=4,
                inferred_kind="NUMERIC",
                summary=NumericSummaryModel(min=1.0, max=5.0),
            )
            for column_name in column_names
        ],
    )


def _build_causal_spec() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "treatment_spec": {
                "kind": "binary",
                "column": "treatment",
                "treated": "1",
                "control": "0",
            },
            "outcome_spec": {
                "kind": "continuous",
                "column": "outcome",
                "unit": "days",
            },
            "covariates": ["age", "sex_code"],
            "effect_modifiers": [],
            "experiment_type": "OBSERVATIONAL",
        }
    )


def test_validate_plan_rejects_incompatible_numeric_preset() -> None:
    plan = TransformPlan.model_validate(
        {
            "columns": [
                {
                    "column": "age",
                    "role": "covariate",
                    "encoding": {"preset": "cat_onehot"},
                }
            ]
        }
    )
    summary = _build_numeric_summary("age")

    issues = _validate_plan_against_constraints(
        plan=plan,
        dataset_summary=summary,
        eligible_cols={"age"},
        expected_covariate_cols={"age"},
        expected_effect_modifier_cols=set(),
        treatment_col="treatment",
        outcome_col="outcome",
    )

    incompatibility_issue = next(
        issue for issue in issues if issue.message == "Encoding plan has column type and preset incompatibilities."
    )
    assert incompatibility_issue.severity == "FAIL"
    assert incompatibility_issue.evidence == {
        "incompatibilities": [
            {
                "column": "age",
                "inferred_kind": "NUMERIC",
                "preset": "cat_onehot",
            }
        ]
    }


def test_validate_plan_accepts_compatible_numeric_preset() -> None:
    plan = TransformPlan.model_validate(
        {
            "columns": [
                {
                    "column": "age",
                    "role": "covariate",
                    "encoding": {"preset": "num_standard"},
                }
            ]
        }
    )
    summary = _build_numeric_summary("age")

    issues = _validate_plan_against_constraints(
        plan=plan,
        dataset_summary=summary,
        eligible_cols={"age"},
        expected_covariate_cols={"age"},
        expected_effect_modifier_cols=set(),
        treatment_col="treatment",
        outcome_col="outcome",
    )

    assert issues == []


def test_user_plan_input_accepts_legacy_user_message_key() -> None:
    payload = UserPlanInput.model_validate(
        {
            "needs_user_input": True,
            "user_message": "Need clarification for encoding choices.",
        }
    )

    assert payload.needs_user_input is True
    assert payload.message == "Need clarification for encoding choices."


def test_generate_encoding_plan_requests_user_input_after_repeated_invalid_plan_attempts() -> None:
    llm = _GeneratePlanLLMStub(
        plan_response=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "age",
                        "role": "covariate",
                        "encoding": {"preset": "num_standard"},
                    },
                    {
                        "column": "sex_code",
                        "role": "covariate",
                        "encoding": {"preset": "cat_onehot"},
                    },
                ]
            }
        )
    )

    discussion, plan = _generate_encoding_plan(
        llm=llm,
        causal_specs=_build_causal_spec(),
        selected_model="econml.dml.LinearDML",
        dataset_summary=_build_numeric_summary("age", "sex_code"),
        prev_training_error=None,
        documentation="fit docs",
        history=None,
    )

    assert plan is None
    assert discussion.needs_user_input is True
    assert "column types" in discussion.message.lower()
    assert llm.triage_calls == 2
    assert llm.plan_calls == 2
    assert len(llm.plan_prompts) == 2
    assert "following issues" in llm.plan_prompts[1]
