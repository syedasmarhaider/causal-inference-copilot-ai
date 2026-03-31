from __future__ import annotations

from python.implementation.workflows.nodes.model_train.model_train_node import (
    _validate_plan_against_constraints,
)
from python.implementation.workflows.tools.causal.encoding_plan import TransformPlan
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
    NumericColumnProfileModel,
    NumericSummaryModel,
)


def _build_numeric_summary(column_name: str) -> DatasetSummaryModel:
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
        ],
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
