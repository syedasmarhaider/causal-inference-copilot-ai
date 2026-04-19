from __future__ import annotations

import pandas as pd

from python.domain.models.validation import ValidationIssueModel
from python.implementation.workflows.nodes.data_compilation.data_compilation_valiation import (
    _build_user_suggestion_message,
    validate_data_compilation,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec


def _causal_spec() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "treatment_spec": {
                "kind": "binary",
                "column": "treatment",
                "treated": "drug",
                "control": "control",
            },
            "outcome_spec": {
                "kind": "binary",
                "column": "outcome",
                "event": "event",
                "non_event": "non_event",
            },
            "covariates": ["age"],
            "effect_modifiers": ["segment"],
            "experiment_type": "OBSERVATIONAL",
            "id_col": "patient_id",
        }
    )


def test_build_user_suggestion_message_mentions_locked_identifier_column() -> None:
    message = _build_user_suggestion_message(
        causal_spec=_causal_spec(),
        issues=[
            ValidationIssueModel(
                severity="FAIL",
                message="Transform plan must not include identifier, treatment, or outcome columns, but found: patient_id (identifier).",
                fix_hint="Remove patient_id from the transform plan.",
            )
        ],
    )

    assert message is not None
    assert "Locked identifier column: patient_id" in message


def test_validate_data_compilation_fails_when_covariate_missingness_remains() -> None:
    rows: list[dict[str, object]] = []
    for index in range(60):
        rows.append(
            {
                "patient_id": f"p{index}",
                "treatment": "drug" if index % 2 == 0 else "control",
                "outcome": "event" if index % 3 == 0 else "non_event",
                "age": None if index == 0 else 30 + index,
                "segment": "A" if index % 2 == 0 else "B",
            }
        )
    dataframe = pd.DataFrame(rows)
    transform_plan = TransformPlan.model_validate(
        {
            "columns": [
                {
                    "column": "age",
                    "role": "covariate",
                    "encoding": {"preset": "num_standard"},
                },
                {
                    "column": "segment",
                    "role": "effect_modifier",
                    "encoding": {"preset": "cat_onehot"},
                },
            ]
        }
    )

    result = validate_data_compilation(
        candidate_df=dataframe,
        causal_spec=_causal_spec(),
        transform_plan=transform_plan,
    )

    assert any(
        issue.severity == "FAIL"
        and "Protocol-scope column 'age' (covariate) still contains missing values after cleaning."
        in issue.message
        for issue in result.validation_errors
    )
    assert result.user_suggestion_message is None
