from __future__ import annotations

from python.domain.models.validation import ValidationIssueModel
from python.implementation.workflows.nodes.data_compilation.data_compilation_valiation import (
    _build_user_suggestion_message,
)
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
