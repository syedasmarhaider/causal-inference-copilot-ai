from __future__ import annotations

import pandas as pd

from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.validation import (
    validation_backdoor as validator_module,
)
from python.implementation.workflows.tools.causal.validation.validation_backdoor import (
    validate_backdoor,
)


def _build_dataframe() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(40):
        rows.append(
            {
                "treatment": "1" if index % 2 == 0 else "0",
                "outcome": float(index + 1),
                "age": 30 + index,
                "segment": "A" if index < 20 else "B",
            }
        )
    return pd.DataFrame(rows)


def _build_causal_spec(*, experiment_type: str, covariates: list[str], effect_modifiers: list[str]) -> CausalSpec:
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
                "unit": "score",
            },
            "covariates": covariates,
            "effect_modifiers": effect_modifiers,
            "experiment_type": experiment_type,
        }
    )


def test_validate_backdoor_accepts_rct_without_covariates() -> None:
    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=[], effect_modifiers=[]),
        dataframe=_build_dataframe(),
        transform_plan=None,
    )

    assert report.status == "WARN"
    assert any(issue.message == "RCT has no covariates; this is acceptable but limits precision gains." for issue in report.issues)
    assert not any(issue.severity == "FAIL" for issue in report.issues)


def test_validate_backdoor_fails_observational_without_covariates() -> None:
    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="OBSERVATIONAL", covariates=[], effect_modifiers=[]),
        dataframe=_build_dataframe(),
        transform_plan=None,
    )

    assert report.status == "FAIL"
    assert any(issue.message == "Observational studies require at least one covariate for adjustment." for issue in report.issues)


def test_validate_backdoor_reports_transform_compile_failure_as_issue() -> None:
    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=[], effect_modifiers=["segment"]),
        dataframe=_build_dataframe(),
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "segment",
                        "role": "effect_modifier",
                        "encoding": {"preset": "drop"},
                    }
                ]
            }
        ),
    )

    assert report.status == "FAIL"
    compile_issue = next(issue for issue in report.issues if issue.message == "Transform plan failed transformer compilation.")
    assert compile_issue.severity == "FAIL"
    assert "dropped" in str(compile_issue.evidence.get("error")).lower()


def test_validate_backdoor_reports_invalid_treatment_values() -> None:
    dataframe = _build_dataframe()
    dataframe.loc[0, "treatment"] = "2"

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "age",
                        "role": "covariate",
                        "encoding": {"preset": "num_standard"},
                    }
                ]
            }
        ),
    )

    assert report.status == "FAIL"
    treatment_issue = next(
        issue for issue in report.issues if issue.message == "Treatment column contains values outside the declared treated/control literals."
    )
    assert treatment_issue.severity == "FAIL"
    assert treatment_issue.evidence["unexpected_values"] == ["num:2.0"]


def test_validate_backdoor_converts_internal_step_error_to_fail_issue(monkeypatch) -> None:
    def _boom(*args: object, **kwargs: object) -> list[object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(validator_module, "_validate_causal_spec", _boom)

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=[], effect_modifiers=[]),
        dataframe=_build_dataframe(),
        transform_plan=None,
    )

    guarded_issue = next(issue for issue in report.issues if issue.message == "Causal spec validation failed unexpectedly.")
    assert guarded_issue.severity == "FAIL"
    assert guarded_issue.evidence["step"] == "causal spec"
    assert "RuntimeError('boom')" in str(guarded_issue.evidence["error"])
