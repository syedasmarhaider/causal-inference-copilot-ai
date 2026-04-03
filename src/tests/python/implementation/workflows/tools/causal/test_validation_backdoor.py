from __future__ import annotations

import pandas as pd
import pytest

from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.encoding.encoding_plan import (
    DateTimeEpochParams,
    TransformPlan,
)
from python.implementation.workflows.tools.causal.validation import (
    validation_backdoor_tool as validator_module,
)
from python.implementation.workflows.tools.causal.validation.validation_backdoor_tool import (
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
                "segment": "A" if index % 4 in {0, 1} else "B",
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


def _build_binary_outcome_causal_spec(*, experiment_type: str, covariates: list[str], effect_modifiers: list[str]) -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "treatment_spec": {
                "kind": "binary",
                "column": "treatment",
                "treated": "1",
                "control": "0",
            },
            "outcome_spec": {
                "kind": "binary",
                "column": "outcome",
                "event": "1",
                "non_event": "0",
            },
            "covariates": covariates,
            "effect_modifiers": effect_modifiers,
            "experiment_type": experiment_type,
        }
    )


def _get_issue(report, message_prefix: str):
    return next(issue for issue in report.issues if issue.message.startswith(message_prefix))


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
    assert any(issue.message == "Observational studies require covariate for adjustment." for issue in report.issues)


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


def test_validate_backdoor_effect_modifier_missing_with_passthrough_fails() -> None:
    dataframe = _build_dataframe()
    dataframe.loc[0, "segment"] = None

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=[], effect_modifiers=["segment"]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "segment",
                        "role": "effect_modifier",
                        "encoding": {"preset": "passthrough"},
                    }
                ]
            }
        ),
    )

    issue = _get_issue(report, "Effect modifier has missing values but the transform preset does not explicitly handle them.")
    assert issue.severity == "FAIL"
    assert issue.evidence["preset"] == "passthrough"


def test_validate_backdoor_effect_modifier_missing_with_cat_onehot_impute_token_has_no_missingness_failure() -> None:
    dataframe = _build_dataframe()
    dataframe.loc[0, "segment"] = None

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=[], effect_modifiers=["segment"]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "segment",
                        "role": "effect_modifier",
                        "encoding": {
                            "preset": "cat_onehot",
                            "missing": "impute_token",
                            "missing_token": "__MISSING__",
                        },
                    }
                ]
            }
        ),
    )

    assert not any(
        issue.message == "Effect modifier has missing values but the transform preset does not explicitly handle them."
        for issue in report.issues
    )


def test_validate_backdoor_effect_modifier_missing_with_map_binary_error_fails() -> None:
    dataframe = _build_dataframe()
    dataframe["flag"] = ["Y" if index % 2 == 0 else "N" for index in range(len(dataframe))]
    dataframe.loc[0, "flag"] = None

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=[], effect_modifiers=["flag"]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "flag",
                        "role": "effect_modifier",
                        "encoding": {
                            "preset": "map_binary",
                            "mapping": {"Y": 1.0, "N": 0.0},
                            "missing": "error",
                            "allow_unknown": False,
                        },
                    }
                ]
            }
        ),
    )

    issue = _get_issue(report, "Effect modifier has missing values but the transform preset does not explicitly handle them.")
    assert issue.severity == "FAIL"
    assert issue.evidence["preset"] == "map_binary"


def test_validate_backdoor_effect_modifier_missing_with_map_ordinal_error_fails() -> None:
    dataframe = _build_dataframe()
    dataframe["level"] = ["low" if index % 3 == 0 else "high" for index in range(len(dataframe))]
    dataframe.loc[0, "level"] = None

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=[], effect_modifiers=["level"]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "level",
                        "role": "effect_modifier",
                        "encoding": {
                            "preset": "map_ordinal",
                            "order": ["low", "high"],
                            "missing": "error",
                            "allow_unknown": False,
                        },
                    }
                ]
            }
        ),
    )

    issue = _get_issue(report, "Effect modifier has missing values but the transform preset does not explicitly handle them.")
    assert issue.severity == "FAIL"
    assert issue.evidence["preset"] == "map_ordinal"


def test_validate_backdoor_effect_modifier_missing_with_datetime_preset_fails() -> None:
    dataframe = _build_dataframe()
    dataframe["visit_dt"] = pd.date_range("2025-01-01", periods=len(dataframe), freq="D")
    dataframe.loc[0, "visit_dt"] = pd.NaT

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=[], effect_modifiers=["visit_dt"]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "visit_dt",
                        "role": "effect_modifier",
                        "encoding": {"preset": "datetime_epoch_seconds"},
                    }
                ]
            }
        ),
    )

    issue = _get_issue(report, "Effect modifier has missing values but the transform preset does not explicitly handle them.")
    assert issue.severity == "FAIL"
    assert issue.evidence["preset"] == "datetime_epoch_seconds"


def test_validate_backdoor_covariate_missing_with_passthrough_warns() -> None:
    dataframe = _build_dataframe()
    dataframe.loc[0, "segment"] = None

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["segment"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "segment",
                        "role": "covariate",
                        "encoding": {"preset": "passthrough"},
                    }
                ]
            }
        ),
    )

    issue = _get_issue(report, "Covariate has missing values but the transform preset does not explicitly handle them.")
    assert issue.severity == "WARN"
    assert issue.evidence["preset"] == "passthrough"


def test_validate_backdoor_covariate_missing_with_numeric_imputer_has_no_missingness_warning() -> None:
    dataframe = _build_dataframe()
    dataframe.loc[0, "age"] = None

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

    assert not any(
        issue.message == "Covariate has missing values but the transform preset does not explicitly handle them."
        for issue in report.issues
    )


def test_validate_backdoor_warns_when_category_level_exists_only_in_treated_arm() -> None:
    dataframe = _build_dataframe()
    dataframe["segment"] = "shared"
    dataframe.loc[[0, 2, 4], "segment"] = "treated_only"

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=[], effect_modifiers=["segment"]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "segment",
                        "role": "effect_modifier",
                        "encoding": {"preset": "cat_onehot"},
                    }
                ]
            }
        ),
    )

    issue = next(
        issue
        for issue in report.issues
        if issue.message == "Categorical or mapped column has levels observed in only one treatment arm."
    )
    assert issue.severity == "WARN"
    assert issue.evidence["levels_missing_by_arm"][0]["missing_arms"] == ["control"]


def test_validate_backdoor_warns_when_category_level_exists_only_in_control_arm() -> None:
    dataframe = _build_dataframe()
    dataframe["segment"] = "shared"
    dataframe.loc[[1, 3, 5], "segment"] = "control_only"

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["segment"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "segment",
                        "role": "covariate",
                        "encoding": {"preset": "cat_onehot"},
                    }
                ]
            }
        ),
    )

    issue = next(
        issue
        for issue in report.issues
        if issue.message == "Categorical or mapped column has levels observed in only one treatment arm."
    )
    assert issue.severity == "WARN"
    assert issue.evidence["levels_missing_by_arm"][0]["missing_arms"] == ["treated"]


def test_validate_backdoor_warns_for_low_cardinality_numeric_with_numeric_preset() -> None:
    dataframe = _build_dataframe()
    dataframe["score_code"] = [index % 5 for index in range(len(dataframe))]

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["score_code"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "score_code",
                        "role": "covariate",
                        "encoding": {"preset": "num_standard"},
                    }
                ]
            }
        ),
    )

    issue = next(
        issue
        for issue in report.issues
        if issue.message == "Numeric column has low cardinality and may actually represent coded categories."
    )
    assert issue.severity == "WARN"
    assert issue.evidence["distinct_non_null_count"] == 5


def test_validate_backdoor_does_not_add_low_cardinality_warning_for_numeric_column_with_categorical_preset() -> None:
    dataframe = _build_dataframe()
    dataframe["score_code"] = [index % 5 for index in range(len(dataframe))]

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["score_code"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "score_code",
                        "role": "covariate",
                        "encoding": {"preset": "cat_onehot"},
                    }
                ]
            }
        ),
    )

    assert any(issue.message == "Transform plan preset is incompatible with the observed dataframe column type." for issue in report.issues)
    assert not any(
        issue.message == "Numeric column has low cardinality and may actually represent coded categories."
        for issue in report.issues
    )


def test_validate_backdoor_reports_empty_dataframe_and_missing_columns() -> None:
    dataframe = pd.DataFrame(columns=["treatment", "outcome"])

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=None,
    )

    assert report.status == "FAIL"
    assert any(issue.message == "Dataframe has no rows." for issue in report.issues)
    assert any(issue.message == "Dataset has very few rows for causal estimation." for issue in report.issues)
    assert any(issue.message == "Dataframe is missing columns referenced by the causal spec." for issue in report.issues)


def test_validate_backdoor_allows_missing_transform_plan_for_numeric_only_eligible_columns() -> None:
    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=_build_dataframe(),
        transform_plan=None,
    )

    assert not any(
        issue.message == "Transform plan is required for non-numeric covariates or effect modifiers."
        for issue in report.issues
    )


def test_validate_backdoor_requires_transform_plan_for_non_numeric_eligible_columns() -> None:
    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["segment"], effect_modifiers=[]),
        dataframe=_build_dataframe(),
        transform_plan=None,
    )

    issue = _get_issue(report, "Transform plan is required for non-numeric covariates or effect modifiers.")
    assert issue.severity == "FAIL"
    assert issue.evidence["non_numeric_columns"] == [
        {"column": "segment", "inferred_kind": "CATEGORICAL"}
    ]


def test_validate_backdoor_reports_duplicate_dataframe_columns() -> None:
    dataframe = _build_dataframe()[["treatment", "outcome", "age", "segment"]].copy()
    dataframe.columns = ["treatment", "outcome", "dup", "dup"]

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=[], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=None,
    )

    issue = _get_issue(report, "Dataframe contains duplicate column names.")
    assert issue.severity == "FAIL"
    assert issue.evidence["duplicate_columns"] == ["dup"]


def test_validate_backdoor_reports_causal_spec_overlap_anomalies() -> None:
    report = validate_backdoor(
        causal_spec=_build_causal_spec(
            experiment_type="RCT",
            covariates=["age", "treatment"],
            effect_modifiers=["age", "outcome"],
        ),
        dataframe=_build_dataframe(),
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {"column": "age", "role": "effect_modifier", "encoding": {"preset": "cat_onehot"}},
                    {"column": "treatment", "role": "covariate", "encoding": {"preset": "passthrough"}},
                    {"column": "outcome", "role": "effect_modifier", "encoding": {"preset": "passthrough"}},
                ]
            }
        ),
    )

    assert any(issue.message == "Covariates and effect modifiers overlap." for issue in report.issues)
    assert any(
        issue.message == "Covariates and effect modifiers must not include treatment or outcome columns."
        for issue in report.issues
    )


def test_validate_backdoor_fails_treatment_missing_values() -> None:
    dataframe = _build_dataframe()
    dataframe.loc[0, "treatment"] = None

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}}]}
        ),
    )

    issue = _get_issue(report, "Treatment column contains missing values.")
    assert issue.severity == "FAIL"


def test_validate_backdoor_fails_when_one_treatment_arm_missing() -> None:
    dataframe = _build_dataframe()
    dataframe["treatment"] = "1"

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}}]}
        ),
    )

    issue = _get_issue(report, "Both treatment arms must be present in the dataframe.")
    assert issue.severity == "FAIL"


def test_validate_backdoor_fails_treatment_low_support_and_imbalance() -> None:
    dataframe = _build_dataframe()
    dataframe["treatment"] = ["1"] * 35 + ["0"] * 5

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="OBSERVATIONAL", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}}]}
        ),
    )

    assert any(issue.message == "One treatment arm has a low row count." for issue in report.issues)
    assert any(issue.message == "Treatment-arm imbalance suggests a positivity risk." for issue in report.issues)


def test_validate_backdoor_fails_for_continuous_outcome_missingness() -> None:
    dataframe = _build_dataframe()
    dataframe.loc[0, "outcome"] = None

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}}]}
        ),
    )

    issue = _get_issue(report, "Outcome column has missingness.")
    assert issue.severity == "FAIL"


def test_validate_backdoor_fails_for_continuous_outcome_non_numeric_values() -> None:
    dataframe = _build_dataframe()
    dataframe["outcome"] = ["bad" if index == 0 else str(index) for index in range(len(dataframe))]

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}}]}
        ),
    )

    issue = _get_issue(report, "Continuous outcome column contains non-numeric values.")
    assert issue.severity == "FAIL"


def test_validate_backdoor_warns_for_continuous_outcome_low_variation() -> None:
    dataframe = _build_dataframe()
    dataframe["outcome"] = [1.0, 2.0, 3.0, 4.0] * 10

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}}]}
        ),
    )

    issue = _get_issue(report, "Continuous outcome has very low numeric variation.")
    assert issue.severity == "WARN"


def test_validate_backdoor_fails_for_binary_outcome_with_unexpected_values() -> None:
    dataframe = _build_dataframe()
    dataframe["outcome"] = ["2" if index == 0 else ("1" if index % 2 == 0 else "0") for index in range(len(dataframe))]

    report = validate_backdoor(
        causal_spec=_build_binary_outcome_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}}]}
        ),
    )

    issue = _get_issue(report, "Binary outcome column contains values outside event/non-event literals.")
    assert issue.severity == "FAIL"


def test_validate_backdoor_fails_for_binary_outcome_with_only_one_class() -> None:
    dataframe = _build_dataframe()
    dataframe["outcome"] = "1"

    report = validate_backdoor(
        causal_spec=_build_binary_outcome_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}}]}
        ),
    )

    issue = _get_issue(report, "Binary outcome must contain both event and non-event observations.")
    assert issue.severity == "FAIL"


def test_validate_backdoor_reports_low_binary_outcome_event_count_and_sparse_arms() -> None:
    dataframe = _build_dataframe()
    dataframe["outcome"] = ["1" if index in {0, 2, 4, 6, 8} else "0" for index in range(len(dataframe))]

    report = validate_backdoor(
        causal_spec=_build_binary_outcome_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}}]}
        ),
    )

    assert any(issue.message == "Outcome event count is low." for issue in report.issues)
    assert any(issue.message == "Some treatment arms have very few observed events." for issue in report.issues)


def test_validate_backdoor_fails_when_plan_is_provided_without_eligible_columns() -> None:
    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=[], effect_modifiers=[]),
        dataframe=_build_dataframe(),
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}}]}
        ),
    )

    issue = _get_issue(report, "Transform plan was provided even though there are no covariates or effect modifiers to encode.")
    assert issue.severity == "FAIL"


def test_validate_backdoor_reports_plan_structure_errors() -> None:
    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=["segment"]),
        dataframe=_build_dataframe(),
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {"column": "treatment", "role": "covariate", "encoding": {"preset": "passthrough"}},
                    {"column": "age", "role": "effect_modifier", "encoding": {"preset": "num_standard"}},
                    {"column": "extra_col", "role": "covariate", "encoding": {"preset": "num_standard"}},
                ]
            }
        ),
    )

    assert any(issue.message == "Transform plan must not include treatment or outcome columns." for issue in report.issues)
    assert any(issue.message == "Transform plan is missing eligible columns." for issue in report.issues)
    assert any(issue.message == "Transform plan contains non-eligible columns." for issue in report.issues)
    assert any(issue.message == "Transform plan assigns the wrong role to a column." for issue in report.issues)


def test_validate_backdoor_accepts_partial_transform_plan_when_only_non_numeric_columns_are_covered() -> None:
    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=["segment"]),
        dataframe=_build_dataframe(),
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "segment",
                        "role": "effect_modifier",
                        "encoding": {"preset": "cat_onehot"},
                    }
                ]
            }
        ),
    )

    assert not any(issue.message == "Transform plan is missing eligible columns." for issue in report.issues)
    assert not any(issue.message == "Transform plan failed transformer compilation." for issue in report.issues)


def test_validate_backdoor_fails_when_partial_transform_plan_omits_non_numeric_covariate() -> None:
    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["segment"], effect_modifiers=["age"]),
        dataframe=_build_dataframe(),
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "age",
                        "role": "effect_modifier",
                        "encoding": {"preset": "num_standard"},
                    }
                ]
            }
        ),
    )

    issue = _get_issue(report, "Transform plan is missing eligible columns.")
    assert issue.severity == "FAIL"
    assert issue.evidence["missing_columns"] == ["segment"]


def test_validate_backdoor_fails_when_partial_transform_plan_omits_non_numeric_effect_modifier() -> None:
    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=["segment"]),
        dataframe=_build_dataframe(),
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

    issue = _get_issue(report, "Transform plan is missing eligible columns.")
    assert issue.severity == "FAIL"
    assert issue.evidence["missing_columns"] == ["segment"]


def test_validate_backdoor_fails_for_num_log1p_invalid_values() -> None:
    dataframe = _build_dataframe()
    dataframe["age"] = [-2.0] + [float(index) for index in range(1, len(dataframe))]

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "age", "role": "covariate", "encoding": {"preset": "num_log1p"}}]}
        ),
    )

    issue = _get_issue(report, "num_log1p preset cannot be applied because some values are <= -1.")
    assert issue.severity == "FAIL"


def test_validate_backdoor_reports_datetime_parse_failures() -> None:
    dataframe = _build_dataframe()
    dataframe["visit_dt"] = ["not-a-date" if index == 0 else f"2025-01-{(index % 28) + 1:02d}" for index in range(len(dataframe))]

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=[], effect_modifiers=["visit_dt"]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "visit_dt",
                        "role": "effect_modifier",
                        "encoding": {"preset": "datetime_epoch_seconds", "errors": "raise"},
                    }
                ]
            }
        ),
    )

    issue = _get_issue(report, "Transform plan preset is incompatible with the observed dataframe column type.")
    assert issue.severity == "FAIL"
    assert issue.evidence["preset"] == "datetime_epoch_seconds"


def test_validate_encoding_semantics_reports_datetime_parse_failures_for_datetime_branch() -> None:
    dataframe = _build_dataframe()
    dataframe["visit_dt"] = [f"2025-01-{(index % 28) + 1:02d}" for index in range(len(dataframe))]
    dataframe.loc[5, "visit_dt"] = "not-a-date"

    issues = validator_module._validate_encoding_semantics(
        dataframe=dataframe,
        treatment_spec=_build_causal_spec(experiment_type="RCT", covariates=[], effect_modifiers=[]).treatment_spec,
        column="visit_dt",
        role="effect_modifier",
        inferred_kind="DATETIME",
        encoding=DateTimeEpochParams(preset="datetime_epoch_seconds", errors="raise"),
    )

    issue = next(issue for issue in issues if issue.message.startswith("datetime_epoch_seconds preset cannot parse some datetime values."))
    assert issue.severity == "FAIL"


def test_validate_backdoor_reports_map_binary_missing_token_not_covered() -> None:
    dataframe = _build_dataframe()
    dataframe["flag"] = ["Y" if index % 2 == 0 else "N" for index in range(len(dataframe))]

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["flag"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "flag",
                        "role": "covariate",
                        "encoding": {
                            "preset": "map_binary",
                            "mapping": {"Y": 1.0, "N": 0.0},
                            "missing": "impute_token",
                            "missing_token": "__MISSING__",
                            "allow_unknown": False,
                        },
                    }
                ]
            }
        ),
    )

    issue = _get_issue(report, "map_binary missing_token is not covered by the declared mapping/order.")
    assert issue.severity == "FAIL"


@pytest.mark.parametrize(
    ("preset_payload", "expected_message", "expected_severity"),
    [
        (
            {
                "preset": "map_binary",
                "mapping": {"Y": 1.0, "N": 0.0},
                "missing": "error",
                "allow_unknown": False,
            },
            "map_binary does not cover all observed non-missing values.",
            "FAIL",
        ),
        (
            {
                "preset": "map_ordinal",
                "order": ["low", "high"],
                "missing": "as_unknown",
                "allow_unknown": True,
                "unknown_value": -1,
            },
            "map_ordinal will send some observed values through unknown handling.",
            "WARN",
        ),
    ],
)
def test_validate_backdoor_reports_mapping_anomalies(
    preset_payload: dict[str, object],
    expected_message: str,
    expected_severity: str,
) -> None:
    dataframe = _build_dataframe()
    dataframe["mapped_col"] = ["Y" if index % 3 == 0 else "N" for index in range(len(dataframe))]
    dataframe.loc[0, "mapped_col"] = "UNKNOWN"
    if preset_payload["preset"] == "map_ordinal":
        dataframe["mapped_col"] = ["low" if index % 3 == 0 else "high" for index in range(len(dataframe))]
        dataframe.loc[0, "mapped_col"] = "mystery"

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["mapped_col"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "mapped_col", "role": "covariate", "encoding": preset_payload}]}
        ),
    )

    issue = _get_issue(report, expected_message)
    assert issue.severity == expected_severity


def test_validate_backdoor_warns_for_cat_onehot_max_categories_exceeded() -> None:
    dataframe = _build_dataframe()
    dataframe["segment"] = [f"group_{index}" for index in range(len(dataframe))]

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["segment"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {
                "columns": [
                    {
                        "column": "segment",
                        "role": "covariate",
                        "encoding": {"preset": "cat_onehot", "max_categories": 5},
                    }
                ]
            }
        ),
    )

    issue = _get_issue(report, "cat_onehot preset sees more categories than max_categories.")
    assert issue.severity == "WARN"


def test_validate_backdoor_reports_real_world_column_name_mismatch() -> None:
    dataframe = _build_dataframe().rename(columns={"age": "age_years"})

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}}]}
        ),
    )

    issue = _get_issue(report, "Dataframe is missing columns referenced by the causal spec.")
    assert issue.severity == "FAIL"
    assert issue.evidence["missing_columns"] == ["age"]


def test_validate_backdoor_reports_real_world_dtype_mismatch_for_numeric_preset() -> None:
    dataframe = _build_dataframe()
    dataframe["age"] = ["young" if index % 2 == 0 else "old" for index in range(len(dataframe))]

    report = validate_backdoor(
        causal_spec=_build_causal_spec(experiment_type="RCT", covariates=["age"], effect_modifiers=[]),
        dataframe=dataframe,
        transform_plan=TransformPlan.model_validate(
            {"columns": [{"column": "age", "role": "covariate", "encoding": {"preset": "num_standard"}}]}
        ),
    )

    issue = _get_issue(report, "Transform plan preset is incompatible with the observed dataframe column type.")
    assert issue.severity == "FAIL"
    assert issue.evidence["column"] == "age"
    assert issue.evidence["preset"] == "num_standard"


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
