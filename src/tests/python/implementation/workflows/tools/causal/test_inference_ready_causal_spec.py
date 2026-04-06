from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import (
    TransformPlan,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


def _causal_spec_payload(
    *,
    covariates: list[str] | None = None,
    effect_modifiers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "treatment_spec": {
            "kind": "binary",
            "column": "treatment",
            "treated": "drug",
            "control": "placebo",
        },
        "outcome_spec": {
            "kind": "continuous",
            "column": "outcome",
            "unit": "score",
        },
        "covariates": covariates if covariates is not None else ["age", "income"],
        "effect_modifiers": effect_modifiers if effect_modifiers is not None else ["segment"],
        "experiment_type": "OBSERVATIONAL",
    }


def _transform_plan_payload(*, columns: list[dict[str, Any]]) -> dict[str, Any]:
    return {"columns": columns}


def _numeric_profile(name: str, *, n_missing: int = 0) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "float64",
        "n_rows": 10,
        "n_missing": n_missing,
        "missing_rate": n_missing / 10,
        "distinct_count": 10,
        "inferred_kind": "NUMERIC",
        "summary": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.1, "quantiles": None},
    }


def _categorical_profile(name: str, values: list[str], *, n_missing: int = 0) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "object",
        "n_rows": 10,
        "n_missing": n_missing,
        "missing_rate": n_missing / 10,
        "distinct_count": len(values),
        "inferred_kind": "CATEGORICAL",
        "summary": {
            "top_categories": [{"value": value, "count": 5} for value in values],
            "other_count": 0,
        },
    }


def _summary_model(*profiles: dict[str, Any]) -> DatasetSummaryModel:
    return DatasetSummaryModel.model_validate({"n_rows": 10, "profiles": list(profiles)})


def _num_standard(column: str, role: str) -> dict[str, Any]:
    return {
        "column": column,
        "role": role,
        "encoding": {"preset": "num_standard"},
    }


def _cat_onehot(column: str, role: str) -> dict[str, Any]:
    return {
        "column": column,
        "role": role,
        "encoding": {"preset": "cat_onehot"},
    }


def _build_causal_spec(
    *,
    covariates: list[str] | None = None,
    effect_modifiers: list[str] | None = None,
) -> CausalSpec:
    return CausalSpec.model_validate(
        _causal_spec_payload(
            covariates=covariates,
            effect_modifiers=effect_modifiers,
        )
    )


def _build_transform_plan(*, columns: list[dict[str, Any]]) -> TransformPlan:
    return TransformPlan.model_validate(_transform_plan_payload(columns=columns))


def _build_data_summary(
    *,
    age_missing: int = 0,
    income_missing: int = 0,
    include_income: bool = True,
) -> DatasetSummaryModel:
    profiles: list[dict[str, Any]] = [
        _categorical_profile("treatment", ["drug", "placebo"]),
        _numeric_profile("outcome"),
        _numeric_profile("age", n_missing=age_missing),
        _categorical_profile("segment", ["A", "B"]),
    ]
    if include_income:
        profiles.insert(3, _numeric_profile("income", n_missing=income_missing))
    return _summary_model(*profiles)


def test_valid_wrapper_derives_orders_from_transformation_plan() -> None:
    wrapper = InferenceReadyCausalSpec(
        causal_spec=_build_causal_spec(),
        transformation_plan=_build_transform_plan(
            columns=[
                _cat_onehot("segment", "effect_modifier"),
                _num_standard("income", "covariate"),
                _num_standard("age", "covariate"),
            ]
        ),
        data_summary=_build_data_summary(),
    )

    assert wrapper.get_effect_modifiers_order() == ["segment"]
    assert wrapper.get_covariates_order() == ["income", "age"]
    assert wrapper.has_adjustment_columns() is True
    assert wrapper.has_covariates() is True
    assert wrapper.has_effect_modifiers() is True
    assert wrapper.is_covariates_missing() is False
    assert wrapper.is_effect_modifiers_missing() is False


def test_valid_wrapper_allows_only_covariates() -> None:
    wrapper = InferenceReadyCausalSpec(
        causal_spec=_build_causal_spec(covariates=["age"], effect_modifiers=[]),
        transformation_plan=_build_transform_plan(columns=[_num_standard("age", "covariate")]),
        data_summary=_build_data_summary(include_income=False),
    )

    assert wrapper.get_covariates_order() == ["age"]
    assert wrapper.get_effect_modifiers_order() == []
    assert wrapper.has_covariates() is True
    assert wrapper.has_effect_modifiers() is False


def test_valid_wrapper_allows_only_effect_modifiers() -> None:
    wrapper = InferenceReadyCausalSpec(
        causal_spec=_build_causal_spec(covariates=[], effect_modifiers=["segment", "tier"]),
        transformation_plan=_build_transform_plan(
            columns=[
                _cat_onehot("tier", "effect_modifier"),
                _cat_onehot("segment", "effect_modifier"),
            ]
        ),
        data_summary=_summary_model(
            _categorical_profile("treatment", ["drug", "placebo"]),
            _numeric_profile("outcome"),
            _categorical_profile("segment", ["A", "B"]),
            _categorical_profile("tier", ["gold", "silver"]),
        ),
    )

    assert wrapper.get_covariates_order() == []
    assert wrapper.get_effect_modifiers_order() == ["tier", "segment"]
    assert wrapper.has_covariates() is False
    assert wrapper.has_effect_modifiers() is True


def test_wrapper_rejects_missing_covariate_in_plan() -> None:
    with pytest.raises(ValidationError, match=r"missing causal_spec covariates"):
        InferenceReadyCausalSpec(
            causal_spec=_build_causal_spec(),
            transformation_plan=_build_transform_plan(
                columns=[
                    _cat_onehot("segment", "effect_modifier"),
                    _num_standard("age", "covariate"),
                ]
            ),
            data_summary=_build_data_summary(),
        )


def test_wrapper_rejects_missing_effect_modifier_in_plan() -> None:
    with pytest.raises(ValidationError, match=r"missing causal_spec effect_modifiers"):
        InferenceReadyCausalSpec(
            causal_spec=_build_causal_spec(),
            transformation_plan=_build_transform_plan(
                columns=[
                    _num_standard("age", "covariate"),
                    _num_standard("income", "covariate"),
                ]
            ),
            data_summary=_build_data_summary(),
        )


def test_wrapper_rejects_extra_plan_columns() -> None:
    with pytest.raises(ValidationError, match=r"contains columns outside causal_spec"):
        InferenceReadyCausalSpec(
            causal_spec=_build_causal_spec(),
            transformation_plan=_build_transform_plan(
                columns=[
                    _cat_onehot("segment", "effect_modifier"),
                    _num_standard("income", "covariate"),
                    _num_standard("age", "covariate"),
                    _num_standard("score", "covariate"),
                ]
            ),
            data_summary=_build_data_summary(),
        )


def test_wrapper_rejects_plan_including_treatment_or_outcome() -> None:
    with pytest.raises(ValidationError, match=r"must not include treatment or outcome"):
        InferenceReadyCausalSpec(
            causal_spec=_build_causal_spec(),
            transformation_plan=_build_transform_plan(
                columns=[
                    _cat_onehot("segment", "effect_modifier"),
                    _num_standard("income", "covariate"),
                    _num_standard("age", "covariate"),
                    _cat_onehot("treatment", "covariate"),
                ]
            ),
            data_summary=_build_data_summary(),
        )

    with pytest.raises(ValidationError, match=r"must not include treatment or outcome"):
        InferenceReadyCausalSpec(
            causal_spec=_build_causal_spec(),
            transformation_plan=_build_transform_plan(
                columns=[
                    _cat_onehot("segment", "effect_modifier"),
                    _num_standard("income", "covariate"),
                    _num_standard("age", "covariate"),
                    _num_standard("outcome", "covariate"),
                ]
            ),
            data_summary=_build_data_summary(),
        )


def test_wrapper_rejects_wrong_plan_roles() -> None:
    with pytest.raises(ValidationError, match=r"assigned roles inconsistent with causal_spec"):
        InferenceReadyCausalSpec(
            causal_spec=_build_causal_spec(),
            transformation_plan=_build_transform_plan(
                columns=[
                    _cat_onehot("segment", "covariate"),
                    _num_standard("income", "covariate"),
                    _num_standard("age", "effect_modifier"),
                ]
            ),
            data_summary=_build_data_summary(),
        )


def test_wrapper_rejects_empty_adjustment_set() -> None:
    with pytest.raises(
        ValidationError,
        match=r"requires at least one covariate or effect_modifier",
    ):
        InferenceReadyCausalSpec(
            causal_spec=_build_causal_spec(covariates=[], effect_modifiers=[]),
            transformation_plan=_build_transform_plan(columns=[_num_standard("age", "covariate")]),
            data_summary=_build_data_summary(include_income=False),
        )


def test_is_covariates_missing_returns_true_when_any_covariate_profile_has_missing_values() -> None:
    wrapper = InferenceReadyCausalSpec(
        causal_spec=_build_causal_spec(),
        transformation_plan=_build_transform_plan(
            columns=[
                _cat_onehot("segment", "effect_modifier"),
                _num_standard("income", "covariate"),
                _num_standard("age", "covariate"),
            ]
        ),
        data_summary=_build_data_summary(age_missing=2),
    )

    assert wrapper.is_covariates_missing() is True


def test_wrapper_rejects_data_summary_missing_referenced_columns() -> None:
    with pytest.raises(
        ValidationError, match=r"data_summary is missing causal_spec/transformation_plan columns"
    ):
        InferenceReadyCausalSpec(
            causal_spec=_build_causal_spec(),
            transformation_plan=_build_transform_plan(
                columns=[
                    _cat_onehot("segment", "effect_modifier"),
                    _num_standard("income", "covariate"),
                    _num_standard("age", "covariate"),
                ]
            ),
            data_summary=_build_data_summary(include_income=False),
        )


def test_wrapper_classifies_missingness_handling_from_summary_and_plan() -> None:
    wrapper = InferenceReadyCausalSpec(
        causal_spec=_build_causal_spec(),
        transformation_plan=_build_transform_plan(
            columns=[
                {
                    "column": "segment",
                    "role": "effect_modifier",
                    "encoding": {"preset": "cat_onehot", "missing": "error"},
                },
                {
                    "column": "income",
                    "role": "covariate",
                    "encoding": {"preset": "passthrough"},
                },
                {
                    "column": "age",
                    "role": "covariate",
                    "encoding": {"preset": "num_standard"},
                },
            ]
        ),
        data_summary=_summary_model(
            _categorical_profile("treatment", ["drug", "placebo"]),
            _numeric_profile("outcome"),
            _numeric_profile("age", n_missing=1),
            _numeric_profile("income", n_missing=2),
            _categorical_profile("segment", ["A", "B"], n_missing=3),
        ),
    )

    assert wrapper.get_covariates_with_missing() == ["income", "age"]
    assert wrapper.get_effect_modifiers_with_missing() == ["segment"]
    assert wrapper.get_covariates_with_unhandled_missing() == ["income"]
    assert wrapper.get_covariates_with_forbidden_missing() == []
    assert wrapper.get_effect_modifiers_with_unhandled_missing() == []
    assert wrapper.get_effect_modifiers_with_forbidden_missing() == ["segment"]
    assert wrapper.requires_allow_missing_for_covariates() is True
    assert wrapper.has_unhandled_missing_effect_modifiers() is False


def test_wrapper_asserts_forbidden_missingness_via_high_level_methods() -> None:
    wrapper = InferenceReadyCausalSpec(
        causal_spec=_build_causal_spec(),
        transformation_plan=_build_transform_plan(
            columns=[
                {
                    "column": "segment",
                    "role": "effect_modifier",
                    "encoding": {"preset": "cat_onehot", "missing": "error"},
                },
                {
                    "column": "income",
                    "role": "covariate",
                    "encoding": {"preset": "passthrough"},
                },
                {
                    "column": "age",
                    "role": "covariate",
                    "encoding": {"preset": "cat_onehot", "missing": "error"},
                },
            ]
        ),
        data_summary=_summary_model(
            _categorical_profile("treatment", ["drug", "placebo"]),
            _numeric_profile("outcome"),
            _categorical_profile("age", ["young", "old"], n_missing=1),
            _numeric_profile("income", n_missing=2),
            _categorical_profile("segment", ["A", "B"], n_missing=3),
        ),
    )

    with pytest.raises(ValueError, match=r"Covariates contain missing values"):
        wrapper.assert_covariates_missingness_is_allowed()

    with pytest.raises(ValueError, match=r"Effect modifiers contain missing values"):
        wrapper.assert_effect_modifiers_missingness_is_allowed()
