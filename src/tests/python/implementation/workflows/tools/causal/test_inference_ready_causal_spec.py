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
    )

    assert wrapper.get_effect_modifiers_order() == ["segment"]
    assert wrapper.get_covariates_order() == ["income", "age"]


def test_valid_wrapper_allows_only_covariates() -> None:
    wrapper = InferenceReadyCausalSpec(
        causal_spec=_build_causal_spec(covariates=["age"], effect_modifiers=[]),
        transformation_plan=_build_transform_plan(
            columns=[_num_standard("age", "covariate")]
        ),
    )

    assert wrapper.get_covariates_order() == ["age"]
    assert wrapper.get_effect_modifiers_order() == []


def test_valid_wrapper_allows_only_effect_modifiers() -> None:
    wrapper = InferenceReadyCausalSpec(
        causal_spec=_build_causal_spec(covariates=[], effect_modifiers=["segment", "tier"]),
        transformation_plan=_build_transform_plan(
            columns=[
                _cat_onehot("tier", "effect_modifier"),
                _cat_onehot("segment", "effect_modifier"),
            ]
        ),
    )

    assert wrapper.get_covariates_order() == []
    assert wrapper.get_effect_modifiers_order() == ["tier", "segment"]


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
        )


def test_wrapper_rejects_empty_adjustment_set() -> None:
    with pytest.raises(
        ValidationError,
        match=r"requires at least one covariate or effect_modifier",
    ):
        InferenceReadyCausalSpec(
            causal_spec=_build_causal_spec(covariates=[], effect_modifiers=[]),
            transformation_plan=_build_transform_plan(
                columns=[_num_standard("age", "covariate")]
            ),
        )
