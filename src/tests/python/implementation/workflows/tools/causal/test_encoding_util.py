from __future__ import annotations

from python.implementation.workflows.tools.causal.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.encoding_util import compile_plan_to_transformers


def test_compile_allows_empty_effect_modifiers_when_covariates_exist() -> None:
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

    compiled = compile_plan_to_transformers(
        plan=plan,
        effect_modifiers=[],
        covariates=["age"],
        dense_output=True,
        require_full_coverage=True,
    )

    assert compiled.pre_X is None
    assert compiled.pre_XW is not None
