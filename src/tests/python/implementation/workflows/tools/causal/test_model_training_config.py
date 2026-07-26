from __future__ import annotations

import pytest

from python.implementation.workflows.tools.causal.inference.econml import (
    model_training_config,
)
from python.implementation.workflows.tools.causal.inference.econml.dml._base_run_dml import (
    _configured_run_seed as configured_dml_run_seed,
)
from python.implementation.workflows.tools.causal.inference.econml.dml._base_run_dml import (
    set_causal_forest_defaults as set_dml_causal_forest_defaults,
)
from python.implementation.workflows.tools.causal.inference.econml.dml.validate_dml import (
    resolve_outer_cv_folds as resolve_dml_outer_cv_folds,
)
from python.implementation.workflows.tools.causal.inference.econml.dr._base_run_dr import (
    _configured_run_seed as configured_dr_run_seed,
)
from python.implementation.workflows.tools.causal.inference.econml.dr._base_run_dr import (
    set_causal_forest_defaults as set_dr_causal_forest_defaults,
)
from python.implementation.workflows.tools.causal.inference.econml.dr.validate_dr import (
    resolve_outer_cv_folds as resolve_dr_outer_cv_folds,
)
from python.implementation.workflows.tools.causal.inference.econml.model_training_config import (
    CausalForestTrainingConfig,
    ModelTrainingConfig,
)
from python.implementation.workflows.tools.causal.inference.econml.utils import (
    ModelSpecError,
)


def test_global_model_training_config_exports_requested_values() -> None:
    config = model_training_config.MODEL_TRAINING_CONFIG

    assert config.run_seed == 1729
    assert config.outer_cv_cate_folds == 10
    assert config.causal_forest == CausalForestTrainingConfig(
        n_estimators=10_000,
        subforest_size=10,
        max_samples=0.50,
        min_samples_leaf=50,
        min_balancedness_tol=0.45,
    )
    assert config.run_seed == model_training_config.PRECISION_MEDICINE_RUN_SEED
    assert (
        config.outer_cv_cate_folds == model_training_config.PRECISION_MEDICINE_ENABLE_OUTER_CV_CATE
    )
    assert config.causal_forest.n_estimators == model_training_config.CAUSAL_FOREST_N_ESTIMATORS
    assert config.causal_forest.subforest_size == model_training_config.CAUSAL_FOREST_SUBFOREST_SIZE
    assert config.causal_forest.max_samples == model_training_config.CAUSAL_FOREST_MAX_SAMPLES
    assert (
        config.causal_forest.min_samples_leaf
        == model_training_config.CAUSAL_FOREST_MIN_SAMPLES_LEAF
    )
    assert (
        config.causal_forest.min_balancedness_tol
        == model_training_config.CAUSAL_FOREST_MIN_BALANCEDNESS_TOL
    )


def test_seed_and_outer_folds_ignore_retired_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("PRECISION_MEDICINE_RUN_SEED", "999")
    monkeypatch.setenv("PRECISION_MEDICINE_ENABLE_OUTER_CV_CATE", "2")

    assert configured_dml_run_seed() == 1729
    assert configured_dr_run_seed() == 1729
    assert resolve_dml_outer_cv_folds() == 10
    assert resolve_dr_outer_cv_folds() == 10


def test_dml_and_dr_forest_helpers_use_the_shared_config() -> None:
    supported = {
        "random_state",
        "cv",
        "mc_iters",
        "mc_agg",
        "n_estimators",
        "subforest_size",
        "max_samples",
        "min_samples_leaf",
        "honest",
        "inference",
        "criterion",
        "min_balancedness_tol",
        "n_jobs",
    }
    init_map = {name: object() for name in supported}
    dml_defaults: dict[str, object] = {}
    dr_defaults: dict[str, object] = {}

    set_dml_causal_forest_defaults(dml_defaults, init_map, run_seed=1729)
    set_dr_causal_forest_defaults(dr_defaults, init_map, run_seed=1729)

    expected_forest = model_training_config.MODEL_TRAINING_CONFIG.causal_forest.as_metadata()
    for key, expected in expected_forest.items():
        assert dml_defaults[key] == expected
        assert dr_defaults[key] == expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_estimators": 10.0}, "n_estimators"),
        ({"subforest_size": 0}, "subforest_size"),
        (
            {"n_estimators": 10_001, "subforest_size": 10},
            "divisible by subforest_size",
        ),
        ({"max_samples": 0.51}, "max_samples"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"min_balancedness_tol": 0.51}, "min_balancedness_tol"),
    ],
)
def test_causal_forest_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CausalForestTrainingConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_seed": -1},
        {"run_seed": 2**32},
        {"outer_cv_cate_folds": 1},
        {"outer_cv_cate_folds": 11},
    ],
)
def test_model_training_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ModelTrainingConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("resolver", [resolve_dml_outer_cv_folds, resolve_dr_outer_cv_folds])
def test_outer_fold_resolver_allows_explicit_test_override(resolver) -> None:
    assert resolver(configured=2) == 2
    with pytest.raises(ModelSpecError, match="outer_cv_cate_folds"):
        resolver(configured=1)
