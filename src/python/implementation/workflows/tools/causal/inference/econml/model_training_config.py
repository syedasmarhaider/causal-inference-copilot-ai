from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_MAX_SKLEARN_SEED = 2**32 - 1


@dataclass(frozen=True, slots=True)
class CausalForestTrainingConfig:
    """Shared forest settings for CausalForestDML and ForestDRLearner."""

    n_estimators: int = 2_000
    subforest_size: int = 4
    max_samples: float = 0.45
    min_samples_leaf: int = 20
    min_balancedness_tol: float = 0.45

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_estimators, bool)
            or not isinstance(self.n_estimators, int)
            or self.n_estimators < 1
        ):
            raise ValueError("causal-forest n_estimators must be a positive integer.")
        if (
            isinstance(self.subforest_size, bool)
            or not isinstance(self.subforest_size, int)
            or self.subforest_size < 1
        ):
            raise ValueError("causal-forest subforest_size must be a positive integer.")
        if self.n_estimators % self.subforest_size != 0:
            raise ValueError(
                "causal-forest n_estimators must be divisible by subforest_size "
                "for bootstrap-of-little-bags inference."
            )
        if (
            isinstance(self.max_samples, bool)
            or not isinstance(self.max_samples, (int, float))
            or not 0.0 < self.max_samples <= 0.5
        ):
            raise ValueError(
                "causal-forest max_samples must be in (0, 0.5] when forest inference is enabled."
            )
        if (
            isinstance(self.min_samples_leaf, bool)
            or not isinstance(self.min_samples_leaf, int)
            or self.min_samples_leaf < 1
        ):
            raise ValueError("causal-forest min_samples_leaf must be a positive integer.")
        if (
            isinstance(self.min_balancedness_tol, bool)
            or not isinstance(self.min_balancedness_tol, (int, float))
            or not 0.0 <= self.min_balancedness_tol <= 0.5
        ):
            raise ValueError("causal-forest min_balancedness_tol must be in [0, 0.5].")

    def as_metadata(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "subforest_size": self.subforest_size,
            "max_samples": self.max_samples,
            "min_samples_leaf": self.min_samples_leaf,
            "min_balancedness_tol": self.min_balancedness_tol,
        }


@dataclass(frozen=True, slots=True)
class ModelTrainingConfig:
    """Application-wide statistical training defaults.

    This is intentionally code configuration rather than environment
    configuration. A later clinician-selectable workflow can construct and
    persist a request-scoped instance without changing estimator internals.
    """

    run_seed: int | None = 1729
    outer_cv_cate_folds: int = 10
    causal_forest: CausalForestTrainingConfig = field(default_factory=CausalForestTrainingConfig)

    def __post_init__(self) -> None:
        if self.run_seed is not None and (
            isinstance(self.run_seed, bool)
            or not isinstance(self.run_seed, int)
            or not 0 <= self.run_seed <= _MAX_SKLEARN_SEED
        ):
            raise ValueError(f"run_seed must be None or an integer in [0, {_MAX_SKLEARN_SEED}].")
        if (
            isinstance(self.outer_cv_cate_folds, bool)
            or not isinstance(self.outer_cv_cate_folds, int)
            or not 2 <= self.outer_cv_cate_folds <= 10
        ):
            raise ValueError("outer_cv_cate_folds must be an integer from 2 to 10.")

    def as_metadata(self) -> dict[str, Any]:
        return {
            "run_seed": self.run_seed,
            "outer_cv_cate_folds": self.outer_cv_cate_folds,
            "causal_forest": self.causal_forest.as_metadata(),
        }


MODEL_TRAINING_CONFIG = ModelTrainingConfig()

# Public flat exports retained for simple imports and configuration inspection.
# They are derived from MODEL_TRAINING_CONFIG, which remains the single source
# of truth used by the estimator and validation adapters.
PRECISION_MEDICINE_RUN_SEED = MODEL_TRAINING_CONFIG.run_seed
PRECISION_MEDICINE_ENABLE_OUTER_CV_CATE = MODEL_TRAINING_CONFIG.outer_cv_cate_folds

CAUSAL_FOREST_N_ESTIMATORS = MODEL_TRAINING_CONFIG.causal_forest.n_estimators
CAUSAL_FOREST_SUBFOREST_SIZE = MODEL_TRAINING_CONFIG.causal_forest.subforest_size
CAUSAL_FOREST_MAX_SAMPLES = MODEL_TRAINING_CONFIG.causal_forest.max_samples
CAUSAL_FOREST_MIN_SAMPLES_LEAF = MODEL_TRAINING_CONFIG.causal_forest.min_samples_leaf
CAUSAL_FOREST_MIN_BALANCEDNESS_TOL = MODEL_TRAINING_CONFIG.causal_forest.min_balancedness_tol


__all__ = [
    "CAUSAL_FOREST_MAX_SAMPLES",
    "CAUSAL_FOREST_MIN_BALANCEDNESS_TOL",
    "CAUSAL_FOREST_MIN_SAMPLES_LEAF",
    "CAUSAL_FOREST_N_ESTIMATORS",
    "CAUSAL_FOREST_SUBFOREST_SIZE",
    "MODEL_TRAINING_CONFIG",
    "PRECISION_MEDICINE_ENABLE_OUTER_CV_CATE",
    "PRECISION_MEDICINE_RUN_SEED",
    "CausalForestTrainingConfig",
    "ModelTrainingConfig",
]
