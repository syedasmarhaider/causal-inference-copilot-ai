from __future__ import annotations

from collections.abc import Sequence

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer

from python.implementation.workflows.tools.causal.inference.econml.dml.shared_nuisance_models import (
    _CrossValidatedBinaryOutcomeRegressor,
    _CrossValidatedProbabilityClassifier,
    _CrossValidatedRegressor,
    _ProbabilityScoredClassifier,
)
from python.implementation.workflows.tools.causal.inference.econml.dr._base_run_dr import (
    _build_propensity_candidates,
    _build_regression_candidates,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec

_DRTESTER_SELECTION_CV = 5


def _probability_scored_candidates(
    candidates: Sequence[BaseEstimator],
) -> list[BaseEstimator]:
    """Compare DRTester classifiers using probabilities rather than accuracy."""
    return [_ProbabilityScoredClassifier(model=candidate) for candidate in candidates]


def get_drtester_models_for_t_and_y(
    specs: CausalSpec,
    *,
    pre_XW: ColumnTransformer,
    n_xw: int,
    missingness: bool,
    random_state: int | None = None,
) -> tuple[BaseEstimator, BaseEstimator]:
    """Build validation-only nuisance selectors from the DR candidate families.

    The fitted DRLearner is not changed. These selectors are cloned and fitted
    only by ``DRTester`` while it constructs held-out doubly robust scores.

    DRLearner's outcome candidates normally receive ``[X, W, T]``. DRTester
    instead fits one outcome regression per treatment arm, so ``n_xw`` covers
    the complete input and the same candidate families operate on ``[X, W]``.
    """
    propensity_candidates = _build_propensity_candidates(
        pre_XW=pre_XW,
        missingness_W=missingness,
        random_state=random_state,
        n_jobs=1,
    )
    outcome_candidates = _build_regression_candidates(
        pre_XW=pre_XW,
        n_xw=n_xw,
        discrete_outcome=specs.outcome_spec.kind == "binary",
        missingness_W=missingness,
        random_state=random_state,
        n_jobs=1,
    )

    model_propensity: BaseEstimator = _CrossValidatedProbabilityClassifier(
        candidates=_probability_scored_candidates(propensity_candidates),
        cv=_DRTESTER_SELECTION_CV,
        random_state=random_state,
    )
    if specs.outcome_spec.kind == "binary":
        model_regression: BaseEstimator = _CrossValidatedBinaryOutcomeRegressor(
            candidates=_probability_scored_candidates(outcome_candidates),
            cv=_DRTESTER_SELECTION_CV,
            random_state=random_state,
        )
    else:
        model_regression = _CrossValidatedRegressor(
            candidates=outcome_candidates,
            cv=_DRTESTER_SELECTION_CV,
            random_state=random_state,
        )

    return model_regression, model_propensity


__all__ = ["get_drtester_models_for_t_and_y"]
