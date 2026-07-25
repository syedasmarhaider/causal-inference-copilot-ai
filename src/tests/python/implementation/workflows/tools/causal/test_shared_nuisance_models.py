from __future__ import annotations

import numpy as np
import pytest
from econml.validate import DRTester
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression

from python.implementation.workflows.tools.causal.inference.econml.dml import (
    shared_nuisance_models as nuisance,
)


class _ScoredProbabilityClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, *, probability: float, score_value: float) -> None:
        self.probability = probability
        self.score_value = score_value

    def fit(self, X: np.ndarray, y: np.ndarray) -> _ScoredProbabilityClassifier:
        _ = (X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = np.full(len(X), self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        _ = (X, y)
        return self.score_value


class _ScoredRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, *, prediction: float, score_value: float) -> None:
        self.prediction = prediction
        self.score_value = score_value

    def fit(self, X: np.ndarray, y: np.ndarray) -> _ScoredRegressor:
        _ = (X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.prediction, dtype=float)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        _ = (X, y)
        return self.score_value


def _features(row_count: int) -> np.ndarray:
    return np.column_stack(
        [
            np.linspace(-2.0, 2.0, row_count),
            np.sin(np.linspace(0.0, 6.0, row_count)),
        ]
    )


def test_binary_outcome_selector_uses_all_candidates_and_returns_probability() -> None:
    selector = nuisance._CrossValidatedBinaryOutcomeRegressor(
        candidates=[
            _ScoredProbabilityClassifier(probability=0.2, score_value=0.1),
            _ScoredProbabilityClassifier(probability=0.73, score_value=0.9),
        ],
        cv=5,
        random_state=11,
    )

    selector.fit(_features(24), np.resize(np.array([0, 1]), 24))

    assert selector.candidate_scores_ == pytest.approx([0.1, 0.9])
    assert selector.predict(_features(4)) == pytest.approx(np.full(4, 0.73))


def test_propensity_selector_refits_the_highest_scoring_candidate() -> None:
    selector = nuisance._CrossValidatedProbabilityClassifier(
        candidates=[
            _ScoredProbabilityClassifier(probability=0.25, score_value=0.2),
            _ScoredProbabilityClassifier(probability=0.8, score_value=0.8),
        ],
        cv=5,
        random_state=13,
    )

    selector.fit(_features(24), np.resize(np.array([0, 1]), 24))
    probabilities = selector.predict_proba(_features(4))

    assert selector.candidate_scores_ == pytest.approx([0.2, 0.8])
    assert probabilities[:, 1] == pytest.approx(np.full(4, 0.8))
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(4))


def test_continuous_outcome_selector_refits_the_best_candidate() -> None:
    selector = nuisance._CrossValidatedRegressor(
        candidates=[
            _ScoredRegressor(prediction=-2.0, score_value=-0.3),
            _ScoredRegressor(prediction=1.5, score_value=0.4),
        ],
        cv=4,
        random_state=17,
    )

    selector.fit(_features(24), np.linspace(-1.0, 1.0, 24))

    assert selector.candidate_scores_ == pytest.approx([-0.3, 0.4])
    assert selector.predict(_features(3)) == pytest.approx(np.full(3, 1.5))


def test_binary_outcome_selector_rejects_one_class_arm() -> None:
    selector = nuisance._CrossValidatedBinaryOutcomeRegressor(
        candidates=[_ScoredProbabilityClassifier(probability=0.5, score_value=0.0)]
    )

    with pytest.raises(ValueError, match="both classes 0 and 1"):
        selector.fit(_features(8), np.zeros(8, dtype=int))


def test_drtester_factory_keeps_the_complete_candidate_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome_candidates = [
        _ScoredProbabilityClassifier(probability=0.3, score_value=0.1),
        _ScoredProbabilityClassifier(probability=0.7, score_value=0.9),
    ]
    treatment_candidates = [
        _ScoredProbabilityClassifier(probability=0.4, score_value=0.2),
        _ScoredProbabilityClassifier(probability=0.6, score_value=0.8),
    ]

    class _OutcomeSpec:
        kind = "binary"

    class _CausalSpec:
        outcome_spec = _OutcomeSpec()

    monkeypatch.setattr(
        nuisance,
        "get_default_models_for_t_and_y",
        lambda *args, **kwargs: {
            "model_y": outcome_candidates,
            "model_t": treatment_candidates,
        },
    )

    model_regression, model_propensity = nuisance.get_drtester_models_for_t_and_y(
        _CausalSpec(),  # type: ignore[arg-type]
        pre_XW=object(),  # type: ignore[arg-type]
        missingness=False,
        random_state=23,
    )

    assert model_regression.candidates is outcome_candidates  # type: ignore[attr-defined]
    assert model_propensity.candidates is treatment_candidates  # type: ignore[attr-defined]


def test_real_drtester_can_clone_and_fit_the_new_selectors() -> None:
    def candidate() -> nuisance._ProbabilityScoredClassifier:
        return nuisance._ProbabilityScoredClassifier(
            LogisticRegression(max_iter=1_000, random_state=19)
        )

    model_regression = nuisance._CrossValidatedBinaryOutcomeRegressor(
        candidates=[candidate()],
        cv=3,
        random_state=19,
    )
    model_propensity = nuisance._CrossValidatedProbabilityClassifier(
        candidates=[candidate()],
        cv=3,
        random_state=19,
    )
    tester = DRTester(
        model_regression=model_regression,
        model_propensity=model_propensity,
        cate=None,
        cv=3,
    )
    train_rows = 160
    validation_rows = 40

    tester.fit_nuisance(
        Xval=_features(validation_rows),
        Dval=np.resize(np.array([0, 1]), validation_rows),
        yval=np.resize(np.array([0, 0, 1, 1]), validation_rows),
        Xtrain=_features(train_rows),
        Dtrain=np.resize(np.array([0, 1, 0, 1]), train_rows),
        ytrain=np.resize(np.array([0, 0, 1, 1]), train_rows),
    )

    assert tester.dr_train_.shape == (train_rows, 1)
    assert tester.dr_val_.shape == (validation_rows, 1)
    assert np.isfinite(tester.dr_train_).all()
    assert np.isfinite(tester.dr_val_).all()
