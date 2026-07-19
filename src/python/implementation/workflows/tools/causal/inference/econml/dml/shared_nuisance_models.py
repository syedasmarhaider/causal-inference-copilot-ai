from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec

# =============================================================================
# Scientifically conservative nuisance-model defaults
# =============================================================================

# Nuisance forests are fit repeatedly during model selection, cross-fitting, and
# Monte Carlo cross-fitting. Five hundred trees is a strong stability/runtime
# compromise; the final causal forest can independently use 1,000 trees.
_NUISANCE_N_ESTIMATORS = 500
_NUISANCE_MIN_SAMPLES_LEAF = 20

# Use a small fixed grid rather than one arbitrary regularization strength.
# EconML compares the resulting models using out-of-fold negative log loss.
_LOGISTIC_C_VALUES = (0.1, 1.0, 10.0)
_LOGISTIC_MAX_ITER = 10_000
_LOGISTIC_TOL = 1e-4

# Sparse and dense regularized linear regression candidates.
_LINEAR_MAX_ITER = 10_000
_RIDGE_ALPHAS = np.logspace(-4, 4, 25)

# Conservative histogram-gradient-boosting defaults.
_HGB_LEARNING_RATE = 0.05
_HGB_MAX_ITER = 400
_HGB_MAX_LEAF_NODES = 15
_HGB_MIN_SAMPLES_LEAF = 20
_HGB_L2_REGULARIZATION = 1.0
_HGB_VALIDATION_FRACTION = 0.10
_HGB_N_ITER_NO_CHANGE = 20
_HGB_TOL = 1e-7


class _ToDense(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to dense arrays for dense-only estimators."""

    def fit(
        self,
        X,
        y=None,
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        _ = (X, y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        if issparse(X):
            return X.toarray()  # type: ignore[no-any-return]
        return np.asarray(X)


class _ProbabilityScoredClassifier(ClassifierMixin, BaseEstimator):
    """
    Classifier adapter whose score is negative log loss.

    EconML selects the best model from a list by comparing each candidate's
    out-of-fold ``score``. Standard sklearn classifiers return accuracy, which
    is inappropriate for selecting nuisance models whose main role is to
    estimate conditional probabilities. This adapter ensures that logistic,
    Random Forest, Extra Trees, and HGB candidates are compared on the same
    proper probability-scoring rule.

    Larger values remain better because this returns ``-log_loss``.
    """

    def __init__(self, model: BaseEstimator):
        self.model = model

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
        **fit_params: Any,
    ) -> _ProbabilityScoredClassifier:
        if sample_weight is None:
            self.model.fit(X, y, **fit_params)
        else:
            self.model.fit(
                X,
                y,
                sample_weight=sample_weight,
                **fit_params,
            )

        self.classes_ = np.asarray(self.model.classes_)

        if hasattr(self.model, "n_features_in_"):
            self.n_features_in_ = int(self.model.n_features_in_)

        return self

    def predict(self, X: Any) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        predict_proba = getattr(self.model, "predict_proba", None)
        if not callable(predict_proba):
            raise AttributeError(f"{type(self.model).__name__} does not implement predict_proba().")
        return predict_proba(X)

    def score(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> float:
        probabilities = self.predict_proba(X)
        return -float(
            log_loss(
                y,
                probabilities,
                labels=self.classes_,
                sample_weight=sample_weight,
            )
        )


def _wrap_with_pre(
    *,
    pre_XW: ColumnTransformer,
    model: BaseEstimator,
    require_dense: bool,
) -> BaseEstimator:
    """Wrap preprocessing and optional sparse-to-dense conversion around a model."""
    steps: list[tuple[str, BaseEstimator]] = [("pre", pre_XW)]
    if require_dense:
        steps.append(("dense", _ToDense()))
    steps.append(("model", model))
    return Pipeline(steps)


def _as_probability_scored_classifier(model: BaseEstimator) -> BaseEstimator:
    """Wrap a probabilistic classifier in the shared negative-log-loss scorer."""
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError(
            f"Discrete nuisance estimator {type(model).__name__} must implement " "predict_proba()."
        )
    return _ProbabilityScoredClassifier(model=model)


def _make_hgb_classifier(
    *,
    random_state: int | None,
) -> BaseEstimator:
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=_HGB_LEARNING_RATE,
        max_iter=_HGB_MAX_ITER,
        max_leaf_nodes=_HGB_MAX_LEAF_NODES,
        max_depth=None,
        min_samples_leaf=_HGB_MIN_SAMPLES_LEAF,
        l2_regularization=_HGB_L2_REGULARIZATION,
        early_stopping=True,
        scoring="loss",
        validation_fraction=_HGB_VALIDATION_FRACTION,
        n_iter_no_change=_HGB_N_ITER_NO_CHANGE,
        tol=_HGB_TOL,
        class_weight=None,
        random_state=random_state,
    )
    return _as_probability_scored_classifier(model)


def _make_hgb_regressor(
    *,
    random_state: int | None,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=_HGB_LEARNING_RATE,
        max_iter=_HGB_MAX_ITER,
        max_leaf_nodes=_HGB_MAX_LEAF_NODES,
        max_depth=None,
        min_samples_leaf=_HGB_MIN_SAMPLES_LEAF,
        l2_regularization=_HGB_L2_REGULARIZATION,
        early_stopping=True,
        scoring="loss",
        validation_fraction=_HGB_VALIDATION_FRACTION,
        n_iter_no_change=_HGB_N_ITER_NO_CHANGE,
        tol=_HGB_TOL,
        random_state=random_state,
    )


def _make_logistic_candidates(
    *,
    random_state: int | None,
    n_jobs: int | None,
) -> Sequence[BaseEstimator]:
    """
    Build a small regularization grid for probabilistic linear nuisance models.

    Each fixed-C candidate is selected by EconML using out-of-fold negative
    log loss through ``_ProbabilityScoredClassifier``. ``class_weight=None`` is
    intentional: balanced class weights would change the fitted probability
    target away from the observed treatment/outcome distribution.
    """
    return [
        _as_probability_scored_classifier(
            LogisticRegression(
                penalty="l2",
                solver="saga",
                C=c_value,
                max_iter=_LOGISTIC_MAX_ITER,
                tol=_LOGISTIC_TOL,
                class_weight=None,
                n_jobs=n_jobs,
                random_state=random_state,
            )
        )
        for c_value in _LOGISTIC_C_VALUES
    ]


def _make_tree_classifier_candidates(
    *,
    random_state: int | None,
    n_jobs: int | None,
) -> Sequence[BaseEstimator]:
    et = ExtraTreesClassifier(
        n_estimators=_NUISANCE_N_ESTIMATORS,
        criterion="log_loss",
        min_samples_leaf=_NUISANCE_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        bootstrap=False,
        class_weight=None,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rf = RandomForestClassifier(
        n_estimators=_NUISANCE_N_ESTIMATORS,
        criterion="log_loss",
        min_samples_leaf=_NUISANCE_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return [
        _as_probability_scored_classifier(et),
        _as_probability_scored_classifier(rf),
    ]


def _make_tree_regressor_candidates(
    *,
    random_state: int | None,
    n_jobs: int | None,
) -> Sequence[BaseEstimator]:
    et = ExtraTreesRegressor(
        n_estimators=_NUISANCE_N_ESTIMATORS,
        criterion="squared_error",
        min_samples_leaf=_NUISANCE_MIN_SAMPLES_LEAF,
        max_features=0.7,
        bootstrap=False,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rf = RandomForestRegressor(
        n_estimators=_NUISANCE_N_ESTIMATORS,
        criterion="squared_error",
        min_samples_leaf=_NUISANCE_MIN_SAMPLES_LEAF,
        max_features=0.7,
        bootstrap=True,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return [et, rf]


def _normalize_model_spec_to_wrapped_list(
    *,
    spec_value: str | BaseEstimator | Sequence[str | BaseEstimator],
    pre_XW: ColumnTransformer,
    is_discrete: bool,
    missingness: bool,
    random_state: int | None,
    n_jobs: int | None,
) -> Sequence[BaseEstimator]:
    """
    Normalize a nuisance-model specification to wrapped sklearn estimators.

    Accepted specifications:
      - Keywords: ``auto``, ``auto_plus``, ``automl``, ``automl_plus``,
        ``linear``, ``forest``, ``trees``, ``gbf``, ``hgb``, ``boosting``.
      - A single sklearn-compatible estimator.
      - A sequence containing keywords and/or estimators.

    Missing-data behavior:
      - If unresolved W missingness is present, only HGB is returned because it
        is explicitly NaN-tolerant across supported sklearn versions.
      - Otherwise, a diverse linear/tree/boosting candidate library is used.

    Selection-score behavior:
      - Discrete candidates expose negative log loss through ``score``.
      - Continuous candidates expose their standard regression score (R²);
        RidgeCV is explicitly configured to select regularization by R² so its
        score remains comparable to the other continuous candidates.
    """
    missing_present = bool(missingness)

    def build_boosting_candidates_nan_safe() -> Sequence[BaseEstimator]:
        model: BaseEstimator
        if is_discrete:
            model = _make_hgb_classifier(random_state=random_state)
        else:
            model = _make_hgb_regressor(random_state=random_state)

        return [
            _wrap_with_pre(
                pre_XW=pre_XW,
                model=model,
                require_dense=True,
            )
        ]

    def build_linear_candidates() -> Sequence[BaseEstimator]:
        if missing_present:
            return build_boosting_candidates_nan_safe()

        if is_discrete:
            return [
                _wrap_with_pre(
                    pre_XW=pre_XW,
                    model=model,
                    require_dense=False,
                )
                for model in _make_logistic_candidates(
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
            ]

        lasso = WeightedLassoCVWrapper(
            random_state=random_state,
            max_iter=_LINEAR_MAX_ITER,
        )
        ridge = RidgeCV(
            alphas=_RIDGE_ALPHAS,
            scoring="r2",
            cv=None,
        )
        return [
            _wrap_with_pre(
                pre_XW=pre_XW,
                model=lasso,  # type: ignore[arg-type]
                require_dense=False,
            ),
            _wrap_with_pre(
                pre_XW=pre_XW,
                model=ridge,
                require_dense=False,
            ),
        ]

    def build_tree_candidates() -> Sequence[BaseEstimator]:
        if missing_present:
            return build_boosting_candidates_nan_safe()

        if is_discrete:
            models = _make_tree_classifier_candidates(
                random_state=random_state,
                n_jobs=n_jobs,
            )
        else:
            models = _make_tree_regressor_candidates(
                random_state=random_state,
                n_jobs=n_jobs,
            )

        return [
            _wrap_with_pre(
                pre_XW=pre_XW,
                model=model,
                require_dense=True,
            )
            for model in models
        ]

    def build_boosting_candidates() -> Sequence[BaseEstimator]:
        return build_boosting_candidates_nan_safe()

    def build_default_candidates() -> Sequence[BaseEstimator]:
        if missing_present:
            return build_boosting_candidates_nan_safe()
        return [
            *build_linear_candidates(),
            *build_tree_candidates(),
            *build_boosting_candidates(),
        ]

    def candidates_for_keyword(key: str) -> Sequence[BaseEstimator]:
        normalized = key.strip().lower()

        if normalized in {"auto", "auto_plus", "automl", "automl_plus"}:
            return build_default_candidates()
        if normalized == "linear":
            return build_linear_candidates()
        if normalized in {"forest", "trees"}:
            return build_tree_candidates()
        if normalized in {"gbf", "hgb", "boosting"}:
            return build_boosting_candidates()

        raise ValueError(f"Unknown model keyword: {key!r}")

    if isinstance(spec_value, (str, BaseEstimator)):
        items: list[str | BaseEstimator] = [spec_value]
    else:
        items = list(spec_value)

    out: list[BaseEstimator] = []

    for item in items:
        if isinstance(item, str):
            out.extend(candidates_for_keyword(item))
            continue

        model: BaseEstimator = item
        if is_discrete:
            model = _as_probability_scored_classifier(model)

        out.append(
            _wrap_with_pre(
                pre_XW=pre_XW,
                model=model,
                require_dense=True,
            )
        )

    if not out:
        raise ValueError("Empty nuisance model candidate list.")

    return out


def get_default_models_for_t_and_y(
    specs: CausalSpec,
    pre_XW: ColumnTransformer,
    missingness: bool,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """
    Build the default DML nuisance candidate libraries.

    ``model_t`` estimates E[T | X, W] or class probabilities for discrete T.
    ``model_y`` estimates E[Y | X, W] or class probabilities for discrete Y.

    The returned lists are consumed by EconML's built-in first-stage model
    selector. Preprocessing remains inside every candidate pipeline, preventing
    preprocessing leakage across cross-fitting folds.
    """
    discrete_treatment = specs.treatment_spec.kind in {"binary", "categorical"}
    discrete_outcome = specs.outcome_spec.kind == "binary"

    default_model_y: str | BaseEstimator | Sequence[str | BaseEstimator] = "auto_plus"
    default_model_t: str | BaseEstimator | Sequence[str | BaseEstimator] = "auto_plus"

    model_y = list(
        _normalize_model_spec_to_wrapped_list(
            spec_value=default_model_y,
            pre_XW=pre_XW,
            is_discrete=discrete_outcome,
            missingness=missingness,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    )

    model_t = list(
        _normalize_model_spec_to_wrapped_list(
            spec_value=default_model_t,
            pre_XW=pre_XW,
            is_discrete=discrete_treatment,
            missingness=missingness,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    )

    return {
        "model_y": model_y,
        "model_t": model_t,
    }


def get_drtester_models_for_t_and_y(
    specs: CausalSpec,
    pre_XW: ColumnTransformer,
    missingness: bool,
    random_state: int | None = None,
) -> tuple[BaseEstimator, BaseEstimator]:
    """Return one outcome and one propensity nuisance model for ``DRTester``.

    EconML DML accepts candidate lists and performs its own selection. DRTester
    accepts one estimator for each nuisance task, so this takes the first shared
    (regularized linear) candidate from the same preprocessed libraries.
    """
    models = get_default_models_for_t_and_y(
        specs,
        pre_XW=pre_XW,
        missingness=missingness,
        random_state=random_state,
    )
    model_y = models["model_y"]
    model_t = models["model_t"]
    return model_y[0], model_t[0]


__all__ = [
    "get_drtester_models_for_t_and_y",
    "get_default_models_for_t_and_y",
]
