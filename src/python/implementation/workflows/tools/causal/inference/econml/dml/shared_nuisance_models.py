from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin
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
from sklearn.pipeline import Pipeline

from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec


class _ToDense(BaseEstimator, TransformerMixin):
    """Convert sparse -> dense for models that don't accept sparse."""

    def fit(
        self, X, y=None
    ):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return self

    def transform(self, X: Any) -> np.ndarray:
        if issparse(X):
            return X.toarray()  # type: ignore[no-any-return]
        return X


def _wrap_with_pre(
    *,
    pre_XW: ColumnTransformer,
    model: BaseEstimator,
    require_dense: bool,
) -> BaseEstimator:
    steps: list[tuple[str, BaseEstimator]] = [("pre", pre_XW)]
    if require_dense:
        steps.append(("dense", _ToDense()))
    steps.append(("model", model))
    return Pipeline(steps)


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
    Accepts: keyword ('auto', 'automl', 'linear'...), estimator, or list of these.
    Returns: list of fully wrapped sklearn estimators (Pipeline(pre -> [dense] -> model)).

    missingness:
      - True: restrict to NaN-tolerant candidates (HGB), avoiding models that error on NaNs.
      - False: usual candidate menu.
    """
    missing_present = missingness

    def build_boosting_candidates_nan_safe() -> Sequence[BaseEstimator]:
        if is_discrete:
            hgb = HistGradientBoostingClassifier(
                random_state=random_state,
                max_depth=None,
                learning_rate=0.05,
                max_iter=4000,
                early_stopping=True,
            )
            return [_wrap_with_pre(pre_XW=pre_XW, model=hgb, require_dense=True)]

        hgb = HistGradientBoostingRegressor(
            random_state=random_state,
            max_depth=None,
            learning_rate=0.05,
            max_iter=4000,
            early_stopping=True,
        )
        return [_wrap_with_pre(pre_XW=pre_XW, model=hgb, require_dense=True)]

    def build_linear_candidates() -> Sequence[BaseEstimator]:
        if missing_present:
            return build_boosting_candidates_nan_safe()

        if is_discrete:
            lr = LogisticRegression(
                penalty="l2",
                solver="saga",
                max_iter=10000,
                C=0.1,
                class_weight="balanced",
                n_jobs=n_jobs if n_jobs is not None else -1,
                random_state=random_state,
            )
            return [_wrap_with_pre(pre_XW=pre_XW, model=lr, require_dense=False)]

        lasso = WeightedLassoCVWrapper(random_state=random_state)
        ridge = RidgeCV(alphas=np.logspace(-4, 4, 25))
        return [
            _wrap_with_pre(pre_XW=pre_XW, model=lasso, require_dense=False),  # type: ignore[arg-type]
            _wrap_with_pre(pre_XW=pre_XW, model=ridge, require_dense=False),
        ]

    def build_tree_candidates() -> Sequence[BaseEstimator]:
        if missing_present:
            return build_boosting_candidates_nan_safe()

        if is_discrete:
            et = ExtraTreesClassifier(
                n_estimators=400,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            rf = RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            return [
                _wrap_with_pre(pre_XW=pre_XW, model=et, require_dense=True),
                _wrap_with_pre(pre_XW=pre_XW, model=rf, require_dense=True),
            ]

        et = ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        rf = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        return [
            _wrap_with_pre(pre_XW=pre_XW, model=et, require_dense=True),
            _wrap_with_pre(pre_XW=pre_XW, model=rf, require_dense=True),
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
        k = key.lower()
        if k in ("auto", "auto_plus"):
            return build_default_candidates()
        if k in ("automl", "automl_plus"):
            return build_default_candidates()
        if k == "linear":
            return build_linear_candidates()
        if k in ("forest", "trees"):
            return build_tree_candidates()
        if k in ("gbf", "hgb", "boosting"):
            return build_boosting_candidates()
        raise ValueError(f"Unknown model keyword: {key!r}")

    items: list[str | BaseEstimator]
    items = [spec_value] if isinstance(spec_value, (str, BaseEstimator)) else list(spec_value)

    out: list[BaseEstimator] = []
    for item in items:
        if isinstance(item, str):
            out.extend(candidates_for_keyword(item))
        else:
            out.append(
                _wrap_with_pre(
                    pre_XW=pre_XW,
                    model=item,
                    require_dense=True,
                )
            )  # pyright: ignore[reportArgumentType]

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
    disc_t = specs.treatment_spec.kind in ("binary", "categorical")
    disc_y = specs.outcome_spec.kind == "binary"

    default_model_y: str | BaseEstimator | Sequence[str | BaseEstimator] = "auto_plus"
    default_model_t: str | BaseEstimator | Sequence[str | BaseEstimator] = "auto_plus"

    model_y = list(
        _normalize_model_spec_to_wrapped_list(
            spec_value=default_model_y,
            pre_XW=pre_XW,
            is_discrete=disc_y,
            missingness=missingness,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    )
    model_t = list(
        _normalize_model_spec_to_wrapped_list(
            spec_value=default_model_t,
            pre_XW=pre_XW,
            is_discrete=disc_t,
            missingness=missingness,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    )

    return {"model_y": model_y, "model_t": model_t}
