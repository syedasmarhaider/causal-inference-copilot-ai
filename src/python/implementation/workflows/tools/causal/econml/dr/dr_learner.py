from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import inspect
import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence
from uuid import UUID
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import issparse  # type: ignore[import]

from econml.dr import ForestDRLearner, LinearDRLearner, SparseLinearDRLearner
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelRecord, ModelsRepo
from python.implementation.workflows.tools.causal.causal_command import (
    ATECommand,
    ATEModelResult,
    ATESuccess,
    CATECommand,
    CATEModelResult,
    CATESuccess,
    CommandFailure,
    CommandType,
    ErrorInfo,
    FitCommand,
    FitSuccess,
)
from python.implementation.workflows.tools.causal.causal_model import (
    CausalCommand,
    CausalModel,
    CausalResult,
)
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.econml.models_info import (
    get_forest_dr_learner_causal_model_info,
    get_linear_dr_learner_causal_model_info,
    get_sparse_linear_dr_learner_causal_model_info,
)
from python.implementation.workflows.tools.causal.econml.utils import (
    ModelSpecError,
    build_init_fit_options_param_maps,
    get_input_params_from_spec,
    get_treatment_t0_t1_from_spec,
    has_missing,
    is_missing_handled,
    now_utc,
    raise_if_x_rows_not_exactly_match_fit_x_cols,
    required_init_keys,
    serialize_inference_obj,
)
from python.implementation.workflows.tools.causal.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.encoding_util import EncodingUtil
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


# =============================================================================
# Helpers
# =============================================================================

class _ToDense(BaseEstimator, TransformerMixin):
    """Convert sparse -> dense for models that don't accept sparse."""

    def fit(self, X, y=None):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return self

    def transform(self, X: Any) -> np.ndarray:
        if issparse(X):
            return X.toarray()  # type: ignore[no-any-return]
        return np.asarray(X)


def _shape_as_list(x: Any) -> Optional[List[int]]:
    if x is None:
        return None
    if hasattr(x, "shape"):
        return list(x.shape)
    return list(np.asarray(x).shape)


def _width(x: Any) -> int:
    if x is None:
        return 0
    if hasattr(x, "shape") and len(x.shape) == 2:
        return int(x.shape[1])
    arr = np.asarray(x)
    if arr.ndim == 1:
        return 1
    return int(arr.shape[1])


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _safe_required_init_keys(estimator_cls: Any, *, init_map: Dict[str, Any]) -> List[str]:
    """
    Reflection sometimes surfaces pseudo-parameters like args/kwargs.
    Filter them out so adapters don't fail spuriously.
    """
    keys = list(required_init_keys(estimator_cls, init_map=init_map))
    return [k for k in keys if k not in ("args", "kwargs")]


def _supports_param(init_map: Dict[str, Any], name: str) -> bool:
    return name in init_map


def _set_if_supported(
    defaults: Dict[str, Any],
    init_map: Dict[str, Any],
    name: str,
    value: Any,
) -> None:
    if value is None:
        return
    if _supports_param(init_map, name):
        defaults[name] = value


def _treatment_categories_from_spec(specs: CausalSpec) -> Any:
    """
    Build explicit category order so EconML's baseline matches the causal spec.
    First entry is control/baseline.
    """
    ts = specs.treatment_spec
    kind = getattr(ts, "kind", None)

    if kind == "binary":
        control = getattr(ts, "control", None)
        treated = getattr(ts, "treated", None)
        if control is not None and treated is not None:
            return [control, treated]

        control_values = list(getattr(ts, "control_values", []) or [])
        treated_values = list(getattr(ts, "treated_values", []) or [])
        if len(control_values) == 1 and len(treated_values) == 1:
            return [control_values[0], treated_values[0]]

        return "auto"

    if kind == "categorical":
        baseline = getattr(ts, "baseline", None)
        levels = list(getattr(ts, "levels", []) or [])
        if baseline is not None:
            if baseline in levels:
                return [baseline] + [v for v in levels if v != baseline]
            if levels:
                return [baseline] + levels

        control_values = list(getattr(ts, "control_values", []) or [])
        treated_values = list(getattr(ts, "treated_values", []) or [])
        if control_values:
            baseline2 = control_values[0]
            rest = [v for v in treated_values if v != baseline2]
            return [baseline2] + rest if rest else [baseline2]

        return "auto"

    return "auto"


def _split_first_block_and_tail(
    X: Any,
    *,
    n_first: int,
) -> tuple[Any, Any]:
    """
    Split a 2D matrix-like object into [0:n_first] and [n_first:].
    Preserves DataFrame head when input is a DataFrame.
    """
    if isinstance(X, pd.DataFrame):
        if X.ndim != 2:
            raise ValueError(f"Expected 2D DataFrame input, got ndim={X.ndim}")
        head = X.iloc[:, :n_first]
        tail = X.iloc[:, n_first:] if X.shape[1] > n_first else None
        return head, tail

    if issparse(X):
        X2 = X.tocsr()
        if X2.ndim != 2:
            raise ValueError(f"Expected 2D sparse input, got ndim={X2.ndim}")
        head = X2[:, :n_first]
        tail = X2[:, n_first:] if X2.shape[1] > n_first else None
        return head, tail

    X2 = np.asarray(X)
    if X2.ndim != 2:
        raise ValueError(f"Expected 2D matrix input, got ndim={X2.ndim}")
    head = X2[:, :n_first]
    tail = X2[:, n_first:] if X2.shape[1] > n_first else None
    return head, tail


def _tail_to_csr_numeric(tail: Any) -> sp.csr_matrix:
    if tail is None:
        return sp.csr_matrix((0, 0))
    if isinstance(tail, pd.DataFrame):
        return sp.csr_matrix(tail.to_numpy(dtype=float))
    if issparse(tail):
        return tail.tocsr()
    return sp.csr_matrix(np.asarray(tail, dtype=float))


def _tail_to_dense(tail: Any) -> np.ndarray:
    if tail is None:
        return np.empty((0, 0), dtype=float)
    if isinstance(tail, pd.DataFrame):
        return tail.to_numpy()
    if issparse(tail):
        return tail.toarray()
    return np.asarray(tail)


class _TransformFirstBlockPassthroughTail(BaseEstimator, TransformerMixin):
    """
    Apply `pre_XW` only to the first `n_xw` columns and pass the remaining tail through.

    This is required for DR nuisance regression, which is trained on:
        concat([X, W, onehot(T_excl_baseline)])

    Only the X/W block should be transformed by pre_XW.
    """

    def __init__(self, *, pre_XW: ColumnTransformer, n_xw: int):
        self.pre_XW = pre_XW
        self.n_xw = int(n_xw)

    def fit(self, X, y=None):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        X_head, _ = _split_first_block_and_tail(X, n_first=self.n_xw)
        self.pre_XW.fit(X_head, y)
        return self

    def transform(self, X: Any) -> Any:
        X_head, X_tail = _split_first_block_and_tail(X, n_first=self.n_xw)
        head_tx = self.pre_XW.transform(X_head)

        if X_tail is None or getattr(X_tail, "shape", (0, 0))[1] == 0:
            return head_tx

        if issparse(head_tx):
            return sp.hstack([head_tx, _tail_to_csr_numeric(X_tail)], format="csr")

        return np.hstack([np.asarray(head_tx), _tail_to_dense(X_tail)])


def _wrap_xw_model(
    *,
    pre_XW: ColumnTransformer,
    model: BaseEstimator,
    require_dense: bool,
) -> BaseEstimator:
    """Wrapper for nuisance models trained on concat([X, W])."""
    steps: List[tuple[str, BaseEstimator]] = [("pre", pre_XW)]
    if require_dense:
        steps.append(("dense", _ToDense()))
    steps.append(("model", model))
    return Pipeline(steps)


def _wrap_xw_plus_t_model(
    *,
    pre_XW: ColumnTransformer,
    n_xw: int,
    model: BaseEstimator,
    require_dense: bool,
) -> BaseEstimator:
    """Wrapper for nuisance models trained on concat([X, W, onehot(T_excl_baseline)])."""
    steps: List[tuple[str, BaseEstimator]] = [
        ("pre_xw_only", _TransformFirstBlockPassthroughTail(pre_XW=pre_XW, n_xw=n_xw))
    ]
    if require_dense:
        steps.append(("dense", _ToDense()))
    steps.append(("model", model))
    return Pipeline(steps)


# =============================================================================
# Default nuisance candidates
# =============================================================================

def _build_propensity_candidates(
    *,
    pre_XW: ColumnTransformer,
    missingness_W: bool,
    random_state: Optional[int],
    n_jobs: Optional[int],
) -> Sequence[BaseEstimator]:
    """
    Propensity nuisance: classifier for Pr[T=t | X, W].
    If W-missingness is present, restrict to NaN-tolerant candidates.
    """
    if missingness_W:
        hgb = HistGradientBoostingClassifier(
            random_state=random_state,
            max_depth=None,
            learning_rate=0.05,
            max_iter=400,
            early_stopping=True,
        )
        return [_wrap_xw_model(pre_XW=pre_XW, model=hgb, require_dense=True)]

    lr = LogisticRegressionCV(
        max_iter=2000,
        solver="lbfgs",
        n_jobs=n_jobs,
        random_state=random_state,
    )
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
    hgb = HistGradientBoostingClassifier(
        random_state=random_state,
        max_depth=None,
        learning_rate=0.05,
        max_iter=400,
        early_stopping=True,
    )
    return [
        _wrap_xw_model(pre_XW=pre_XW, model=lr, require_dense=False),
        _wrap_xw_model(pre_XW=pre_XW, model=et, require_dense=True),
        _wrap_xw_model(pre_XW=pre_XW, model=rf, require_dense=True),
        _wrap_xw_model(pre_XW=pre_XW, model=hgb, require_dense=True),
    ]


def _build_regression_candidates(
    *,
    pre_XW: ColumnTransformer,
    n_xw: int,
    discrete_outcome: bool,
    missingness_W: bool,
    random_state: Optional[int],
    n_jobs: Optional[int],
) -> Sequence[BaseEstimator]:
    """
    Outcome nuisance: estimator for E[Y | X, W, T], trained on concat([X, W, onehot(T_excl_baseline)]).
    If W-missingness is present, restrict to NaN-tolerant candidates.
    """
    if missingness_W:
        if discrete_outcome:
            hgb = HistGradientBoostingClassifier(
                random_state=random_state,
                max_depth=None,
                learning_rate=0.05,
                max_iter=400,
                early_stopping=True,
            )
            return [
                _wrap_xw_plus_t_model(
                    pre_XW=pre_XW,
                    n_xw=n_xw,
                    model=hgb,
                    require_dense=True,
                )
            ]

        hgb = HistGradientBoostingRegressor(
            random_state=random_state,
            max_depth=None,
            learning_rate=0.05,
            max_iter=400,
            early_stopping=True,
        )
        return [
            _wrap_xw_plus_t_model(
                pre_XW=pre_XW,
                n_xw=n_xw,
                model=hgb,
                require_dense=True,
            )
        ]

    if discrete_outcome:
        lr = LogisticRegressionCV(
            max_iter=2000,
            solver="lbfgs",
            n_jobs=n_jobs,
            random_state=random_state,
        )
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
        hgb = HistGradientBoostingClassifier(
            random_state=random_state,
            max_depth=None,
            learning_rate=0.05,
            max_iter=400,
            early_stopping=True,
        )
        return [
            _wrap_xw_plus_t_model(pre_XW=pre_XW, n_xw=n_xw, model=lr, require_dense=False),
            _wrap_xw_plus_t_model(pre_XW=pre_XW, n_xw=n_xw, model=et, require_dense=True),
            _wrap_xw_plus_t_model(pre_XW=pre_XW, n_xw=n_xw, model=rf, require_dense=True),
            _wrap_xw_plus_t_model(pre_XW=pre_XW, n_xw=n_xw, model=hgb, require_dense=True),
        ]

    lasso = WeightedLassoCVWrapper(random_state=random_state)
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 25))
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
    hgb = HistGradientBoostingRegressor(
        random_state=random_state,
        max_depth=None,
        learning_rate=0.05,
        max_iter=400,
        early_stopping=True,
    )
    return [
        _wrap_xw_plus_t_model(pre_XW=pre_XW, n_xw=n_xw, model=lasso, require_dense=False),  # type: ignore[arg-type]
        _wrap_xw_plus_t_model(pre_XW=pre_XW, n_xw=n_xw, model=ridge, require_dense=False),
        _wrap_xw_plus_t_model(pre_XW=pre_XW, n_xw=n_xw, model=et, require_dense=True),
        _wrap_xw_plus_t_model(pre_XW=pre_XW, n_xw=n_xw, model=rf, require_dense=True),
        _wrap_xw_plus_t_model(pre_XW=pre_XW, n_xw=n_xw, model=hgb, require_dense=True),
    ]


# =============================================================================
# Base adapter shared by concrete DR learners
# =============================================================================

@dataclass(frozen=True, slots=True)
class _BaseDRLearnerAdapter(CausalModel):
    data_repo: DataRepo
    models_repo: ModelsRepo
    encoding_util: EncodingUtil

    ESTIMATOR_CLS: ClassVar[Any]
    BACKEND_NAME: ClassVar[str]
    INFO: ClassVar[str]

    def get_info(self) -> str:
        return self.INFO

    def get_command_info(self, command: CommandType) -> str | None:
        match command:
            case "FIT":
                fit_doc = inspect.getdoc(self.ESTIMATOR_CLS.fit) or ""
                base_doc = inspect.getdoc(self.ESTIMATOR_CLS) or ""
                return base_doc + fit_doc
            case "ATE":
                ate_doc = inspect.getdoc(self.ESTIMATOR_CLS.ate) or ""
                return ate_doc
            case "CATE":
                effect_doc = inspect.getdoc(self.ESTIMATOR_CLS.effect) or ""
                return effect_doc
            case _:
                return None

    def execute(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: CausalCommand,
    ) -> CausalResult:
        started = now_utc()
        try:
            df = self.data_repo.get_csv_data(
                user_id,
                conversation_id,
                command.dataset_id,
                limit=None,
            )
        except Exception as e:
            logging.exception(e)
            return CommandFailure(
                run_id=command.run_id,
                started_at=started,
                finished_at=now_utc(),
                error=ErrorInfo(
                    code="DATASET_NOT_FOUND",
                    message="Failed to load dataset.",
                    details={"dataset_id": str(command.dataset_id), "exception": repr(e)},
                ),
                warnings=[],
                meta={},
            )

        if isinstance(command, FitCommand):
            return self._fit(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
                df=df,
                started_at=started,
            )
        if isinstance(command, ATECommand):
            return self._ate(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
                df=df,
                started_at=started,
            )
        if isinstance(command, CATECommand):  # pyright: ignore[reportUnnecessaryIsInstance]
            return self._cate(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
                started_at=started,
            )

        raise ValueError(f"Unsupported command type: {type(command)}")

    # -------------------------------------------------------------------------
    # FIT
    # -------------------------------------------------------------------------

    def _fit(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: FitCommand,
        df: pd.DataFrame,
        started_at: datetime,
    ) -> CausalResult:
        try:
            specs: CausalSpec = command.causal_specs
            data_summary: DatasetSummaryModel = command.data_summary
            transformation_plan: Optional[TransformPlan] = command.transformation_plan

            effect_modifiers_order: List[str] = list(
                command.order_effect_modifiers or specs.effect_modifiers or []
            )
            covariates_order: List[str] = list(
                command.order_covariates or specs.covariates or []
            )

            if specs.treatment_spec.kind not in ("binary", "categorical"):
                raise ModelSpecError(
                    f"{self.BACKEND_NAME} supports only binary/categorical treatments."
                )

            plan = (
                self.encoding_util.compile(
                    plan=transformation_plan,
                    effect_modifiers_order=effect_modifiers_order,
                    covariates_order=covariates_order,
                    dense_output=True,
                )
                if transformation_plan is not None
                else None
            )

            pre_x = plan.pre_X if plan is not None else None
            pre_xw = plan.pre_XW if plan is not None else None

            if pre_x is None and len(specs.effect_modifiers or []) > 0:
                raise ModelSpecError(
                    "Spec declares effect modifiers but no pre_X transformer was provided."
                )
            if pre_xw is None and (len(specs.covariates or []) + len(specs.effect_modifiers or [])) > 0:
                raise ModelSpecError(
                    "Spec declares covariates and/or effect modifiers but no pre_XW transformer was provided."
                )

            Y, T, X, W, col_meta = get_input_params_from_spec(
                df,
                specs,
                effect_modifiers_order=effect_modifiers_order,
                covariates_order=covariates_order,
            )

            miss = {
                "Y": has_missing(Y),
                "T": has_missing(T),
                "X": has_missing(X),
                "W": has_missing(W),
            }
            if miss["Y"] or miss["T"]:
                raise ModelSpecError(
                    f"Y/T contain missing values; must be fixed upstream. missing={miss}"
                )

            missingness_X = (
                len(specs.effect_modifiers or []) > 0
                and miss["X"]
                and (
                    transformation_plan is None
                    or not is_missing_handled(
                        plan=transformation_plan,
                        summary=data_summary,
                        col_name_list=specs.effect_modifiers,
                    )
                )
            )
            missingness_W = (
                len(specs.covariates or []) > 0
                and miss["W"]
                and (
                    transformation_plan is None
                    or not is_missing_handled(
                        plan=transformation_plan,
                        summary=data_summary,
                        col_name_list=specs.covariates,
                    )
                )
            )

            if missingness_X:
                raise ModelSpecError(
                    f"{self.BACKEND_NAME} does not support missing values in X via allow_missing. "
                    "Impute/clean X upstream before fit."
                )

            n_xw = _width(X) + _width(W)

            maps = build_init_fit_options_param_maps(
                self.ESTIMATOR_CLS,
                fit_include_names={
                    "cache_values",
                    "inference",
                    "sample_weight",
                    "freq_weight",
                    "sample_var",
                    "groups",
                },
            )
            init_map = maps["init"]

            defaults: Dict[str, Any] = {}
            discrete_outcome = specs.outcome_spec.kind == "binary"

            if discrete_outcome:
                _set_if_supported(defaults, init_map, "discrete_outcome", True)

            _set_if_supported(defaults, init_map, "categories", _treatment_categories_from_spec(specs))
            _set_if_supported(defaults, init_map, "allow_missing", missingness_W)

            if pre_xw is not None:
                _set_if_supported(
                    defaults,
                    init_map,
                    "model_propensity",
                    list(
                        _build_propensity_candidates(
                            pre_XW=pre_xw,
                            missingness_W=missingness_W,
                            random_state=None,
                            n_jobs=None,
                        )
                    ),
                )
                _set_if_supported(
                    defaults,
                    init_map,
                    "model_regression",
                    list(
                        _build_regression_candidates(
                            pre_XW=pre_xw,
                            n_xw=n_xw,
                            discrete_outcome=discrete_outcome,
                            missingness_W=missingness_W,
                            random_state=None,
                            n_jobs=None,
                        )
                    ),
                )

            if pre_x is not None:
                _set_if_supported(defaults, init_map, "featurizer", pre_x)

            required_keys = _safe_required_init_keys(self.ESTIMATOR_CLS, init_map=init_map)
            missing_required = [k for k in required_keys if k not in defaults]
            if missing_required:
                raise ModelSpecError(
                    f"Missing required {self.BACKEND_NAME} __init__ parameters: {missing_required}. "
                    "(Adapter is not exposing command.options yet.)"
                )

            est = self.ESTIMATOR_CLS(**defaults)

            fit_warnings: List[str] = []
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                est.fit(Y=Y, T=T, X=X, W=W)  # pyright: ignore[reportUnknownMemberType]
            fit_warnings = [f"{w.category.__name__}: {str(w.message)}" for w in ws]

            artifacts: Dict[str, Any] = {
                "n": int(df.shape[0]),
                "y_shape": _shape_as_list(Y),
                "t_shape": _shape_as_list(T),
                "x_shape": _shape_as_list(X),
                "w_shape": _shape_as_list(W),
            }
            for attr in ("score_", "nuisance_scores_propensity", "nuisance_scores_regression"):
                try:
                    if hasattr(est, attr):
                        artifacts[attr] = _to_jsonable(getattr(est, attr))
                except Exception:
                    pass

            fit_meta: Dict[str, Any] = {
                "warnings": fit_warnings,
                "meta": {
                    "backend": self.BACKEND_NAME,
                    "n": int(df.shape[0]),
                    "columns": col_meta,
                    "used_init_kwargs": defaults,
                    "spec_semantics_applied": sorted(list(required_keys)),
                },
                "artifacts": artifacts,
            }

            model_id = command.run_id
            self.models_repo.save_model(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
                model=est,
                metadata=fit_meta,
            )

            finished = now_utc()
            return FitSuccess(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=finished,
                warnings=fit_meta.get("warnings", []),
                meta=fit_meta.get("meta", {}),
                fitted_model_id=model_id,
                artifacts=fit_meta.get("artifacts", {}),
            )

        except ModelSpecError as e:
            return CommandFailure(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(code="OPTIONS_INVALID", message=str(e), details={}),
                warnings=[],
                meta={},
            )
        except Exception as e:
            logging.exception(e)
            return CommandFailure(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(
                    code="ESTIMATOR_ERROR",
                    message=f"{self.BACKEND_NAME}.fit failed.",
                    details={"exception": repr(e)},
                ),
                warnings=[],
                meta={},
            )

    # -------------------------------------------------------------------------
    # ATE
    # -------------------------------------------------------------------------

    def _ate(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: ATECommand,
        df: pd.DataFrame,
        started_at: datetime,
    ) -> CausalResult:
        try:
            warnings_list: List[str] = []
            spec: CausalSpec = command.causal_specs
            effect_modifiers_order: List[str] = list(
                command.order_effect_modifiers or spec.effect_modifiers or []
            )
            covariates_order: List[str] = list(
                command.order_covariates or spec.covariates or []
            )

            model_record: ModelRecord | None = self.models_repo.load_model(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=command.fitted_model_id,
            )
            if model_record is None:
                raise ModelSpecError(
                    f"Fitted model with id {command.fitted_model_id} not found."
                )

            est = model_record.model

            t0, t1 = get_treatment_t0_t1_from_spec(
                spec,
                is_global_counter_factual=False,
            )

            _, _, X, _, _ = get_input_params_from_spec(
                df,
                spec,
                effect_modifiers_order=effect_modifiers_order,
                covariates_order=covariates_order,
            )

            if t1 == t0:
                raise ModelSpecError(
                    f"Invalid contrast: t1 value {t1} is the same as t0 baseline {t0}."
                )

            item: Dict[ATEModelResult, Any] = {"for_treatment": {"t0": t0, "t1": t1}}
            item["ate"] = est.ate(X=X, T0=t0, T1=t1)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]

            try:
                ate_interval = est.ate_interval(
                    X=X,
                    T0=t0,
                    T1=t1,
                    alpha=command.inputs.alpha,
                )  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
                if ate_interval is None:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: ate_interval returned None")
                    item["ate_interval"] = None
                else:
                    item["ate_interval"] = ate_interval
            except Exception as e:
                warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                item["ate_interval"] = None

            try:
                inf = est.ate_inference(X=X, T0=t0, T1=t1)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
                if inf is None:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: ate_inference returned None")
                    item["ate_inference"] = None
                else:
                    item["ate_inference"] = serialize_inference_obj(inf)
            except Exception as e:
                warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                item["ate_inference"] = None

            finished = now_utc()
            return ATESuccess(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=finished,
                warnings=warnings_list,
                meta={
                    "backend": self.BACKEND_NAME,
                    "n": int(df.shape[0]),
                    "x_cols": spec.effect_modifiers if spec.effect_modifiers else None,
                    "contrast_kind": "single_pair",
                    "t0": t0,
                },
                fitted_model_id=command.fitted_model_id,
                contrast={"t0": t0, "t1": t1},
                ate=[item],
            )

        except Exception as e:
            logging.exception(e)
            return CommandFailure(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(
                    code="ESTIMATOR_ERROR",
                    message="ATE computation failed.",
                    details={"exception": repr(e)},
                ),
                warnings=[],
                meta={},
            )

    # -------------------------------------------------------------------------
    # CATE
    # -------------------------------------------------------------------------

    def _cate(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: CATECommand,
        started_at: datetime,
    ) -> CausalResult:
        warnings_list: List[str] = []
        try:
            model_record: ModelRecord | None = self.models_repo.load_model(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=command.fitted_model_id,
            )
            if model_record is None:
                return CommandFailure(
                    run_id=command.run_id,
                    started_at=started_at,
                    finished_at=now_utc(),
                    error=ErrorInfo(
                        code="MODEL_NOT_FOUND",
                        message="Fitted model not found.",
                        details={"fitted_model_id": str(command.fitted_model_id)},
                    ),
                    warnings=[],
                    meta={},
                )

            est = model_record.model
            spec: CausalSpec = command.causal_specs
            effect_modifiers_order: List[str] = list(
                command.order_effect_modifiers or spec.effect_modifiers or []
            )

            X_df = command.inputs.x_rows
            x_cols = spec.effect_modifiers
            raise_if_x_rows_not_exactly_match_fit_x_cols(x_rows=X_df, x_cols=x_cols)

            # Keep DataFrame columns intact for featurizer=pre_X.
            X_query = X_df[effect_modifiers_order].copy() if effect_modifiers_order else None

            if X_query is None or X_query.shape[1] == 0:
                return CommandFailure(
                    run_id=command.run_id,
                    started_at=started_at,
                    finished_at=now_utc(),
                    error=ErrorInfo(
                        code="OPTIONS_INVALID",
                        message="CATE requires non-empty X for effect modification; none provided.",
                        details={},
                    ),
                    warnings=[],
                    meta={},
                )

            t0, t1 = get_treatment_t0_t1_from_spec(
                spec,
                is_global_counter_factual=command.inputs.counterfactual,
            )

            effects: Dict[CATEModelResult, Any] = {"for_treatment": {"t0": t0, "t1": t1}}

            try:
                effects["cate"] = est.effect(X_query, T0=t0, T1=t1)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
            except Exception as e:
                return CommandFailure(
                    run_id=command.run_id,
                    started_at=started_at,
                    finished_at=now_utc(),
                    error=ErrorInfo(
                        code="ESTIMATOR_ERROR",
                        message="CATE computation failed (effect).",
                        details={"exception": repr(e)},
                    ),
                    warnings=[],
                    meta={},
                )

            try:
                interval = est.effect_interval(
                    X_query,
                    T0=t0,
                    T1=t1,
                    alpha=command.inputs.alpha,
                )  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
                if interval is None:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: effect_interval returned None")
                    effects["cate_interval"] = None
                else:
                    effects["cate_interval"] = interval
            except Exception as e:
                warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                effects["cate_interval"] = None

            try:
                inf = est.effect_inference(X_query, T0=t0, T1=t1)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
                if inf is None:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: effect_inference returned None")
                    effects["cate_inference"] = None
                else:
                    effects["cate_inference"] = serialize_inference_obj(inf)
            except Exception as e:
                warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                effects["cate_inference"] = None

            if effects.get("cate") is None:
                return CommandFailure(
                    run_id=command.run_id,
                    started_at=started_at,
                    finished_at=now_utc(),
                    error=ErrorInfo(
                        code="ESTIMATOR_ERROR",
                        message="CATE computation failed: effect returned None.",
                        details={},
                    ),
                    warnings=[],
                    meta={},
                )

            finished = now_utc()
            return CATESuccess(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=finished,
                warnings=warnings_list,
                meta={
                    "backend": self.BACKEND_NAME,
                    "row_count": int(getattr(X_query, "shape", [len(command.inputs.x_rows)])[0]),
                },
                fitted_model_id=command.fitted_model_id,
                x_cols=x_cols,
                effects=effects,
            )

        except ModelSpecError as e:
            return CommandFailure(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(code="OPTIONS_INVALID", message=str(e), details={}),
                warnings=[],
                meta={},
            )
        except Exception as e:
            logging.exception(e)
            return CommandFailure(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(
                    code="ESTIMATOR_ERROR",
                    message="CATE computation failed.",
                    details={"exception": repr(e)},
                ),
                warnings=[],
                meta={},
            )


# =============================================================================
# Concrete adapters
# =============================================================================

@dataclass(frozen=True, slots=True)
class LinearDRLearnerCausalModel(_BaseDRLearnerAdapter):
    ESTIMATOR_CLS: ClassVar[Any] = LinearDRLearner
    BACKEND_NAME: ClassVar[str] = "econml.dr.LinearDRLearner"
    INFO: ClassVar[str] = get_linear_dr_learner_causal_model_info()


@dataclass(frozen=True, slots=True)
class ForestDRLearnerCausalModel(_BaseDRLearnerAdapter):
    ESTIMATOR_CLS: ClassVar[Any] = ForestDRLearner
    BACKEND_NAME: ClassVar[str] = "econml.dr.ForestDRLearner"
    INFO: ClassVar[str] = get_forest_dr_learner_causal_model_info()


@dataclass(frozen=True, slots=True)
class SparseLinearDRLearnerCausalModel(_BaseDRLearnerAdapter):
    ESTIMATOR_CLS: ClassVar[Any] = SparseLinearDRLearner
    BACKEND_NAME: ClassVar[str] = "econml.dr.SparseLinearDRLearner"
    INFO: ClassVar[str] = get_sparse_linear_dr_learner_causal_model_info()