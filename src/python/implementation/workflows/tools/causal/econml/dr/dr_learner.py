from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import inspect
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID
import warnings

import numpy as np
import pandas as pd

import scipy.sparse as sp
from scipy.sparse import issparse  # type: ignore[import]

from econml.dr import DRLearner, ForestDRLearner, LinearDRLearner, SparseLinearDRLearner
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
from python.implementation.workflows.tools.causal.econml.models_info import get_forest_dr_learner_causal_model_info, get_linear_dr_learner_causal_model_info, get_sparse_linear_dr_learner_causal_model_info
from python.implementation.workflows.tools.causal.econml.utils import (
    ModelSpecError,
    build_init_fit_options_param_maps,
    categorical_t0_t1_pairs,
    get_input_params_from_spec,
    has_missing,
    is_missing_handled,
    now_utc,
    raise_if_x_rows_not_exactly_match_fit_x_cols,
    required_init_keys,
    serialize_inference_obj,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.tools.common.model.encoding_plan import TransformPlan
from python.implementation.workflows.tools.common.model.encoding_plan import TransformPlan

# =============================================================================
# Helpers: sparse/dense + "transform XW only, passthrough tail"
# =============================================================================

class _ToDense(BaseEstimator, TransformerMixin):
    """Convert sparse -> dense for models that don't accept sparse."""
    def fit(self, X, y=None):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return self

    def transform(self, X: Any) -> np.ndarray:
        if issparse(X):
            return X.toarray()  # type: ignore[no-any-return]
        return np.asarray(X)


# FIX(3): normalize input to a sliceable 2D matrix (dense ndarray or CSR)
def _as_2d_matrix(X: Any):
    if issparse(X):
        X2 = X.tocsr()
    else:
        X2 = np.asarray(X)
    if getattr(X2, "ndim", 0) != 2:
        raise ValueError(f"Expected 2D matrix input, got ndim={getattr(X2, 'ndim', None)}")
    return X2


class _TransformFirstBlockPassthroughTail(BaseEstimator, TransformerMixin):
    """
    Applies `pre_XW` to the first `n_xw` columns of input, and passes through
    any remaining columns unchanged.

    WHY (DRLearner):
      - propensity is trained on concat([X, W]) -> exactly n_xw cols
      - regression is trained on concat([X, W, onehot(T_excl_baseline)]) -> tail cols exist
    """
    def __init__(self, *, pre_XW: ColumnTransformer, n_xw: int):
        self.pre_XW = pre_XW
        self.n_xw = int(n_xw)

    # FIX(3): normalize X before slicing
    def fit(self, X, y=None):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        X_arr = _as_2d_matrix(X)
        X_head = X_arr[:, : self.n_xw]
        self.pre_XW.fit(X_head, y)
        return self

    # FIX(3): normalize X before slicing + robust hstack
    def transform(self, X: Any):
        X_arr = _as_2d_matrix(X)
        X_head = X_arr[:, : self.n_xw]
        X_tail = X_arr[:, self.n_xw :] if X_arr.shape[1] > self.n_xw else None

        head_tx = self.pre_XW.transform(X_head)

        if X_tail is None or X_tail.shape[1] == 0:
            return head_tx

        # hstack head + tail while preserving sparsity when possible
        if issparse(head_tx):
            if issparse(X_tail):
                tail_sparse = X_tail.tocsr()
            else:
                # tail should be numeric one-hot; force numeric conversion early
                try:
                    tail_dense = np.asarray(X_tail, dtype=float)
                except Exception as e:
                    raise ValueError(f"DRLearner tail block is not numeric; cannot csr_matrix it. {e!r}") from e
                tail_sparse = sp.csr_matrix(tail_dense)
            return sp.hstack([head_tx, tail_sparse], format="csr")

        # head is dense
        if issparse(X_tail):
            tail_dense2 = X_tail.toarray()
        else:
            tail_dense2 = np.asarray(X_tail)
        return np.hstack([np.asarray(head_tx), tail_dense2])


def _wrap_xw_model(
    *,
    pre_XW: ColumnTransformer,
    model: BaseEstimator,
    require_dense: bool,
) -> BaseEstimator:
    """For models trained on concat([X, W])."""
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
    """For models trained on concat([X, W, onehot(T_excl_baseline)])."""
    steps: List[tuple[str, BaseEstimator]] = [
        ("pre_xw_only", _TransformFirstBlockPassthroughTail(pre_XW=pre_XW, n_xw=n_xw))
    ]
    if require_dense:
        steps.append(("dense", _ToDense()))
    steps.append(("model", model))
    return Pipeline(steps)


# =============================================================================
# Default nuisance candidate builders (propensity + regression)
# =============================================================================

def _build_propensity_candidates(
    *,
    pre_XW: ColumnTransformer,
    missingness_W: bool,
    random_state: Optional[int],
    n_jobs: Optional[int],
) -> Sequence[BaseEstimator]:
    """
    Propensity model: classifier for Pr[T=t | X,W].
    If missingness is present, restrict to NaN-tolerant models (HGB).
    """
    missing_present = missingness_W

    if missing_present:
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
    Regression nuisance for E[Y | X,W,T] trained on concat([X,W, onehot(T_excl_baseline)]).
    If missingness is present, restrict to NaN-tolerant HGB.
    """
    missing_present = missingness_W

    if missing_present:
        if discrete_outcome:
            hgb = HistGradientBoostingClassifier(
                random_state=random_state,
                max_depth=None,
                learning_rate=0.05,
                max_iter=400,
                early_stopping=True,
            )
            return [_wrap_xw_plus_t_model(pre_XW=pre_XW, n_xw=n_xw, model=hgb, require_dense=True)]
        hgb = HistGradientBoostingRegressor(
            random_state=random_state,
            max_depth=None,
            learning_rate=0.05,
            max_iter=400,
            early_stopping=True,
        )
        return [_wrap_xw_plus_t_model(pre_XW=pre_XW, n_xw=n_xw, model=hgb, require_dense=True)]

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


def _safe_required_init_keys(estimator_cls: Any, *, init_map: Dict[str, Any]) -> List[str]:
    """
    Your `required_init_keys()` currently returns ['args','kwargs'] for some estimators in tests.
    Filter those out here so adapters don't fail spuriously.
    """
    keys = list(required_init_keys(estimator_cls, init_map=init_map))
    return [k for k in keys if k not in ("args", "kwargs")]


def _categories_from_spec(specs: CausalSpec) -> Union[str, List[Any]]:
    """
    Ensure baseline ordering matches your spec when treatment is discrete.
    EconML uses the first category as control/baseline.
    """
    if specs.T.kind == "binary":
        if len(specs.T.control_values) == 1 and len(specs.T.treated_values) == 1:
            return [specs.T.control_values[0], specs.T.treated_values[0]]
        return "auto"

    if specs.T.kind == "categorical":
        baseline = getattr(specs.T, "baseline", None)
        levels = list(getattr(specs.T, "levels", []) or [])
        if baseline is not None and baseline in levels:
            return [baseline] + [v for v in levels if v != baseline]
        return "auto"

    return "auto"


# =============================================================================
# Base adapter shared logic (DRLearner family)
# =============================================================================

@dataclass(frozen=True, slots=True)
class _BaseDRLearnerAdapter(CausalModel):
    data_repo: DataRepo
    models_repo: ModelsRepo

    ESTIMATOR_CLS: Any = DRLearner
    BACKEND_NAME: str = "econml.dr.DRLearner"
    INFO: str = "No info provided."

    def get_info(self) -> str:
        return self.INFO
    
    def get_command_info(self, command: CommandType) -> str | None:
        match command:
            case "FIT":
                fit_doc = inspect.getdoc(self.ESTIMATOR_CLS.fit) or "" # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                base_doc = inspect.getdoc(self.ESTIMATOR_CLS) or ""
                return base_doc + fit_doc
            case "ATE":
                ate_doc = inspect.getdoc(self.ESTIMATOR_CLS.ate) or "" # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                return ate_doc
            case "CATE":
                effect_doc = inspect.getdoc(self.ESTIMATOR_CLS.effect) or "" # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
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
            return self._fit(user_id=user_id, conversation_id=conversation_id, command=command, df=df, started_at=started)
        if isinstance(command, ATECommand):
            return self._ate(user_id=user_id, conversation_id=conversation_id, command=command, df=df, started_at=started)
        if isinstance(command, CATECommand):  # pyright: ignore[reportUnnecessaryIsInstance]
            return self._cate(user_id=user_id, conversation_id=conversation_id, command=command, started_at=started)

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
            specs: CausalSpec = command.protocol_specs
            pre_x: ColumnTransformer | None = command.inputs.pre_X
            pre_xw: ColumnTransformer | None = command.inputs.pre_XW
            order_X: Optional[List[str]] = command.order_X
            order_W: Optional[List[str]] = command.order_W
            data_summary: DatasetSummaryModel = command.data_summary
            transformation_plan: Optional[TransformPlan] = command.transformation_plan

            # DRLearner assumes discrete treatments
            if specs.T.kind not in ("binary", "categorical"):
                raise ModelSpecError(f"{self.BACKEND_NAME} supports only binary/categorical treatments.")

            if pre_x is None and len(specs.X or []) > 0:
                raise ModelSpecError("Spec declares effect modifiers (spec.X) but no pre_X transformer provided.")
            if pre_xw is None and (len(specs.W or []) + len(specs.X or [])) > 0:
                raise ModelSpecError("Spec declares controls (spec.W) and/or effect modifiers (spec.X) but no pre_XW transformer provided.")

            Y, T, X, W, col_meta = get_input_params_from_spec(df, specs, order_X=order_X, order_W=order_W)

            miss = {"Y": has_missing(Y), "T": has_missing(T), "X": has_missing(X), "W": has_missing(W)}
            if miss["Y"] or miss["T"]:
                raise ModelSpecError(f"Y/T contain missing values; must be fixed upstream. missing={miss}")

            missingness_X = len(specs.X or []) > 0 and miss["X"] and (transformation_plan is  None or not is_missing_handled(plan=transformation_plan,summary=data_summary, col_name_list=specs.X))
            missingness_W = len(specs.W or []) > 0 and miss["W"] and (transformation_plan is  None or not is_missing_handled(plan=transformation_plan,summary=data_summary, col_name_list=specs.W))
            if missingness_X:
                raise ModelSpecError(
                    f"{self.BACKEND_NAME} does not support missing values in X. "
                    f"Impute/clean X upstream. missing={miss}"
                )
                
            # Figure out raw n_xw for the regression nuisance wrapper
            n_x = int(np.asarray(X).shape[1]) if X is not None else 0
            n_w = int(np.asarray(W).shape[1]) if W is not None else 0
            n_xw = n_x + n_w

            maps = build_init_fit_options_param_maps(
                self.ESTIMATOR_CLS,
                fit_include_names={"cache_values", "inference", "sample_weight", "freq_weight", "sample_var", "groups"},
            )
            init_map = maps["init"]

            defaults: Dict[str, Any] = {}

            disc_y = specs.Y.kind == "binary"
            if disc_y:
                defaults["discrete_outcome"] = True

            defaults["categories"] = _categories_from_spec(specs)

            if pre_xw is not None:
                defaults["model_propensity"] = list(
                    _build_propensity_candidates(
                        pre_XW=pre_xw,
                        missingness_W=missingness_W,
                        random_state=None,
                        n_jobs=None,
                    )
                )
                defaults["model_regression"] = list(
                    _build_regression_candidates(
                        pre_XW=pre_xw,
                        n_xw=n_xw,
                        discrete_outcome=disc_y,
                        missingness_W=missingness_W,
                        random_state=None,
                        n_jobs=None,
                    )
                )

            if pre_x is not None:
                defaults["featurizer"] = pre_x

            defaults["allow_missing"] = missingness_W

            required_keys = _safe_required_init_keys(self.ESTIMATOR_CLS, init_map=init_map)
            missing_required = [k for k in required_keys if k not in defaults]
            if missing_required:
                raise ModelSpecError(
                    f"Missing required {self.BACKEND_NAME} __init__ parameters: {missing_required}. "
                    f"(Adapter is not exposing command.options yet.)"
                )

            est = self.ESTIMATOR_CLS(**defaults)

            fit_warnings: list[str] = []
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                est.fit(Y, T, X=X, W=W)  # pyright: ignore[reportUnknownMemberType]
            fit_warnings = [f"{w.category.__name__}: {str(w.message)}" for w in ws]

            n = int(df.shape[0])

            artifacts: Dict[str, Any] = {
                "n": n,
                "y_shape": list(np.asarray(Y).shape),
                "t_shape": list(np.asarray(T).shape),
                "x_shape": (list(np.asarray(X).shape) if X is not None else None),
                "w_shape": (list(np.asarray(W).shape) if W is not None else None),
            }
            for attr in ("score_", "nuisance_scores_propensity", "nuisance_scores_regression"):
                try:
                    if hasattr(est, attr):
                        val = getattr(est, attr)
                        artifacts[attr] = np.asarray(val).tolist() if isinstance(val, (np.ndarray,)) else val
                except Exception:
                    pass

            fit_meta: Dict[str, Any] = {
                "warnings": fit_warnings,
                "meta": {
                    "backend": self.BACKEND_NAME,
                    "n": n,
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
            return CommandFailure(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(code="ESTIMATOR_ERROR", message=f"{self.BACKEND_NAME}.fit failed.", details={"exception": repr(e)}),
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
            spec: CausalSpec = command.protocol_specs

            model_record: ModelRecord | None = self.models_repo.load_model(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=command.fitted_model_id,
            )
            if model_record is None:
                raise ModelSpecError(f"Fitted model with id {command.fitted_model_id} not found.")

            est = model_record.model

            if spec.T.kind == "binary":
                if len(spec.T.control_values) != 1 or len(spec.T.treated_values) != 1:
                    raise ModelSpecError("Binary ATE requires exactly one control_value and one treated_value (or pre-normalize T).")
                t0 = spec.T.control_values[0]
                t1s = [spec.T.treated_values[0]]
            elif spec.T.kind == "categorical":
                t0, t1s = categorical_t0_t1_pairs(spec)
            else:
                raise ModelSpecError(f"Unsupported treatment kind {spec.T.kind!r} for ATE.")

            effects: List[Dict[ATEModelResult, Any]] = []
            X_for_ate = df[spec.X] if spec.X else None

            for t1_val in t1s:
                if t1_val == t0:
                    raise ModelSpecError(f"Invalid contrast: t1 value {t1_val} is the same as t0 baseline {t0}.")

                item: Dict[ATEModelResult, Any] = {"for_treatment": {"t0": t0, "t1": t1_val}}
                item["ate"] = est.ate(X=X_for_ate, T0=t0, T1=t1_val)

                try:
                    lo, hi = est.ate_interval(X=X_for_ate, T0=t0, T1=t1_val, alpha=command.inputs.alpha)  # pyright: ignore
                    item["ate_interval"] = (list(lo), list(hi)) if lo is not None and hi is not None else None  # pyright: ignore
                    if item["ate_interval"] is None:
                        warnings_list.append("INFERENCE_NOT_AVAILABLE: ate_interval returned None")
                except Exception as e:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                    item["ate_interval"] = None

                try:
                    inf = est.ate_inference(X=X_for_ate, T0=t0, T1=t1_val)  # pyright: ignore
                    item["ate_inference"] = serialize_inference_obj(inf) if inf is not None else None
                    if inf is None:
                        warnings_list.append("INFERENCE_NOT_AVAILABLE: ate_inference returned None")
                except Exception as e:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                    item["ate_inference"] = None

                effects.append(item)

            finished = now_utc()
            return ATESuccess(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=finished,
                warnings=warnings_list,
                meta={
                    "backend": self.BACKEND_NAME,
                    "n": int(df.shape[0]),
                    "x_cols": spec.X if spec.X else None,
                    "contrast_kind": "baseline_vs_all",
                    "t0": t0,
                },
                fitted_model_id=command.fitted_model_id,
                contrast={"t0": t0, "t1": "vs_all"},
                ate=effects,
            )

        except Exception as e:
            return CommandFailure(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(code="ESTIMATOR_ERROR", message="ATE computation failed.", details={"exception": repr(e)}),
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
                    error=ErrorInfo(code="MODEL_NOT_FOUND", message="Fitted model not found.", details={"fitted_model_id": str(command.fitted_model_id)}),
                    warnings=[],
                    meta={},
                )

            est = model_record.model
            spec: CausalSpec = command.protocol_specs

            X_query = command.inputs.x_rows
            x_cols = spec.X
            raise_if_x_rows_not_exactly_match_fit_x_cols(x_rows=X_query, x_cols=x_cols)

            effects: List[Dict[CATEModelResult, Any]] = []

            if spec.T.kind == "binary":
                if len(spec.T.control_values) != 1 or len(spec.T.treated_values) != 1:
                    return CommandFailure(
                        run_id=command.run_id,
                        started_at=started_at,
                        finished_at=now_utc(),
                        error=ErrorInfo(
                            code="OPTIONS_INVALID",
                            message="Binary treatment for CATE requires exactly one control_value and one treated_value (or pre-normalize T upstream).",
                            details={"control_values": list(spec.T.control_values), "treated_values": list(spec.T.treated_values)},
                        ),
                        warnings=[],
                        meta={},
                    )
                t0 = spec.T.control_values[0]
                t1s = [spec.T.treated_values[0]]
            elif spec.T.kind == "categorical":
                t0, t1s = categorical_t0_t1_pairs(spec)
            else:
                return CommandFailure(
                    run_id=command.run_id,
                    started_at=started_at,
                    finished_at=now_utc(),
                    error=ErrorInfo(code="UNSUPPORTED_QUERY", message=f"Unsupported treatment kind {spec.T.kind!r} for CATE.", details={}),
                    warnings=[],
                    meta={},
                )

            for t1_val in t1s:
                if t1_val == t0:
                    continue

                item: Dict[CATEModelResult, Any] = {"for_treatment": {"t0": t0, "t1": t1_val}}

                try:
                    item["cate"] = est.effect(X_query, T0=t0, T1=t1_val)  # pyright: ignore
                except Exception as e:
                    return CommandFailure(
                        run_id=command.run_id,
                        started_at=started_at,
                        finished_at=now_utc(),
                        error=ErrorInfo(code="ESTIMATOR_ERROR", message="CATE computation failed (effect).", details={"exception": repr(e)}),
                        warnings=[],
                        meta={},
                    )

                try:
                    lo, hi = est.effect_interval(X_query, T0=t0, T1=t1_val, alpha=command.inputs.alpha)  # pyright: ignore
                    item["cate_interval"] = (list(lo), list(hi)) if lo is not None and hi is not None else None  # pyright: ignore
                    if item["cate_interval"] is None:
                        warnings_list.append("INFERENCE_NOT_AVAILABLE: effect_interval returned None")
                except Exception as e:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                    item["cate_interval"] = None

                try:
                    inf = est.effect_inference(X_query, T0=t0, T1=t1_val)  # pyright: ignore
                    item["cate_inference"] = serialize_inference_obj(inf) if inf is not None else None
                    if inf is None:
                        warnings_list.append("INFERENCE_NOT_AVAILABLE: effect_inference returned None")
                except Exception as e:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                    item["cate_inference"] = None

                effects.append(item)

            finished = now_utc()
            return CATESuccess(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=finished,
                warnings=warnings_list,
                meta={"backend": self.BACKEND_NAME, "row_count": int(getattr(X_query, "shape", [len(X_query)])[0])},
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
            return CommandFailure(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(code="ESTIMATOR_ERROR", message="CATE computation failed.", details={"exception": repr(e)}),
                warnings=[],
                meta={},
            )


# =============================================================================
# Concrete adapters
# =============================================================================

@dataclass(frozen=True, slots=True)
class LinearDRLearnerCausalModel(_BaseDRLearnerAdapter):
    ESTIMATOR_CLS: Any = LinearDRLearner
    BACKEND_NAME: str = "econml.dr.LinearDRLearner"
    INFO :str = get_linear_dr_learner_causal_model_info()


@dataclass(frozen=True, slots=True)
class ForestDRLearnerCausalModel(_BaseDRLearnerAdapter):
    ESTIMATOR_CLS: Any = ForestDRLearner
    BACKEND_NAME: str = "econml.dr.ForestDRLearner"
    INFO :str = get_forest_dr_learner_causal_model_info()

@dataclass(frozen=True, slots=True)
class SparseLinearDRLearnerCausalModel(_BaseDRLearnerAdapter):
    ESTIMATOR_CLS: Any = SparseLinearDRLearner
    BACKEND_NAME: str = "econml.dr.SparseLinearDRLearner"
    INFO :str = get_sparse_linear_dr_learner_causal_model_info()