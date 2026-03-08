from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import issparse  # type: ignore[import]

from econml.dml import CausalForestDML # pyright: ignore[reportMissingTypeStubs]
import inspect

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

from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper

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
from python.implementation.workflows.tools.causal.causal_model import CausalCommand, CausalModel, CausalResult
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec

# NOTE: you should ideally add causal_forest_dml_causal_model_info()
from python.implementation.workflows.tools.causal.econml.models_info import get_causal_forest_dml_causal_model_info

from python.implementation.workflows.tools.causal.econml.utils import (
    ModelSpecError,
    build_init_fit_options_param_maps,
    get_input_params_from_spec,
    has_missing,
    is_missing_handled,
    now_utc,
    raise_if_x_rows_not_exactly_match_fit_x_cols,
    required_init_keys,
    serialize_inference_obj,
)
from python.implementation.workflows.tools.causal.encoding_util import EncodingUtil
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.tools.causal.encoding_plan import TransformPlan


# =============================================================================
# Model spec and options (unchanged)
# =============================================================================

class _ToDense(BaseEstimator, TransformerMixin):
    """Convert sparse -> dense for models that don't accept sparse."""
    def fit(self, X, y=None):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
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
    steps: List[tuple[str, BaseEstimator]] = [("pre", pre_XW)]
    if require_dense:
        steps.append(("dense", _ToDense()))
    steps.append(("model", model))
    return Pipeline(steps)


def _normalize_model_spec_to_wrapped_list(
    *,
    spec_value: Union[str, BaseEstimator, Sequence[Union[str, BaseEstimator]]],
    pre_XW: ColumnTransformer,
    is_discrete: bool,
    missingness: bool,
    random_state: Optional[int],
    n_jobs: Optional[int],
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
        # NaN-safe fallback (dense); good when you truly cannot impute upstream.
        if is_discrete:
            hgb = HistGradientBoostingClassifier(
                random_state=random_state,
                max_depth=None,
                learning_rate=0.05,
                max_iter=400,
                early_stopping=True,
            )
            return [_wrap_with_pre(pre_XW=pre_XW, model=hgb, require_dense=True)]
        hgb = HistGradientBoostingRegressor(
            random_state=random_state,
            max_depth=None,
            learning_rate=0.05,
            max_iter=400,
            early_stopping=True,
        )
        return [_wrap_with_pre(pre_XW=pre_XW, model=hgb, require_dense=True)]

    def build_linear_candidates() -> Sequence[BaseEstimator]:
        if missing_present:
            return build_boosting_candidates_nan_safe()

        if is_discrete:
            lr = LogisticRegressionCV(
                max_iter=2000,
                solver="lbfgs",
                n_jobs=n_jobs,
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

        # WARNING: these densify; can explode RAM if pre_XW creates huge sparse one-hot matrices.
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

    # Normalize input to list
    items: List[Union[str, BaseEstimator]]
    if isinstance(spec_value, (str, BaseEstimator)):
        items = [spec_value]
    else:
        items = list(spec_value)

    out: list[BaseEstimator] = []
    for item in items:
        if isinstance(item, str):
            out.extend(candidates_for_keyword(item))
        else:
            # User provided a concrete estimator/pipeline; we still wrap to enforce pre_XW.
            out.append(_wrap_with_pre(pre_XW=pre_XW, model=item, require_dense=True))  # pyright: ignore[reportArgumentType]

    if not out:
        raise ValueError("Empty nuisance model candidate list.")
    return out


def _get_default_models_for_t_and_y(
    specs: CausalSpec,
    pre_XW: ColumnTransformer,
    *,
    missingness: bool,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    disc_t = specs.treatment_spec.kind in ("binary", "categorical")
    disc_y = specs.outcome_spec.kind == "binary"

    default_model_y: Union[str, BaseEstimator, Sequence[Union[str, BaseEstimator]]] = "auto_plus"
    default_model_t: Union[str, BaseEstimator, Sequence[Union[str, BaseEstimator]]] = "auto_plus"

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


# =============================================================================
# CausalForestDML adapter
# =============================================================================

@dataclass(frozen=True, slots=True)
class CausalForestDMLCausalModel(CausalModel):
    data_repo: DataRepo
    models_repo: ModelsRepo
    encoding_util: EncodingUtil

    def get_info(self) -> str:
        return get_causal_forest_dml_causal_model_info()

    def get_command_info(self, command: CommandType) -> str | None:
        match command:
            case "FIT":
                fit_doc = inspect.getdoc(CausalForestDML.fit) or "" # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                base_doc = inspect.getdoc(CausalForestDML) or ""
                return base_doc + fit_doc
            case "ATE":
                ate_doc = inspect.getdoc(CausalForestDML.ate) or "" # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                return ate_doc
            case "CATE":
                effect_doc = inspect.getdoc(CausalForestDML.effect) or "" # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
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
                    message="Failed to load dataset for FIT.",
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
            covariates_order = specs.covariates or []
            effect_modifiers_order = specs.effect_modifiers
            plan = self.encoding_util.compile(
                plan=transformation_plan,
                effect_modifiers_order=effect_modifiers_order,
                covariates_order=covariates_order,
                dense_output=True,
            ) if transformation_plan is not None else None
            
            pre_x = plan.pre_X if plan is not None else None
            pre_xw = plan.pre_XW if plan is not None else None

            if pre_x is None and len(specs.effect_modifiers or []) > 0:
                raise ModelSpecError(
                    "Spec declares effect modifiers (spec.X) but no pre_X transformer provided in inputs."
                )
            if pre_xw is None and (len(specs.covariates or []) + len(specs.effect_modifiers or [])) > 0:
                raise ModelSpecError(
                    "Spec declares controls (spec.W) and/or effect modifiers (spec.X) but no pre_XW transformer provided in inputs."
                )

            Y, T, X, W, col_meta = get_input_params_from_spec(df, specs, effect_modifiers_order=effect_modifiers_order, covariates_order=covariates_order)

            miss = {"Y": has_missing(Y), "T": has_missing(T), "X": has_missing(X), "W": has_missing(W)}
            if miss["Y"] or miss["T"]:
                raise ModelSpecError(f"Y/T contain missing values; must be fixed upstream. missing={miss}")

            # CHANGED (Forest): CausalForestDML does NOT allow missing X via allow_missing;
            # allow_missing only applies to W. If X contains NaNs, force upstream imputation/cleaning.
            missingness_X = len(specs.effect_modifiers or []) > 0 and miss["X"] and (transformation_plan is  None or not is_missing_handled(plan=transformation_plan,summary=data_summary, col_name_list=specs.effect_modifiers))
            missingness_W = len(specs.covariates or []) > 0 and miss["W"] and (transformation_plan is  None or not is_missing_handled(plan=transformation_plan,summary=data_summary, col_name_list=specs.covariates))
            if missingness_X:
                raise ModelSpecError(
                    "CausalForestDML does not support missing values in X via allow_missing "
                    "(only W is allowed). Impute/clean X upstream before fit."
                )

            maps = build_init_fit_options_param_maps(
                CausalForestDML,  # CHANGED (Forest): estimator class
                fit_include_names={"cache_values", "inference", "sample_weight", "groups"},
            )
            init_map = maps["init"]

            defaults: Dict[str, Any] = {}

            disc_t = specs.T.kind in ("binary", "categorical")
            disc_y = specs.Y.kind == "binary"
            if disc_t:
                defaults["discrete_treatment"] = True
            if disc_y:
                defaults["discrete_outcome"] = True

            # CHANGED (Forest): allow_missing only controls W-missing acceptance
            # (safe to set True always; it just relaxes W checks).
            defaults["allow_missing"] = missingness_W

            if pre_xw is not None:
                defaults.update(
                    _get_default_models_for_t_and_y(
                        specs,
                        pre_XW=pre_xw,
                        missingness=missingness_W,
                    )
                )

            if pre_x is not None:
                # CHANGED (Forest): featurizer affects the forest splitting/features (not linear regression)
                defaults["featurizer"] = pre_x

            # Required init enforcement (likely empty for CausalForestDML since it has defaults everywhere,
            # but we keep your pattern)
            required_keys = required_init_keys(CausalForestDML, init_map=init_map)  # CHANGED (Forest)
            missing_required = [k for k in required_keys if k not in defaults]
            if missing_required:
                raise ModelSpecError(
                    f"Missing required CausalForestDML __init__ parameters: {missing_required}. "
                    f"(Adapter is not exposing command.options yet.)"
                )

            # CHANGED (Forest): estimator instance
            est = CausalForestDML(**defaults)

            fit_warnings: list[str] = []
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                # NOTE: we are not passing fit-time options; command.options disabled
                est.fit(Y, T, X=X, W=W)  # pyright: ignore[reportUnknownMemberType]
            fit_warnings = [f"{w.category.__name__}: {str(w.message)}" for w in ws]

            # Optional forest artifacts
            forest_artifacts: Dict[str, Any] = {}
            try:
                if hasattr(est, "feature_importances_"):
                    forest_artifacts["feature_importances_"] = list(getattr(est, "feature_importances_"))
            except Exception:
                pass
            # ate_ exists only in discrete_treatment + drate scenarios
            try:
                if hasattr(est, "ate_"):
                    forest_artifacts["ate_"] = np.asarray(getattr(est, "ate_")).tolist()
                if hasattr(est, "ate_stderr_"):
                    forest_artifacts["ate_stderr_"] = np.asarray(getattr(est, "ate_stderr_")).tolist()
            except Exception:
                pass

            n = int(df.shape[0])
            fit_meta: Dict[str, Any] = {
                "warnings": fit_warnings,
                "meta": {
                    "backend": "econml.dml.CausalForestDML",  # CHANGED (Forest)
                    "n": n,
                    "columns": col_meta,
                    "used_init_kwargs": defaults,
                    "spec_semantics_applied": sorted(list(required_keys)),
                },
                "artifacts": {
                    "n": n,
                    "y_shape": list(np.asarray(Y).shape),
                    "t_shape": list(np.asarray(T).shape),
                    "x_shape": (list(np.asarray(X).shape) if X is not None else None),
                    "w_shape": (list(np.asarray(W).shape) if W is not None else None),
                    **forest_artifacts,
                },
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
                error=ErrorInfo(code="ESTIMATOR_ERROR", message="EconML CausalForestDML.fit failed.", details={"exception": repr(e)}),
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
            order_X: Optional[List[str]] = command.order_X
            order_W: Optional[List[str]] = command.order_W

            model_record: ModelRecord | None = self.models_repo.load_model(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=command.fitted_model_id,
            )
            if model_record is None:
                raise ModelSpecError(f"Fitted model with id {command.fitted_model_id} not found.")

            # CHANGED (Forest): estimator type
            est: CausalForestDML = model_record.model

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
            _, _, X, _, _ = get_input_params_from_spec(df, spec, order_X=order_X, order_W=order_W)

            for t1_val in t1s:
                if t1_val == t0:
                    raise ModelSpecError(f"Invalid contrast: t1 value {t1_val} is the same as t0 baseline {t0}.")

                item: Dict[ATEModelResult, Any] = {"for_treatment": {"t0": t0, "t1": t1_val}}
                item["ate"] = est.ate(X=X, T0=t0, T1=t1_val)
                logging_info = f"Computed ATE for contrast t0={t0} vs t1={t1_val}: {item['ate']}"
                logging.info(logging_info)
                
                try:
                    ate_interval = est.ate_interval(X=X, T0=t0, T1=t1_val, alpha=command.inputs.alpha) 
                    if ate_interval is not None:
                        item["ate_interval"] = ate_interval
                    else:
                        item["ate_interval"] = None
                        warnings_list.append("INFERENCE_NOT_AVAILABLE: ate_interval returned None")
                except Exception as e:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                    item["ate_interval"] = None

                try:
                    inference = est.ate_inference(X=X, T0=t0, T1=t1_val)  # pyright: ignore
                    item["ate_inference"] = serialize_inference_obj(inference) if inference is not None else None
                    if inference is None:
                        warnings_list.append("INFERENCE_NOT_AVAILABLE: ate_inference returned None")
                except Exception as e:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                    item["ate_inference"] = None

                effects.append(item)

            if not effects:
                return CommandFailure(
                    run_id=command.run_id,
                    started_at=started_at,
                    finished_at=now_utc(),
                    error=ErrorInfo(code="OPTIONS_INVALID", message="No valid categorical contrasts found (baseline vs all).", details={}),
                    warnings=[],
                    meta={},
                )

            finished = now_utc()
            return ATESuccess(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=finished,
                warnings=warnings_list,
                meta={
                    "backend": "econml.dml.CausalForestDML",  # CHANGED (Forest)
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
            logging.exception(e)
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
                    error=ErrorInfo(
                        code="MODEL_NOT_FOUND",
                        message="Fitted model not found.",
                        details={"fitted_model_id": str(command.fitted_model_id)},
                    ),
                    warnings=[],
                    meta={},
                )

            # CHANGED (Forest): estimator type
            est: CausalForestDML = model_record.model
            spec: CausalSpec = command.protocol_specs
            X_order: Optional[List[str]] = command.order_X

            X_df = command.inputs.x_rows
            x_cols = spec.X
            raise_if_x_rows_not_exactly_match_fit_x_cols(x_rows=X_df, x_cols=x_cols)
            X_query = X_df[X_order].to_numpy() if X_order else None
            
            if X_query is None or X_query.shape[1] == 0:
                return CommandFailure(
                    run_id=command.run_id,
                    started_at=started_at,
                    finished_at=now_utc(),
                    error=ErrorInfo(code="OPTIONS_INVALID", message="CATE requires non-empty X for effect modification; none provided.", details={}),
                    warnings=[],
                    meta={},
                )

    
            if spec.T.kind == "binary":
                if len(spec.T.control_values) != 1 or len(spec.T.treated_values) != 1:
                    return CommandFailure(
                        run_id=command.run_id,
                        started_at=started_at,
                        finished_at=now_utc(),
                        error=ErrorInfo(
                            code="OPTIONS_INVALID",
                            message="Binary treatment for CATE requires exactly one control_value and one treated_value (or pre-normalize T upstream).",
                            details={
                                "control_values": list(spec.T.control_values),
                                "treated_values": list(spec.T.treated_values),
                            },
                        ),
                        warnings=[],
                        meta={},
                    )
                t0 = command.inputs.t0
                t1 = command.inputs.t1
                if t0 == t1 or t0 is None or t1 is None:
                    return CommandFailure(
                        run_id=command.run_id,
                        started_at=started_at,
                        finished_at=now_utc(),
                        error=ErrorInfo(code="OPTIONS_INVALID", message=f"Invalid contrast: t0 value {t0} is the same as t1 value {t1}.", details={}),
                        warnings=[],
                        meta={},
                    )
                    
                
            else:
                return CommandFailure(
                    run_id=command.run_id,
                    started_at=started_at,
                    finished_at=now_utc(),
                    error=ErrorInfo(code="UNSUPPORTED_QUERY", message=f"Unsupported treatment kind {spec.T.kind!r} for CATE.", details={}),
                    warnings=[],
                    meta={},
                )

  

            effects: Dict[CATEModelResult, Any] = {"for_treatment": {"t0": t0, "t1": t1}}

            try:
                effects["cate"] = est.effect(X_query, T0=t0, T1=t1)  # pyright: ignore
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
                interval = est.effect_interval(X_query, T0=t0, T1=t1, alpha=command.inputs.alpha)  # pyright: ignore
                if interval is None:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: effect_interval returned None")
                    effects["cate_interval"] = None
                else:
                        effects["cate_interval"] = interval
            except Exception as e:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                    effects["cate_interval"] = None

            try:
                    inf = est.effect_inference(X_query, T0=t0, T1=t1_val)  # pyright: ignore
                    if inf is None:
                        warnings_list.append("INFERENCE_NOT_AVAILABLE: effect_inference returned None")
                        effects["cate_inference"] = None
                    else:
                        effects["cate_inference"] = serialize_inference_obj(inf)
            except Exception as e:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                    effects["cate_inference"] = None


            if effects.get("cate", None) is None:
                return CommandFailure(
                    run_id=command.run_id,
                    started_at=started_at,
                    finished_at=now_utc(),
                    error=ErrorInfo(code="OPTIONS_INVALID", message="No valid contrasts produced for CATE.", details={}),
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
                    "backend": "econml.dml.CausalForestDML",
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
            return CommandFailure(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(code="ESTIMATOR_ERROR", message="CATE computation failed.", details={"exception": repr(e)}),
                warnings=[],
                meta={},
            )