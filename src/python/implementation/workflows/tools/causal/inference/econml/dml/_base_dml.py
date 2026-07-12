from __future__ import annotations

import inspect
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from python.domain.repo.models_repo import ModelRecord, ModelsRepo
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.causal.encoding.encoding_util import EncodingUtil
from python.implementation.workflows.tools.causal.inference.causal_command import (
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
from python.implementation.workflows.tools.causal.inference.causal_model import (
    CausalCommand,
    CausalModel,
    CausalResult,
)
from python.implementation.workflows.tools.causal.inference.econml.dml.shared_nuisance_models import (
    get_default_models_for_t_and_y as _get_default_models_for_t_and_y,
)
from python.implementation.workflows.tools.causal.inference.econml.utils import (
    ModelSpecError,
    build_init_fit_options_param_maps,
    get_input_params_from_spec,
    get_treatment_t0_t1_from_spec,
    now_utc,
    raise_if_x_rows_not_exactly_match_fit_x_cols,
    required_init_keys,
    serialize_econml_sensitivity_analysis,
    serialize_inference_obj,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec

log = get_logger(__name__)

# =============================================================================
# Reproducibility and scientifically conservative DML defaults
# =============================================================================

_RUN_SEED_ENV = "PRECISION_MEDICINE_RUN_SEED"
_MAX_SKLEARN_SEED = 2**32 - 1

_DML_CV_FOLDS = 5
_DML_MC_ITERS = 1
_DML_MC_AGG = "median"

_SPARSE_LINEAR_DML_MAX_ITER = 10_000
_SPARSE_LINEAR_DML_TOL = 1e-4

_CAUSAL_FOREST_N_ESTIMATORS = 1_000
_CAUSAL_FOREST_SUBFOREST_SIZE = 4
_CAUSAL_FOREST_MAX_SAMPLES = 0.45
_CAUSAL_FOREST_MIN_SAMPLES_LEAF = 20
_CAUSAL_FOREST_MIN_BALANCEDNESS_TOL = 0.45


def _shape_as_list(x: Any) -> list[int] | None:
    if x is None:
        return None
    if hasattr(x, "shape"):
        return list(x.shape)
    return list(np.asarray(x).shape)


def _ndim_or_none(x: Any) -> int | None:
    if x is None:
        return None
    if hasattr(x, "ndim"):
        return int(x.ndim)
    return int(np.asarray(x).ndim)


def _estimator_debug_name(value: Any) -> str:
    if isinstance(value, Pipeline):
        try:
            final_step = value.steps[-1][1]
            return f"Pipeline(..., {type(final_step).__name__})"
        except Exception:
            return "Pipeline"
    return type(value).__name__


def _debug_init_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in defaults.items():
        if isinstance(value, list):
            out[key] = [_estimator_debug_name(item) for item in value]
        elif isinstance(value, BaseEstimator):
            out[key] = _estimator_debug_name(value)
        else:
            out[key] = value
    return out


def _dtype_or_none(x: Any) -> str | None:
    if x is None:
        return None
    if hasattr(x, "dtype"):
        return str(x.dtype)
    try:
        return str(np.asarray(x).dtype)
    except Exception:
        return None


def _inference_attr_debug(value: Any) -> dict[str, Any]:
    try:
        arr = np.asarray(value)
        return {
            "type": type(value).__name__,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
    except Exception as exc:
        return {"type": type(value).__name__, "error": repr(exc)}


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_required_init_keys(estimator_cls: Any, *, init_map: dict[str, Any]) -> list[str]:
    keys = list(required_init_keys(estimator_cls, init_map=init_map))
    return [k for k in keys if k not in ("args", "kwargs")]


def _supports_param(init_map: dict[str, Any], name: str) -> bool:
    return name in init_map


def _set_if_supported(
    defaults: dict[str, Any],
    init_map: dict[str, Any],
    name: str,
    value: Any,
) -> None:
    if value is None:
        return
    if _supports_param(init_map, name):
        defaults[name] = value


def _configured_run_seed() -> int | None:
    """
    Resolve the analysis seed from PRECISION_MEDICINE_RUN_SEED.

    Examples:
        PRECISION_MEDICINE_RUN_SEED=1729
            Reproduce the primary manuscript analysis.

        PRECISION_MEDICINE_RUN_SEED=2718
            Run a prespecified seed-sensitivity analysis.

        PRECISION_MEDICINE_RUN_SEED=none
            Run an explicitly unseeded exploratory analysis.

    The default is 1729 so production and manuscript executions are seeded
    unless unseeded behavior is requested explicitly.
    """
    raw_value = os.getenv(_RUN_SEED_ENV)
    if raw_value is None:
        return None

    normalized = raw_value.strip().lower()
    if normalized in {"", "none", "null", "random", "unseeded"}:
        return None

    try:
        seed = int(normalized)
    except ValueError as exc:
        raise ModelSpecError(
            f"{_RUN_SEED_ENV} must be an integer or one of "
            "{'none', 'null', 'random', 'unseeded'}. "
            f"Received: {raw_value!r}."
        ) from exc

    if not 0 <= seed <= _MAX_SKLEARN_SEED:
        raise ModelSpecError(
            f"{_RUN_SEED_ENV} must be between 0 and {_MAX_SKLEARN_SEED}, "
            f"inclusive. Received: {seed}."
        )

    return seed


def set_dml_defaults(
    defaults: dict[str, Any],
    init_map: dict[str, Any],
    *,
    run_seed: int | None,
) -> None:
    """
    Apply common cross-fitting defaults to every EconML DML estimator.

    Five-fold cross-fitting preserves more training data per nuisance fit than
    the two-fold library default. Repeating the nuisance stage three times and
    aggregating by the median reduces sensitivity to a single random partition.
    """
    _set_if_supported(defaults, init_map, "random_state", run_seed)
    _set_if_supported(defaults, init_map, "cv", _DML_CV_FOLDS)
    _set_if_supported(defaults, init_map, "mc_iters", _DML_MC_ITERS)
    _set_if_supported(defaults, init_map, "mc_agg", _DML_MC_AGG)


def set_sparse_linear_dml_defaults(
    defaults: dict[str, Any],
    init_map: dict[str, Any],
    *,
    run_seed: int | None,
) -> None:
    """
    Extend common DML defaults for SparseLinearDML.

    A larger iteration budget reduces avoidable non-convergence in the
    debiased-lasso final stage without changing its estimand.
    """
    set_dml_defaults(defaults, init_map, run_seed=run_seed)
    _set_if_supported(
        defaults,
        init_map,
        "max_iter",
        _SPARSE_LINEAR_DML_MAX_ITER,
    )
    _set_if_supported(defaults, init_map, "tol", _SPARSE_LINEAR_DML_TOL)


def set_causal_forest_defaults(
    defaults: dict[str, Any],
    init_map: dict[str, Any],
    *,
    run_seed: int | None,
) -> None:
    """
    Extend common DML defaults for CausalForestDML.

    The forest remains honest and uses subsampling compatible with EconML's
    built-in forest inference. The larger tree count and moderate leaf size
    reduce Monte Carlo noise and avoid highly local, unstable CATE estimates.
    """
    set_dml_defaults(defaults, init_map, run_seed=run_seed)

    _set_if_supported(
        defaults,
        init_map,
        "n_estimators",
        _CAUSAL_FOREST_N_ESTIMATORS,
    )
    _set_if_supported(
        defaults,
        init_map,
        "subforest_size",
        _CAUSAL_FOREST_SUBFOREST_SIZE,
    )
    _set_if_supported(
        defaults,
        init_map,
        "max_samples",
        _CAUSAL_FOREST_MAX_SAMPLES,
    )
    _set_if_supported(
        defaults,
        init_map,
        "min_samples_leaf",
        _CAUSAL_FOREST_MIN_SAMPLES_LEAF,
    )
    _set_if_supported(defaults, init_map, "honest", True)
    _set_if_supported(defaults, init_map, "inference", True)
    _set_if_supported(defaults, init_map, "criterion", "mse")
    _set_if_supported(
        defaults,
        init_map,
        "min_balancedness_tol",
        _CAUSAL_FOREST_MIN_BALANCEDNESS_TOL,
    )
    _set_if_supported(defaults, init_map, "n_jobs", -1)


def _raise_if_x_not_numeric(X: Any) -> None:
    if X is None:
        return

    if isinstance(X, pd.DataFrame):
        bad = [c for c in X.columns if not is_numeric_dtype(X[c])]
        if bad:
            raise ModelSpecError(
                "KernelDML requires numeric X (no strings/datetimes). "
                f"Non-numeric X columns: {bad}. Encode/transform X upstream."
            )
        return

    arr = np.asarray(X)
    if arr.dtype == object:
        raise ModelSpecError(
            "KernelDML requires numeric X; got object dtype. Encode/transform X upstream."
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise ModelSpecError(f"KernelDML requires numeric X; got dtype={arr.dtype}.")


@dataclass(frozen=True, slots=True)
class _ResolvedInferenceContext:
    inference_ready_spec: InferenceReadyCausalSpec
    specs: CausalSpec
    effect_modifiers_order: list[str]
    covariates_order: list[str]


def _resolve_inference_context(
    command: FitCommand | ATECommand | CATECommand,
) -> _ResolvedInferenceContext:
    inference_ready_spec = command.inference_ready_spec
    return _ResolvedInferenceContext(
        inference_ready_spec=inference_ready_spec,
        specs=inference_ready_spec.causal_spec,
        effect_modifiers_order=inference_ready_spec.get_effect_modifiers_order(),
        covariates_order=inference_ready_spec.get_covariates_order(),
    )


@dataclass(frozen=True, slots=True)
class _BaseDMLAdapter(CausalModel):
    models_repo: ModelsRepo
    encoding_util: EncodingUtil

    ESTIMATOR_CLS: ClassVar[Any]
    BACKEND_NAME: ClassVar[str]
    INFO: ClassVar[str]
    FIT_INCLUDE_NAMES: ClassVar[set[str]]
    USE_PRE_X_AS_FEATURIZER: ClassVar[bool] = True
    REQUIRE_NUMERIC_X: ClassVar[bool] = False
    DROP_FIRST_EFFECT_MODIFIER_ONEHOT: ClassVar[bool] = False

    def get_info(self) -> str:
        return self.INFO

    def get_command_info(self, command: CommandType) -> str | None:
        match command:
            case "FIT":
                fit_doc = inspect.getdoc(self.ESTIMATOR_CLS.fit) or ""
                base_doc = inspect.getdoc(self.ESTIMATOR_CLS) or ""
                return base_doc + fit_doc
            case "ATE":
                return inspect.getdoc(self.ESTIMATOR_CLS.ate) or ""
            case "CATE":
                return inspect.getdoc(self.ESTIMATOR_CLS.effect) or ""
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
        df = command.df

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
            ctx = _resolve_inference_context(command)
            inference_ready_spec = ctx.inference_ready_spec
            specs = ctx.specs

            plan = self.encoding_util.compile(
                plan=inference_ready_spec.transformation_plan,
                effect_modifiers_order=ctx.effect_modifiers_order,
                covariates_order=ctx.covariates_order,
                dense_output=True,
                drop_first_effect_modifier_onehot=self.DROP_FIRST_EFFECT_MODIFIER_ONEHOT,
            )

            pre_x = plan.pre_X
            pre_xw = plan.pre_XW

            if pre_x is None and inference_ready_spec.has_effect_modifiers():
                raise ModelSpecError(
                    "Spec declares effect modifiers but no pre_X transformer was provided."
                )
            if pre_xw is None and inference_ready_spec.has_adjustment_columns():
                raise ModelSpecError(
                    "Spec declares covariates and/or effect modifiers but no pre_XW transformer was provided."
                )

            Y, T, X, W, col_meta = get_input_params_from_spec(
                df,
                specs,
                effect_modifiers_order=ctx.effect_modifiers_order,
                covariates_order=ctx.covariates_order,
            )

            try:
                inference_ready_spec.assert_effect_modifiers_missingness_is_allowed()
                inference_ready_spec.assert_covariates_missingness_is_allowed()
            except ValueError as exc:
                raise ModelSpecError(str(exc)) from exc

            missingness_X = inference_ready_spec.has_unhandled_missing_effect_modifiers()
            missingness_W = inference_ready_spec.requires_allow_missing_for_covariates()
            if missingness_X:
                raise ModelSpecError(
                    f"{self.BACKEND_NAME} does not support missing values in X via allow_missing "
                    "(only W is allowed). Impute/clean X upstream before fit."
                )

            self._validate_fit_x(X)

            maps = build_init_fit_options_param_maps(
                self.ESTIMATOR_CLS,
                fit_include_names=self.FIT_INCLUDE_NAMES,
            )
            init_map = maps["init"]
            defaults: dict[str, Any] = {}
            run_seed = _configured_run_seed()

            _set_if_supported(
                defaults,
                init_map,
                "discrete_treatment",
                specs.treatment_spec.kind in ("binary", "categorical"),
            )
            _set_if_supported(
                defaults,
                init_map,
                "discrete_outcome",
                specs.outcome_spec.kind == "binary",
            )
            _set_if_supported(defaults, init_map, "allow_missing", missingness_W)
            # Apply the same seeded cross-fitting policy to every DML estimator.
            # Class-specific helpers extend, rather than replace, these defaults.
            if self.BACKEND_NAME == "econml.dml.CausalForestDML":
                set_causal_forest_defaults(
                    defaults,
                    init_map,
                    run_seed=run_seed,
                )
            elif self.BACKEND_NAME == "econml.dml.SparseLinearDML":
                set_sparse_linear_dml_defaults(
                    defaults,
                    init_map,
                    run_seed=run_seed,
                )
            else:
                set_dml_defaults(
                    defaults,
                    init_map,
                    run_seed=run_seed,
                )

            if pre_xw is not None:
                nuisance_defaults = _get_default_models_for_t_and_y(
                    specs,
                    pre_XW=pre_xw,
                    missingness=missingness_W,
                    random_state=run_seed,
                )
                for key, value in nuisance_defaults.items():
                    _set_if_supported(defaults, init_map, key, value)

            if self.USE_PRE_X_AS_FEATURIZER and pre_x is not None:
                _set_if_supported(defaults, init_map, "featurizer", pre_x)

            self._extend_init_defaults(
                defaults=defaults,
                init_map=init_map,
                specs=specs,
                pre_x=pre_x,
                pre_xw=pre_xw,
                X=X,
                W=W,
            )

            required_keys = _safe_required_init_keys(self.ESTIMATOR_CLS, init_map=init_map)
            missing_required = [k for k in required_keys if k not in defaults]
            if missing_required:
                raise ModelSpecError(
                    f"Missing required {self.BACKEND_NAME} __init__ parameters: {missing_required}. "
                    "(Adapter is not exposing command.options yet.)"
                )

            est = self.ESTIMATOR_CLS(**defaults)
            log.info(
                "%s fit prepared",
                self.BACKEND_NAME,
                backend=self.BACKEND_NAME,
                estimator_cls=getattr(self.ESTIMATOR_CLS, "__name__", str(self.ESTIMATOR_CLS)),
                run_seed=run_seed,
                n=int(df.shape[0]),
                y_shape=_shape_as_list(Y),
                y_ndim=_ndim_or_none(Y),
                t_shape=_shape_as_list(T),
                t_ndim=_ndim_or_none(T),
                x_shape=_shape_as_list(X),
                x_ndim=_ndim_or_none(X),
                w_shape=_shape_as_list(W),
                w_ndim=_ndim_or_none(W),
                allow_missing_requested=bool(defaults.get("allow_missing")),
                missingness_X=missingness_X,
                missingness_W=missingness_W,
                x_columns=ctx.effect_modifiers_order,
                w_columns=ctx.covariates_order,
                init_defaults=_debug_init_defaults(defaults),
            )

            fit_warnings: list[str] = []
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                est.fit(Y, T, X=X, W=W)  # pyright: ignore[reportUnknownMemberType]
            fit_warnings = [f"{w.category.__name__}: {str(w.message)}" for w in ws]
            fit_warning_details = [
                {
                    "category": w.category.__name__,
                    "message": str(w.message),
                    "filename": w.filename,
                    "lineno": int(w.lineno),
                }
                for w in ws
            ]
            for warning_detail in fit_warning_details:
                log.warning(
                    "%s fit warning",
                    self.BACKEND_NAME,
                    backend=self.BACKEND_NAME,
                    warning_category=warning_detail["category"],
                    warning_message=warning_detail["message"],
                    warning_filename=warning_detail["filename"],
                    warning_lineno=warning_detail["lineno"],
                )
            log.info(
                "%s fit completed",
                self.BACKEND_NAME,
                backend=self.BACKEND_NAME,
                warning_count=len(fit_warnings),
                score_available=hasattr(est, "score_"),
            )

            artifacts: dict[str, Any] = {
                "n": int(df.shape[0]),
                "y_shape": _shape_as_list(Y),
                "y_ndim": _ndim_or_none(Y),
                "t_shape": _shape_as_list(T),
                "t_ndim": _ndim_or_none(T),
                "x_shape": _shape_as_list(X),
                "x_ndim": _ndim_or_none(X),
                "w_shape": _shape_as_list(W),
                "w_ndim": _ndim_or_none(W),
                "fit_warning_details": fit_warning_details,
                **self._extra_fit_artifacts(est),
            }

            fit_meta: dict[str, Any] = {
                "warnings": fit_warnings,
                "meta": {
                    "backend": self.BACKEND_NAME,
                    "n": int(df.shape[0]),
                    "run_seed": run_seed,
                    "run_seed_env": _RUN_SEED_ENV,
                    "dml_cross_fitting": {
                        "cv": defaults.get("cv"),
                        "mc_iters": defaults.get("mc_iters"),
                        "mc_agg": defaults.get("mc_agg"),
                    },
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
            log.exception("%s command failed", self.BACKEND_NAME, error=e)
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
            warnings_list: list[str] = []
            ctx = _resolve_inference_context(command)

            model_record: ModelRecord | None = self.models_repo.load_model(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=command.fitted_model_id,
            )
            if model_record is None:
                raise ModelSpecError(f"Fitted model with id {command.fitted_model_id} not found.")

            est = model_record.model
            t0, t1 = get_treatment_t0_t1_from_spec(
                ctx.specs,
                is_global_counter_factual=False,
            )

            Y, T, X, _, _ = get_input_params_from_spec(
                df,
                ctx.specs,
                effect_modifiers_order=ctx.effect_modifiers_order,
                covariates_order=ctx.covariates_order,
            )
            self._validate_ate_x(X)

            item: dict[ATEModelResult, Any] = {"for_treatment": {"t0": t0, "t1": t1}}
            log.info(
                "%s ATE inference input dtypes",
                self.BACKEND_NAME,
                backend=self.BACKEND_NAME,
                fitted_model_id=str(command.fitted_model_id),
                X_type=str(type(X)),
                X_dtype=_dtype_or_none(X),
                Y_dtype=_dtype_or_none(Y),
                T_dtype=_dtype_or_none(T),
            )
            item["ate"] = est.ate(
                X=X, T0=t0, T1=t1
            )  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
            log.info(
                "%s ATE point estimate",
                self.BACKEND_NAME,
                backend=self.BACKEND_NAME,
                fitted_model_id=str(command.fitted_model_id),
                ate=_to_jsonable(np.asarray(item["ate"])),
                ate_type=str(type(item["ate"])),
                ate_dtype=_dtype_or_none(item["ate"]),
            )

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
                    log.info(
                        "%s ATE interval object",
                        self.BACKEND_NAME,
                        backend=self.BACKEND_NAME,
                        fitted_model_id=str(command.fitted_model_id),
                        interval_type=str(type(ate_interval)),
                        interval_dtype=_dtype_or_none(ate_interval),
                    )
            except Exception as e:
                log.exception(
                    "%s EconML ATE interval failed",
                    self.BACKEND_NAME,
                    backend=self.BACKEND_NAME,
                    fitted_model_id=str(command.fitted_model_id),
                    exception=repr(e),
                )
                warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                item["ate_interval"] = None

            try:
                inf = est.ate_inference(
                    X=X, T0=t0, T1=t1
                )  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
                if inf is None:
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: ate_inference returned None")
                    item["ate_inference"] = None
                else:
                    log.info(
                        "%s ATE inference object",
                        self.BACKEND_NAME,
                        backend=self.BACKEND_NAME,
                        fitted_model_id=str(command.fitted_model_id),
                        inf_type=str(type(inf)),
                        point_type=str(type(getattr(inf, "point_estimate", None))),
                        point_dtype=_dtype_or_none(getattr(inf, "point_estimate", None)),
                        stderr_type=str(type(getattr(inf, "stderr", None))),
                        stderr_dtype=_dtype_or_none(getattr(inf, "stderr", None)),
                        point_debug=_inference_attr_debug(getattr(inf, "point_estimate", None)),
                        stderr_debug=_inference_attr_debug(getattr(inf, "stderr", None)),
                    )
                    item["ate_inference"] = serialize_inference_obj(inf)
            except Exception as e:
                log.exception(
                    "%s EconML ATE inference failed",
                    self.BACKEND_NAME,
                    backend=self.BACKEND_NAME,
                    fitted_model_id=str(command.fitted_model_id),
                    exception=repr(e),
                )
                warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                item["ate_inference"] = None

            sensitivity_fields, sensitivity_warnings = serialize_econml_sensitivity_analysis(
                est,
                treatment_value=t1,
                alpha=command.inputs.alpha,
            )
            item.update(sensitivity_fields)
            warnings_list.extend(sensitivity_warnings)

            finished = now_utc()
            return ATESuccess(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=finished,
                warnings=warnings_list,
                meta={
                    "backend": self.BACKEND_NAME,
                    "n": int(df.shape[0]),
                    "x_cols": ctx.effect_modifiers_order or None,
                    "contrast_kind": "single_pair",
                    "t0": t0,
                },
                fitted_model_id=command.fitted_model_id,
                contrast={"t0": t0, "t1": t1},
                ate=[item],
            )
        except Exception as e:
            log.exception("%s command failed", self.BACKEND_NAME, error=e)
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

    def _cate(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: CATECommand,
        started_at: datetime,
    ) -> CausalResult:
        warnings_list: list[str] = []
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
            ctx = _resolve_inference_context(command)
            x_cols = ctx.effect_modifiers_order
            X_df = command.inputs.x_rows
            raise_if_x_rows_not_exactly_match_fit_x_cols(x_rows=X_df, x_cols=x_cols)

            X_query_df = X_df[x_cols].copy() if x_cols else None
            if X_query_df is None or X_query_df.shape[1] == 0:
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

            X_query = self._prepare_cate_query(X_query_df)
            t0, t1 = get_treatment_t0_t1_from_spec(
                ctx.specs,
                is_global_counter_factual=command.inputs.counterfactual,
            )

            effects: dict[CATEModelResult, Any] = {"for_treatment": {"t0": t0, "t1": t1}}
            try:
                effects["cate"] = est.effect(
                    X_query, T0=t0, T1=t1
                )  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
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
                inf = est.effect_inference(
                    X_query, T0=t0, T1=t1
                )  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
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
            log.exception("%s command failed", self.BACKEND_NAME, error=e)
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

    def _validate_fit_x(self, X: Any) -> None:
        if self.REQUIRE_NUMERIC_X:
            _raise_if_x_not_numeric(X)

    def _validate_ate_x(self, X: Any) -> None:
        if self.REQUIRE_NUMERIC_X:
            _raise_if_x_not_numeric(X)

    def _prepare_cate_query(self, X_query_df: pd.DataFrame) -> Any:
        if self.REQUIRE_NUMERIC_X:
            _raise_if_x_not_numeric(X_query_df)
        return X_query_df

    def _extend_init_defaults(
        self,
        *,
        defaults: dict[str, Any],
        init_map: dict[str, Any],
        specs: CausalSpec,
        pre_x: Any,
        pre_xw: Any,
        X: Any,
        W: Any,
    ) -> None:
        _ = (defaults, init_map, specs, pre_x, pre_xw, X, W)

    def _extra_fit_artifacts(self, est: Any) -> dict[str, Any]:
        _ = est
        return {}


__all__ = [
    "_BaseDMLAdapter",
    "_configured_run_seed",
    "_to_jsonable",
    "set_causal_forest_defaults",
    "set_dml_defaults",
    "set_sparse_linear_dml_defaults",
]