from __future__ import annotations

import os
import warnings as py_warnings
from dataclasses import dataclass
from multiprocessing import Manager
from queue import Empty
from threading import Event, Thread
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from econml.validate import DRTester
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from statsmodels.api import OLS

from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.tools.causal.inference.causal_command import (
    CATECommand,
    CATEInputs,
    CATESuccess,
    CommandFailure,
    ErrorInfo,
    FitCommand,
    FitInputs,
    FitSuccess,
    ValidateCommand,
    ValidateResult,
    ValidateSuccess,
)
from python.implementation.workflows.tools.causal.inference.econml import (
    model_training_config,
)
from python.implementation.workflows.tools.causal.inference.econml.dml._base_run_dml import (
    _BaseRunDML,
    _configured_run_seed,
)
from python.implementation.workflows.tools.causal.inference.econml.dml.shared_nuisance_models import (
    get_drtester_models_for_t_and_y,
)
from python.implementation.workflows.tools.causal.inference.econml.utils import (
    ModelSpecError,
    get_input_params_from_spec,
    normalize_drtester_treatment_pair,
    now_utc,
)

log = get_logger(__name__)

_OUTER_CV_N_JOBS_ENV = "PRECISION_MEDICINE_OUTER_CV_CATE_N_JOBS"
_INNER_DML_CV = 5
_VALIDATION_GROUPS = 4
_UPLIFT_BOOTSTRAP_REPETITIONS = 1_000
_PROPENSITY_NUMERICAL_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class _FoldResult:
    dataframe: pd.DataFrame
    warnings: list[str]
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _HeldOutDRResult:
    dr_outcome: np.ndarray
    propensity: np.ndarray
    propensity_used: np.ndarray
    propensity_clipped: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    residual_correction: np.ndarray
    treatment_binary: np.ndarray
    warnings: list[str]
    diagnostics: dict[str, Any]


class _ValidationFoldError(RuntimeError):
    pass


def _exclusive_quantile_groups(
    values: np.ndarray,
    *,
    n_groups: int,
) -> np.ndarray:
    """Assign each row to exactly one stable, approximately equal-sized group."""
    values_1d = np.asarray(values, dtype=float).reshape(-1)
    n_rows = len(values_1d)
    if n_groups < 2:
        raise ValueError("Pooled OOF calibration requires at least two groups.")
    if n_rows < n_groups:
        raise _ValidationFoldError("Pooled OOF calibration requires at least one row per group.")

    order = np.argsort(values_1d, kind="stable")
    ordered_groups = np.minimum(
        np.arange(n_rows, dtype=np.int64) * n_groups // n_rows,
        n_groups - 1,
    )
    groups = np.empty(n_rows, dtype=np.int64)
    groups[order] = ordered_groups
    return groups


def _exclusive_calibration_r_squared(
    *,
    cate_oof: np.ndarray,
    dr_oof: np.ndarray,
    n_groups: int,
) -> float:
    """Calculate EconML's calibration score with exclusive group membership."""
    groups = _exclusive_quantile_groups(cate_oof, n_groups=n_groups)
    counts = np.bincount(groups, minlength=n_groups).astype(float)
    gate = np.bincount(groups, weights=dr_oof, minlength=n_groups) / counts
    grouped_cate = np.bincount(groups, weights=cate_oof, minlength=n_groups) / counts
    probabilities = counts / float(len(groups))
    grouped_error = float(np.sum(np.abs(gate - grouped_cate) * probabilities))
    baseline_error = float(np.sum(np.abs(gate - float(np.mean(dr_oof))) * probabilities))
    if np.isclose(baseline_error, 0.0):
        return float("nan")
    return 1.0 - grouped_error / baseline_error


@dataclass(frozen=True, slots=True)
class _BaseValidateDML:
    """Outer-CV held-out CATE and DR validation for one ``_BaseRunDML``."""

    run_dml: _BaseRunDML
    n_jobs: int | None = None
    outer_cv_folds: int | None = None
    # Retained only for constructor compatibility with older dependency-injection
    # tests. The cross-fitted evaluator no longer fabricates DRTester state.
    dr_tester_cls: Any = DRTester

    def execute(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: ValidateCommand,
    ) -> ValidateResult:
        started_at = now_utc()
        fit_command = command.fit_command
        try:
            n_splits = resolve_outer_cv_folds(configured=self.outer_cv_folds)
            effect_modifiers = fit_command.inference_ready_spec.get_effect_modifiers_order()
            if not effect_modifiers:
                raise ModelSpecError("Outer-CV validation requires at least one effect modifier.")

            _, treatment, _, _, _ = get_input_params_from_spec(
                fit_command.df,
                fit_command.inference_ready_spec.causal_spec,
                effect_modifiers_order=effect_modifiers,
                covariates_order=fit_command.inference_ready_spec.get_covariates_order(),
            )
            treatment_1d = np.asarray(treatment).reshape(-1)
            _validate_fold_counts(treatment=treatment_1d, n_splits=n_splits)
            splits = list(
                StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=_configured_run_seed(),
                ).split(np.zeros(len(fit_command.df)), treatment_1d)
            )
            n_jobs = resolve_outer_cv_n_jobs(n_splits=n_splits, configured=self.n_jobs)
            log.info(
                "%s outer-CV DR validation started",
                self.run_dml.BACKEND_NAME,
                backend=self.run_dml.BACKEND_NAME,
                outer_cv_folds=n_splits,
                outer_cv_n_jobs=n_jobs,
                inner_dml_cv=_INNER_DML_CV,
                row_count=len(fit_command.df),
            )
            results = self._run_folds(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
                splits=splits,
                n_jobs=n_jobs,
            )
            validation_dataframe = pd.concat(
                [result.dataframe for result in results], ignore_index=True
            ).sort_values("effect_row", kind="stable", ignore_index=True)
            _validate_oof_coverage(validation_dataframe, expected_rows=len(fit_command.df))
            validation_dataframe = _add_within_fold_ranking_columns(
                validation_dataframe,
                n_groups=_VALIDATION_GROUPS,
            )
            dr_test_summary, gate_summary, validation_diagnostics = self._evaluate_cross_fitted_oof(
                validation_dataframe=validation_dataframe,
                treatment=treatment_1d,
            )
            warnings = [warning for result in results for warning in result.warnings]
            nuisance_diagnostics = [result.diagnostics for result in results]
            log.info(
                "%s outer-CV DR validation completed",
                self.run_dml.BACKEND_NAME,
                backend=self.run_dml.BACKEND_NAME,
                outer_cv_folds=n_splits,
                row_count=len(validation_dataframe),
            )
            return ValidateSuccess(
                run_id=fit_command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                validation_dataframe=validation_dataframe,
                dr_test_summary=dr_test_summary,
                warnings=warnings,
                meta={
                    "backend": self.run_dml.BACKEND_NAME,
                    "outer_cv_folds": n_splits,
                    "outer_cv_n_jobs": n_jobs,
                    "inner_dml_cv": _INNER_DML_CV,
                    "row_count": len(validation_dataframe),
                    "validation_columns": list(validation_dataframe.columns),
                    "dr_test_columns": list(dr_test_summary.columns),
                    "dr_evaluation_scope": "fold_aware_oof",
                    "gate_grouping": "within_outer_fold_percentile_rank",
                    "gate_summary": gate_summary,
                    "validation_diagnostics": validation_diagnostics,
                    "dr_nuisance_diagnostics": nuisance_diagnostics,
                },
            )
        except ModelSpecError as exc:
            return CommandFailure(
                run_id=fit_command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(code="OPTIONS_INVALID", message=str(exc), details={}),
                warnings=[],
                meta={},
            )
        except Exception as exc:
            log.exception(
                "%s outer-CV DR validation failed",
                self.run_dml.BACKEND_NAME,
                backend=self.run_dml.BACKEND_NAME,
                error=exc,
            )
            return CommandFailure(
                run_id=fit_command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(
                    code="ESTIMATOR_ERROR",
                    message=f"{self.run_dml.BACKEND_NAME} outer-CV validation failed.",
                    details={"exception": repr(exc)},
                ),
                warnings=[],
                meta={},
            )

    def _run_folds(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: ValidateCommand,
        splits: list[tuple[np.ndarray, np.ndarray]],
        n_jobs: int,
    ) -> list[_FoldResult]:
        if n_jobs == 1:
            return [
                self._run_fold(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    command=command,
                    outer_fold=fold,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    progress_queue=None,
                )
                for fold, (train_indices, test_indices) in enumerate(splits, start=1)
            ]

        with Manager() as manager:
            progress_queue = manager.Queue()
            stop_event = Event()
            log_thread = Thread(
                target=_forward_progress_logs,
                args=(progress_queue, stop_event, self.run_dml.BACKEND_NAME),
                daemon=True,
            )
            log_thread.start()
            try:
                return Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(_run_validate_fold)(
                        validator=self,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        command=command,
                        outer_fold=fold,
                        train_indices=train_indices,
                        test_indices=test_indices,
                        progress_queue=progress_queue,
                    )
                    for fold, (train_indices, test_indices) in enumerate(splits, start=1)
                )
            finally:
                stop_event.set()
                log_thread.join(timeout=2)

    def _run_fold(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: ValidateCommand,
        outer_fold: int,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        progress_queue: Any | None,
    ) -> _FoldResult:
        fit_command = command.fit_command
        train_df = fit_command.df.iloc[train_indices].reset_index(drop=True).copy()
        test_df = fit_command.df.iloc[test_indices].reset_index(drop=True).copy()
        _emit_progress(progress_queue, "training_started", outer_fold, train_df, test_df)
        model_id: UUID | None = None
        try:
            fit_result = self.run_dml.fit(
                user_id=user_id,
                conversation_id=conversation_id,
                command=FitCommand(
                    model_name=fit_command.model_name,
                    df=train_df,
                    run_id=uuid4(),
                    inference_ready_spec=fit_command.inference_ready_spec,
                    options=dict(fit_command.options),
                    inputs=FitInputs(model_spec=fit_command.inputs.model_spec),
                ),
                df=train_df,
                started_at=now_utc(),
            )
            if not isinstance(fit_result, FitSuccess):
                raise _ValidationFoldError(_failure_message("DML FIT", fit_result))
            model_id = fit_result.fitted_model_id
            _emit_progress(progress_queue, "training_completed", outer_fold, train_df, test_df)

            effect_modifiers = fit_command.inference_ready_spec.get_effect_modifiers_order()
            cate_result = self.run_dml.cate(
                user_id=user_id,
                conversation_id=conversation_id,
                command=CATECommand(
                    model_name=fit_command.model_name,
                    df=test_df,
                    run_id=uuid4(),
                    inference_ready_spec=fit_command.inference_ready_spec,
                    options=dict(fit_command.options),
                    fitted_model_id=model_id,
                    inputs=CATEInputs(
                        x_rows=test_df.loc[:, effect_modifiers].reset_index(drop=True).copy()
                    ),
                ),
                started_at=now_utc(),
            )
            if not isinstance(cate_result, CATESuccess):
                raise _ValidationFoldError(_failure_message("DML CATE", cate_result))
            cate, cate_lower, cate_upper = _extract_cate(cate_result, len(test_df))

            _emit_progress(progress_queue, "dr_validation_started", outer_fold, train_df, test_df)
            dr_result = self._fit_held_out_dr_scores(
                command=fit_command,
                train_df=train_df,
                test_df=test_df,
                outer_fold=outer_fold,
            )
            if len(dr_result.dr_outcome) != len(test_df):
                raise _ValidationFoldError("Held-out DR construction returned the wrong row count.")
            _emit_progress(progress_queue, "fold_completed", outer_fold, train_df, test_df)
            return _FoldResult(
                dataframe=pd.DataFrame(
                    {
                        "effect_row": test_indices.astype(int, copy=False) + 1,
                        "outer_fold": outer_fold,
                        "treatment_oof": dr_result.treatment_binary,
                        "cate_oof": cate,
                        "cate_oof_lower": cate_lower,
                        "cate_oof_upper": cate_upper,
                        "mu0_oof": dr_result.mu0,
                        "mu1_oof": dr_result.mu1,
                        "propensity_oof": dr_result.propensity,
                        "propensity_used_oof": dr_result.propensity_used,
                        "propensity_clipped_oof": dr_result.propensity_clipped,
                        "dr_residual_correction_oof": dr_result.residual_correction,
                        "dr_outcome_oof": dr_result.dr_outcome,
                    }
                ),
                warnings=[
                    *fit_result.warnings,
                    *cate_result.warnings,
                    *dr_result.warnings,
                ],
                diagnostics=dr_result.diagnostics,
            )
        finally:
            if model_id is not None:
                self.run_dml.models_repo.delete_model(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    model_id=model_id,
                )

    def _fit_held_out_dr_scores(
        self,
        *,
        command: FitCommand,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        outer_fold: int,
    ) -> _HeldOutDRResult:
        """Fit nuisances on the complete outer-training sample and score outer-test rows.

        No training-sample DR outcomes are created. The returned DR score for
        every row is based only on nuisance predictions from models that did not
        use that row during fitting.
        """
        spec = command.inference_ready_spec
        effect_modifiers = spec.get_effect_modifiers_order()
        covariates = spec.get_covariates_order()
        y_train, t_train, x_train, w_train, _ = get_input_params_from_spec(
            train_df,
            spec.causal_spec,
            effect_modifiers_order=effect_modifiers,
            covariates_order=covariates,
        )
        y_test, t_test, x_test, w_test, _ = get_input_params_from_spec(
            test_df,
            spec.causal_spec,
            effect_modifiers_order=effect_modifiers,
            covariates_order=covariates,
        )
        t_train, t_test = normalize_drtester_treatment_pair(
            train=t_train,
            validation=t_test,
        )
        if x_train is None or x_test is None:
            raise _ValidationFoldError("DR validation requires non-empty effect modifiers.")

        y_train_1d = np.asarray(y_train, dtype=float).reshape(-1)
        y_test_1d = np.asarray(y_test, dtype=float).reshape(-1)
        t_train_1d = np.asarray(t_train).reshape(-1)
        t_test_1d = np.asarray(t_test).reshape(-1)
        control_value, treated_value = _resolve_binary_treatment_values(
            t_train=t_train_1d,
            t_test=t_test_1d,
        )
        treatment_binary = (t_test_1d == treated_value).astype(float)

        plan = self.run_dml.encoding_util.compile(
            plan=spec.transformation_plan,
            effect_modifiers_order=effect_modifiers,
            covariates_order=covariates,
            dense_output=True,
            drop_first_effect_modifier_onehot=self.run_dml.DROP_FIRST_EFFECT_MODIFIER_ONEHOT,
        )
        model_regression, model_propensity = get_drtester_models_for_t_and_y(
            spec.causal_spec,
            pre_XW=plan.pre_XW,
            missingness=spec.requires_allow_missing_for_covariates(),
            random_state=_configured_run_seed(),
        )
        xw_train = _combine_x_w(x_train, w_train)
        xw_test = _combine_x_w(x_test, w_test)

        fit_warning_details: list[dict[str, Any]] = []
        fitted_propensity, propensity_warnings = _fit_model_with_diagnostics(
            model=clone(model_propensity),
            X=xw_train,
            y=t_train_1d,
            outer_fold=outer_fold,
            role="propensity",
            treatment_arm="all",
        )
        fit_warning_details.extend(propensity_warnings)

        control_mask = t_train_1d == control_value
        treated_mask = t_train_1d == treated_value
        if not np.any(control_mask) or not np.any(treated_mask):
            raise _ValidationFoldError(
                "Each outer-training fold must contain both treatment groups."
            )

        fitted_mu0, mu0_warnings = _fit_model_with_diagnostics(
            model=clone(model_regression),
            X=_take_rows(xw_train, np.flatnonzero(control_mask)),
            y=y_train_1d[control_mask],
            outer_fold=outer_fold,
            role="outcome",
            treatment_arm=str(control_value),
        )
        fitted_mu1, mu1_warnings = _fit_model_with_diagnostics(
            model=clone(model_regression),
            X=_take_rows(xw_train, np.flatnonzero(treated_mask)),
            y=y_train_1d[treated_mask],
            outer_fold=outer_fold,
            role="outcome",
            treatment_arm=str(treated_value),
        )
        fit_warning_details.extend(mu0_warnings)
        fit_warning_details.extend(mu1_warnings)
        treatment_counts = _treatment_counts(t_train_1d)
        outcome_by_treatment = {
            str(control_value): _outcome_counts(y_train_1d[control_mask]),
            str(treated_value): _outcome_counts(y_train_1d[treated_mask]),
        }
        for warning_detail in fit_warning_details:
            warning_detail.setdefault("treatment_counts", treatment_counts)
            warning_detail.setdefault("outcome_by_treatment", outcome_by_treatment)

        propensity = _predict_treated_probability(
            fitted_propensity,
            xw_test,
            treated_value=treated_value,
        )
        mu0 = np.asarray(fitted_mu0.predict(xw_test), dtype=float).reshape(-1)
        mu1 = np.asarray(fitted_mu1.predict(xw_test), dtype=float).reshape(-1)
        _validate_nuisance_predictions(
            propensity=propensity,
            mu0=mu0,
            mu1=mu1,
            expected_rows=len(test_df),
        )

        propensity_used = np.clip(
            propensity,
            _PROPENSITY_NUMERICAL_EPSILON,
            1.0 - _PROPENSITY_NUMERICAL_EPSILON,
        )
        propensity_clipped = ~np.isclose(propensity, propensity_used, rtol=0.0, atol=0.0)
        if np.any(propensity_clipped):
            fit_warning_details.append(
                {
                    "outer_fold": outer_fold,
                    "role": "propensity",
                    "treatment_arm": "all",
                    "warning_class": "NumericalPropensityClipping",
                    "message": (
                        f"Clipped {int(np.sum(propensity_clipped))} held-out propensity "
                        f"predictions to [{_PROPENSITY_NUMERICAL_EPSILON}, "
                        f"{1.0 - _PROPENSITY_NUMERICAL_EPSILON}]."
                    ),
                    "estimator": _fitted_estimator_name(fitted_propensity),
                    "n_iter": _extract_n_iter(fitted_propensity),
                    "n_rows": int(len(t_train_1d)),
                    "outcome_counts": _outcome_counts(y_train_1d),
                    "treatment_counts": _treatment_counts(t_train_1d),
                }
            )

        residual_correction = treatment_binary * (y_test_1d - mu1) / propensity_used - (
            1.0 - treatment_binary
        ) * (y_test_1d - mu0) / (1.0 - propensity_used)
        dr_outcome = mu1 - mu0 + residual_correction
        if not np.all(np.isfinite(dr_outcome)):
            raise _ValidationFoldError("Held-out DR outcomes contain non-finite values.")

        warning_strings = [_warning_detail_to_string(item) for item in fit_warning_details]
        diagnostics = {
            "outer_fold": outer_fold,
            "train_rows": int(len(train_df)),
            "held_out_rows": int(len(test_df)),
            "control_value": _json_scalar(control_value),
            "treated_value": _json_scalar(treated_value),
            "training_treatment_counts": treatment_counts,
            "training_outcome_counts": _outcome_counts(y_train_1d),
            "training_outcome_by_treatment": outcome_by_treatment,
            "propensity_model": _model_fit_diagnostic(fitted_propensity),
            "outcome_model_control": _model_fit_diagnostic(fitted_mu0),
            "outcome_model_treated": _model_fit_diagnostic(fitted_mu1),
            "warning_details": fit_warning_details,
            "held_out_propensity": _propensity_diagnostics(
                propensity=propensity,
                treatment=treatment_binary,
            ),
            "held_out_dr_score": _distribution_diagnostics(dr_outcome),
        }
        return _HeldOutDRResult(
            dr_outcome=dr_outcome,
            propensity=propensity,
            propensity_used=propensity_used,
            propensity_clipped=propensity_clipped,
            mu0=mu0,
            mu1=mu1,
            residual_correction=residual_correction,
            treatment_binary=treatment_binary,
            warnings=warning_strings,
            diagnostics=diagnostics,
        )

    def _evaluate_cross_fitted_oof(
        self,
        *,
        validation_dataframe: pd.DataFrame,
        treatment: np.ndarray,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
        """Evaluate completed OOF CATE/DR pairs without fabricating DRTester state."""
        cate_oof = np.asarray(validation_dataframe["cate_oof"], dtype=float).reshape(-1)
        dr_oof = np.asarray(validation_dataframe["dr_outcome_oof"], dtype=float).reshape(-1)
        outer_fold = np.asarray(validation_dataframe["outer_fold"]).reshape(-1)
        group = np.asarray(validation_dataframe["cate_quartile_within_fold"], dtype=int)

        if len(cate_oof) != len(dr_oof) or len(cate_oof) != len(treatment):
            raise _ValidationFoldError("OOF evaluation received inconsistent row counts.")
        if not np.all(np.isfinite(cate_oof)):
            raise _ValidationFoldError("OOF CATE predictions contain non-finite values.")
        if not np.all(np.isfinite(dr_oof)):
            raise _ValidationFoldError("OOF DR outcomes contain non-finite values.")

        treatments = np.sort(np.unique(np.asarray(treatment).reshape(-1)))
        if len(treatments) != 2:
            raise _ValidationFoldError("OOF DR evaluation currently requires binary treatment.")

        blp_est, blp_se, blp_pval = _fold_adjusted_blp(
            cate_oof=cate_oof,
            dr_oof=dr_oof,
            outer_fold=outer_fold,
        )
        qini_est, qini_se, qini_pval, autoc_est, autoc_se, autoc_pval = (
            _fold_aggregated_uplift_with_bootstrap(
                cate_oof=cate_oof,
                dr_oof=dr_oof,
                outer_fold=outer_fold,
                n_bootstrap=_UPLIFT_BOOTSTRAP_REPETITIONS,
                random_state=_configured_run_seed(),
            )
        )
        cal_r_squared = _calibration_r_squared_from_groups(
            cate_oof=cate_oof,
            dr_oof=dr_oof,
            groups=group,
        )
        gate_summary = _gate_summary(
            validation_dataframe,
            n_groups=_VALIDATION_GROUPS,
        )
        validation_diagnostics = {
            "propensity": _propensity_diagnostics(
                propensity=np.asarray(validation_dataframe["propensity_oof"], dtype=float),
                treatment=np.asarray(validation_dataframe["treatment_oof"], dtype=float),
            ),
            "dr_score": _distribution_diagnostics(dr_oof),
            "cate": _distribution_diagnostics(cate_oof),
            "fold_mean_cate": {
                str(int(fold)): float(np.mean(cate_oof[outer_fold == fold]))
                for fold in np.unique(outer_fold)
            },
            "fold_mean_dr": {
                str(int(fold)): float(np.mean(dr_oof[outer_fold == fold]))
                for fold in np.unique(outer_fold)
            },
            "uplift_bootstrap_scope": "resample_completed_oof_rows_within_outer_fold",
            "uplift_p_values": "two_sided_normal_approximation",
            "blp_covariance": "HC3",
            "blp_fold_fixed_effects": True,
        }

        summary = pd.DataFrame(
            [
                {
                    "evaluation_scope": "fold_aware_oof",
                    "treatment": _json_scalar(treatments[1]),
                    "blp_est": blp_est,
                    "blp_se": blp_se,
                    "blp_pval": blp_pval,
                    "qini_est": qini_est,
                    "qini_se": qini_se,
                    "qini_pval": qini_pval,
                    "autoc_est": autoc_est,
                    "autoc_se": autoc_se,
                    "autoc_pval": autoc_pval,
                    "cal_r_squared": cal_r_squared,
                }
            ]
        )
        return summary, gate_summary, validation_diagnostics


def _take_rows(X: Any, indices: np.ndarray) -> Any:
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return X[indices]


def _resolve_binary_treatment_values(
    *,
    t_train: np.ndarray,
    t_test: np.ndarray,
) -> tuple[Any, Any]:
    train_values = np.sort(np.unique(t_train))
    combined_values = np.sort(np.unique(np.concatenate([t_train, t_test])))
    if len(train_values) != 2 or len(combined_values) != 2:
        raise _ValidationFoldError(
            "Held-out DR construction requires exactly two treatment values."
        )
    if not np.array_equal(train_values, combined_values):
        raise _ValidationFoldError(
            "Outer-training and combined train/test treatment values are inconsistent."
        )
    return train_values[0], train_values[1]


def _fit_model_with_diagnostics(
    *,
    model: Any,
    X: Any,
    y: np.ndarray,
    outer_fold: int,
    role: str,
    treatment_arm: str,
) -> tuple[Any, list[dict[str, Any]]]:
    warning_details: list[dict[str, Any]] = []
    with py_warnings.catch_warnings(record=True) as captured:
        py_warnings.simplefilter("always")
        fitted = model.fit(X, y)
    for warning in captured:
        warning_details.append(
            {
                "outer_fold": outer_fold,
                "role": role,
                "treatment_arm": treatment_arm,
                "warning_class": warning.category.__name__,
                "message": str(warning.message),
                "filename": warning.filename,
                "lineno": int(warning.lineno),
                "estimator": _fitted_estimator_name(fitted),
                "n_iter": _extract_n_iter(fitted),
                "n_rows": int(len(y)),
                "outcome_counts": _outcome_counts(y),
            }
        )
    return fitted, warning_details


def _predict_treated_probability(
    fitted_propensity: Any,
    X: Any,
    *,
    treated_value: Any,
) -> np.ndarray:
    probabilities = np.asarray(fitted_propensity.predict_proba(X), dtype=float)
    classes = np.asarray(fitted_propensity.classes_)
    matching = np.flatnonzero(classes == treated_value)
    if probabilities.ndim != 2 or len(matching) != 1:
        raise _ValidationFoldError(
            "Propensity model did not expose exactly one probability column for treatment."
        )
    propensity = probabilities[:, int(matching[0])]
    if not np.all(np.isfinite(propensity)):
        raise _ValidationFoldError("Propensity predictions contain non-finite values.")
    return propensity


def _validate_nuisance_predictions(
    *,
    propensity: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    expected_rows: int,
) -> None:
    if not (len(propensity) == len(mu0) == len(mu1) == expected_rows):
        raise _ValidationFoldError("Nuisance models returned inconsistent held-out row counts.")
    if np.any((propensity < 0.0) | (propensity > 1.0)):
        raise _ValidationFoldError("Propensity predictions must lie in [0, 1].")
    if not np.all(np.isfinite(mu0)) or not np.all(np.isfinite(mu1)):
        raise _ValidationFoldError("Outcome nuisance predictions contain non-finite values.")


def _add_within_fold_ranking_columns(
    dataframe: pd.DataFrame,
    *,
    n_groups: int,
) -> pd.DataFrame:
    out = dataframe.copy()
    quartiles = np.empty(len(out), dtype=np.int64)
    percentile_ranks = np.empty(len(out), dtype=float)
    for fold in np.sort(out["outer_fold"].unique()):
        row_indices = np.flatnonzero(out["outer_fold"].to_numpy() == fold)
        fold_values = out.iloc[row_indices]["cate_oof"].to_numpy(dtype=float)
        fold_groups = _exclusive_quantile_groups(fold_values, n_groups=n_groups)
        quartiles[row_indices] = fold_groups + 1
        order = np.argsort(fold_values, kind="stable")
        ranks = np.empty(len(order), dtype=float)
        ranks[order] = (np.arange(len(order), dtype=float) + 0.5) / len(order)
        percentile_ranks[row_indices] = ranks
    out["cate_percentile_within_fold"] = percentile_ranks
    out["cate_quartile_within_fold"] = quartiles
    return out


def _calibration_r_squared_from_groups(
    *,
    cate_oof: np.ndarray,
    dr_oof: np.ndarray,
    groups: np.ndarray,
) -> float:
    unique_groups = np.sort(np.unique(groups))
    overall_dr = float(np.mean(dr_oof))
    grouped_error = 0.0
    baseline_error = 0.0
    for group in unique_groups:
        mask = groups == group
        probability = float(np.mean(mask))
        gate = float(np.mean(dr_oof[mask]))
        grouped_cate = float(np.mean(cate_oof[mask]))
        grouped_error += probability * abs(gate - grouped_cate)
        baseline_error += probability * abs(gate - overall_dr)
    if np.isclose(baseline_error, 0.0):
        return float("nan")
    return float(1.0 - grouped_error / baseline_error)


def _fold_adjusted_blp(
    *,
    cate_oof: np.ndarray,
    dr_oof: np.ndarray,
    outer_fold: np.ndarray,
) -> tuple[float, float, float]:
    folds = np.sort(np.unique(outer_fold))
    fold_dummies = [(outer_fold == fold).astype(float) for fold in folds[1:]]
    columns = [np.ones(len(cate_oof), dtype=float), cate_oof, *fold_dummies]
    design = np.column_stack(columns)
    fitted = OLS(dr_oof, design).fit(cov_type="HC3")
    return float(fitted.params[1]), float(fitted.bse[1]), float(fitted.pvalues[1])


def _fold_uplift_coefficients(
    *,
    cate: np.ndarray,
    dr: np.ndarray,
    quantiles: np.ndarray,
) -> tuple[float, float]:
    overall = float(np.mean(dr))
    toc_values: list[float] = []
    qini_values: list[float] = []
    for quantile in quantiles:
        threshold = float(np.quantile(cate, quantile))
        selected = cate >= threshold
        selected_probability = float(np.mean(selected))
        if selected_probability <= 0.0:
            toc_values.append(0.0)
            qini_values.append(0.0)
            continue
        toc = float(np.mean(dr[selected]) - overall)
        toc_values.append(toc)
        qini_values.append(selected_probability * toc)
    autoc = float(np.trapezoid(np.asarray(toc_values), quantiles))
    qini = float(np.trapezoid(np.asarray(qini_values), quantiles))
    return qini, autoc


def _aggregate_fold_uplift(
    *,
    cate_oof: np.ndarray,
    dr_oof: np.ndarray,
    outer_fold: np.ndarray,
) -> tuple[float, float]:
    quantiles = np.linspace(0.05, 0.95, 50)
    qini_total = 0.0
    autoc_total = 0.0
    n_total = float(len(cate_oof))
    for fold in np.sort(np.unique(outer_fold)):
        mask = outer_fold == fold
        weight = float(np.sum(mask)) / n_total
        qini_fold, autoc_fold = _fold_uplift_coefficients(
            cate=cate_oof[mask],
            dr=dr_oof[mask],
            quantiles=quantiles,
        )
        qini_total += weight * qini_fold
        autoc_total += weight * autoc_fold
    return qini_total, autoc_total


def _fold_aggregated_uplift_with_bootstrap(
    *,
    cate_oof: np.ndarray,
    dr_oof: np.ndarray,
    outer_fold: np.ndarray,
    n_bootstrap: int,
    random_state: int | None,
) -> tuple[float, float, float, float, float, float]:
    qini_est, autoc_est = _aggregate_fold_uplift(
        cate_oof=cate_oof,
        dr_oof=dr_oof,
        outer_fold=outer_fold,
    )
    rng = np.random.default_rng(random_state)
    qini_boot = np.empty(n_bootstrap, dtype=float)
    autoc_boot = np.empty(n_bootstrap, dtype=float)
    unique_folds = np.sort(np.unique(outer_fold))
    fold_indices = {fold: np.flatnonzero(outer_fold == fold) for fold in unique_folds}
    for bootstrap_index in range(n_bootstrap):
        sampled_indices = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in fold_indices.values()
            ]
        )
        qini_boot[bootstrap_index], autoc_boot[bootstrap_index] = _aggregate_fold_uplift(
            cate_oof=cate_oof[sampled_indices],
            dr_oof=dr_oof[sampled_indices],
            outer_fold=outer_fold[sampled_indices],
        )
    qini_se = float(np.std(qini_boot, ddof=1))
    autoc_se = float(np.std(autoc_boot, ddof=1))
    qini_pval = _two_sided_normal_pvalue(qini_est, qini_se)
    autoc_pval = _two_sided_normal_pvalue(autoc_est, autoc_se)
    return qini_est, qini_se, qini_pval, autoc_est, autoc_se, autoc_pval


def _two_sided_normal_pvalue(estimate: float, standard_error: float) -> float:
    if not np.isfinite(standard_error) or standard_error <= 0.0:
        return float("nan")
    return float(2.0 * stats.norm.sf(abs(estimate / standard_error)))


def _gate_summary(
    dataframe: pd.DataFrame,
    *,
    n_groups: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in range(1, n_groups + 1):
        rows = dataframe.loc[dataframe["cate_quartile_within_fold"] == group]
        dr = rows["dr_outcome_oof"].to_numpy(dtype=float)
        cate = rows["cate_oof"].to_numpy(dtype=float)
        n_rows = len(rows)
        standard_error = float(np.std(dr, ddof=1) / np.sqrt(n_rows))
        critical = float(stats.t.ppf(0.975, df=n_rows - 1))
        estimate = float(np.mean(dr))
        out.append(
            {
                "quartile": f"Q{group}",
                "n": n_rows,
                "mean_cate_oof": float(np.mean(cate)),
                "dr_gate": estimate,
                "dr_gate_se": standard_error,
                "dr_gate_ci_lower": estimate - critical * standard_error,
                "dr_gate_ci_upper": estimate + critical * standard_error,
            }
        )
    return out


def _propensity_diagnostics(
    *,
    propensity: np.ndarray,
    treatment: np.ndarray,
) -> dict[str, Any]:
    propensity_1d = np.asarray(propensity, dtype=float).reshape(-1)
    treatment_1d = np.asarray(treatment, dtype=float).reshape(-1)
    used = np.clip(
        propensity_1d,
        _PROPENSITY_NUMERICAL_EPSILON,
        1.0 - _PROPENSITY_NUMERICAL_EPSILON,
    )
    observed_weight = treatment_1d / used + (1.0 - treatment_1d) / (1.0 - used)
    diagnostics = _distribution_diagnostics(propensity_1d)
    diagnostics.update(
        {
            "fraction_below_0_05": float(np.mean(propensity_1d < 0.05)),
            "fraction_above_0_95": float(np.mean(propensity_1d > 0.95)),
            "fraction_outside_0_05_0_95": float(
                np.mean((propensity_1d < 0.05) | (propensity_1d > 0.95))
            ),
            "effective_sample_size_overall": _effective_sample_size(observed_weight),
            "effective_sample_size_control": _effective_sample_size(
                observed_weight[treatment_1d == 0.0]
            ),
            "effective_sample_size_treated": _effective_sample_size(
                observed_weight[treatment_1d == 1.0]
            ),
        }
    )
    return diagnostics


def _effective_sample_size(weights: np.ndarray) -> float:
    weights_1d = np.asarray(weights, dtype=float).reshape(-1)
    if len(weights_1d) == 0:
        return float("nan")
    denominator = float(np.sum(np.square(weights_1d)))
    if np.isclose(denominator, 0.0):
        return float("nan")
    return float(np.square(np.sum(weights_1d)) / denominator)


def _distribution_diagnostics(values: np.ndarray) -> dict[str, Any]:
    values_1d = np.asarray(values, dtype=float).reshape(-1)
    quantiles = np.quantile(values_1d, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "n": int(len(values_1d)),
        "mean": float(np.mean(values_1d)),
        "standard_deviation": float(np.std(values_1d, ddof=1)),
        "minimum": float(np.min(values_1d)),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p95": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "maximum": float(np.max(values_1d)),
    }


def _model_fit_diagnostic(model: Any) -> dict[str, Any]:
    return {
        "estimator": _fitted_estimator_name(model),
        "selected_model_name": getattr(model, "selected_model_name_", None),
        "n_iter": _extract_n_iter(model),
        "candidate_scores": _json_array_or_none(getattr(model, "candidate_scores_", None)),
    }


def _fitted_estimator_name(model: Any) -> str:
    selected_name = getattr(model, "selected_model_name_", None)
    if selected_name is not None:
        return str(selected_name)
    selected = getattr(model, "selected_model_", None)
    if selected is not None:
        return type(selected).__name__
    return type(model).__name__


def _extract_n_iter(model: Any) -> Any:
    visited: set[int] = set()

    def visit(value: Any) -> Any:
        if value is None or id(value) in visited:
            return None
        visited.add(id(value))
        n_iter = getattr(value, "n_iter_", None)
        if n_iter is not None:
            return _json_array_or_none(n_iter)
        selected = getattr(value, "selected_model_", None)
        found = visit(selected)
        if found is not None:
            return found
        named_steps = getattr(value, "named_steps", None)
        if named_steps is not None:
            for step in reversed(list(named_steps.values())):
                found = visit(step)
                if found is not None:
                    return found
        inner = getattr(value, "model", None)
        return visit(inner)

    return visit(model)


def _outcome_counts(y: np.ndarray) -> dict[str, int] | dict[str, float]:
    values = np.asarray(y).reshape(-1)
    unique, counts = np.unique(values, return_counts=True)
    if len(unique) <= 20:
        return {
            str(_json_scalar(key)): int(count) for key, count in zip(unique, counts, strict=True)
        }
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values.astype(float))),
    }


def _treatment_counts(treatment: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(treatment).reshape(-1), return_counts=True)
    return {str(_json_scalar(key)): int(count) for key, count in zip(unique, counts, strict=True)}


def _warning_detail_to_string(detail: dict[str, Any]) -> str:
    return (
        f"DR_NUISANCE_WARNING outer_fold={detail.get('outer_fold')} "
        f"role={detail.get('role')} arm={detail.get('treatment_arm')} "
        f"estimator={detail.get('estimator')} "
        f"warning={detail.get('warning_class')}: {detail.get('message')}"
    )


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_array_or_none(value: Any) -> Any:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        return _json_scalar(array.item())
    return array.tolist()


def resolve_outer_cv_folds(*, configured: int | None = None) -> int:
    folds = (
        model_training_config.MODEL_TRAINING_CONFIG.outer_cv_cate_folds
        if configured is None
        else configured
    )
    if isinstance(folds, bool) or not isinstance(folds, int) or not 2 <= folds <= 10:
        raise ModelSpecError(
            "Model training config outer_cv_cate_folds must be an integer from 2 to 10. "
            f"Received: {folds!r}."
        )
    return folds


def resolve_outer_cv_n_jobs(*, n_splits: int, configured: int | None = None) -> int:
    if configured is not None:
        jobs = configured
    else:
        raw_value = os.getenv(_OUTER_CV_N_JOBS_ENV)
        if raw_value is None:
            jobs = n_splits
        else:
            try:
                jobs = int(raw_value.strip())
            except ValueError as exc:
                raise ModelSpecError(
                    f"{_OUTER_CV_N_JOBS_ENV} must be an integer from 1 to {n_splits}. "
                    f"Received: {raw_value!r}."
                ) from exc
    if not 1 <= jobs <= n_splits:
        raise ModelSpecError(
            f"Outer-CV worker count must be from 1 to {n_splits}. Received: {jobs!r}."
        )
    return jobs


def _run_validate_fold(**kwargs: Any) -> _FoldResult:
    return kwargs.pop("validator")._run_fold(**kwargs)


def _forward_progress_logs(progress_queue: Any, stop_event: Event, backend_name: str) -> None:
    while not stop_event.is_set():
        try:
            event = progress_queue.get(timeout=0.2)
        except Empty:
            continue
        log.info(
            "%s outer-CV validation %s",
            backend_name,
            event["event"].replace("_", " "),
            backend=backend_name,
            outer_fold=event["outer_fold"],
            train_rows=event["train_rows"],
            held_out_rows=event["held_out_rows"],
        )


def _emit_progress(
    progress_queue: Any | None,
    event: str,
    outer_fold: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    payload = {
        "event": event,
        "outer_fold": outer_fold,
        "train_rows": len(train_df),
        "held_out_rows": len(test_df),
    }
    if progress_queue is None:
        log.info("outer-CV validation %s", event.replace("_", " "), **payload)
    else:
        progress_queue.put(payload)


def _combine_x_w(X: Any, W: Any) -> Any:
    if W is None:
        return X
    if X is None:
        return W
    if isinstance(X, pd.DataFrame) and isinstance(W, pd.DataFrame):
        return pd.concat([X.reset_index(drop=True), W.reset_index(drop=True)], axis=1)
    return np.hstack([np.asarray(X), np.asarray(W)])


def _extract_cate(
    result: CATESuccess, expected_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cate = np.asarray(result.effects["cate"], dtype=float).reshape(-1)
    if len(cate) != expected_rows:
        raise _ValidationFoldError("DML CATE returned the wrong held-out row count.")
    interval = result.effects.get("cate_interval")
    if interval is None:
        unavailable = np.full(expected_rows, np.nan)
        return cate, unavailable, unavailable.copy()
    lower, upper = interval
    lower_array = np.asarray(lower, dtype=float).reshape(-1)
    upper_array = np.asarray(upper, dtype=float).reshape(-1)
    if len(lower_array) != expected_rows or len(upper_array) != expected_rows:
        raise _ValidationFoldError("DML CATE returned an invalid held-out confidence interval.")
    return cate, lower_array, upper_array


def _validate_fold_counts(*, treatment: np.ndarray, n_splits: int) -> None:
    _, counts = np.unique(treatment, return_counts=True)
    if counts.size < 2 or int(counts.min()) < n_splits:
        raise ModelSpecError(
            f"Outer-CV with {n_splits} folds requires at least {n_splits} rows in every treatment group."
        )


def _validate_oof_coverage(dataframe: pd.DataFrame, *, expected_rows: int) -> None:
    expected = np.arange(1, expected_rows + 1)
    if not np.array_equal(dataframe["effect_row"].to_numpy(), expected):
        raise _ValidationFoldError("Outer-CV did not produce one held-out result for each row.")


def _failure_message(step: str, result: Any) -> str:
    if isinstance(result, CommandFailure):
        return f"{step} failed: {result.error.message}"
    return f"{step} returned unexpected {type(result).__name__}."


__all__ = ["resolve_outer_cv_folds", "resolve_outer_cv_n_jobs"]
