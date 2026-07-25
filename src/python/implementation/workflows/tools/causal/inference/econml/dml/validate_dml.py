from __future__ import annotations

import os
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
from sklearn.model_selection import StratifiedKFold

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
    normalize_drtester_cate_predictions,
    normalize_drtester_treatment_pair,
    now_utc,
)

log = get_logger(__name__)

_OUTER_CV_ENV = "PRECISION_MEDICINE_ENABLE_OUTER_CV_CATE"
_OUTER_CV_N_JOBS_ENV = "PRECISION_MEDICINE_OUTER_CV_CATE_N_JOBS"
_INNER_DML_CV = 5


@dataclass(frozen=True, slots=True)
class _FoldResult:
    dataframe: pd.DataFrame
    warnings: list[str]


class _ValidationFoldError(RuntimeError):
    pass


class _CATEOnCombinedFeatures:
    """Expose a fitted DML estimator as DRTester's CATE object."""

    def __init__(self, *, estimator: Any, effect_modifier_columns: list[str]) -> None:
        self._estimator = estimator
        self._effect_modifier_columns = effect_modifier_columns

    def effect(self, X: Any, T0: Any = None, T1: Any = None) -> Any:
        if isinstance(X, pd.DataFrame):
            x_effect = X.loc[:, self._effect_modifier_columns]
        else:
            x_effect = np.asarray(X)[:, : len(self._effect_modifier_columns)]
        return normalize_drtester_cate_predictions(
            self._estimator.effect(x_effect, T0=T0, T1=T1),
            expected_rows=len(x_effect),
        )


@dataclass(frozen=True, slots=True)
class _BaseValidateDML:
    """Outer-CV held-out CATE and DR validation for one ``_BaseRunDML``."""

    run_dml: _BaseRunDML
    n_jobs: int | None = None
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
            n_splits = resolve_outer_cv_folds()
            if n_splits == 1:
                raise ModelSpecError(
                    f"{_OUTER_CV_ENV} must be set to an integer from 2 to 10 for VALIDATE."
                )
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
            dr_test_summary = self._evaluate_pooled_oof(
                validation_dataframe=validation_dataframe,
                treatment=treatment_1d,
            )
            warnings = [warning for result in results for warning in result.warnings]
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
                    "dr_evaluation_scope": "pooled_oof",
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

            record = self.run_dml.models_repo.load_model(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
            )
            if record is None:
                raise _ValidationFoldError(
                    "Temporary fold model was not available for DR validation."
                )
            _emit_progress(progress_queue, "dr_validation_started", outer_fold, train_df, test_df)
            dr_outcome = self._run_drtester(
                command=fit_command,
                train_df=train_df,
                test_df=test_df,
                fitted_estimator=record.model,
            )
            if len(dr_outcome) != len(test_df):
                raise _ValidationFoldError("DRTester returned the wrong held-out row count.")
            _emit_progress(progress_queue, "fold_completed", outer_fold, train_df, test_df)
            return _FoldResult(
                dataframe=pd.DataFrame(
                    {
                        "effect_row": test_indices.astype(int, copy=False) + 1,
                        "outer_fold": outer_fold,
                        "cate_oof": cate,
                        "cate_oof_lower": cate_lower,
                        "cate_oof_upper": cate_upper,
                        "dr_outcome_oof": dr_outcome,
                    }
                ),
                warnings=[*fit_result.warnings, *cate_result.warnings],
            )
        finally:
            if model_id is not None:
                self.run_dml.models_repo.delete_model(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    model_id=model_id,
                )

    def _run_drtester(
        self,
        *,
        command: FitCommand,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        fitted_estimator: Any,
    ) -> np.ndarray:
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
        tester = self.dr_tester_cls(
            model_regression=model_regression,
            model_propensity=model_propensity,
            cate=_CATEOnCombinedFeatures(
                estimator=fitted_estimator,
                effect_modifier_columns=effect_modifiers,
            ),
            cv=_INNER_DML_CV,
        )
        xw_train = _combine_x_w(x_train, w_train)
        xw_test = _combine_x_w(x_test, w_test)
        tester.fit_nuisance(
            Xval=xw_test,
            Dval=t_test,
            yval=y_test,
            Xtrain=xw_train,
            Dtrain=t_train,
            ytrain=y_train,
        )
        return np.asarray(tester.dr_val_, dtype=float).reshape(-1)

    def _evaluate_pooled_oof(
        self,
        *,
        validation_dataframe: pd.DataFrame,
        treatment: np.ndarray,
    ) -> pd.DataFrame:
        """Run one DRTester evaluation on all matched OOF CATE/DR pairs.

        The primary and nuisance predictions remain outer-fold held out. The
        pooled evaluation uses those completed OOF vectors only; it does not
        refit or alter the primary CATE estimator.
        """
        cate_oof = np.asarray(
            validation_dataframe["cate_oof"],
            dtype=float,
        ).reshape(-1)
        dr_oof = np.asarray(
            validation_dataframe["dr_outcome_oof"],
            dtype=float,
        ).reshape(-1)

        if len(cate_oof) != len(dr_oof) or len(cate_oof) != len(treatment):
            raise _ValidationFoldError(
                "Pooled OOF evaluation received inconsistent row counts."
            )
        if not np.all(np.isfinite(cate_oof)):
            raise _ValidationFoldError(
                "Pooled OOF CATE predictions contain non-finite values."
            )
        if not np.all(np.isfinite(dr_oof)):
            raise _ValidationFoldError(
                "Pooled OOF DR outcomes contain non-finite values."
            )

        treatments = np.sort(np.unique(np.asarray(treatment).reshape(-1)))
        if len(treatments) != 2:
            raise _ValidationFoldError(
                "Pooled OOF DRTester evaluation currently requires binary treatment."
            )

        # DRTester has no public constructor for already-computed OOF values.
        # Populate only the documented evaluation attributes, then call its
        # official BLP/calibration/uplift evaluation methods.
        tester = self.dr_tester_cls(
            model_regression=None,
            model_propensity=None,
            cate=None,
            cv=_INNER_DML_CV,
        )
        tester.Dval = np.asarray(treatment).reshape(-1)
        tester.treatments = treatments
        tester.n_treat = 1
        tester.fit_on_train = True
        tester.dr_val_ = dr_oof.reshape(-1, 1)
        tester.cate_preds_val_ = cate_oof.reshape(-1, 1)

        # Pooled OOF predictions define the empirical score distribution used
        # for quartile boundaries and uplift thresholds. Every prediction was
        # generated without its own row in training.
        tester.cate_preds_train_ = cate_oof.reshape(-1, 1)
        tester.ate_val = np.asarray([float(np.mean(dr_oof))])

        summary = tester.evaluate_all(n_bootstrap=1_000).summary().copy()
        summary.insert(0, "evaluation_scope", "pooled_oof")
        return summary


def resolve_outer_cv_folds() -> int:
    raw_value = os.getenv(_OUTER_CV_ENV)
    if raw_value is None:
        return 1
    try:
        folds = int(raw_value.strip())
    except ValueError as exc:
        raise ModelSpecError(
            f"{_OUTER_CV_ENV} must be an integer from 2 to 10 when set. Received: {raw_value!r}."
        ) from exc
    if not 2 <= folds <= 10:
        raise ModelSpecError(
            f"{_OUTER_CV_ENV} must be an integer from 2 to 10 when set. Received: {raw_value!r}."
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
