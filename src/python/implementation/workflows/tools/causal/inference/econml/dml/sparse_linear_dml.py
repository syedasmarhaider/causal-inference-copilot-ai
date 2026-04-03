from __future__ import annotations

import inspect
from python.implementation.service.logging.default_logging import get_logger
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
from econml.dml import SparseLinearDML

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelRecord, ModelsRepo
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
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.inference.econml.models_info import (
    get_sparse_linear_dml_causal_model_info,
)
from python.implementation.workflows.tools.causal.inference.econml.dml.shared_nuisance_models import (
    get_default_models_for_t_and_y as _get_default_models_for_t_and_y,
)
from python.implementation.workflows.tools.causal.inference.econml.utils import (
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
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.encoding.encoding_util import EncodingUtil
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel

log = get_logger(__name__)

# =============================================================================
# SparseLinearDML adapter
# =============================================================================

@dataclass(frozen=True, slots=True)
class SparseLinearDMLCausalModel(CausalModel):
    data_repo: DataRepo
    models_repo: ModelsRepo
    encoding_util: EncodingUtil

    def get_info(self) -> str:
        return get_sparse_linear_dml_causal_model_info()

    def get_command_info(self, command: CommandType) -> str | None:
        match command:
            case "FIT":
                fit_doc = inspect.getdoc(SparseLinearDML.fit) or ""  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                base_doc = inspect.getdoc(SparseLinearDML) or ""
                return base_doc + fit_doc
            case "ATE":
                ate_doc = inspect.getdoc(SparseLinearDML.ate) or ""  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                return ate_doc
            case "CATE":
                effect_doc = inspect.getdoc(SparseLinearDML.effect) or ""  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
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
            log.exception("SparseLinearDML command failed", error=e)
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
            transformation_plan: TransformPlan | None = command.transformation_plan

            covariates_order: list[str] = list(specs.covariates or [])
            effect_modifiers_order: list[str] = list(specs.effect_modifiers or [])

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
                    "Spec declares effect modifiers (spec.effect_modifiers) but no pre_X transformer was provided."
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
                raise ModelSpecError(f"Y/T contain missing values; must be fixed upstream. missing={miss}")

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
                    "SparseLinearDML does not support missing values in X via allow_missing "
                    "(only W is allowed). Impute/clean X upstream before fit."
                )

            maps = build_init_fit_options_param_maps(
                SparseLinearDML,
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

            defaults: dict[str, Any] = {}

            disc_t = specs.treatment_spec.kind in ("binary", "categorical")
            disc_y = specs.outcome_spec.kind == "binary"

            if disc_t:
                defaults["discrete_treatment"] = True
            if disc_y:
                defaults["discrete_outcome"] = True

            # Same contract you used elsewhere: allow_missing only relaxes W.
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
                defaults["featurizer"] = pre_x

            required_keys = required_init_keys(SparseLinearDML, init_map=init_map)
            missing_required = [k for k in required_keys if k not in defaults]
            if missing_required:
                raise ModelSpecError(
                    f"Missing required SparseLinearDML __init__ parameters: {missing_required}. "
                    f"(Adapter is not exposing command.options yet.)"
                )

            est = SparseLinearDML(**defaults)

            fit_warnings: list[str] = []
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                est.fit(Y, T, X=X, W=W)  # pyright: ignore[reportUnknownMemberType]
            fit_warnings = [f"{w.category.__name__}: {str(w.message)}" for w in ws]

            n = int(df.shape[0])
            fit_meta: dict[str, Any] = {
                "warnings": fit_warnings,
                "meta": {
                    "backend": "econml.dml.SparseLinearDML",
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
            log.exception("SparseLinearDML command failed", error=e)
            return CommandFailure(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(
                    code="ESTIMATOR_ERROR",
                    message="EconML SparseLinearDML.fit failed.",
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
            warnings_list: list[str] = []
            spec: CausalSpec = command.causal_specs
            order_effect_modifiers: list[str] = list(
                command.order_effect_modifiers or spec.effect_modifiers or []
            )
            order_covariates: list[str] = list(
                command.order_covariates or spec.covariates or []
            )

            model_record: ModelRecord | None = self.models_repo.load_model(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=command.fitted_model_id,
            )
            if model_record is None:
                raise ModelSpecError(f"Fitted model with id {command.fitted_model_id} not found.")

            est: SparseLinearDML = model_record.model

            t0, t1 = get_treatment_t0_t1_from_spec(
                spec,
                is_global_counter_factual=False,
            )

            _, _, X, _, _ = get_input_params_from_spec(
                df,
                spec,
                effect_modifiers_order=order_effect_modifiers,
                covariates_order=order_covariates,
            )

            if t1 == t0:
                raise ModelSpecError(f"Invalid contrast: t1 value {t1} is the same as t0 baseline {t0}.")

            effects: list[dict[ATEModelResult, Any]] = []

            item: dict[ATEModelResult, Any] = {"for_treatment": {"t0": t0, "t1": t1}}
            item["ate"] = est.ate(X=X, T0=t0, T1=t1)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
            log.info("Computed ATE for contrast t0=%s vs t1=%s: %s", t0, t1, item["ate"])

            try:
                ate_interval = est.ate_interval(
                    X=X,
                    T0=t0,
                    T1=t1,
                    alpha=command.inputs.alpha,
                )  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
                if ate_interval is not None:
                    item["ate_interval"] = ate_interval
                else:
                    item["ate_interval"] = None
                    warnings_list.append("INFERENCE_NOT_AVAILABLE: ate_interval returned None")
            except Exception as e:
                warnings_list.append("INFERENCE_NOT_AVAILABLE: " + repr(e))
                item["ate_interval"] = None

            try:
                inference = est.ate_inference(X=X, T0=t0, T1=t1)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]
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
                    error=ErrorInfo(
                        code="OPTIONS_INVALID",
                        message="No valid categorical contrasts found (baseline vs all).",
                        details={},
                    ),
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
                    "backend": "econml.dml.SparseLinearDML",
                    "n": int(df.shape[0]),
                    "x_cols": spec.effect_modifiers if spec.effect_modifiers else None,
                    "contrast_kind": "baseline_vs_all",
                    "t0": t0,
                },
                fitted_model_id=command.fitted_model_id,
                contrast={"t0": t0, "t1": "vs_all"},
                ate=effects,
            )

        except Exception as e:
            log.exception("SparseLinearDML command failed", error=e)
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

            est: SparseLinearDML = model_record.model
            spec: CausalSpec = command.causal_specs
            effect_modifiers_order: list[str] = list(
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

            effects: dict[CATEModelResult, Any] = {"for_treatment": {"t0": t0, "t1": t1}}

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
                        code="OPTIONS_INVALID",
                        message="No valid contrasts produced for CATE.",
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
                    "backend": "econml.dml.SparseLinearDML",
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
            log.exception("SparseLinearDML command failed", error=e)
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
