from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import UUID

from econml.dml.dml import LinearDML
import numpy as np
import pandas as pd

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelRecord, ModelsRepo
from python.implementation.workflows.tools.causal.causal_command import (
    ATECommand,
    ATESuccess,
    CommandFailure,
    ErrorInfo,
    FitCommand,
    FitSuccess,
)
from python.implementation.workflows.tools.causal.causal_model import CausalCommand, CausalModel, CausalResult
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.econml.dml.dml_info import  linear_dml_causal_model_info
from python.implementation.workflows.tools.causal.econml.utils import ModelSpecError, build_init_fit_options_param_maps, get_input_params_from_spec, has_missing, now_utc, required_init_keys, resolve_contrast, serialize_inference_obj, split_flat_options, validate_semantic_consistency

@dataclass(frozen=True, slots=True)
class LinearDMLCausalModel(CausalModel):
    data_repo: DataRepo
    models_repo: ModelsRepo
    

    def get_info(self) -> Dict[str, Any]:
        return linear_dml_causal_model_info()

    def execute(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: CausalCommand,
    ) -> CausalResult:
        started = now_utc()
        # Load data (this stays here as you requested)
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
        if isinstance(command, (ATECommand)):    
            return self._ate(
                user_id=user_id,
                conversation_id=conversation_id,
                command=command,
                started_at=started,
            )
        raise ValueError(f"Unsupported command type: {type(command)}")
    
    # -------------------------------------------------------------------------
    # private
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
            spec: CausalSpec = command.transformed_protocol_specs
            options: Mapping[str, Any] =command.options or {}
            Y, T, X, W, col_meta = get_input_params_from_spec(df, spec)

            # 2) Missingness: keep strict for Y/T
            miss = {"Y": has_missing(Y), "T": has_missing(T), "X": has_missing(X), "W": has_missing(W)}
            if miss["Y"] or miss["T"]:
                raise ModelSpecError(f"Y/T contain missing values; must be fixed upstream. missing={miss}")

            maps = build_init_fit_options_param_maps(
                LinearDML,
                fit_include_names={"cache_values", "inference", "sample_weight", "freq_weight", "sample_var", "groups"},
            )

            init_map = maps["init"]
            fit_map = maps["fit"]


            init_kwargs_default = self._get_default_init_options(spec)
            init_kwargs_from_user, fit_kwargs = split_flat_options(
                options=options,
                init_map=init_map,
                fit_map=fit_map,
            )
            
            init_kwargs_default.update(init_kwargs_from_user)
            validate_semantic_consistency(spec, init_kwargs_default)

            # 6) Enforce required init args (no defaults injected by us)
            required_keys = required_init_keys(LinearDML, init_map=init_map)
            required_keys.add("discrete_treatment") if spec.T.kind in ("binary", "categorical") else None
            required_keys.add("discrete_outcome") if spec.Y.kind == "binary" else None
            missing_required = [k for k in required_keys if k not in init_kwargs_default]
            if missing_required:
                raise ModelSpecError(
                    f"Missing required DML __init__ parameters: {missing_required}. "
                    f"Provide them in options. (This adapter does not inject defaults.)"
                )

            # 7) If X/W missing, require allow_missing=True
            allow_missing = bool(init_kwargs_default.get("allow_missing", False))
            if (miss["X"] or miss["W"]) and not allow_missing:
                raise ModelSpecError(f"X/W contain missing values but allow_missing is not True in options. missing={miss}")

            # 8) Fit
            est = LinearDML(**init_kwargs_default)
            est.fit(Y, T, X=X, W=W, **fit_kwargs) # pyright: ignore[reportUnknownMemberType]

            # 9) Meta
            n = int(df.shape[0])
            fit_meta: Dict[str, Any] = {
                "warnings": [],
                "meta": {
                    "backend": "econml.dml.DML",
                    "n": n,
                    "columns": col_meta,
                    "used_init_kwargs": sorted(list(init_kwargs_default.keys())),
                    "used_fit_kwargs": sorted(list(fit_kwargs.keys())),
                    "provided_options": dict(options),
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

            # 10) Persist model (idempotent by run_id)
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
            # If persist fails, it's also an exception; we return ARTIFACT_PERSIST_FAILED only if
            # we *know* fit succeeded and save failed. Everything else is ESTIMATOR_ERROR.
            #
            # If you want exact split, we can wrap save_model() in its own try/except block.
            return CommandFailure(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(code="ESTIMATOR_ERROR", message="EconML Linear DML.fit failed.", details={"exception": repr(e)}),
                warnings=[],
                meta={},
            )
            
            
    def _ate(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: ATECommand,
        started_at: datetime,
    ) -> CausalResult:
        warnings: List[str] = []
            # 1) load fitted model + metadata
            # You need a corresponding ModelsRepo API; adapt to your interface.
        spec: CausalSpec = command.transformed_protocol_specs    
        model_record: ModelRecord | None = self.models_repo.load_model(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=command.fitted_model_id,
        )
        if model_record is None:
                raise ModelSpecError(f"Fitted model with id {command.fitted_model_id} not found.")
            
        df = self.data_repo.get_csv_data(
                user_id,
                conversation_id,
                command.dataset_id,
                limit=None,
        )
        est = model_record.model
     
        # 3) build X from stored column meta
        _, _, X, _, _ = get_input_params_from_spec(df, spec)

        # 4) resolve contrast
        try:
            # For now: default assumes binary encoding control=0, treated=1.
            # later we will store the mapping in fit_meta and resolve from there.
            t0, t1, contrast_norm = resolve_contrast(command.inputs, default_binary=True)
        except ModelSpecError as e:
            return CommandFailure(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(code="OPTIONS_INVALID", message=str(e), details={}),
                warnings=[],
                meta={},
            )

        # 5) compute ATE (+ optional interval/inference)
        try:
            ate = est.ate(X=X, T0=t0, T1=t1)
        except Exception as e:
            return CommandFailure(
                run_id=command.run_id,
                started_at=started_at,
                finished_at=now_utc(),
                error=ErrorInfo(code="ESTIMATOR_ERROR", message="ATE computation failed.", details={"exception": repr(e)}),
                warnings=[],
                meta={},
            )

        ate_interval: Optional[Tuple[Any, Any]] = None
        if command.inputs.return_interval:
            try:
                ate_interval = est.ate_interval(X=X, T0=t0, T1=t1, alpha=command.inputs.alpha)
            except Exception:
                warnings.append("INFERENCE_NOT_AVAILABLE")

        ate_inference: Optional[Dict[str, Any]] = None
        if command.inputs.return_inference:
            try:
                inf = est.ate_inference(X=X, T0=t0, T1=t1)
                ate_inference = serialize_inference_obj(inf)
            except Exception:
                warnings.append("INFERENCE_NOT_AVAILABLE")

        finished = now_utc()
        return ATESuccess(
            run_id=command.run_id,
            started_at=started_at,
            finished_at=finished,
            warnings=warnings,
            meta={"backend": "econml.dml.LinearDML", "n": int(df.shape[0]), "x_cols": spec.X if spec.X else None},
            fitted_model_id=command.fitted_model_id,
            contrast=contrast_norm,
            ate=ate,
            ate_interval=ate_interval,
            ate_inference=ate_inference,
            artifacts={},
        )
            
    def _get_default_init_options(self, specs: CausalSpec) -> Dict[str, Any]:
        """
        Merge BaseCommand.options with FitInputs.model_spec if present.
        model_spec overrides.
        """
        opts: Dict[str, Any] = {}
        if specs.T.kind in ("binary", "categorical"):
            opts["discrete_treatment"] = True
        if specs.Y.kind == "binary":
            opts["discrete_outcome"] = True
        
        opts["model_y"] = "auto" 
        opts["model_t"] = "auto"
        
        return opts  


    

    