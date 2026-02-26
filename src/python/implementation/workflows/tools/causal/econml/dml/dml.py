from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import UUID

from econml.dml.dml import DML
import numpy as np
import pandas as pd

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.implementation.workflows.tools.causal.causal_command import (
    BaseCommand,
    BaseResult,
    CommandFailure,
    ErrorInfo,
    FitCommand,
    FitSuccess,
)
from python.implementation.workflows.tools.causal.causal_model import CausalModel
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.econml.dml.dml_info import get_dml_info
from python.implementation.workflows.tools.causal.econml.utils import build_init_fit_param_maps, split_flat_options

# =============================================================================
# Errors / time
# =============================================================================

class ModelSpecError(ValueError):
    pass


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
# =============================================================================
# CausalSpec -> columns / arrays (strict to your Pydantic schema)
# =============================================================================

def _validate_columns_exist(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ModelSpecError(f"Dataset missing required columns: {missing}")


def _has_missing(arr: Any) -> bool:
    if arr is None:
        return False
    a = np.asarray(arr)
    try:
        return bool(np.isnan(a).any())
    except Exception:
        return bool(pd.isna(a).any())


def _materialize_from_spec(df: pd.DataFrame, spec: CausalSpec) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    y_col = spec.Y.column
    t_col = spec.T.column
    x_cols = list(spec.X or [])
    w_cols = list(spec.W or [])

    _validate_columns_exist(df, [y_col, t_col] + x_cols + w_cols)

    y: np.ndarray = df[[y_col]].to_numpy()
    t: np.ndarray = df[[t_col]].to_numpy()
    x: Optional[np.ndarray] = df[x_cols].to_numpy() if x_cols else None
    w: Optional[np.ndarray] = df[w_cols].to_numpy() if w_cols else None

    # squeeze singleton dims
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if t.ndim == 2 and t.shape[1] == 1:
        t = t[:, 0]

    meta: Dict[str, Any] = {"y": y_col, "t": t_col, "x": x_cols, "w": w_cols}
    return y, t, x, w, meta



# ALWAYS assumes categories if treatment are categorical first would treated as control
def _semantic_required_init_overrides(spec: CausalSpec) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    t_kind = spec.T.kind
    if t_kind in ("binary", "categorical"):
        overrides["discrete_treatment"] = True

    y_kind = spec.Y.kind
    if y_kind == "binary":
        overrides["discrete_outcome"] = True
        
    return overrides


def _validate_semantic_consistency(spec: CausalSpec, init_kwargs: Mapping[str, Any]) -> None:
    """
    If the user provided options contradict the declared CausalSpec, fail fast.
    Keep it minimal.
    """
    t_kind = getattr(spec.T, "kind", None)
    y_kind = getattr(spec.Y, "kind", None)

    if y_kind == "binary" and "discrete_outcome" in init_kwargs and not bool(init_kwargs["discrete_outcome"]):
        raise ModelSpecError("Spec declares binary outcome but options.discrete_outcome is False.")

    if t_kind in ("binary", "categorical") and "discrete_treatment" in init_kwargs and not bool(init_kwargs["discrete_treatment"]):
        raise ModelSpecError("Spec declares discrete treatment but options.discrete_treatment is False.")

    if t_kind == "categorical":
        baseline = getattr(spec.T, "baseline", None)
        if baseline is not None and "categories" in init_kwargs:
            cats = init_kwargs["categories"]
            if isinstance(cats, list) and cats:
                if cats[0] != baseline:
                    raise ModelSpecError(
                        f"Spec baseline={baseline!r} must be the FIRST category (control). Got categories[0]={cats[0]!r}."
                    )


@dataclass(frozen=True, slots=True)
class DMLO(CausalModel):
    data_repo: DataRepo
    models_repo: ModelsRepo
    
    def get_info(self) -> Dict[str, Any]:
        return get_dml_info()      


    def execute(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: BaseCommand,
    ) -> BaseResult:
        started = now_utc()

        if not isinstance(command, FitCommand):
            return CommandFailure(
                run_id=command.run_id,
                started_at=started,
                finished_at=now_utc(),
                error=ErrorInfo(code="UNSUPPORTED_COMMAND", message=f"DML supports only FIT; got {type(command).__name__}.", details={}),
                warnings=[],
                meta={},
            )

        # Load data
        try:
            df = self.data_repo.get_csv_data(user_id, conversation_id, command.dataset_id, limit=None)
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

        # Execute fit
        try:
            est, fit_meta = self.fit(
                df=df,
                spec=command.inputs.transformed_protocol_specs,
                options=self._effective_options(command),
            )
        except ModelSpecError as e:
            return CommandFailure(
                run_id=command.run_id,
                started_at=started,
                finished_at=now_utc(),
                error=ErrorInfo(code="OPTIONS_INVALID", message=str(e), details={}),
                warnings=[],
                meta={},
            )
        except Exception as e:
            return CommandFailure(
                run_id=command.run_id,
                started_at=started,
                finished_at=now_utc(),
                error=ErrorInfo(code="ESTIMATOR_ERROR", message="EconML DML.fit failed.", details={"exception": repr(e)}),
                warnings=[],
                meta={},
            )

        # Persist model (idempotent by run_id)
        model_id = command.run_id
        try:
            self.models_repo.save_model(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
                model=est,
                metadata=fit_meta,
            )
        except Exception as e:
            return CommandFailure(
                run_id=command.run_id,
                started_at=started,
                finished_at=now_utc(),
                error=ErrorInfo(
                    code="ARTIFACT_PERSIST_FAILED",
                    message="Fit succeeded but persisting model failed.",
                    details={"model_id": str(model_id), "exception": repr(e)},
                ),
                warnings=fit_meta.get("warnings", []),
                meta=fit_meta.get("meta", {}),
            )

        finished = now_utc()
        return FitSuccess(
            run_id=command.run_id,
            started_at=started,
            finished_at=finished,
            warnings=fit_meta.get("warnings", []),
            meta=fit_meta.get("meta", {}),
            fitted_model_id=model_id,
            artifacts=fit_meta.get("artifacts", {}),
        )

    def _effective_options(self, command: FitCommand) -> Dict[str, Any]:
        """
        Merge BaseCommand.options with FitInputs.model_spec if present.
        model_spec overrides.
        """
        opts: Dict[str, Any] = dict(getattr(command, "options", None) or {})
        inputs = getattr(command, "inputs", None)
        model_spec = getattr(inputs, "model_spec", None) if inputs is not None else None
        if isinstance(model_spec, Mapping):
            opts.update(dict(model_spec))
        return opts

    # -------------------------------------------------------------------------
    # Fit (separate, simple, strict, no defaults)
    # -------------------------------------------------------------------------

    def fit(
        self,
        *,
        df: pd.DataFrame,
        spec: CausalSpec,
        options: Mapping[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        # 1) Build matrices
        Y, T, X, W, col_meta = _materialize_from_spec(df, spec)

        # 2) Missingness: keep strict for Y/T
        miss = {"Y": _has_missing(Y), "T": _has_missing(T), "X": _has_missing(X), "W": _has_missing(W)}
        if miss["Y"] or miss["T"]:
            raise ModelSpecError(f"Y/T contain missing values; must be fixed upstream. missing={miss}")
        
        maps = build_init_fit_param_maps(
            DML,
            fit_include_names={"cache_values", "inference", "sample_weight", "freq_weight", "sample_var", "groups"},
        )
        
        init_map = maps["init"]
        fit_map = maps["fit"]

        # 4) Strict split flat options -> init vs fit
        init_kwargs, fit_kwargs = split_flat_options(options, init_map=init_map, fit_map=fit_map)

        # 5) Apply SPEC-required semantics (NOT defaults)
        _validate_semantic_consistency(spec, init_kwargs)
        required_semantic = _semantic_required_init_overrides(spec)
        for k, v in required_semantic.items():
            init_kwargs.setdefault(k, v)

        # 6) Enforce required init args (no defaults injected by us)
        req = required_init_keys(init_map)
        missing_required = [k for k in req if k not in init_kwargs]
        if missing_required:
            raise ModelSpecError(
                f"Missing required DML __init__ parameters: {missing_required}. "
                f"Provide them in options. (This adapter does not inject defaults.)"
            )

        # 7) If X/W missing, require allow_missing=True (either user provided or spec-required)
        allow_missing = bool(init_kwargs.get("allow_missing", False))
        if (miss["X"] or miss["W"]) and not allow_missing:
            raise ModelSpecError(f"X/W contain missing values but allow_missing is not True in options. missing={miss}")

        # 8) Fit
        est = DML(**init_kwargs)
        est.fit(Y, T, X=X, W=W, **fit_kwargs)

        # 9) Meta
        n = int(df.shape[0])
        fit_meta: Dict[str, Any] = {
            "warnings": [],
            "meta": {
                "backend": "econml.dml.DML",
                "n": n,
                "columns": col_meta,
                "used_init_kwargs": sorted(list(init_kwargs.keys())),
                "used_fit_kwargs": sorted(list(fit_kwargs.keys())),
                "provided_options": dict(options),
                "spec_semantics_applied": sorted(list(required_semantic.keys())),
            },
            "artifacts": {
                "n": n,
                "y_shape": list(np.asarray(Y).shape),
                "t_shape": list(np.asarray(T).shape),
                "x_shape": (list(np.asarray(X).shape) if X is not None else None),
                "w_shape": (list(np.asarray(W).shape) if W is not None else None),
            },
        }
        return est, fit_meta