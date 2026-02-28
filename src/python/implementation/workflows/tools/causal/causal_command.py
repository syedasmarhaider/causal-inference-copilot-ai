from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal,  Optional,  Union
from uuid import UUID

import pandas as pd
from sklearn.compose import ColumnTransformer

from python.implementation.workflows.tools.causal.causal_spec import CausalSpec

CommandType = Literal["FIT", "EFFECT", "INTERVAL"]

# =============================================================================
# Meta / base command
# =============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseCommand:
    model_name: str
    dataset_id: UUID
    run_id: UUID
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    protocol_specs: CausalSpec
    options: Dict[str, Any] = field(default_factory=lambda: {})


# =============================================================================
# FIT
# =============================================================================

@dataclass(frozen=True, slots=True)
class FitInputs:
    model_spec: Optional[Dict[str, Any]] = None
    missingness_mode: MissingnessMode = "none"
    pre_X:  Optional[ColumnTransformer] = None
    pre_XW: Optional[ColumnTransformer] = None
    


@dataclass(frozen=True, slots=True)
class FitCommand(BaseCommand):
    inputs: FitInputs
    command: Literal["FIT"] = field(init=False, default="FIT")

# =============================================================================
# Results (success + failure)
# =============================================================================

ResultStatus = Literal["SUCCEEDED", "FAILED"]


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    code: str                 # e.g., "DATA_VALIDATION_FAILED", "NO_OVERLAP", "ESTIMATOR_ERROR"
    message: str              # short human-friendly message
    details: Dict[str, Any] = field(default_factory=lambda: {})  # structured debug payload


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseResult:
    run_id: UUID
    status: ResultStatus
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    warnings: List[str] = field(default_factory=lambda: [])
    meta: Dict[str, Any] = field(default_factory=lambda: {})  # metrics, logs pointers, etc.


@dataclass(frozen=True, slots=True)
class CommandFailure(BaseResult):
    error: ErrorInfo
    status: Literal["FAILED"] = field(init=False, default="FAILED")


# ---- FIT result ----

@dataclass(frozen=True, slots=True)
class FitSuccess(BaseResult):
    fitted_model_id: UUID
    # keep it flexible: training metrics, featurizer info, nuisance model details, etc.
    artifacts: Dict[str, Any] = field(default_factory=lambda: {})
    status: Literal["SUCCEEDED"] = field(init=False, default="SUCCEEDED")


FitResult = Union[FitSuccess, CommandFailure]



# =============================================================================
# Effect Command and Result
# =============================================================================
MissingnessMode = Literal["none", "present"]
@dataclass(frozen=True, slots=True)
class ATEInputsModel:
    alpha: float = 0.05
    

@dataclass(frozen=True, slots=True, kw_only=True)
class ATECommand(BaseCommand):
    fitted_model_id: UUID
    inputs: ATEInputsModel
    command: Literal["ATE"] = field(init=False, default="ATE")

# =============================================================================
# ATE Result
# =============================================================================

ATEModelResult = Literal["for_treatment","ate", "ate_interval", "ate_inference"]

@dataclass(frozen=True, slots=True)
class ATESuccess(BaseResult):
    """
    ATE (point) is always returned.
    interval/inference are optional depending on request + estimator support.
    """
    fitted_model_id: UUID
    contrast: Dict[str, Any]
    ate: List[Dict[ATEModelResult, Any]]                      
    artifacts: Dict[str, Any] = field(default_factory=lambda: {})
    status: Literal["SUCCEEDED"] = field(init=False, default="SUCCEEDED")


ATEResult = Union[ATESuccess, CommandFailure]


# =============================================================================
# CATE Command Inputs (Pydantic)
# =============================================================================

FilterOp = Literal["==", "!=","in", "not_in", ">=", "<=", ">", "<"]

# ----------------------------
# Models
# ----------------------------
@dataclass(frozen=True, slots=True)
class CATEInputs:
    x_rows: pd.DataFrame   # already-transformed X in exact training columns/order
    alpha: float = 0.05

@dataclass(frozen=True, slots=True, kw_only=True)
class CATECommand(BaseCommand):
    fitted_model_id: UUID
    run_id: UUID
    inputs: CATEInputs
    command: Literal["CATE"] = field(init=False, default="CATE")


# =============================================================================
# CATE Result
# =============================================================================
CATEModelResult = Literal["for_treatment","cate", "cate_interval", "cate_inference"]
@dataclass(frozen=True, slots=True)
class CATESuccess(BaseResult):
    fitted_model_id: UUID
    x_cols: List[str]
    # one item per contrast (binary -> 1 item, categorical -> many items)
    effects: List[Dict[CATEModelResult, Any]] = field(default_factory=lambda: [])
    status: Literal["SUCCEEDED"] = field(init=False, default="SUCCEEDED")


CATEResult = Union[CATESuccess, CommandFailure]  