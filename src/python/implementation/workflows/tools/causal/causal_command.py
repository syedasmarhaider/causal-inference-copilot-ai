from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal,  Optional,  Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.utils.utils import ScalarValue

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
    transformed_protocol_specs: CausalSpec
    options: Dict[str, Any] = field(default_factory=lambda: {})


# =============================================================================
# FIT
# =============================================================================

@dataclass(frozen=True, slots=True)
class FitInputs:
    model_spec: Optional[Dict[str, Any]] = None
    


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

class ATEInputsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    alpha: float = Field(0.05, gt=0.0, lt=1.0)

@dataclass(frozen=True, slots=True, kw_only=True)
class ATECommand(BaseCommand):
    fitted_model_id: UUID
    input: ATEInputsModel
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
    ate: List[dict[ATEModelResult,Any]]                      
    artifacts: Dict[str, Any] = field(default_factory=lambda: {})
    status: Literal["SUCCEEDED"] = field(init=False, default="SUCCEEDED")


ATEResult = Union[ATESuccess, CommandFailure]


# =============================================================================
# CATE Command Inputs (Pydantic)
# =============================================================================

class CATEInputsModel(BaseModel):
    """
    Caller must provide X_query explicitly as rows of feature->value mappings.

    IMPORTANT:
      - Each row must contain ALL feature names used during training (x_cols)
      - No extra keys allowed
      - Values must be numeric/bool (already-transformed feature space)
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    x_rows: List[Dict[str, ScalarValue]] = Field(..., min_length=1)
    alpha: float = Field(0.05, gt=0.0, lt=1.0)

@dataclass(frozen=True, slots=True, kw_only=True)
class CATECommand(BaseCommand):
    fitted_model_id: UUID
    inputs: CATEInputsModel
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