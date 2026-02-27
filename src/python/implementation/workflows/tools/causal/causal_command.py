from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal,  Optional, Tuple,  Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

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
class TreatmentArmSymbolicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["symbolic"]
    value: Literal["control", "treated"]

class TreatmentArmValueModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["value"]
    value: Union[str, int, float]

TreatmentArmRefModel = Union[TreatmentArmSymbolicModel, TreatmentArmValueModel]

class TreatmentContrastModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    t0: TreatmentArmRefModel
    t1: TreatmentArmRefModel

class ATEInputsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    mode: Literal["default", "contrast", "baseline_vs_all"] = "default"
    contrast: Optional[TreatmentContrastModel] = None
    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    return_interval: bool = True
    return_inference: bool = True

@dataclass(frozen=True, slots=True, kw_only=True)
class ATECommand(BaseCommand):
    fitted_model_id: UUID
    inputs: ATEInputsModel
    command: Literal["ATE"] = field(init=False, default="ATE")

# =============================================================================
# ATE Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class ATESuccess(BaseResult):
    """
    ATE (point) is always returned.
    interval/inference are optional depending on request + estimator support.
    """
    fitted_model_id: UUID
    contrast: Dict[str, Any]                 # normalized: {"t0": ..., "t1": ...}
    ate: Any                                 # float or np-like, keep Any for multioutput
    ate_interval: Optional[Tuple[Any, Any]] = None
    ate_inference: Optional[Dict[str, Any]] = None  # serialize inference summary you choose
    artifacts: Dict[str, Any] = field(default_factory=lambda: {})
    status: Literal["SUCCEEDED"] = field(init=False, default="SUCCEEDED")


ATEResult = Union[ATESuccess, CommandFailure]


# =============================================================================
# CATE Command Inputs (Pydantic)
# =============================================================================

class CateQueryRowsModel(BaseModel):
    """
    Compute CATE for specific dataset row indices (0-based).
    This avoids needing "patient profile -> transform" plumbing in v1.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    mode: Literal["rows"]
    row_indices: List[int] = Field(..., min_length=1)
    max_rows: int = Field(200, ge=1)

class CateQuerySampleModel(BaseModel):
    """
    Compute CATE for a random sample of dataset rows (for UI preview).
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    mode: Literal["sample"]
    n: int = Field(50, ge=1, le=500)
    seed: Optional[int] = None

CateQueryModel = Union[CateQueryRowsModel, CateQuerySampleModel]


class CATEInputsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    mode: Literal["default", "contrast"] = "default"
    contrast: Optional[TreatmentContrastModel] = None

    query: CateQueryModel = Field(..., discriminator="mode")

    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    return_interval: bool = True
    return_inference: bool = False


# =============================================================================
# CATE Command
# =============================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class CATECommand(BaseCommand):
    fitted_model_id: UUID
    inputs: CATEInputsModel
    command: Literal["CATE"] = field(init=False, default="CATE")


# =============================================================================
# CATE Result
# =============================================================================

@dataclass(frozen=True, slots=True)
class CATESuccess(BaseResult):
    """
    Returns per-row effect estimates for the requested rows/sample.
    """
    fitted_model_id: UUID
    contrast: Dict[str, Any]                      # normalized {"t0": ..., "t1": ...}

    # which rows were evaluated (always resolved to explicit indices)
    row_indices: List[int]

    # effects aligned with row_indices (length m)
    cate: List[Any]

    # optional intervals aligned with row_indices
    cate_interval: Optional[Tuple[List[Any], List[Any]]] = None

    # optional inference payload (you decide serialization)
    cate_inference: Optional[Dict[str, Any]] = None

    artifacts: Dict[str, Any] = field(default_factory=lambda: {})
    status: Literal["SUCCEEDED"] = field(init=False, default="SUCCEEDED")


CATEResult = Union[CATESuccess, CommandFailure]  