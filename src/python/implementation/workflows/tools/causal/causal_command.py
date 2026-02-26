from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal,  Optional,  Union
from uuid import UUID

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
    options: Dict[str, Any] = field(default_factory=lambda: {})


# =============================================================================
# FIT
# =============================================================================

@dataclass(frozen=True, slots=True)
class FitInputs:
    transformed_protocol_specs: CausalSpec
    model_spec: Optional[Dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class FitCommand(BaseCommand):
    inputs: FitInputs
    command: Literal["FIT"] = field(init=False, default="FIT")

CausalCommand = Union[FitCommand]

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

CausalResult = Union[FitResult]