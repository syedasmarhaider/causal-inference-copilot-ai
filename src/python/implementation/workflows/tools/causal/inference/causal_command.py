from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

import pandas as pd

from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel

CommandType = Literal["FIT", "ATE", "CATE"]

# =============================================================================
# Meta / base command
# =============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseCommand:
    model_name: str
    dataset_id: UUID
    run_id: UUID
    order_effect_modifiers: list[str] | None = None
    order_covariates: list[str] | None = None
    data_summary: DatasetSummaryModel
    transformation_plan: TransformPlan | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    causal_specs: CausalSpec
    options: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# FIT
# =============================================================================

@dataclass(frozen=True, slots=True)
class FitInputs:
    model_spec: dict[str, Any] | None = None

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
    details: dict[str, Any] = field(default_factory=dict)  # structured debug payload


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseResult:
    run_id: UUID
    status: ResultStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)  # metrics, logs pointers, etc.


@dataclass(frozen=True, slots=True)
class CommandFailure(BaseResult):
    error: ErrorInfo
    status: Literal["FAILED"] = field(init=False, default="FAILED")


# ---- FIT result ----

@dataclass(frozen=True, slots=True)
class FitSuccess(BaseResult):
    fitted_model_id: UUID
    # keep it flexible: training metrics, featurizer info, nuisance model details, etc.
    artifacts: dict[str, Any] = field(default_factory=dict)
    status: Literal["SUCCEEDED"] = field(init=False, default="SUCCEEDED")


FitResult = FitSuccess | CommandFailure



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
    contrast: dict[str, Any]
    ate: list[dict[ATEModelResult, Any]]                      
    artifacts: dict[str, Any] = field(default_factory=dict)
    status: Literal["SUCCEEDED"] = field(init=False, default="SUCCEEDED")


ATEResult = ATESuccess | CommandFailure


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
    counterfactual: bool = False
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
    x_cols: list[str]
    # one item per contrast (binary -> 1 item, categorical -> many items)
    effects: dict[CATEModelResult, Any] = field(default_factory=dict)
    status: Literal["SUCCEEDED"] = field(init=False, default="SUCCEEDED")


CATEResult = CATESuccess | CommandFailure
