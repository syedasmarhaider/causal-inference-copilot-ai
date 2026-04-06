from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

import pandas as pd

from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)

CommandType = Literal["FIT", "ATE", "CATE"]
ResultStatus = Literal["SUCCEEDED", "FAILED"]
ATEModelResult = Literal["for_treatment", "ate", "ate_interval", "ate_inference"]
CATEModelResult = Literal["for_treatment", "cate", "cate_interval", "cate_inference"]


def _now_utc() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class BaseCommand:
    model_name: str
    df: pd.DataFrame = field(repr=False)
    run_id: UUID
    inference_ready_spec: InferenceReadyCausalSpec
    created_at: datetime = field(default_factory=_now_utc)
    options: dict[str, Any] = field(
        default_factory=dict
    )  # pyright: ignore[reportUnknownVariableType]


@dataclass(frozen=True, slots=True, eq=False)
class FitInputs:
    model_spec: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class FitCommand(BaseCommand):
    inputs: FitInputs
    command: Literal["FIT"] = field(init=False, default="FIT")


@dataclass(frozen=True, slots=True, eq=False)
class ErrorInfo:
    code: str
    message: str
    details: dict[str, Any] = field(
        default_factory=dict
    )  # pyright: ignore[reportUnknownVariableType]


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class BaseResult:
    run_id: UUID
    status: ResultStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)  # type: ignore


@dataclass(frozen=True, slots=True, eq=False)
class CommandFailure(BaseResult):
    error: ErrorInfo
    status: Literal["FAILED"] = field(init=False, default="FAILED")


@dataclass(frozen=True, slots=True, eq=False)
class FitSuccess(BaseResult):
    fitted_model_id: UUID
    artifacts: dict[str, Any] = field(default_factory=dict)  # type: ignore
    status: Literal["SUCCEEDED"] = field(init=False, default="SUCCEEDED")


FitResult = FitSuccess | CommandFailure


@dataclass(frozen=True, slots=True, eq=False)
class ATEInputsModel:
    alpha: float = 0.05


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class ATECommand(BaseCommand):
    fitted_model_id: UUID
    inputs: ATEInputsModel
    command: Literal["ATE"] = field(init=False, default="ATE")


@dataclass(frozen=True, slots=True, eq=False)
class ATESuccess(BaseResult):
    fitted_model_id: UUID
    contrast: dict[str, Any]
    ate: list[dict[ATEModelResult, Any]]
    artifacts: dict[str, Any] = field(default_factory=dict)  # type: ignore
    status: Literal["SUCCEEDED"] = field(init=False, default="SUCCEEDED")


ATEResult = ATESuccess | CommandFailure


@dataclass(frozen=True, slots=True, eq=False)
class CATEInputs:
    x_rows: pd.DataFrame = field(repr=False)
    counterfactual: bool = False
    alpha: float = 0.05


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CATECommand(BaseCommand):
    fitted_model_id: UUID
    inputs: CATEInputs
    command: Literal["CATE"] = field(init=False, default="CATE")


@dataclass(frozen=True, slots=True, eq=False)
class CATESuccess(BaseResult):
    fitted_model_id: UUID
    x_cols: list[str]
    effects: dict[CATEModelResult, Any] = field(default_factory=dict)  # type: ignore
    status: Literal["SUCCEEDED"] = field(init=False, default="SUCCEEDED")


CATEResult = CATESuccess | CommandFailure
