from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AnalysisKind = Literal[
    "descriptive",
    "correlation",
    "linear_regression",
    "logistic_regression",
    "propensity_score",
    "chi_squared",
    "ttest",
]


class AnalyticsPlanModel(BaseModel):
    """LLM output: which analysis to run and with what columns."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    analysis_type: AnalysisKind
    columns: list[str] = Field(default_factory=list)
    target: str | None = None
    predictors: list[str] = Field(default_factory=list)
    treatment: str | None = None
    covariates: list[str] = Field(default_factory=list)
    group_by: str | None = None
    column_a: str | None = None
    column_b: str | None = None
    numeric_column: str | None = None
    group_column: str | None = None

    @model_validator(mode="after")
    def _check(self) -> AnalyticsPlanModel:
        k = self.analysis_type
        if k == "descriptive" and not self.columns:
            raise ValueError("descriptive requires 'columns'")
        if k == "correlation" and len(self.columns) < 2:
            raise ValueError("correlation requires >= 2 columns")
        if k in ("linear_regression", "logistic_regression"):
            if not self.target or not self.predictors:
                raise ValueError(f"{k} requires 'target' and 'predictors'")
        if k == "propensity_score":
            if not self.treatment or not self.covariates:
                raise ValueError("propensity_score requires 'treatment' and 'covariates'")
        if k == "chi_squared" and (not self.column_a or not self.column_b):
            raise ValueError("chi_squared requires 'column_a' and 'column_b'")
        if k == "ttest" and (not self.numeric_column or not self.group_column):
            raise ValueError("ttest requires 'numeric_column' and 'group_column'")
        return self


class AnalyticsResultModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    analysis_type: AnalysisKind
    summary: str
    tables: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
