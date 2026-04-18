from __future__ import annotations
from pydantic import Field

from litellm import ConfigDict
import pandas as pd
from pydantic import BaseModel

from python.domain.models.models import NonEmptyStr
from python.domain.models.validation import ValidationIssueModel

class CausalSpecDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    treatment_column: NonEmptyStr
    outcome_column: NonEmptyStr
    covariates: list[NonEmptyStr] = Field(default_factory=list)
    effect_modifiers: list[NonEmptyStr] = Field(default_factory=list)
    
    def validate_against_dataframe(self, *, df: pd.DataFrame) -> ValidationIssueModel | None:
        if self.treatment_column not in df.columns:
            return ValidationIssueModel(
                severity="FAIL",
                message=f'Treatment column "{self.treatment_column}" not found in dataset',
            )
        if self.outcome_column not in df.columns:
            return ValidationIssueModel(
                severity="FAIL",
                message=f'Outcome column "{self.outcome_column}" not found in dataset',
            )
        missing_covariates = [col for col in self.covariates if col not in df.columns]
        if missing_covariates:
            return ValidationIssueModel(
                severity="WARN",
                message=f'Covariate columns not found in dataset: {", ".join(missing_covariates)}',
            )
        missing_effect_modifiers = [
            col for col in self.effect_modifiers if col not in df.columns
        ]
        if missing_effect_modifiers:
            return ValidationIssueModel(
                severity="WARN",
                message=f'Effect modifier columns not found in dataset: {", ".join(missing_effect_modifiers)}',
            )
        return None
        
