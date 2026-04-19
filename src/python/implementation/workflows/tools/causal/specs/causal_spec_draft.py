from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, ClassVar

import pandas as pd
from litellm import ConfigDict
from pydantic import BaseModel, Field, model_validator

from python.domain.models.models import NonEmptyStr
from python.domain.models.validation import ValidationIssueModel
from python.domain.service.llm_service import LLMConfig, LLMService
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


ID_COL_AUTO_FILL = "__auto_id__"

class CausalSpecDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    SUMMARY_FIELD_NAMES: ClassVar[tuple[str, ...] | None] = None

    id_col: NonEmptyStr = Field(default=ID_COL_AUTO_FILL)
    treatment_column: NonEmptyStr
    outcome_column: NonEmptyStr
    covariates: list[NonEmptyStr] = Field(default_factory=list)
    effect_modifiers: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_against_summary(self) -> CausalSpecDraft:
        summary_field_names = type(self).SUMMARY_FIELD_NAMES
        if summary_field_names is None:
            return self

        known_columns = set(summary_field_names)
        treatment_column = str(self.treatment_column).strip()
        outcome_column = str(self.outcome_column).strip()
        covariates = [str(column).strip() for column in self.covariates]
        effect_modifiers = [str(column).strip() for column in self.effect_modifiers]
        referenced_columns = [
            treatment_column,
            outcome_column,
            *covariates,
            *effect_modifiers,
        ]
        missing_columns = sorted({column for column in referenced_columns if column not in known_columns})
        if missing_columns:
            raise ValueError(
                "causal draft references unknown dataset_summary columns: "
                f"{missing_columns}"
            )

        if treatment_column == outcome_column:
            raise ValueError("treatment and outcome must be different columns")

        duplicate_covariates = _find_duplicates(covariates)
        if duplicate_covariates:
            raise ValueError(f"covariates contain duplicates: {duplicate_covariates}")

        duplicate_effect_modifiers = _find_duplicates(effect_modifiers)
        if duplicate_effect_modifiers:
            raise ValueError(
                f"effect_modifiers contain duplicates: {duplicate_effect_modifiers}"
            )

        overlap = sorted(set(covariates).intersection(effect_modifiers))
        if overlap:
            raise ValueError(f"covariates and effect_modifiers overlap: {overlap}")

        protected_overlap = sorted(
            {
                column
                for column in covariates + effect_modifiers
                if column in {treatment_column, outcome_column}
            }
        )
        if protected_overlap:
            raise ValueError(
                "covariates and effect_modifiers must not include treatment or outcome "
                f"columns: {protected_overlap}"
            )
        return self

    @classmethod
    def for_dataset_summary(cls, dataset_summary: DatasetSummaryModel) -> type[CausalSpecDraft]:
        field_names = _extract_summary_field_names(dataset_summary)
        if not field_names:
            raise ValueError("dataset_summary must contain at least one non-empty column name")

        return type(
            f"{cls.__name__}ForFields_{len(field_names)}",
            (cls,),
            {
                "__module__": cls.__module__,
                "SUMMARY_FIELD_NAMES": field_names,
            },
        )

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
                message=(
                    "Effect modifier columns not found in dataset: "
                    f'{", ".join(missing_effect_modifiers)}'
                ),
            )
        return None


def compile_causal_spec_draft_from_discussion(
    *,
    llm: LLMService,
    protocol_discussion: str,
    dataset_summary: DatasetSummaryModel,
    retry_feedback: str | None = None,
    previous_draft: CausalSpecDraft | None = None,
) -> CausalSpecDraft:
    schema = CausalSpecDraft.for_dataset_summary(dataset_summary)
    user_payload: dict[str, Any] = {
        "protocol_discussion": protocol_discussion,
        "dataset_summary": dataset_summary.model_dump(mode="json"),
    }
    if previous_draft is not None:
        user_payload["previous_draft"] = previous_draft.model_dump(mode="json")
    if retry_feedback is not None:
        user_payload["retry_feedback"] = retry_feedback

    return llm.generate_json(
        schema=schema,
        system_prompt=get_compile_causal_spec_draft_prompt(),
        user_prompt=json.dumps(user_payload, ensure_ascii=False),
        config=LLMConfig(model="basic", temperature=0.0),
        history=None,
        max_attempts=2,
    )


def get_compile_causal_spec_draft_prompt() -> str:
    return """
You are compiling a strict causal draft from a confirmed protocol discussion.

Inputs:
- protocol_discussion: authoritative confirmed protocol text
- dataset_summary: authoritative dataset metadata summary with exact column names
- previous_draft: optional prior draft that failed dataframe validation
- retry_feedback: optional validation feedback that must be fixed

Task:
- Return the best grounded CausalSpecDraft using exact dataset_summary column names.

Rules:
- Use only columns that appear exactly in dataset_summary.
- Never invent, rename, normalize, or paraphrase column names.
- treatment_column and outcome_column must be explicit and different.
- covariates and effect_modifiers are optional but must be grounded in the confirmed protocol discussion.
- Remove duplicates.
- Do not place treatment or outcome inside covariates or effect_modifiers.
- Do not let covariates and effect_modifiers overlap.
- If retry_feedback is present, fix that issue directly in the next draft.
- Prefer an empty list over guessing an unclear covariate or effect modifier.

Output:
Return only JSON matching the CausalSpecDraft schema.
""".strip()


def _extract_summary_field_names(dataset_summary: DatasetSummaryModel) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for profile in dataset_summary.profiles:
        name = str(profile.name).strip()
        if not name or name in seen:
            continue
        names.append(name)
        seen.add(name)
    return tuple(names)


def _find_duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
            continue
        seen.add(value)
    return duplicates


__all__ = [
    "CausalSpecDraft",
    "compile_causal_spec_draft_from_discussion",
    "get_compile_causal_spec_draft_prompt",
]
