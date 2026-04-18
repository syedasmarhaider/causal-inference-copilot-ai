from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.service.llm_service import LLMConfig, LLMService
from python.implementation.workflows.nodes.data_compilation.data_compilation_prompts import (
    data_compilation_causal_semantics_prompt,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import (
    BinaryTreatmentSpecModel,
    CausalSpec,
    ContinuousOutcomeSpecModel,
    BinaryOutcomeSpecModel,
)
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import CausalSpecDraft
from python.implementation.workflows.tools.causal.specs.causal_specs_tool import (
    CausalSpecsTool,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.tools.common.model.data_summary import (
    BooleanColumnProfileModel,
    CategoricalColumnProfileModel,
    DatetimeColumnProfileModel,
    NumericColumnProfileModel,
    OtherColumnProfileModel,
)
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import DatasetProfilingTool


@dataclass(frozen=True)
class CleaningResult:
    cleaned_data_summary: DatasetSummaryModel
    pd_cleaned: pd.DataFrame
    causal: CausalSpec


class _TreatmentSemanticsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    treated: str = Field(..., min_length=1)
    control: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_distinct_labels(self) -> "_TreatmentSemanticsModel":
        if self.treated == self.control:
            raise ValueError("treated and control must be different")
        return self


class _BinaryOutcomeSemanticsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["binary"]
    event: str = Field(..., min_length=1)
    non_event: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_binary_outcome(self) -> "_BinaryOutcomeSemanticsModel":
        if self.event == self.non_event:
            raise ValueError("event and non_event must be different")
        return self


class _ContinuousOutcomeSemanticsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["continuous"]
    unit: str | None = None
    clip_min: float | None = None
    clip_max: float | None = None

    @model_validator(mode="after")
    def _validate_continuous_outcome(self) -> "_ContinuousOutcomeSemanticsModel":
        if (
            self.clip_min is not None
            and self.clip_max is not None
            and self.clip_min > self.clip_max
        ):
            raise ValueError("clip_min must be <= clip_max")
        return self


_OutcomeSemanticsModel = Annotated[
    _BinaryOutcomeSemanticsModel | _ContinuousOutcomeSemanticsModel,
    Field(discriminator="kind"),
]


class _CausalSemanticsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    treatment: _TreatmentSemanticsModel
    outcome: _OutcomeSemanticsModel
    experiment_type: Literal["RCT", "OBSERVATIONAL"]


def cleaning(
    *,
    protocol_discussion: str | None,
    cleaning_instructions: str,
    draft_causal_spec: CausalSpecDraft,
    data_summary: DatasetSummaryModel,
    to_clean_df: pd.DataFrame,
    datasetProfilingTool: DatasetProfilingTool,
    dataManipulationTool: DataManipulationTool,
    llm: LLMService,
) -> CleaningResult:
    _ = data_summary

    draft_scope_columns = _draft_scope_columns(draft_causal_spec)
    _ensure_draft_matches_dataframe(
        draft_causal_spec=draft_causal_spec,
        dataframe=to_clean_df,
        context="input dataframe",
    )

    scoped_df = to_clean_df.loc[:, draft_scope_columns].copy()
    scoped_summary = _profile_dataset(
        dataset_profiling_tool=datasetProfilingTool,
        dataframe=scoped_df,
    )

    effective_instructions = _build_manipulation_instructions(
        protocol_discussion=protocol_discussion,
        cleaning_instructions=cleaning_instructions,
        draft_scope_columns=draft_scope_columns,
    )
    cleaned_candidate_df = scoped_df.copy()
    if effective_instructions:
        cleaned_candidate_df = dataManipulationTool.manipulate(
            dataframe=scoped_df,
            table_name="protocol_scope_df",
            data_summary=datasetProfilingTool.dataset_summary_to_json(scoped_summary),
            instructions=effective_instructions,
            retry_attempts=3,
        )

    _ensure_draft_matches_dataframe(
        draft_causal_spec=draft_causal_spec,
        dataframe=cleaned_candidate_df,
        context="cleaned dataframe",
    )
    cleaned_df = cleaned_candidate_df.loc[:, draft_scope_columns].copy()
    cleaned_summary = _profile_dataset(
        dataset_profiling_tool=datasetProfilingTool,
        dataframe=cleaned_df,
    )
    causal_spec = compile_causal_spec_from_cleaned_summary(
        llm=llm,
        cleaned_summary=cleaned_summary,
        draft_causal_spec=draft_causal_spec,
        protocol_discussion=protocol_discussion,
    )

    return CleaningResult(
        cleaned_data_summary=cleaned_summary,
        pd_cleaned=cleaned_df,
        causal=causal_spec,
    )


def _draft_scope_columns(draft_causal_spec: CausalSpecDraft) -> list[str]:
    ordered_columns = [
        str(draft_causal_spec.treatment_column).strip(),
        str(draft_causal_spec.outcome_column).strip(),
        *(str(column).strip() for column in draft_causal_spec.covariates),
        *(str(column).strip() for column in draft_causal_spec.effect_modifiers),
    ]
    deduped_columns: list[str] = []
    for column in ordered_columns:
        if column and column not in deduped_columns:
            deduped_columns.append(column)
    return deduped_columns


def _ensure_draft_matches_dataframe(
    *,
    draft_causal_spec: CausalSpecDraft,
    dataframe: pd.DataFrame,
    context: str,
) -> None:
    validation_issue = draft_causal_spec.validate_against_dataframe(df=dataframe)
    if validation_issue is None:
        return
    raise ValueError(
        f"{context} does not satisfy draft causal spec: "
        f"[{validation_issue.severity}] {validation_issue.message}"
    )


def _build_manipulation_instructions(
    *,
    protocol_discussion: str | None,
    cleaning_instructions: str,
    draft_scope_columns: Sequence[str],
) -> str:
    normalized_cleaning_instructions = _normalize_text(cleaning_instructions)
    normalized_protocol_discussion = _normalize_text(protocol_discussion)
    if not normalized_cleaning_instructions and not normalized_protocol_discussion:
        return ""

    parts: list[str] = []
    if normalized_cleaning_instructions:
        parts.extend(
            [
                "Confirmed protocol cleaning instructions:",
                normalized_cleaning_instructions,
            ]
        )
    if normalized_protocol_discussion:
        if parts:
            parts.append("")
        parts.extend(
            [
                "Confirmed protocol discussion:",
                normalized_protocol_discussion,
            ]
        )

    if parts:
        parts.extend(
            [
                "",
                "Preserve exactly these columns and do not require or create any additional columns:",
                ", ".join(draft_scope_columns),
                "The cleaned dataframe must keep all of these draft columns available.",
            ]
        )
    return "\n".join(parts).strip()


def _normalize_text(raw: str | None) -> str:
    if raw is None:
        return ""
    return raw.strip()


def _profile_dataset(
    *,
    dataset_profiling_tool: DatasetProfilingTool,
    dataframe: pd.DataFrame,
) -> DatasetSummaryModel:
    return dataset_profiling_tool.extract_dataset_summary(
        dataframe,
        max_categories=200,
        sample_distinct=200,
        compute_quantiles=False,
        strict=True,
    )


def compile_causal_spec_from_cleaned_summary(
    *,
    llm: LLMService,
    cleaned_summary: DatasetSummaryModel,
    draft_causal_spec: CausalSpecDraft,
    protocol_discussion: str | None,
    retry_feedback: str | None = None,
) -> CausalSpec:
    causal_specs_tool = CausalSpecsTool()
    compile_feedback = _normalize_text(retry_feedback) or None

    for attempt in range(2):
        semantics = _compile_causal_semantics_once(
            llm=llm,
            cleaned_summary=cleaned_summary,
            draft_causal_spec=draft_causal_spec,
            protocol_discussion=protocol_discussion,
            compile_feedback=compile_feedback,
        )
        causal_spec = _assemble_causal_spec(
            draft_causal_spec=draft_causal_spec,
            semantics=semantics,
        )
        try:
            return causal_specs_tool.post_validate_backdoor_spec(
                causal_spec=causal_spec,
                data_summary=cleaned_summary,
            )
        except Exception as exc:
            if attempt == 1:
                raise ValueError(
                    "compiled causal spec semantics remained invalid after retry: "
                    f"{compile_feedback or str(exc)}"
                ) from exc
            compile_feedback = _merge_compile_feedback(
                compile_feedback=compile_feedback,
                compile_issue=str(exc).strip() or exc.__class__.__name__,
            )

    raise ValueError(
        "compiled causal spec semantics remained invalid after retry: "
        f"{compile_feedback}"
    )


def _merge_compile_feedback(
    *,
    compile_feedback: str | None,
    compile_issue: str,
) -> str:
    if not compile_feedback:
        return compile_issue
    return f"{compile_feedback}\n\nAlso fix this issue: {compile_issue}"


def _compile_causal_semantics_once(
    *,
    llm: LLMService,
    cleaned_summary: DatasetSummaryModel,
    draft_causal_spec: CausalSpecDraft,
    protocol_discussion: str | None,
    compile_feedback: str | None,
) -> _CausalSemanticsModel:
    context_payload: dict[str, object] = {
        "draft_causal_spec": draft_causal_spec.model_dump(mode="json"),
        "treatment_column_profile": _summary_profile_payload(
            summary=cleaned_summary,
            column=str(draft_causal_spec.treatment_column).strip(),
        ),
        "outcome_column_profile": _summary_profile_payload(
            summary=cleaned_summary,
            column=str(draft_causal_spec.outcome_column).strip(),
        ),
    }
    normalized_protocol_discussion = _normalize_text(protocol_discussion)
    if normalized_protocol_discussion:
        context_payload["protocol_discussion"] = normalized_protocol_discussion
    if compile_feedback:
        context_payload["compile_feedback"] = compile_feedback

    return llm.generate_json(
        schema=_CausalSemanticsModel,
        system_prompt=data_compilation_causal_semantics_prompt(),
        user_prompt=json.dumps(context_payload, ensure_ascii=False),
        config=LLMConfig(model="pro", temperature=0.1),
        history=None,
        max_attempts=3,
    )


def _assemble_causal_spec(
    *,
    draft_causal_spec: CausalSpecDraft,
    semantics: _CausalSemanticsModel,
) -> CausalSpec:
    treatment_column = str(draft_causal_spec.treatment_column).strip()
    outcome_column = str(draft_causal_spec.outcome_column).strip()

    treatment_spec = BinaryTreatmentSpecModel(
        kind="binary",
        column=treatment_column,
        treated=semantics.treatment.treated,
        control=semantics.treatment.control,
    )

    if isinstance(semantics.outcome, _BinaryOutcomeSemanticsModel):
        outcome_spec = BinaryOutcomeSpecModel(
            kind="binary",
            column=outcome_column,
            event=semantics.outcome.event,
            non_event=semantics.outcome.non_event,
        )
    else:
        outcome_spec = ContinuousOutcomeSpecModel(
            kind="continuous",
            column=outcome_column,
            unit=semantics.outcome.unit,
            clip_min=semantics.outcome.clip_min,
            clip_max=semantics.outcome.clip_max,
        )

    return CausalSpec(
        treatment_spec=treatment_spec,
        outcome_spec=outcome_spec,
        covariates=[str(column).strip() for column in draft_causal_spec.covariates],
        effect_modifiers=[
            str(column).strip() for column in draft_causal_spec.effect_modifiers
        ],
        experiment_type=semantics.experiment_type,
    )


def _summary_profile_payload(
    *,
    summary: DatasetSummaryModel,
    column: str,
) -> dict[str, Any]:
    profiles_by_name = {
        str(profile.name).strip(): profile
        for profile in summary.profiles
        if str(profile.name).strip()
    }
    profile = profiles_by_name.get(column)
    if profile is None:
        raise ValueError(f"dataset summary is missing required column '{column}'")
    return _column_prompt_payload(profile)


def _column_prompt_payload(
    profile: (
        NumericColumnProfileModel
        | DatetimeColumnProfileModel
        | BooleanColumnProfileModel
        | CategoricalColumnProfileModel
        | OtherColumnProfileModel
    ),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": str(profile.name).strip(),
        "kind": str(profile.inferred_kind),
        "dtype": profile.dtype,
        "missing_rate": profile.missing_rate,
        "distinct_count": profile.distinct_count,
    }

    if isinstance(profile, NumericColumnProfileModel):
        payload["range"] = {
            "min": profile.summary.min,
            "max": profile.summary.max,
        }
        return payload

    if isinstance(profile, DatetimeColumnProfileModel):
        payload["range"] = {
            "min": profile.summary.min,
            "max": profile.summary.max,
        }
        return payload

    if isinstance(profile, BooleanColumnProfileModel):
        payload["known_values"] = list(profile.summary.counts.keys())
        return payload

    if isinstance(profile, CategoricalColumnProfileModel):
        payload["known_values"] = [item.value for item in profile.summary.top_categories]
        return payload

    if isinstance(profile, OtherColumnProfileModel):
        payload["sample_values"] = list(profile.summary.distinct_values_sample)
        return payload

    return payload
