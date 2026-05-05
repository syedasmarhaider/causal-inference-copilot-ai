from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.service.llm_service import LLMConfig, LLMService
from python.implementation.workflows.nodes.data_compilation.data_compilation_prompts import (
    data_compilation_causal_semantics_prompt,
    data_compilation_data_manipulation_plan_prompt,
    data_compilation_simple_transform_prompt,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import (
    BinaryOutcomeSpecModel,
    BinaryTreatmentSpecModel,
    CausalSpec,
    ContinuousOutcomeSpecModel,
)
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    ID_COL_AUTO_FILL,
    CausalSpecDraft,
)
from python.implementation.workflows.tools.causal.specs.causal_specs_tool import (
    CausalSpecsTool,
)
from python.implementation.workflows.tools.common.model.data_summary import (
    BooleanColumnProfileModel,
    CategoricalColumnProfileModel,
    DatasetSummaryModel,
    DatetimeColumnProfileModel,
    NumericColumnProfileModel,
    OtherColumnProfileModel,
)
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.tools.simple_data_transformation_tool.simple_data_transformation_tool import (
    ColumnTransformationSpec,
    SimpleDataTransformationSpec,
    SimpleDataTransformationTool,
)


@dataclass(frozen=True)
class CleaningResult:
    cleaned_data_summary: DatasetSummaryModel
    pd_cleaned: pd.DataFrame
    causal: CausalSpec
    missingness_decisions: MissingnessDecisionList


ColumnRole = Literal["treatment", "outcome", "covariate", "effect_modifier"]
MissingnessResolution = Literal["none_needed", "drop_rows", "impute"]


class MissingnessDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    column: str = Field(..., min_length=1)
    role: ColumnRole
    missing_count_before: int = Field(..., ge=0)
    resolution: MissingnessResolution
    reason: str = Field(..., min_length=1)
    instruction: str = Field(..., min_length=1)
    missing_count_after: int = Field(..., ge=0)


class MissingnessDecisionList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[MissingnessDecision] = Field(..., min_length=1)


class _SimpleTransformationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[ColumnTransformationSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_columns(self) -> _SimpleTransformationPlan:
        columns = [str(column.column).strip() for column in self.columns]
        duplicates = sorted({column for column in columns if columns.count(column) > 1})
        if duplicates:
            raise ValueError(f"simple transform plan contains duplicate columns: {duplicates}")
        fill_columns = [
            str(column.column).strip() for column in self.columns if column.has_fill_value
        ]
        if fill_columns:
            raise ValueError(
                "simple transform plan must not handle missingness with fill_value; "
                f"leave missingness to SQL cleaning: {fill_columns}"
            )
        return self

    def to_spec(self) -> SimpleDataTransformationSpec | None:
        if not self.columns:
            return None
        return SimpleDataTransformationSpec(columns=self.columns)


class _DataManipulationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    instructions: str | None = None

    @model_validator(mode="after")
    def _normalize_instructions(self) -> _DataManipulationPlan:
        if self.instructions is not None:
            normalized = self.instructions.strip()
            self.instructions = normalized or None
        return self


class _TreatmentSemanticsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    treated: str = Field(..., min_length=1)
    control: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_distinct_labels(self) -> _TreatmentSemanticsModel:
        if self.treated == self.control:
            raise ValueError("treated and control must be different")
        return self


class _BinaryOutcomeSemanticsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["binary"]
    event: str = Field(..., min_length=1)
    non_event: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_binary_outcome(self) -> _BinaryOutcomeSemanticsModel:
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
    def _validate_continuous_outcome(self) -> _ContinuousOutcomeSemanticsModel:
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
    review_recompile_request: str | None,
    draft_causal_spec: CausalSpecDraft,
    data_summary: DatasetSummaryModel,
    to_clean_df: pd.DataFrame,
    datasetProfilingTool: DatasetProfilingTool,
    simpleDataTransformationTool: SimpleDataTransformationTool,
    dataManipulationTool: DataManipulationTool,
    llm: LLMService,
) -> CleaningResult:
    prepared_df, effective_id_col = _resolve_identifier_column(
        dataframe=to_clean_df,
        draft_causal_spec=draft_causal_spec,
    )
    required_columns = _required_columns(
        draft_causal_spec=draft_causal_spec,
        effective_id_col=effective_id_col,
    )
    _ensure_columns_present(
        dataframe=prepared_df,
        columns=required_columns,
        context="input dataframe",
    )
    prepared_summary = _profile_dataset(
        dataset_profiling_tool=datasetProfilingTool,
        dataframe=prepared_df,
    )

    simple_plan = _plan_simple_transformations(
        llm=llm,
        source_summary=data_summary,
        prepared_summary=prepared_summary,
        draft_causal_spec=draft_causal_spec,
        effective_id_col=effective_id_col,
        protocol_discussion=protocol_discussion,
        cleaning_instructions=cleaning_instructions,
        review_recompile_request=review_recompile_request,
    )
    simple_transform_spec = simple_plan.to_spec()
    transformed_df = prepared_df.copy()
    if simple_transform_spec is not None:
        transformed_df = simpleDataTransformationTool.transform(
            dataframe=prepared_df,
            specification=simple_transform_spec,
        )
    transformed_summary = _profile_dataset(
        dataset_profiling_tool=datasetProfilingTool,
        dataframe=transformed_df,
    )

    manipulation_plan = _plan_data_manipulation(
        llm=llm,
        source_summary=data_summary,
        transformed_summary=transformed_summary,
        transformed_df=transformed_df,
        draft_causal_spec=draft_causal_spec,
        effective_id_col=effective_id_col,
        required_columns=required_columns,
        simple_plan=simple_plan,
        protocol_discussion=protocol_discussion,
        cleaning_instructions=cleaning_instructions,
        review_recompile_request=review_recompile_request,
    )
    cleaned_candidate_df = transformed_df.copy()
    if manipulation_plan.instructions:
        cleaned_candidate_df = dataManipulationTool.manipulate(
            dataframe=transformed_df,
            table_name="protocol_scope_df",
            data_summary=datasetProfilingTool.dataset_summary_to_json(transformed_summary),
            instructions=manipulation_plan.instructions,
            retry_attempts=3,
        )

    _ensure_columns_present(
        dataframe=cleaned_candidate_df,
        columns=required_columns,
        context="cleaned dataframe",
    )
    cleaned_df = _project_required_columns(
        dataframe=cleaned_candidate_df,
        columns=required_columns,
    )
    missingness_decisions = _finalize_missingness_decisions(
        draft_causal_spec=draft_causal_spec,
        before_df=prepared_df,
        cleaned_df=cleaned_df,
    )
    cleaned_summary = _profile_dataset(
        dataset_profiling_tool=datasetProfilingTool,
        dataframe=cleaned_df,
    )
    causal_spec = compile_causal_spec_from_cleaned_summary(
        llm=llm,
        cleaned_summary=cleaned_summary,
        draft_causal_spec=draft_causal_spec,
        protocol_discussion=protocol_discussion,
        effective_id_col=effective_id_col,
    )

    return CleaningResult(
        cleaned_data_summary=cleaned_summary,
        pd_cleaned=cleaned_df,
        causal=causal_spec,
        missingness_decisions=missingness_decisions,
    )


def _resolve_identifier_column(
    *,
    dataframe: pd.DataFrame,
    draft_causal_spec: CausalSpecDraft,
) -> tuple[pd.DataFrame, str]:
    draft_id_col = str(draft_causal_spec.id_col).strip()
    if draft_id_col in dataframe.columns:
        identifier = dataframe[draft_id_col]
        if bool(identifier.notna().all()) and bool(identifier.is_unique):
            return dataframe.copy(), draft_id_col

    prepared = dataframe.copy()
    prepared[ID_COL_AUTO_FILL] = pd.RangeIndex(start=1, stop=len(prepared) + 1, step=1)
    return prepared, ID_COL_AUTO_FILL


def _required_columns(
    *,
    draft_causal_spec: CausalSpecDraft,
    effective_id_col: str,
) -> list[str]:
    ordered_columns = [
        effective_id_col,
        str(draft_causal_spec.treatment_column).strip(),
        str(draft_causal_spec.outcome_column).strip(),
        *(str(column).strip() for column in draft_causal_spec.covariates),
        *(str(column).strip() for column in draft_causal_spec.effect_modifiers),
    ]
    return [
        column
        for index, column in enumerate(ordered_columns)
        if column and column not in ordered_columns[:index]
    ]


def _ensure_columns_present(
    *,
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    context: str,
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{context} is missing required column(s): {', '.join(missing)}")


def _project_required_columns(
    *,
    dataframe: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    return dataframe.loc[:, list(columns)].copy()


def _plan_simple_transformations(
    *,
    llm: LLMService,
    source_summary: DatasetSummaryModel,
    prepared_summary: DatasetSummaryModel,
    draft_causal_spec: CausalSpecDraft,
    effective_id_col: str,
    protocol_discussion: str | None,
    cleaning_instructions: str,
    review_recompile_request: str | None,
) -> _SimpleTransformationPlan:
    payload: dict[str, Any] = {
        "confirmed_protocol_discussion": _normalize_text(protocol_discussion),
        "confirmed_protocol_cleaning_instructions": _normalize_text(cleaning_instructions),
        "source_dataset_summary": _dataset_summary_prompt_payload(source_summary),
        "prepared_dataset_summary": _dataset_summary_prompt_payload(prepared_summary),
        "draft_causal_spec": draft_causal_spec.model_dump(mode="json"),
        "effective_id_col": effective_id_col,
        "expected_role_by_column": _expected_role_by_column(draft_causal_spec),
    }
    normalized_review_recompile_request = _normalize_text(review_recompile_request)
    if normalized_review_recompile_request:
        payload["review_recompile_request"] = normalized_review_recompile_request

    plan = llm.generate_json(
        schema=_SimpleTransformationPlan,
        system_prompt=data_compilation_simple_transform_prompt(),
        user_prompt=json.dumps(payload, ensure_ascii=False),
        config=LLMConfig(model="basic", temperature=0.7),
        history=None,
        max_attempts=2,
    )

    prepared_columns = {
        str(profile.name).strip()
        for profile in prepared_summary.profiles
        if str(profile.name).strip()
    }
    unknown_columns = sorted(
        str(column.column).strip()
        for column in plan.columns
        if str(column.column).strip() not in prepared_columns
    )
    if unknown_columns:
        raise ValueError(
            "simple transform plan contains unknown prepared columns: " f"{unknown_columns}"
        )

    transformed_id = [
        str(column.column).strip()
        for column in plan.columns
        if str(column.column).strip() == effective_id_col
    ]
    if transformed_id:
        raise ValueError(
            "simple transform plan must not transform the effective identifier column: "
            f"{effective_id_col}"
        )
    return plan


def _plan_data_manipulation(
    *,
    llm: LLMService,
    source_summary: DatasetSummaryModel,
    transformed_summary: DatasetSummaryModel,
    transformed_df: pd.DataFrame,
    draft_causal_spec: CausalSpecDraft,
    effective_id_col: str,
    required_columns: Sequence[str],
    simple_plan: _SimpleTransformationPlan,
    protocol_discussion: str | None,
    cleaning_instructions: str,
    review_recompile_request: str | None,
) -> _DataManipulationPlan:
    role_by_column = _expected_role_by_column(draft_causal_spec)
    payload: dict[str, Any] = {
        "confirmed_protocol_discussion": _normalize_text(protocol_discussion),
        "confirmed_protocol_cleaning_instructions": _normalize_text(cleaning_instructions),
        "source_dataset_summary": _dataset_summary_prompt_payload(source_summary),
        "transformed_dataset_summary": _dataset_summary_prompt_payload(transformed_summary),
        "draft_causal_spec": draft_causal_spec.model_dump(mode="json"),
        "effective_id_col": effective_id_col,
        "required_final_columns": list(required_columns),
        "expected_role_by_column": role_by_column,
        "simple_transformations_applied": [
            column.model_dump(mode="json", exclude_none=True) for column in simple_plan.columns
        ],
        "required_column_missing_counts": _missing_counts_by_column(
            transformed_df,
            role_by_column,
        ),
    }
    normalized_review_recompile_request = _normalize_text(review_recompile_request)
    if normalized_review_recompile_request:
        payload["review_recompile_request"] = normalized_review_recompile_request

    return llm.generate_json(
        schema=_DataManipulationPlan,
        system_prompt=data_compilation_data_manipulation_plan_prompt(),
        user_prompt=json.dumps(payload, ensure_ascii=False),
        config=LLMConfig(model="basic", temperature=0.4),
        history=None,
        max_attempts=2,
    )


def _finalize_missingness_decisions(
    *,
    draft_causal_spec: CausalSpecDraft,
    before_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
) -> MissingnessDecisionList:
    role_by_column = _expected_role_by_column(draft_causal_spec)
    before_counts = _missing_counts_by_column(before_df, role_by_column)
    after_counts = _missing_counts_by_column(cleaned_df, role_by_column)
    decisions = MissingnessDecisionList(
        decisions=[
            _build_missingness_decision(
                column=column,
                role=role,
                before_count=before_counts.get(column, 0),
                after_count=after_counts.get(column, 0),
                rows_before=len(before_df),
                rows_after=len(cleaned_df),
            )
            for column, role in role_by_column.items()
        ]
    )
    unresolved = [decision for decision in decisions.decisions if decision.missing_count_after > 0]
    if unresolved:
        formatted = ", ".join(
            f"{decision.column}={decision.missing_count_after}" for decision in unresolved
        )
        raise ValueError(
            "cleaned dataframe still contains protocol-scope missing values: " f"{formatted}"
        )
    return decisions


def _build_missingness_decision(
    *,
    column: str,
    role: ColumnRole,
    before_count: int,
    after_count: int,
    rows_before: int,
    rows_after: int,
) -> MissingnessDecision:
    if before_count == 0:
        return MissingnessDecision(
            column=column,
            role=role,
            missing_count_before=before_count,
            resolution="none_needed",
            reason="No missing values were detected before cleaning.",
            instruction="No missingness action was required.",
            missing_count_after=after_count,
        )

    used_row_filtering = rows_after < rows_before
    resolution: MissingnessResolution = "drop_rows" if used_row_filtering else "impute"
    action = "row filtering" if used_row_filtering else "cleaning or imputation"
    return MissingnessDecision(
        column=column,
        role=role,
        missing_count_before=before_count,
        resolution=resolution,
        reason=f"Missing values were resolved during SQL {action}.",
        instruction="Handled by the SQL cleaning step from the protocol cleaning context.",
        missing_count_after=after_count,
    )


def _expected_role_by_column(
    draft_causal_spec: CausalSpecDraft,
) -> dict[str, ColumnRole]:
    role_by_column: dict[str, ColumnRole] = {
        str(draft_causal_spec.treatment_column).strip(): "treatment",
        str(draft_causal_spec.outcome_column).strip(): "outcome",
    }
    for column in draft_causal_spec.covariates:
        normalized = str(column).strip()
        if normalized:
            role_by_column[normalized] = "covariate"
    for column in draft_causal_spec.effect_modifiers:
        normalized = str(column).strip()
        if normalized:
            role_by_column[normalized] = "effect_modifier"
    return role_by_column


def _missing_counts_by_column(
    dataframe: pd.DataFrame,
    role_by_column: dict[str, ColumnRole],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in role_by_column:
        if column not in dataframe.columns:
            counts[column] = 0
            continue
        counts[column] = int(dataframe[column].isna().sum())
    return counts


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


def _dataset_summary_prompt_payload(summary: DatasetSummaryModel) -> dict[str, Any]:
    return {
        "n_rows": summary.n_rows,
        "columns": [_column_prompt_payload(profile) for profile in summary.profiles],
    }


def compile_causal_spec_from_cleaned_summary(
    *,
    llm: LLMService,
    cleaned_summary: DatasetSummaryModel,
    draft_causal_spec: CausalSpecDraft,
    protocol_discussion: str | None,
    retry_feedback: str | None = None,
    effective_id_col: str | None = None,
) -> CausalSpec:
    causal_specs_tool = CausalSpecsTool()
    compile_feedback = _normalize_text(retry_feedback) or None
    resolved_id_col = effective_id_col or _resolve_summary_identifier_column(
        cleaned_summary=cleaned_summary,
        draft_causal_spec=draft_causal_spec,
    )

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
            effective_id_col=resolved_id_col,
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
        "compiled causal spec semantics remained invalid after retry: " f"{compile_feedback}"
    )


def _merge_compile_feedback(
    *,
    compile_feedback: str | None,
    compile_issue: str,
) -> str:
    if not compile_feedback:
        return compile_issue
    return f"{compile_feedback}\n\nAlso fix this issue: {compile_issue}"


def _resolve_summary_identifier_column(
    *,
    cleaned_summary: DatasetSummaryModel,
    draft_causal_spec: CausalSpecDraft,
) -> str:
    summary_columns = {
        str(profile.name).strip()
        for profile in cleaned_summary.profiles
        if str(profile.name).strip()
    }
    draft_id_col = str(draft_causal_spec.id_col).strip()
    if draft_id_col in summary_columns:
        return draft_id_col
    if ID_COL_AUTO_FILL in summary_columns:
        return ID_COL_AUTO_FILL
    return draft_id_col


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
        config=LLMConfig(model="pro", temperature=0.6),
        history=None,
        max_attempts=3,
    )


def _assemble_causal_spec(
    *,
    draft_causal_spec: CausalSpecDraft,
    semantics: _CausalSemanticsModel,
    effective_id_col: str,
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
        effect_modifiers=[str(column).strip() for column in draft_causal_spec.effect_modifiers],
        experiment_type=semantics.experiment_type,
        id_col=effective_id_col,
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
