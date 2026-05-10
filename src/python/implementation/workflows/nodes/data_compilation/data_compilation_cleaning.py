from __future__ import annotations

import json
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.service.llm_service import LLMConfig, LLMService
from python.implementation.workflows.nodes.data_compilation.data_compilation_prompts import (
    data_compilation_adaptive_cleaning_instruction_prompt,
    data_compilation_causal_semantics_prompt,
    data_compilation_missingness_instruction_prompt,
    data_compilation_transformation_instruction_prompt,
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


@dataclass(frozen=True)
class CleaningResult:
    cleaned_data_summary: DatasetSummaryModel
    pd_cleaned: pd.DataFrame
    causal: CausalSpec
    missingness_decisions: MissingnessDecisionList
    cleaning_notes: tuple[str, ...] = ()


ColumnRole = Literal[
    "treatment",
    "outcome",
    "negative_control_outcome",
    "covariate",
    "effect_modifier",
]
MissingnessResolution = Literal["none_needed", "drop_rows", "impute"]
CleaningInstructionAction = Literal["run_instruction", "done"]
CleaningStage = Literal["transformation", "missingness", "cleanup_1", "cleanup_2"]
_PROTOCOL_SCOPE_TABLE = "protocol_scope_df"
_MAX_CLEANING_MANIPULATION_STAGES = 4
_COMPACT_VALUE_LIMIT = 25
_MISSINGNESS_INDICATOR_MIN_COUNT = 50
_MISSINGNESS_INDICATOR_MIN_RATE = 0.01


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


class _CleaningInstructionStep(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: CleaningInstructionAction
    instruction: str | None = None
    reason: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_instruction(self) -> _CleaningInstructionStep:
        if self.action == "run_instruction":
            if not self.instruction or not self.instruction.strip():
                raise ValueError("instruction is required when action=run_instruction")
            self.instruction = self.instruction.strip()
        else:
            self.instruction = None
        self.reason = self.reason.strip()
        return self


@dataclass(frozen=True)
class _ExecutedCleaningInstruction:
    stage: CleaningStage
    instruction: str
    reason: str
    rows_before: int
    rows_after: int
    missing_counts_before: dict[str, int]
    missing_counts_after: dict[str, int]


@dataclass(frozen=True)
class _MissingnessIndicatorSpec:
    source_column: str
    indicator_column: str
    role: ColumnRole
    missing_by_id: pd.Series


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
    negative_control_outcome: _OutcomeSemanticsModel | None = None
    experiment_type: Literal["RCT", "OBSERVATIONAL"]


@dataclass(frozen=True)
class _CleaningFailure:
    error: str
    stage: CleaningStage | None = None
    failed_instruction: str | None = None
    current_summary: DatasetSummaryModel | None = None
    current_missing_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class _SemanticConsistencyIssue:
    column: str
    role: str
    message: str
    allowed_values: tuple[Any, ...]
    unexpected_values: tuple[str, ...] = ()


def cleaning(
    *,
    review_recompile_request: str | None,
    draft_causal_spec: CausalSpecDraft,
    data_summary: DatasetSummaryModel,
    to_clean_df: pd.DataFrame,
    datasetProfilingTool: DatasetProfilingTool,
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

    (
        cleaned_df,
        missingness_decisions,
        cleaning_notes,
        indicator_role_by_column,
    ) = _clean_with_adaptive_manipulation(
        llm=llm,
        prepared_summary=prepared_summary,
        prepared_df=prepared_df,
        before_df=prepared_df,
        draft_causal_spec=draft_causal_spec,
        effective_id_col=effective_id_col,
        required_columns=required_columns,
        protocol_discussion=protocol_discussion,
        cleaning_instructions=cleaning_instructions,
        review_recompile_request=review_recompile_request,
        dataset_profiling_tool=datasetProfilingTool,
        data_manipulation_tool=dataManipulationTool,
    )
    cleaned_df, cleaned_summary, causal_spec, missingness_decisions, semantic_notes = (
        _compile_and_repair_semantic_consistency(
            llm=llm,
            cleaned_df=cleaned_df,
            before_df=prepared_df,
            draft_causal_spec=draft_causal_spec,
            effective_id_col=effective_id_col,
            required_columns=list(cleaned_df.columns),
            indicator_role_by_column=indicator_role_by_column,
            protocol_discussion=protocol_discussion,
            dataset_profiling_tool=datasetProfilingTool,
            data_manipulation_tool=dataManipulationTool,
        )
    )

    return CleaningResult(
        cleaned_data_summary=cleaned_summary,
        pd_cleaned=cleaned_df,
        causal=causal_spec,
        missingness_decisions=missingness_decisions,
        cleaning_notes=(*cleaning_notes, *semantic_notes),
    )


def _compile_and_repair_semantic_consistency(
    *,
    llm: LLMService,
    cleaned_df: pd.DataFrame,
    before_df: pd.DataFrame,
    draft_causal_spec: CausalSpecDraft,
    effective_id_col: str,
    required_columns: Sequence[str],
    indicator_role_by_column: dict[str, ColumnRole],
    protocol_discussion: str | None,
    dataset_profiling_tool: DatasetProfilingTool,
    data_manipulation_tool: DataManipulationTool,
) -> tuple[
    pd.DataFrame,
    DatasetSummaryModel,
    CausalSpec,
    MissingnessDecisionList,
    tuple[str, ...],
]:
    role_by_column = {
        **_expected_role_by_column(draft_causal_spec),
        **indicator_role_by_column,
    }
    current_df = cleaned_df
    semantic_notes: list[str] = []
    current_missingness_decisions = _finalize_missingness_decisions(
        draft_causal_spec=draft_causal_spec,
        before_df=before_df,
        cleaned_df=current_df,
    )

    for attempt_index in range(2):
        cleaned_summary = _profile_dataset(
            dataset_profiling_tool=dataset_profiling_tool,
            dataframe=current_df,
        )
        causal_spec = compile_causal_spec_from_cleaned_summary(
            llm=llm,
            cleaned_summary=cleaned_summary,
            draft_causal_spec=draft_causal_spec,
            protocol_discussion=protocol_discussion,
            effective_id_col=effective_id_col,
            indicator_role_by_column=indicator_role_by_column,
        )
        semantic_issues = _compiled_semantic_consistency_issues(
            dataframe=current_df,
            causal_spec=causal_spec,
        )
        if not semantic_issues:
            return (
                current_df,
                cleaned_summary,
                causal_spec,
                current_missingness_decisions,
                tuple(semantic_notes),
            )

        if attempt_index == 1:
            raise ValueError(
                "cleaned dataframe is inconsistent with compiled causal semantics: "
                f"{_format_semantic_consistency_issues(semantic_issues)}"
            )

        repair_instruction = _semantic_consistency_repair_instruction(
            causal_spec=causal_spec,
            issues=semantic_issues,
        )
        repaired_df = data_manipulation_tool.manipulate(
            dataframe=current_df,
            table_name=_PROTOCOL_SCOPE_TABLE,
            data_summary=_json_dumps(
                _compact_dataset_summary(
                    summary=cleaned_summary,
                    role_by_column=role_by_column,
                    required_columns=required_columns,
                )
            ),
            instructions=repair_instruction,
        )
        _ensure_columns_present(
            dataframe=repaired_df,
            columns=required_columns,
            context="semantic consistency repair output dataframe",
        )
        _ensure_identifier_integrity(
            before_df=current_df,
            after_df=repaired_df,
            effective_id_col=effective_id_col,
            context="semantic consistency repair output dataframe",
        )
        current_df = _project_required_columns(
            dataframe=repaired_df,
            columns=required_columns,
        )
        current_missingness_decisions = _finalize_missingness_decisions(
            draft_causal_spec=draft_causal_spec,
            before_df=before_df,
            cleaned_df=current_df,
        )
        semantic_notes.append(
            "Resolved compiled semantic consistency issue before validation: "
            f"{_format_semantic_consistency_issues(semantic_issues)}"
        )

    raise ValueError("semantic consistency repair failed unexpectedly")


def _compiled_semantic_consistency_issues(
    *,
    dataframe: pd.DataFrame,
    causal_spec: CausalSpec,
) -> list[_SemanticConsistencyIssue]:
    issues: list[_SemanticConsistencyIssue] = []
    treatment = causal_spec.treatment_spec
    issues.extend(
        _binary_literal_consistency_issues(
            dataframe=dataframe,
            column=str(treatment.column),
            role="treatment",
            allowed_values=(treatment.treated, treatment.control),
            outside_message=(
                "Treatment column contains values outside the compiled treated/control literals."
            ),
            missing_class_message=(
                "Treatment column does not contain both compiled treated/control literals."
            ),
        )
    )
    outcome = causal_spec.outcome_spec
    if isinstance(outcome, BinaryOutcomeSpecModel):
        issues.extend(
            _binary_literal_consistency_issues(
                dataframe=dataframe,
                column=str(outcome.column),
                role="outcome",
                allowed_values=(outcome.event, outcome.non_event),
                outside_message=(
                    "Binary outcome column contains values outside the compiled event/non-event literals."
                ),
                missing_class_message=(
                    "Binary outcome column does not contain both compiled event/non-event literals."
                ),
            )
        )
    negative_control = causal_spec.negative_control_outcome
    if isinstance(negative_control, BinaryOutcomeSpecModel):
        issues.extend(
            _binary_literal_consistency_issues(
                dataframe=dataframe,
                column=str(negative_control.column),
                role="negative_control_outcome",
                allowed_values=(negative_control.event, negative_control.non_event),
                outside_message=(
                    "Negative-control outcome column contains values outside the compiled event/non-event literals."
                ),
                missing_class_message=(
                    "Negative-control outcome column does not contain both compiled event/non-event literals."
                ),
            )
        )
    return issues


def _binary_literal_consistency_issues(
    *,
    dataframe: pd.DataFrame,
    column: str,
    role: str,
    allowed_values: tuple[Any, Any],
    outside_message: str,
    missing_class_message: str,
) -> list[_SemanticConsistencyIssue]:
    if column not in dataframe.columns:
        return [
            _SemanticConsistencyIssue(
                column=column,
                role=role,
                message=f"{role} column is missing from the cleaned dataframe.",
                allowed_values=allowed_values,
            )
        ]
    observed = _normalized_discrete_counts(dataframe[column].dropna())
    allowed_keys = {_normalize_discrete_value(value) for value in allowed_values}
    unexpected = sorted(
        _discrete_key_text(key) for key in observed if key not in allowed_keys
    )
    issues: list[_SemanticConsistencyIssue] = []
    if unexpected:
        issues.append(
            _SemanticConsistencyIssue(
                column=column,
                role=role,
                message=outside_message,
                allowed_values=allowed_values,
                unexpected_values=tuple(unexpected),
            )
        )
    missing_allowed = [key for key in allowed_keys if int(observed.get(key, 0)) == 0]
    if missing_allowed:
        issues.append(
            _SemanticConsistencyIssue(
                column=column,
                role=role,
                message=missing_class_message,
                allowed_values=allowed_values,
            )
        )
    return issues


def _semantic_consistency_repair_instruction(
    *,
    causal_spec: CausalSpec,
    issues: Sequence[_SemanticConsistencyIssue],
) -> str:
    issue_lines = []
    for issue in issues:
        line = (
            f"- {issue.column} ({issue.role}): {issue.message} "
            f"Allowed final values: {list(issue.allowed_values)}."
        )
        if issue.unexpected_values:
            line += f" Unexpected observed values: {list(issue.unexpected_values)}."
        issue_lines.append(line)

    return "\n".join(
        [
            "High-priority final semantic consistency repair for the compiled causal spec.",
            "Return the full protocol-scope working dataset after repair.",
            "Map or filter only values that are grounded by the current column values and the compiled causal spec.",
            "Do not change, drop, duplicate, null, or regenerate the effective identifier column.",
            "Do not create new causal columns or change locked causal roles.",
            f"Compiled treatment spec: {causal_spec.treatment_spec.model_dump(mode='json')}.",
            f"Compiled outcome spec: {causal_spec.outcome_spec.model_dump(mode='json')}.",
            f"Compiled negative-control outcome spec: "
            f"{None if causal_spec.negative_control_outcome is None else causal_spec.negative_control_outcome.model_dump(mode='json')}.",
            "Issues to repair before validation:",
            *issue_lines,
        ]
    )


def _format_semantic_consistency_issues(
    issues: Sequence[_SemanticConsistencyIssue],
) -> str:
    return "; ".join(
        f"{issue.column} ({issue.role}): {issue.message}"
        for issue in issues
    )


def _normalize_discrete_value(value: Any) -> tuple[str, Any]:
    try:
        if pd.isna(value):
            return ("na", None)
    except Exception:
        pass

    if isinstance(value, bool):
        return ("bool", bool(value))
    if isinstance(value, numbers.Real):
        return ("num", float(value))
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered == "true":
            return ("bool", True)
        if lowered == "false":
            return ("bool", False)
        try:
            return ("num", float(lowered))
        except ValueError:
            return ("str", lowered)
    return ("str", str(value).strip().lower())


def _normalized_discrete_counts(series: pd.Series) -> dict[tuple[str, Any], int]:
    counts: dict[tuple[str, Any], int] = {}
    for value in series.tolist():
        key = _normalize_discrete_value(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _discrete_key_text(key: tuple[str, Any]) -> str:
    return f"{key[0]}:{key[1]!r}"


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
        *(
            [str(draft_causal_spec.negative_control_outcome).strip()]
            if draft_causal_spec.negative_control_outcome is not None
            else []
        ),
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


def _clean_with_adaptive_manipulation(
    *,
    llm: LLMService,
    prepared_summary: DatasetSummaryModel,
    prepared_df: pd.DataFrame,
    before_df: pd.DataFrame,
    draft_causal_spec: CausalSpecDraft,
    effective_id_col: str,
    required_columns: Sequence[str],
    protocol_discussion: str | None,
    cleaning_instructions: str,
    review_recompile_request: str | None,
    dataset_profiling_tool: DatasetProfilingTool,
    data_manipulation_tool: DataManipulationTool,
) -> tuple[
    pd.DataFrame,
    MissingnessDecisionList,
    tuple[str, ...],
    dict[str, ColumnRole],
]:
    role_by_column = _expected_role_by_column(draft_causal_spec)
    current_role_by_column = dict(role_by_column)
    current_required_columns = list(required_columns)
    current_df = prepared_df.copy()
    current_summary = prepared_summary
    executed_steps: list[_ExecutedCleaningInstruction] = []
    validation_feedback: _CleaningFailure | None = None
    indicator_role_by_column: dict[str, ColumnRole] = {}

    for stage in _cleaning_stage_sequence():
        if stage == "missingness" and not any(
            _missing_counts_by_column(current_df, role_by_column).values()
        ):
            continue

        missingness_indicator_specs: tuple[_MissingnessIndicatorSpec, ...] = ()
        if stage == "missingness":
            missingness_indicator_specs = _missingness_indicator_specs(
                dataframe=current_df,
                role_by_column=role_by_column,
                effective_id_col=effective_id_col,
                reserved_columns=(*current_df.columns, *current_required_columns),
            )

        step = _plan_next_cleaning_instruction(
            llm=llm,
            stage=stage,
            current_summary=current_summary,
            current_df=current_df,
            draft_causal_spec=draft_causal_spec,
            effective_id_col=effective_id_col,
            required_columns=current_required_columns,
            role_by_column=current_role_by_column,
            protocol_discussion=protocol_discussion,
            cleaning_instructions=cleaning_instructions,
            review_recompile_request=review_recompile_request,
            executed_steps=executed_steps,
            validation_feedback=validation_feedback,
        )
        if step.action == "done":
            if stage == "missingness" and any(
                _missing_counts_by_column(current_df, role_by_column).values()
            ):
                validation_feedback = _CleaningFailure(
                    error="missingness step returned done while protocol-scope missingness remains",
                    stage=stage,
                    current_summary=current_summary,
                    current_missing_counts=_missing_counts_by_column(
                        current_df, role_by_column
                    ),
                )
            continue

        try:
            next_df = data_manipulation_tool.manipulate(
                dataframe=current_df,
                table_name=_PROTOCOL_SCOPE_TABLE,
                data_summary=_json_dumps(
                    _compact_dataset_summary(
                        summary=current_summary,
                        role_by_column=role_by_column,
                        required_columns=required_columns,
                    )
                ),
                instructions=step.instruction or "",
            )
            _ensure_columns_present(
                dataframe=next_df,
                columns=current_required_columns,
                context=f"{stage} data manipulation output dataframe",
            )
            _ensure_identifier_integrity(
                before_df=current_df,
                after_df=next_df,
                effective_id_col=effective_id_col,
                context=f"{stage} data manipulation output dataframe",
            )
            if stage == "missingness":
                applied_indicators = _apply_missingness_indicators(
                    dataframe=next_df,
                    indicator_specs=missingness_indicator_specs,
                    effective_id_col=effective_id_col,
                )
                for indicator_column, role in applied_indicators.items():
                    if indicator_column not in current_required_columns:
                        current_required_columns.append(indicator_column)
                    current_role_by_column[indicator_column] = role
                    indicator_role_by_column[indicator_column] = role
            next_summary = _profile_dataset(
                dataset_profiling_tool=dataset_profiling_tool,
                dataframe=next_df,
            )
        except Exception as exc:
            validation_feedback = _CleaningFailure(
                error=str(exc).strip() or exc.__class__.__name__,
                stage=stage,
                failed_instruction=step.instruction,
                current_summary=current_summary,
                current_missing_counts=_missing_counts_by_column(
                    current_df, role_by_column
                ),
            )
            continue

        executed_steps.append(
            _ExecutedCleaningInstruction(
                stage=stage,
                instruction=step.instruction or "",
                reason=step.reason,
                rows_before=len(current_df),
                rows_after=len(next_df),
                missing_counts_before=_missing_counts_by_column(current_df, role_by_column),
                missing_counts_after=_missing_counts_by_column(next_df, role_by_column),
            )
        )
        current_df = next_df
        current_summary = next_summary
        validation_feedback = None

    if validation_feedback is not None:
        raise ValueError(f"adaptive data manipulation cleaning failed: {validation_feedback.error}")

    _ensure_columns_present(
        dataframe=current_df,
        columns=current_required_columns,
        context="cleaned dataframe",
    )
    cleaned_df = _project_required_columns(
        dataframe=current_df,
        columns=current_required_columns,
    )
    try:
        missingness_decisions = _finalize_missingness_decisions(
            draft_causal_spec=draft_causal_spec,
            before_df=before_df,
            cleaned_df=cleaned_df,
        )
    except Exception as exc:
        raise ValueError(
            "adaptive data manipulation cleaning failed: "
            f"{str(exc).strip() or exc.__class__.__name__}"
        ) from exc
    return (
        cleaned_df,
        missingness_decisions,
        _cleaning_notes_from_executed_steps(executed_steps),
        indicator_role_by_column,
    )


def _cleaning_notes_from_executed_steps(
    executed_steps: Sequence[_ExecutedCleaningInstruction],
) -> tuple[str, ...]:
    notes: list[str] = []
    for step in executed_steps:
        reason = step.reason.strip()
        if reason:
            notes.append(f"Cleaning decision ({step.stage}): {reason}")
    return tuple(notes)


def _cleaning_stage_sequence() -> tuple[CleaningStage, ...]:
    stages: tuple[CleaningStage, ...] = (
        "transformation",
        "missingness",
        "cleanup_1",
        "cleanup_2",
    )
    if len(stages) != _MAX_CLEANING_MANIPULATION_STAGES:
        raise ValueError("cleaning stage sequence must match the configured stage limit")
    return stages


def _plan_next_cleaning_instruction(
    *,
    llm: LLMService,
    stage: CleaningStage,
    current_summary: DatasetSummaryModel,
    current_df: pd.DataFrame,
    draft_causal_spec: CausalSpecDraft,
    effective_id_col: str,
    required_columns: Sequence[str],
    role_by_column: dict[str, ColumnRole],
    protocol_discussion: str | None,
    cleaning_instructions: str,
    review_recompile_request: str | None,
    executed_steps: Sequence[_ExecutedCleaningInstruction],
    validation_feedback: _CleaningFailure | None,
) -> _CleaningInstructionStep:
    payload: dict[str, Any] = {
        "stage": stage,
        "confirmed_protocol_discussion": _normalize_text(protocol_discussion),
        "confirmed_protocol_cleaning_instructions": _normalize_text(cleaning_instructions),
        "draft_causal_spec": draft_causal_spec.model_dump(mode="json"),
        "effective_id_col": effective_id_col,
        "required_final_columns": list(required_columns),
        "expected_role_by_column": role_by_column,
        "current_table_name": _PROTOCOL_SCOPE_TABLE,
        "compact_current_dataset_summary": _compact_dataset_summary(
            summary=current_summary,
            role_by_column=role_by_column,
            required_columns=required_columns,
        ),
        "required_column_missing_counts": _missing_counts_by_column(
            current_df, role_by_column
        ),
        "executed_cleaning_instructions": [
            _executed_cleaning_instruction_payload(step) for step in executed_steps
        ],
    }
    normalized_review_recompile_request = _normalize_text(review_recompile_request)
    if normalized_review_recompile_request:
        payload["high_priority_review_recompile_request"] = (
            normalized_review_recompile_request
        )
    if validation_feedback is not None:
        payload["validation_feedback"] = _cleaning_failure_payload(
            feedback=validation_feedback,
            role_by_column=role_by_column,
            required_columns=required_columns,
        )

    return llm.generate_json(
        schema=_CleaningInstructionStep,
        system_prompt=_cleaning_instruction_prompt_for_stage(stage),
        user_prompt=_json_dumps(payload),
        config=LLMConfig(model="basic", temperature=0.4),
        history=None,
        max_attempts=2,
    )


def _cleaning_instruction_prompt_for_stage(stage: CleaningStage) -> str:
    if stage == "transformation":
        return data_compilation_transformation_instruction_prompt()
    if stage == "missingness":
        return data_compilation_missingness_instruction_prompt()
    return data_compilation_adaptive_cleaning_instruction_prompt()


def _executed_cleaning_instruction_payload(
    step: _ExecutedCleaningInstruction,
) -> dict[str, Any]:
    return {
        "stage": step.stage,
        "instruction": step.instruction,
        "reason": step.reason,
        "rows_before": step.rows_before,
        "rows_after": step.rows_after,
        "missing_counts_before": step.missing_counts_before,
        "missing_counts_after": step.missing_counts_after,
    }


def _cleaning_failure_payload(
    *,
    feedback: _CleaningFailure,
    role_by_column: dict[str, ColumnRole],
    required_columns: Sequence[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": feedback.error,
    }
    if feedback.stage is not None:
        payload["stage"] = feedback.stage
    if feedback.failed_instruction is not None:
        payload["failed_instruction"] = feedback.failed_instruction
    if feedback.current_summary is not None:
        payload["current_compact_dataset_summary"] = _compact_dataset_summary(
            summary=feedback.current_summary,
            role_by_column=role_by_column,
            required_columns=required_columns,
        )
    if feedback.current_missing_counts is not None:
        payload["current_required_column_missing_counts"] = feedback.current_missing_counts
    return payload


def _compact_dataset_summary(
    *,
    summary: DatasetSummaryModel,
    role_by_column: dict[str, ColumnRole],
    required_columns: Sequence[str],
) -> dict[str, Any]:
    profiles_by_name = {
        str(profile.name).strip(): profile
        for profile in summary.profiles
        if str(profile.name).strip()
    }
    columns: list[dict[str, Any]] = []
    for column in required_columns:
        profile = profiles_by_name.get(column)
        if profile is None:
            columns.append(
                {
                    "column": column,
                    "role": role_by_column.get(column, "identifier"),
                    "missing": True,
                }
            )
            continue
        columns.append(
            _compact_column_summary_payload(
                profile=profile,
                role=str(role_by_column.get(column, "identifier")),
            )
        )
    return {
        "n_rows": summary.n_rows,
        "columns": columns,
    }


def _compact_column_summary_payload(
    *,
    profile: (
        NumericColumnProfileModel
        | DatetimeColumnProfileModel
        | BooleanColumnProfileModel
        | CategoricalColumnProfileModel
        | OtherColumnProfileModel
    ),
    role: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "column": str(profile.name).strip(),
        "role": role,
        "kind": str(profile.inferred_kind),
        "dtype": profile.dtype,
        "missing_count": profile.n_missing,
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
        payload["known_values"] = list(profile.summary.counts.keys())[:_COMPACT_VALUE_LIMIT]
        return payload

    if isinstance(profile, CategoricalColumnProfileModel):
        payload["known_values"] = [
            item.value for item in profile.summary.top_categories[:_COMPACT_VALUE_LIMIT]
        ]
        return payload

    if isinstance(profile, OtherColumnProfileModel):
        payload["sample_values"] = list(
            profile.summary.distinct_values_sample[:_COMPACT_VALUE_LIMIT]
        )
        return payload

    return payload


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _ensure_identifier_integrity(
    *,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    effective_id_col: str,
    context: str,
) -> None:
    before_id_columns = list(before_df.columns).count(effective_id_col)
    after_id_columns = list(after_df.columns).count(effective_id_col)
    if before_id_columns != 1 or after_id_columns != 1:
        raise ValueError(
            f"{context} must contain exactly one effective identifier column: "
            f"{effective_id_col}"
        )

    before_ids = before_df[effective_id_col]
    after_ids = after_df[effective_id_col]
    if bool(after_ids.isna().any()):
        raise ValueError(
            f"{context} produced null values in effective identifier column: " f"{effective_id_col}"
        )
    duplicated = after_ids[after_ids.duplicated()].head(25).tolist()
    if duplicated:
        raise ValueError(
            f"{context} produced duplicate effective identifier values in "
            f"{effective_id_col}: {duplicated}"
        )
    regenerated = after_ids[~after_ids.isin(before_ids)].head(25).tolist()
    if regenerated:
        raise ValueError(
            f"{context} produced regenerated effective identifier values in "
            f"{effective_id_col}: {regenerated}"
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
    if draft_causal_spec.negative_control_outcome is not None:
        role_by_column[str(draft_causal_spec.negative_control_outcome).strip()] = (
            "negative_control_outcome"
        )
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


def _missingness_indicator_specs(
    *,
    dataframe: pd.DataFrame,
    role_by_column: dict[str, ColumnRole],
    effective_id_col: str,
    reserved_columns: Sequence[str],
) -> tuple[_MissingnessIndicatorSpec, ...]:
    reserved = {str(column) for column in reserved_columns}
    specs: list[_MissingnessIndicatorSpec] = []

    for source_column, role in role_by_column.items():
        if role not in {"covariate", "effect_modifier"}:
            continue
        if source_column not in dataframe.columns:
            continue
        missing_mask = dataframe[source_column].isna()
        missing_count = int(missing_mask.sum())
        missing_rate = float(missing_count / len(dataframe)) if len(dataframe) else 0.0
        if (
            missing_count < _MISSINGNESS_INDICATOR_MIN_COUNT
            or missing_rate < _MISSINGNESS_INDICATOR_MIN_RATE
        ):
            continue
        indicator_column = _missingness_indicator_name(
            source_column=source_column,
            reserved_columns=reserved,
        )
        reserved.add(indicator_column)
        missing_by_id = pd.Series(
            missing_mask.to_numpy(dtype=bool),
            index=dataframe[effective_id_col].tolist(),
        )
        specs.append(
            _MissingnessIndicatorSpec(
                source_column=source_column,
                indicator_column=indicator_column,
                role=role,
                missing_by_id=missing_by_id,
            )
        )

    return tuple(specs)


def _missingness_indicator_name(
    *,
    source_column: str,
    reserved_columns: set[str],
) -> str:
    base_name = f"{source_column}__missing"
    if base_name not in reserved_columns:
        return base_name

    suffix = 2
    while f"{base_name}_{suffix}" in reserved_columns:
        suffix += 1
    return f"{base_name}_{suffix}"


def _apply_missingness_indicators(
    *,
    dataframe: pd.DataFrame,
    indicator_specs: Sequence[_MissingnessIndicatorSpec],
    effective_id_col: str,
) -> dict[str, ColumnRole]:
    applied: dict[str, ColumnRole] = {}
    for spec in indicator_specs:
        if spec.source_column not in dataframe.columns:
            continue
        if int(dataframe[spec.source_column].isna().sum()) > 0:
            continue

        retained_missing = (
            dataframe[effective_id_col]
            .map(spec.missing_by_id)
            .fillna(False)
            .astype(bool)
        )
        if not bool(retained_missing.any()):
            continue

        dataframe[spec.indicator_column] = retained_missing.astype("int64")
        applied[spec.indicator_column] = spec.role
    return applied


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
    effective_id_col: str | None = None,
    indicator_role_by_column: dict[str, ColumnRole] | None = None,
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
            indicator_role_by_column=indicator_role_by_column or {},
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
    if draft_causal_spec.negative_control_outcome is not None:
        context_payload["negative_control_outcome_column_profile"] = _summary_profile_payload(
            summary=cleaned_summary,
            column=str(draft_causal_spec.negative_control_outcome).strip(),
        )
    normalized_protocol_discussion = _normalize_text(protocol_discussion)
    if normalized_protocol_discussion:
        context_payload["protocol_discussion"] = normalized_protocol_discussion
    if compile_feedback:
        context_payload["compile_feedback"] = compile_feedback

    return llm.generate_json(
        schema=_CausalSemanticsModel,
        system_prompt=data_compilation_causal_semantics_prompt(),
        user_prompt=json.dumps(context_payload, ensure_ascii=False),
        config=LLMConfig(model="pro", temperature=0.4),
        history=None,
        max_attempts=3,
    )


def _assemble_causal_spec(
    *,
    draft_causal_spec: CausalSpecDraft,
    semantics: _CausalSemanticsModel,
    effective_id_col: str,
    indicator_role_by_column: dict[str, ColumnRole],
) -> CausalSpec:
    treatment_column = str(draft_causal_spec.treatment_column).strip()
    outcome_column = str(draft_causal_spec.outcome_column).strip()

    treatment_spec = BinaryTreatmentSpecModel(
        kind="binary",
        column=treatment_column,
        treated=semantics.treatment.treated,
        control=semantics.treatment.control,
    )

    outcome_spec = _outcome_spec_from_semantics(
        column=outcome_column,
        semantics=semantics.outcome,
    )
    negative_control_outcome = None
    if draft_causal_spec.negative_control_outcome is not None:
        if semantics.negative_control_outcome is None:
            raise ValueError(
                "negative_control_outcome semantics are required when the causal draft "
                "includes a negative-control outcome column"
            )
        negative_control_outcome = _outcome_spec_from_semantics(
            column=str(draft_causal_spec.negative_control_outcome).strip(),
            semantics=semantics.negative_control_outcome,
        )

    return CausalSpec(
        treatment_spec=treatment_spec,
        outcome_spec=outcome_spec,
        negative_control_outcome=negative_control_outcome,
        covariates=[
            *[str(column).strip() for column in draft_causal_spec.covariates],
            *[
                column
                for column, role in indicator_role_by_column.items()
                if role == "covariate"
            ],
        ],
        effect_modifiers=[
            *[str(column).strip() for column in draft_causal_spec.effect_modifiers],
            *[
                column
                for column, role in indicator_role_by_column.items()
                if role == "effect_modifier"
            ],
        ],
        experiment_type=semantics.experiment_type,
        id_col=effective_id_col,
    )


def _outcome_spec_from_semantics(
    *,
    column: str,
    semantics: _OutcomeSemanticsModel,
) -> BinaryOutcomeSpecModel | ContinuousOutcomeSpecModel:
    if isinstance(semantics, _BinaryOutcomeSemanticsModel):
        return BinaryOutcomeSpecModel(
            kind="binary",
            column=column,
            event=semantics.event,
            non_event=semantics.non_event,
        )
    return ContinuousOutcomeSpecModel(
        kind="continuous",
        column=column,
        unit=semantics.unit,
        clip_min=semantics.clip_min,
        clip_max=semantics.clip_max,
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
