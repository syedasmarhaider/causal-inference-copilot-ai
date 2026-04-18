from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from python.domain.service.llm_service import LLMConfig, LLMService
from python.implementation.workflows.nodes.data_compilation.data_compilation_prompts import (
    data_compilation_causal_spec_prompt,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import CausalSpecDraft
from python.implementation.workflows.tools.causal.specs.causal_specs_tool import (
    CausalSpecsTool,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import DatasetProfilingTool


@dataclass(frozen=True)
class CleaningResult:
    cleaned_data_summary: DatasetSummaryModel
    pd_cleaned: pd.DataFrame
    causal: CausalSpec


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
    causal_spec = _compile_causal_spec_with_retry(
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


def _compile_causal_spec_with_retry(
    *,
    llm: LLMService,
    cleaned_summary: DatasetSummaryModel,
    draft_causal_spec: CausalSpecDraft,
    protocol_discussion: str | None,
) -> CausalSpec:
    causal_specs_tool = CausalSpecsTool()
    compile_feedback: str | None = None

    for _ in range(2):
        causal_spec = _compile_causal_spec_once(
            llm=llm,
            causal_specs_tool=causal_specs_tool,
            cleaned_summary=cleaned_summary,
            draft_causal_spec=draft_causal_spec,
            protocol_discussion=protocol_discussion,
            compile_feedback=compile_feedback,
        )
        mismatch_message = _draft_vs_compiled_spec_mismatch(
            draft_causal_spec=draft_causal_spec,
            causal_spec=causal_spec,
        )
        if mismatch_message is None:
            return causal_specs_tool.post_validate_backdoor_spec(
                causal_spec=causal_spec,
                data_summary=cleaned_summary,
            )

        compile_feedback = mismatch_message

    raise ValueError(
        "compiled causal spec does not match draft causal spec after retry: "
        f"{compile_feedback}"
    )


def _compile_causal_spec_once(
    *,
    llm: LLMService,
    causal_specs_tool: CausalSpecsTool,
    cleaned_summary: DatasetSummaryModel,
    draft_causal_spec: CausalSpecDraft,
    protocol_discussion: str | None,
    compile_feedback: str | None,
) -> CausalSpec:
    context_payload: dict[str, object] = {
        "dataset_summary": cleaned_summary.model_dump(mode="json"),
        "draft_causal_spec": draft_causal_spec.model_dump(mode="json"),
    }
    normalized_protocol_discussion = _normalize_text(protocol_discussion)
    if normalized_protocol_discussion:
        context_payload["protocol_discussion"] = normalized_protocol_discussion
    if compile_feedback:
        context_payload["compile_feedback"] = compile_feedback

    causal_schema = causal_specs_tool.build_backdoor_schema(data_summary=cleaned_summary)
    return llm.generate_json(
        schema=causal_schema,
        system_prompt=data_compilation_causal_spec_prompt(),
        user_prompt=json.dumps(context_payload, ensure_ascii=False),
        config=LLMConfig(model="pro", temperature=0.1),
        history=None,
        max_attempts=3,
    )


def _draft_vs_compiled_spec_mismatch(
    *,
    draft_causal_spec: CausalSpecDraft,
    causal_spec: CausalSpec,
) -> str | None:
    mismatches: list[str] = []

    expected_treatment = str(draft_causal_spec.treatment_column).strip()
    observed_treatment = str(causal_spec.treatment_spec.column).strip()
    if observed_treatment != expected_treatment:
        mismatches.append(
            "treatment column mismatch: "
            f"expected '{expected_treatment}' got '{observed_treatment}'"
        )

    expected_outcome = str(draft_causal_spec.outcome_column).strip()
    observed_outcome = str(causal_spec.outcome_spec.column).strip()
    if observed_outcome != expected_outcome:
        mismatches.append(
            "outcome column mismatch: "
            f"expected '{expected_outcome}' got '{observed_outcome}'"
        )

    expected_covariates = [str(column).strip() for column in draft_causal_spec.covariates]
    observed_covariates = [str(column).strip() for column in causal_spec.covariates]
    if observed_covariates != expected_covariates:
        mismatches.append(
            "covariates mismatch: "
            f"expected {expected_covariates} got {observed_covariates}"
        )

    expected_effect_modifiers = [
        str(column).strip() for column in draft_causal_spec.effect_modifiers
    ]
    observed_effect_modifiers = [
        str(column).strip() for column in causal_spec.effect_modifiers
    ]
    if observed_effect_modifiers != expected_effect_modifiers:
        mismatches.append(
            "effect modifiers mismatch: "
            f"expected {expected_effect_modifiers} got {observed_effect_modifiers}"
        )

    if not mismatches:
        return None
    return "; ".join(mismatches)
