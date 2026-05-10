from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from python.domain.service.llm_service import LLMConfig, LLMService
from python.implementation.workflows.nodes.data_compilation.data_compilation_prompts import (
    data_compilation_data_type_plan_prompt,
    data_compilation_filter_plan_prompt,
    data_compilation_imputation_plan_prompt,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    CausalSpecDraft,
    ID_COL_AUTO_FILL,
)
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)



@dataclass(frozen=True)
class CleaningResult:
    causal: CausalSpec
    pd_cleaned: pd.DataFrame
    cleaned_data_summary: DatasetSummaryModel
    summary_str: str
    cleaning_notes: tuple[str, ...] = ()
    effective_draft: CausalSpecDraft | None = None


def clean(
    data: pd.DataFrame,
    data_summary: DatasetSummaryModel,
    draft: CausalSpecDraft,
    data_maupulation_tools: DataManipulationTool,
    data_profiling_tools: DatasetProfilingTool,
    llm: LLMService,
    protocol_discussion: str = "",
    revised_instructions: str | None = None,
    table_name: str = "protocol_scope_df",
) -> CleaningResult:
    validation_retry_instructions = revised_instructions
    carried_notes: list[str] = []
    last_validation_error: Exception | None = None

    for pipeline_attempt in range(2):
        cleaning_notes = list(carried_notes)
        if pipeline_attempt == 1 and last_validation_error is not None:
            cleaning_notes.append(
                "Reran the full cleaning pipeline after final validation failed: "
                + str(last_validation_error)
            )

        working_data, working_draft = _ensure_effective_id(data=data, draft=draft)
        if working_draft.id_col != draft.id_col:
            cleaning_notes.append(
                f"Identifier column changed from '{draft.id_col}' to "
                f"'{working_draft.id_col}' because the requested identifier was missing, "
                "non-unique, or contained missing values."
            )

        filter_instructions = validation_retry_instructions
        for attempt in range(2):
            try:
                filtered_data = _filter_data(
                    table_name=table_name,
                    data=working_data,
                    draft=working_draft,
                    summary=data_summary,
                    data_maupulation_tools=data_maupulation_tools,
                    llm=llm,
                    revised_instructions=filter_instructions,
                )
                break
            except Exception as exc:
                if attempt == 1:
                    raise ValueError(
                        "target-population filtering LLM stage failed after retry: "
                        + str(exc)
                    ) from exc
                filter_instructions = "\n\n".join(
                    part
                    for part in [
                        validation_retry_instructions,
                        (
                            "Retry feedback for target-population filtering: the "
                            f"previous attempt failed with: {exc}. Revise the filter "
                            "instruction while preserving all draft-required columns."
                        ),
                    ]
                    if part
                )
                cleaning_notes.append(
                    "Retried target-population filtering LLM stage after error: " + str(exc)
                )

        filtered_summary = data_profiling_tools.extract_dataset_summary(
            filtered_data,
            max_categories=200,
            sample_distinct=200,
            compute_quantiles=False,
            strict=True,
        )
        scoped_data = _drop_irrelevant_columns(filtered_data, working_draft)
        scoped_summary = data_profiling_tools.extract_dataset_summary(
            scoped_data,
            max_categories=200,
            sample_distinct=200,
            compute_quantiles=False,
            strict=True,
        )

        datatype_instructions = validation_retry_instructions
        for attempt in range(2):
            try:
                typed_data = _data_type_change(
                    table_name=table_name,
                    data=scoped_data,
                    draft=working_draft,
                    summary=scoped_summary,
                    data_maupulation_tools=data_maupulation_tools,
                    llm=llm,
                    revised_instructions=datatype_instructions,
                )
                break
            except Exception as exc:
                if attempt == 1:
                    raise ValueError(
                        "datatype normalization LLM stage failed after retry: "
                        + str(exc)
                    ) from exc
                datatype_instructions = "\n\n".join(
                    part
                    for part in [
                        validation_retry_instructions,
                        (
                            "Retry feedback for datatype normalization: the previous "
                            f"attempt failed with: {exc}. Revise the datatype instruction "
                            "while preserving all draft-required columns."
                        ),
                    ]
                    if part
                )
                cleaning_notes.append(
                    "Retried datatype normalization LLM stage after error: " + str(exc)
                )

        typed_summary = data_profiling_tools.extract_dataset_summary(
            typed_data,
            max_categories=200,
            sample_distinct=200,
            compute_quantiles=False,
            strict=True,
        )
        col_missingess = _calcaulate_miisingess_per_col_and_get_back_msingnes(
            data=typed_data,
            draft=working_draft,
            summary=typed_summary,
            data_profiling_tools=data_profiling_tools,
        )

        imputation_instructions = validation_retry_instructions
        for attempt in range(2):
            try:
                imputed_data = _impute_missing_values(
                    table_name=table_name,
                    data=typed_data,
                    draft=working_draft,
                    summary=typed_summary,
                    data_maupulation_tools=data_maupulation_tools,
                    llm=llm,
                    revised_instructions=imputation_instructions,
                )
                break
            except Exception as exc:
                if attempt == 1:
                    raise ValueError(
                        "missing-value imputation LLM stage failed after retry: "
                        + str(exc)
                    ) from exc
                imputation_instructions = "\n\n".join(
                    part
                    for part in [
                        validation_retry_instructions,
                        (
                            "Retry feedback for missing-value imputation: the previous "
                            f"attempt failed with: {exc}. Revise the imputation instruction "
                            "while preserving all draft-required columns."
                        ),
                    ]
                    if part
                )
                cleaning_notes.append(
                    "Retried missing-value imputation LLM stage after error: " + str(exc)
                )

        filled_data, filled_draft = _fill_miginess(
            draft=working_draft,
            data=imputed_data,
            col_missingess=col_missingess,
        )
        final_data = _drop_irrelevant_columns(filled_data, filled_draft)
        final_summary = data_profiling_tools.extract_dataset_summary(
            final_data,
            max_categories=200,
            sample_distinct=200,
            compute_quantiles=False,
            strict=True,
        )
        causal_spec = compile_causal_spec_from_draft(
            protocol_discussion=protocol_discussion,
            dataset_summary=final_summary,
            previous_draft=filled_draft,
            retry_feedback=validation_retry_instructions,
        )

        try:
            _validate(
                causal_spec=causal_spec,
                causal_spec_draft=filled_draft,
                data=final_data,
                dataset_summary=final_summary,
            )
        except Exception as exc:
            if pipeline_attempt == 1:
                raise ValueError(
                    "final validation failed after full cleaning retry: " + str(exc)
                ) from exc
            last_validation_error = exc
            validation_retry_instructions = "\n\n".join(
                part
                for part in [
                    revised_instructions,
                    (
                        "Full cleaning retry feedback from final validation: the cleaned "
                        f"data/spec failed validation with: {exc}. Rerun all LLM-planned "
                        "cleaning stages from the original data and correct the issue while "
                        "preserving draft-required columns."
                    ),
                ]
                if part
            )
            carried_notes = cleaning_notes + [
                "Full cleaning pipeline retry triggered by final validation error: "
                + str(exc)
            ]
            continue

        summary_str = cleaning_summary(
            before_data=data,
            after_data=final_data,
            before_summary=data_summary,
            after_summary=final_summary,
            before_draft=draft,
            after_draft=filled_draft,
        )
        if cleaning_notes:
            summary_str = "\n".join(
                [
                    summary_str,
                    "Cleaning notes:",
                    *(f"- {note}" for note in cleaning_notes),
                ]
            )
        _ = filtered_summary
        
        return CleaningResult(
            causal=causal_spec,
            pd_cleaned=final_data,
            cleaned_data_summary=final_summary,
            summary_str=summary_str,
            cleaning_notes=tuple(cleaning_notes),
            effective_draft=filled_draft,
        )

    raise RuntimeError("cleaning pipeline exited without returning a result")


def cleaning(
    *,
    protocol_discussion: str | None,
    cleaning_instructions: str | None,
    review_recompile_request: str | None,
    draft_causal_spec: CausalSpecDraft,
    data_summary: DatasetSummaryModel,
    to_clean_df: pd.DataFrame,
    datasetProfilingTool: DatasetProfilingTool,
    dataManipulationTool: DataManipulationTool,
    llm: LLMService,
) -> CleaningResult:
    revised_parts = [
        str(value).strip()
        for value in [cleaning_instructions, review_recompile_request]
        if value is not None and str(value).strip()
    ]
    return clean(
        data=to_clean_df,
        data_summary=data_summary,
        draft=draft_causal_spec,
        data_maupulation_tools=dataManipulationTool,
        data_profiling_tools=datasetProfilingTool,
        llm=llm,
        protocol_discussion=protocol_discussion or "",
        revised_instructions="\n".join(revised_parts) or None,
    )


def _ensure_effective_id(
    data: pd.DataFrame,
    draft: CausalSpecDraft,
) -> tuple[pd.DataFrame, CausalSpecDraft]:
    working_data = data.copy()
    requested_id_col = str(draft.id_col or "").strip()
    if (
        requested_id_col
        and requested_id_col in working_data.columns
        and int(working_data[requested_id_col].isna().sum()) == 0
        and not bool(working_data[requested_id_col].duplicated().any())
    ):
        return working_data, draft

    working_data[ID_COL_AUTO_FILL] = range(1, len(working_data) + 1)
    if requested_id_col == ID_COL_AUTO_FILL:
        return working_data, draft

    return working_data, draft.model_copy(update={"id_col": ID_COL_AUTO_FILL})


def _filter_data(
    table_name: str,
    data: pd.DataFrame,
    draft: CausalSpecDraft,
    summary: DatasetSummaryModel,
    data_maupulation_tools: DataManipulationTool,
    llm: LLMService,
    revised_instructions: str | None = None,
) -> pd.DataFrame:
    """
    Apply physical dataset filters based on the draft's target population text, if useful and agreed by the user.
    steps.


    1: plan as a text simple text like call llm and generate fitler plan insturction
    2: define prompt in promtp file
    3: generate insturction
    4: pass instruction + data summary in data manupulation tool
    5: get result
    6: validate if cols does not change w.r.t draft
    7: return causal draft
    """
    target_population = str(draft.target_population or "").strip()
    if not target_population:
        return data.copy()

    required_columns: list[str] = []
    for column in [
        draft.treatment_column,
        draft.outcome_column,
        draft.negative_control_outcome,
        *draft.covariates,
        *draft.effect_modifiers,
    ]:
        if column is None:
            continue
        normalized_column = str(column).strip()
        if normalized_column and normalized_column not in required_columns:
            required_columns.append(normalized_column)

    id_col = str(draft.id_col or "").strip()
    if id_col and id_col in data.columns and id_col not in required_columns:
        required_columns.insert(0, id_col)

    planner_payload = {
        "target_population": target_population,
        "draft": draft.model_dump(mode="json"),
        "dataset_summary": summary.model_dump(mode="json"),
        "current_dataframe_columns": [str(column) for column in data.columns],
        "revised_instructions": revised_instructions,
    }
    
    filter_plan_response = llm.generate(
        system_prompt=data_compilation_filter_plan_prompt(),
        user_prompt=json.dumps(planner_payload, ensure_ascii=False),
        config=LLMConfig(model="basic", temperature=0.4),
        history=None,
    )
    
    filter_plan_text = str(filter_plan_response.content or "").strip()
    if not filter_plan_text:
        raise ValueError("target population filter planning returned an empty instruction")

    filtered_data = data_maupulation_tools.manipulate(
        dataframe=data,
        table_name=table_name,
        data_summary=summary.model_dump_json(),
        instructions=filter_plan_text,
    )
    missing_columns = [
        column for column in required_columns if column not in filtered_data.columns
    ]
    
    if missing_columns:
        raise ValueError(
            "target population filter output dataframe is missing required column(s): "
            + ", ".join(missing_columns)
    )

    return filtered_data


def _drop_irrelevant_columns(
    data: pd.DataFrame,
    draft: CausalSpecDraft,
) -> pd.DataFrame:
    """

    drop all irrelavent cols..
    keep tratment outcome covariates effect modifers and negative control outcome if specified. and id cols if specified. drop rest of the cols.
    return data frame.
    """
    columns_to_keep: list[str] = []
    id_col = str(draft.id_col or "").strip()
    if id_col and id_col in data.columns:
        columns_to_keep.append(id_col)

    for column in [
        draft.treatment_column,
        draft.outcome_column,
        draft.negative_control_outcome,
        *draft.covariates,
        *draft.effect_modifiers,
    ]:
        if column is None:
            continue
        normalized_column = str(column).strip()
        if normalized_column and normalized_column not in columns_to_keep:
            columns_to_keep.append(normalized_column)

    missing_columns = [
        column
        for column in columns_to_keep
        if column != id_col and column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "input dataframe is missing required column(s): "
            + ", ".join(missing_columns)
        )

    return data.loc[:, columns_to_keep].copy()


def _data_type_change(
    table_name: str,
    data: pd.DataFrame,
    draft: CausalSpecDraft,
    summary: DatasetSummaryModel,
    data_maupulation_tools: DataManipulationTool,
    llm: LLMService,
    revised_instructions: str | None = None,
) -> pd.DataFrame:
    """
    generate string plan by calling LLM to convert datatypes for best of the knowlwdge and best practices according to machine learning
    and causal inference best practices.
    give instruction to data manipulation tool and get the data with changed datatypes.
    validate if cols does not change w.r.t draft and return data frame.
    """
    required_columns: list[str] = []
    role_by_column: dict[str, str] = {}
    for column, role in [
        (draft.treatment_column, "treatment"),
        (draft.outcome_column, "outcome"),
        (draft.negative_control_outcome, "negative_control_outcome"),
        *((covariate, "covariate") for covariate in draft.covariates),
        *((effect_modifier, "effect_modifier") for effect_modifier in draft.effect_modifiers),
    ]:
        if column is None:
            continue
        normalized_column = str(column).strip()
        if not normalized_column:
            continue
        if normalized_column not in required_columns:
            required_columns.append(normalized_column)
        role_by_column[normalized_column] = role

    id_col = str(draft.id_col or "").strip()
    if id_col and id_col in data.columns and id_col not in required_columns:
        required_columns.insert(0, id_col)
        role_by_column[id_col] = "identifier"

    current_dtype_by_column = {
        str(column): str(dtype) for column, dtype in data.dtypes.to_dict().items()
    }
    planner_payload = {
        "draft": draft.model_dump(mode="json"),
        "dataset_summary": summary.model_dump(mode="json"),
        "current_dataframe_columns": [str(column) for column in data.columns],
        "current_dataframe_dtypes": current_dtype_by_column,
        "required_columns": required_columns,
        "role_by_column": role_by_column,
        "revised_instructions": revised_instructions,
    }

    data_type_plan_response = llm.generate(
        system_prompt=data_compilation_data_type_plan_prompt(),
        user_prompt=json.dumps(planner_payload, ensure_ascii=False),
        config=LLMConfig(model="basic", temperature=0.4),
        history=None,
    )
    data_type_plan_text = str(data_type_plan_response.content or "").strip()
    if not data_type_plan_text:
        raise ValueError("datatype conversion planning returned an empty instruction")

    changed_data = data_maupulation_tools.manipulate(
        dataframe=data,
        table_name=table_name,
        data_summary=summary.model_dump_json(),
        instructions=data_type_plan_text,
    )
    
    missing_columns = [
        column for column in required_columns if column not in changed_data.columns
    ]
    if missing_columns:
        raise ValueError(
            "datatype conversion output dataframe is missing required column(s): "
            + ", ".join(missing_columns)
        )

    return changed_data


def _impute_missing_values(
    table_name: str,
    data: pd.DataFrame,
    draft: CausalSpecDraft,
    summary: DatasetSummaryModel,
    data_maupulation_tools: DataManipulationTool,
    llm: LLMService,
    revised_instructions: str | None = None,
) -> pd.DataFrame:
    """
    generate string plan by calling LLM to impute missing values for best of the knowlwdge and best practices according to machine learning
    and causal inference best practices.
    give instruction to data manipulation tool and get the data with imputed values.
    if covariates and effect modifers are greater than 10 prefer two LLMs calls or loop if greater than 10 then 2 and 20 3 and so on
    validate if cols does not change w.r.t draft and return data frame.
    """
    required_columns: list[str] = []
    role_by_column: dict[str, str] = {}
    for column, role in [
        (draft.treatment_column, "treatment"),
        (draft.outcome_column, "outcome"),
        (draft.negative_control_outcome, "negative_control_outcome"),
        *((covariate, "covariate") for covariate in draft.covariates),
        *((effect_modifier, "effect_modifier") for effect_modifier in draft.effect_modifiers),
    ]:
        if column is None:
            continue
        normalized_column = str(column).strip()
        if not normalized_column:
            continue
        if normalized_column not in required_columns:
            required_columns.append(normalized_column)
        role_by_column[normalized_column] = role

    id_col = str(draft.id_col or "").strip()
    if id_col and id_col in data.columns and id_col not in required_columns:
        required_columns.insert(0, id_col)
        role_by_column[id_col] = "identifier"

    missing_count_by_column = {
        column: int(data[column].isna().sum())
        for column in required_columns
        if column in data.columns
    }
    columns_with_missingness = [
        column for column, count in missing_count_by_column.items() if count > 0
    ]
    if not columns_with_missingness:
        return data.copy()

    feature_columns = [
        str(column).strip()
        for column in [*draft.covariates, *draft.effect_modifiers]
        if str(column).strip()
    ]
    feature_batches = [
        feature_columns[index : index + 10]
        for index in range(0, len(feature_columns), 10)
    ]
    if not feature_batches:
        feature_batches = [[]]

    imputed_data = data.copy()
    for batch_number, feature_batch in enumerate(feature_batches, start=1):
        batch_columns = [
            column
            for column in [
                draft.treatment_column,
                draft.outcome_column,
                draft.negative_control_outcome,
                *feature_batch,
            ]
            if column is not None and str(column).strip()
        ]
        batch_columns = list(dict.fromkeys(str(column).strip() for column in batch_columns))
        batch_missing_counts = {
            column: int(imputed_data[column].isna().sum())
            for column in batch_columns
            if column in imputed_data.columns
        }
        if not any(count > 0 for count in batch_missing_counts.values()):
            continue

        planner_payload = {
            "draft": draft.model_dump(mode="json"),
            "dataset_summary": summary.model_dump(mode="json"),
            "current_dataframe_columns": [str(column) for column in imputed_data.columns],
            "required_columns": required_columns,
            "role_by_column": role_by_column,
            "columns_to_impute_this_batch": batch_columns,
            "batch_number": batch_number,
            "total_batches": len(feature_batches),
            "missing_count_by_column": batch_missing_counts,
            "revised_instructions": revised_instructions,
        }
        imputation_plan_response = llm.generate(
            system_prompt=data_compilation_imputation_plan_prompt(),
            user_prompt=json.dumps(planner_payload, ensure_ascii=False),
            config=LLMConfig(model="basic", temperature=0.4),
            history=None,
        )
        imputation_plan_text = str(imputation_plan_response.content or "").strip()
        if not imputation_plan_text:
            raise ValueError("missing-value imputation planning returned an empty instruction")

        imputed_data = data_maupulation_tools.manipulate(
            dataframe=imputed_data,
            table_name=table_name,
            data_summary=summary.model_dump_json(),
            instructions=imputation_plan_text,
        )

        missing_columns = [
            column for column in required_columns if column not in imputed_data.columns
        ]
        if missing_columns:
            raise ValueError(
                "missing-value imputation output dataframe is missing required column(s): "
                + ", ".join(missing_columns)
            )

    return imputed_data


def _calcaulate_miisingess_per_col_and_get_back_msingnes(
    data: pd.DataFrame,
    draft: CausalSpecDraft,
    summary: DatasetSummaryModel,
    data_profiling_tools: DatasetProfilingTool,
) -> dict[str, dict[str, Any]]:
    """
    calculate missingness per column and return col_missingess so that after impuation we acan check if values are imputeted or not.
    feel free to define the stricuture here.
    """
    profile_by_column = {
        str(profile.name).strip(): profile for profile in summary.profiles
    }

    draft_columns: list[str] = []
    for column in [
        draft.treatment_column,
        draft.outcome_column,
        draft.negative_control_outcome,
        *draft.covariates,
        *draft.effect_modifiers,
    ]:
        if column is None:
            continue
        normalized_column = str(column).strip()
        if normalized_column and normalized_column not in draft_columns:
            draft_columns.append(normalized_column)

    id_col = str(draft.id_col or "").strip()
    missingness_by_column: dict[str, dict[str, Any]] = {}
    for column in draft_columns:
        if column not in data.columns:
            continue
        missing_mask = data[column].isna()
        profile = profile_by_column.get(column)
        missingness_by_column[column] = {
            "missing_count": int(profile.n_missing if profile is not None else missing_mask.sum()),
            "missing_rate": float(
                profile.missing_rate if profile is not None else missing_mask.mean()
            ),
            "missing_index": data.index[missing_mask].tolist(),
            "missing_ids": (
                data.loc[missing_mask, id_col].tolist()
                if id_col and id_col in data.columns
                else []
            ),
        }

    return missingness_by_column


def _fill_miginess(
    draft: CausalSpecDraft,
    data: pd.DataFrame,
    col_missingess: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, CausalSpecDraft]:
    """
     check before and after msingness values and then add mossinges col in dataframe and in draft and return datasframe and new draft.
    """
    updated_data = data.copy()
    updated_covariates = list(draft.covariates)
    updated_effect_modifiers = list(draft.effect_modifiers)
    id_col = str(draft.id_col or "").strip()

    for column, missingness in col_missingess.items():
        before_missing_count = int(missingness.get("missing_count") or 0)
        if before_missing_count <= 0 or column not in updated_data.columns:
            continue

        after_missing_count = int(updated_data[column].isna().sum())
        if after_missing_count > 0:
            continue

        if column not in draft.covariates and column not in draft.effect_modifiers:
            continue

        indicator_column = f"{column}__missing"
        if indicator_column not in updated_data.columns:
            missing_ids = missingness.get("missing_ids") or []
            missing_index = missingness.get("missing_index") or []
            if id_col and id_col in updated_data.columns and missing_ids:
                updated_data[indicator_column] = updated_data[id_col].isin(missing_ids).astype(int)
            else:
                updated_data[indicator_column] = updated_data.index.isin(missing_index).astype(int)

        if column in draft.covariates and indicator_column not in updated_covariates:
            updated_covariates.append(indicator_column)
        if column in draft.effect_modifiers and indicator_column not in updated_effect_modifiers:
            updated_effect_modifiers.append(indicator_column)

    updated_draft = draft.model_copy(
        update={
            "covariates": updated_covariates,
            "effect_modifiers": updated_effect_modifiers,
        }
    )
    return updated_data, updated_draft


def compile_causal_spec_from_draft(
    protocol_discussion: str,
    dataset_summary: DatasetSummaryModel,
    previous_draft: CausalSpecDraft | None = None,
    retry_feedback: str | None = None,
) -> CausalSpec:
    """
    Compile causal specs from causal drafht
    """
    _ = protocol_discussion
    _ = retry_feedback
    if previous_draft is None:
        raise ValueError("previous_draft is required to compile causal spec from draft")

    profile_by_column = {
        str(profile.name).strip(): profile for profile in dataset_summary.profiles
    }

    treatment_profile = profile_by_column.get(str(previous_draft.treatment_column).strip())
    outcome_profile = profile_by_column.get(str(previous_draft.outcome_column).strip())
    if treatment_profile is None:
        raise ValueError(
            f"dataset summary is missing treatment column: {previous_draft.treatment_column}"
        )
    if outcome_profile is None:
        raise ValueError(
            f"dataset summary is missing outcome column: {previous_draft.outcome_column}"
        )

    treatment_values: list[str] = []
    treatment_profile_dump = treatment_profile.model_dump(mode="json")
    treatment_summary = treatment_profile_dump.get("summary", {})
    if "top_categories" in treatment_summary:
        treatment_values = [
            str(item["value"]) for item in treatment_summary.get("top_categories", [])
        ]
    elif "counts" in treatment_summary:
        treatment_values = [str(value) for value in treatment_summary.get("counts", {}).keys()]
    elif "distinct_values_sample" in treatment_summary:
        treatment_values = [
            str(value) for value in treatment_summary.get("distinct_values_sample", [])
        ]
    elif str(treatment_profile.inferred_kind) == "NUMERIC":
        treatment_values = ["1", "0"]
    if len(treatment_values) < 2:
        raise ValueError(
            f"treatment column '{previous_draft.treatment_column}' needs at least two observed values"
        )

    outcome_profile_dump = outcome_profile.model_dump(mode="json")
    outcome_summary = outcome_profile_dump.get("summary", {})
    outcome_values: list[str] = []
    if "top_categories" in outcome_summary:
        outcome_values = [
            str(item["value"]) for item in outcome_summary.get("top_categories", [])
        ]
    elif "counts" in outcome_summary:
        outcome_values = [str(value) for value in outcome_summary.get("counts", {}).keys()]
    elif "distinct_values_sample" in outcome_summary:
        outcome_values = [
            str(value) for value in outcome_summary.get("distinct_values_sample", [])
        ]

    if str(outcome_profile.inferred_kind) == "NUMERIC" and (
        outcome_profile.distinct_count is None or outcome_profile.distinct_count > 2
    ):
        outcome_spec: dict[str, Any] = {
            "kind": "continuous",
            "column": str(previous_draft.outcome_column),
            "unit": None,
            "clip_min": None,
            "clip_max": None,
        }
    else:
        if len(outcome_values) < 2:
            outcome_values = ["1", "0"]
        outcome_spec = {
            "kind": "binary",
            "column": str(previous_draft.outcome_column),
            "event": outcome_values[0],
            "non_event": outcome_values[1],
        }

    negative_control_outcome_spec: dict[str, Any] | None = None
    if previous_draft.negative_control_outcome is not None:
        negative_control_column = str(previous_draft.negative_control_outcome).strip()
        negative_control_profile = profile_by_column.get(negative_control_column)
        if negative_control_profile is None:
            raise ValueError(
                "dataset summary is missing negative-control outcome column: "
                + negative_control_column
            )
        negative_control_dump = negative_control_profile.model_dump(mode="json")
        negative_control_summary = negative_control_dump.get("summary", {})
        negative_control_values: list[str] = []
        if "top_categories" in negative_control_summary:
            negative_control_values = [
                str(item["value"])
                for item in negative_control_summary.get("top_categories", [])
            ]
        elif "counts" in negative_control_summary:
            negative_control_values = [
                str(value) for value in negative_control_summary.get("counts", {}).keys()
            ]
        elif "distinct_values_sample" in negative_control_summary:
            negative_control_values = [
                str(value)
                for value in negative_control_summary.get("distinct_values_sample", [])
            ]

        if str(negative_control_profile.inferred_kind) == "NUMERIC" and (
            negative_control_profile.distinct_count is None
            or negative_control_profile.distinct_count > 2
        ):
            negative_control_outcome_spec = {
                "kind": "continuous",
                "column": negative_control_column,
                "unit": None,
                "clip_min": None,
                "clip_max": None,
            }
        else:
            if len(negative_control_values) < 2:
                negative_control_values = ["1", "0"]
            negative_control_outcome_spec = {
                "kind": "binary",
                "column": negative_control_column,
                "event": negative_control_values[0],
                "non_event": negative_control_values[1],
            }

    payload = {
        "treatment_spec": {
            "kind": "binary",
            "column": str(previous_draft.treatment_column),
            "treated": treatment_values[0],
            "control": treatment_values[1],
        },
        "outcome_spec": outcome_spec,
        "negative_control_outcome": negative_control_outcome_spec,
        "covariates": [str(column) for column in previous_draft.covariates],
        "effect_modifiers": [str(column) for column in previous_draft.effect_modifiers],
        "experiment_type": previous_draft.study_type or "OBSERVATIONAL",
        "id_col": str(previous_draft.id_col),
    }
    return CausalSpec.for_dataset_summary(dataset_summary).model_validate(payload)


def compile_causal_spec_from_cleaned_summary(
    *,
    llm: LLMService,
    cleaned_summary: DatasetSummaryModel,
    draft_causal_spec: CausalSpecDraft,
    protocol_discussion: str,
    retry_feedback: str | None = None,
) -> CausalSpec:
    _ = llm
    return compile_causal_spec_from_draft(
        protocol_discussion=protocol_discussion,
        dataset_summary=cleaned_summary,
        previous_draft=draft_causal_spec,
        retry_feedback=retry_feedback,
    )


def _validate(
    causal_spec: object,
    causal_spec_draft: CausalSpecDraft,
    data: pd.DataFrame,
    dataset_summary: DatasetSummaryModel,
) -> None:
    """validate after cleaning"""
    summary_columns = {
        str(profile.name).strip()
        for profile in dataset_summary.profiles
        if str(profile.name).strip()
    }
    dataframe_columns = {str(column) for column in data.columns}

    draft_required_columns: list[str] = []
    for column in [
        causal_spec_draft.treatment_column,
        causal_spec_draft.outcome_column,
        causal_spec_draft.negative_control_outcome,
        *causal_spec_draft.covariates,
        *causal_spec_draft.effect_modifiers,
    ]:
        if column is None:
            continue
        normalized_column = str(column).strip()
        if normalized_column and normalized_column not in draft_required_columns:
            draft_required_columns.append(normalized_column)

    id_col = str(causal_spec_draft.id_col or "").strip()
    if id_col and id_col in data.columns and id_col not in draft_required_columns:
        draft_required_columns.insert(0, id_col)

    missing_from_data = [
        column for column in draft_required_columns if column not in dataframe_columns
    ]
    if missing_from_data:
        raise ValueError(
            "cleaned dataframe is missing draft column(s): "
            + ", ".join(missing_from_data)
        )

    missing_from_summary = [
        column for column in draft_required_columns if column not in summary_columns
    ]
    if missing_from_summary:
        raise ValueError(
            "dataset summary is missing draft column(s): "
            + ", ".join(missing_from_summary)
        )

    if id_col and id_col in data.columns:
        if int(data[id_col].isna().sum()) > 0:
            raise ValueError(f"identifier column '{id_col}' contains missing values")
        if bool(data[id_col].duplicated().any()):
            raise ValueError(f"identifier column '{id_col}' contains duplicate values")

    missing_after_cleaning = [
        column for column in draft_required_columns if int(data[column].isna().sum()) > 0
    ]
    if missing_after_cleaning:
        raise ValueError(
            "cleaned dataframe still has missing values in draft column(s): "
            + ", ".join(missing_after_cleaning)
        )

    if isinstance(causal_spec, CausalSpec):
        spec_columns: list[str] = [
            str(causal_spec.treatment_spec.column),
            str(causal_spec.outcome_spec.column),
            *(
                [str(causal_spec.negative_control_outcome.column)]
                if causal_spec.negative_control_outcome is not None
                else []
            ),
            *[str(column) for column in causal_spec.covariates],
            *[str(column) for column in causal_spec.effect_modifiers],
        ]
        if str(causal_spec.id_col) in data.columns:
            spec_columns.insert(0, str(causal_spec.id_col))

        spec_missing_from_data = [
            column for column in dict.fromkeys(spec_columns) if column not in dataframe_columns
        ]
        if spec_missing_from_data:
            raise ValueError(
                "cleaned dataframe is missing causal spec column(s): "
                + ", ".join(spec_missing_from_data)
            )


def cleaning_summary(
    before_data: pd.DataFrame,
    after_data: pd.DataFrame,
    before_summary: DatasetSummaryModel,
    after_summary: DatasetSummaryModel,
    before_draft: CausalSpecDraft,
    after_draft: CausalSpecDraft,
) -> str:
    before_columns = [str(column) for column in before_data.columns]
    after_columns = [str(column) for column in after_data.columns]
    before_column_set = set(before_columns)
    after_column_set = set(after_columns)
    removed_columns = [column for column in before_columns if column not in after_column_set]
    added_columns = [column for column in after_columns if column not in before_column_set]
    retained_columns = [column for column in before_columns if column in after_column_set]

    before_profile_by_column = {
        str(profile.name).strip(): profile for profile in before_summary.profiles
    }
    after_profile_by_column = {
        str(profile.name).strip(): profile for profile in after_summary.profiles
    }

    dtype_changes: list[str] = []
    missingness_changes: list[str] = []
    for column in retained_columns:
        before_profile = before_profile_by_column.get(column)
        after_profile = after_profile_by_column.get(column)
        before_dtype = (
            str(before_data[column].dtype)
            if column in before_data.columns
            else str(before_profile.dtype if before_profile is not None else "")
        )
        after_dtype = (
            str(after_data[column].dtype)
            if column in after_data.columns
            else str(after_profile.dtype if after_profile is not None else "")
        )
        if before_dtype != after_dtype:
            dtype_changes.append(f"{column}: {before_dtype} -> {after_dtype}")

        before_missing = (
            int(before_profile.n_missing)
            if before_profile is not None
            else int(before_data[column].isna().sum())
        )
        after_missing = (
            int(after_profile.n_missing)
            if after_profile is not None
            else int(after_data[column].isna().sum())
        )
        if before_missing != after_missing:
            missingness_changes.append(f"{column}: {before_missing} -> {after_missing}")

    added_covariates = [
        str(column)
        for column in after_draft.covariates
        if str(column) not in {str(value) for value in before_draft.covariates}
    ]
    added_effect_modifiers = [
        str(column)
        for column in after_draft.effect_modifiers
        if str(column) not in {str(value) for value in before_draft.effect_modifiers}
    ]

    lines = [
        "Cleaning summary:",
        f"- Rows: {len(before_data)} -> {len(after_data)}",
        f"- Columns: {len(before_columns)} -> {len(after_columns)}",
    ]
    if removed_columns:
        lines.append("- Removed columns: " + ", ".join(removed_columns))
    if added_columns:
        lines.append("- Added columns: " + ", ".join(added_columns))
    if dtype_changes:
        lines.append("- Datatype changes: " + "; ".join(dtype_changes))
    if missingness_changes:
        lines.append("- Missingness changes: " + "; ".join(missingness_changes))
    if added_covariates:
        lines.append("- Added covariates: " + ", ".join(added_covariates))
    if added_effect_modifiers:
        lines.append("- Added effect modifiers: " + ", ".join(added_effect_modifiers))
    if len(lines) == 3:
        lines.append("- No row, column, datatype, missingness, or draft-role changes detected.")

    return "\n".join(lines)
