from __future__ import annotations

import json

import pandas as pd

from python.domain.service.llm_service import LLMConfig, LLMService
from python.implementation.workflows.nodes.data_compilation.data_compilation_prompts import (
    data_compilation_filter_plan_prompt,
)
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    CausalSpecDraft,
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


def clean(
    data: pd.DataFrame,
    data_summary: DatasetSummaryModel,
    draft: CausalSpecDraft,
    data_maupulation_tools: DataManipulationTool,
    data_profiling_tools: DatasetProfilingTool,
) -> pd.DataFrame:
    raise NotImplementedError("clean orchestration will be implemented after the stage functions")


def _filter_data(
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
        table_name="protocol_scope_df",
        data_summary=summary.model_dump_json(),
        instructions=filter_plan_text,
    )
    if not isinstance(filtered_data, pd.DataFrame):
        raise TypeError(
            "target population filter manipulation must return a pandas DataFrame"
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
    raise NotImplementedError("_drop_irrelevant_columns is not implemented yet")


def _data_type_change(
    data: pd.DataFrame,
    draft: CausalSpecDraft,
    data_maupulation_tools: DataManipulationTool,
) -> pd.DataFrame:
    """
    generate string plan by calling LLM to convert datatypes for best of the knowlwdge and best practices according to machine learning
    and causal inference best practices.
    give instruction to data manipulation tool and get the data with changed datatypes.
    validate if cols does not change w.r.t draft and return data frame.
    """
    raise NotImplementedError("_data_type_change is not implemented yet")


def _impute_missing_values(
    data: pd.DataFrame,
    draft: CausalSpecDraft,
    data_maupulation_tools: DataManipulationTool,
) -> pd.DataFrame:
    """
    generate string plan by calling LLM to impute missing values for best of the knowlwdge and best practices according to machine learning
    and causal inference best practices.
    give instruction to data manipulation tool and get the data with imputed values.
    if covariates and effect modifers are greater than 10 prefer two LLMs calls or loop if greater than 10 then 2 and 20 3 and so on
    validate if cols does not change w.r.t draft and return data frame.
    """
    raise NotImplementedError("_impute_missing_values is not implemented yet")


def _calcaulate_miisingess_per_col_and_get_back_msingnes(
    data: pd.DataFrame,
    draft: CausalSpecDraft,
    data_profiling_tools: DatasetProfilingTool,
) -> dict[str, float]:
    """
    calculate missingness per column and return col_missingess so that after impuation we acan check if values are imputeted or not.
    feel free to define the stricuture here.
    """
    raise NotImplementedError(
        "_calcaulate_miisingess_per_col_and_get_back_msingnes is not implemented yet"
    )


def _fill_miginess(
    draft: CausalSpecDraft,
    data: pd.DataFrame,
    col_missingess: dict[str, float],
) -> CausalSpecDraft:
    """
     check before and after msingness values and then add mossinges col in dataframe and in draft and return datasframe and new draft.
    """
    raise NotImplementedError("_fill_miginess is not implemented yet")


def compile_causal_spec_from_draft(
    protocol_discussion: str,
    dataset_summary: DatasetSummaryModel,
    previous_draft: CausalSpecDraft | None = None,
    retry_feedback: str | None = None,
) -> CausalSpecDraft:
    """
    Compile causal specs from causal drafht
    """
    raise NotImplementedError("compile_causal_spec_from_draft is not implemented yet")


def _validate(
    causal_spec: object,
    causal_spec_draft: CausalSpecDraft,
    data: pd.DataFrame,
    dataset_summary: DatasetSummaryModel,
) -> None:
    """validate after cleaning"""
    raise NotImplementedError("_validate is not implemented yet")
