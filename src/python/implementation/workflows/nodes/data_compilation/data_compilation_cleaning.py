







import pandas as pd

from python.implementation.workflows.tools.causal.specs.causal_spec_draft import CausalSpecDraft


def clean(
    data: pd.DataFrame,
    data_summary: ModelDataSummary,
    draft: CausalSpecDraft,
    data_maupulation_tools: DataManipulationTools,
    data_profiling_tools: DataProfilingTools,
    ):
    













def _filter_data(
    data: pd.DataFrame,
    draft: CausalSpecDraft,
    summary: ModelDataSummary,
    data_maupulation_tools: DataManipulationTools,
    ) -> pd.DataFrame:
    """
    Apply physical dataset filters based on the draft's target population text, if useful and agreed by the user.
    steps:
    
    
    1: plan as a text simple text like call llm and generate fitler plan insturction 
    2: define prompt in promtp file
    3: generate insturction
    4: pass instruction + data summary in data manupulation tool
    5: get result 
    6: validate if cols does not change w.r.t draft
    7: return causal draft 
    
    
    
    
    
    """









def _drop_irrelevant_columns(
    data: pd.DataFrame,
    draft: CausalSpecDraft,
    ) -> pd.DataFrame:
    """
    
    drop all irrelavent cols..
    keep tratment outcome covariates effect modifers and negative control outcome if specified. and id cols if specified. drop rest of the cols. 
    return data frame. 
    """
    
    

def _data_type_change(
    data: pd.DataFrame,
    draft: CausalSpecDraft,
    data_maupulation_tools: DataManipulationTools,
    ) -> pd.DataFrame:
    """
    generate string plan by calling LLM to convert datatypes for best of the knowlwdge and best practices according to machine learning
    and causal inference best practices.
    give instruction to data manipulation tool and get the data with changed datatypes.
    validate if cols does not change w.r.t draft and return data frame. 
    """    



def _impute_missing_values(
    data: pd.DataFrame,
    draft: CausalSpecDraft,
    data_maupulation_tools: DataManipulationTools,
    ) -> pd.DataFrame:
    """
    generate string plan by calling LLM to impute missing values for best of the knowlwdge and best practices according to machine learning
    and causal inference best practices.
    give instruction to data manipulation tool and get the data with imputed values.
    if covariates and effect modifers are greater than 10 prefer two LLMs calls or loop if greater than 10 then 2 and 20 3 and so on
    validate if cols does not change w.r.t draft and return data frame. 
    """ 
    
    
 
 def _calcaulate_miisingess_per_col_and_get_back_msingnes(
    data: pd.DataFrame,
    draft: CausalSpecDraft,
    data_profiling_tools: DataProfilingTools,
    ) -> dict[str, float]:
    """
    calculate missingness per column and return col_missingess so that after impuation we acan check if values are imputeted or not.
    feel free to define the stricuture here.
    """
    

def _fill_miginess(
    draft: CausalSpecDraft,
    data: pd.DataFrame,
    col_missingess: dict[str, float],
    ) -> CausalSpecDraft:
    """
     check before and after msingness values and then add mossinges col in dataframe and in draft and return datasframe and new draft.
    """



def compile_causal_spec_from_draft(
    protocol_discussion: str,
    dataset_summary: ModelDataSummary,
    previous_draft: CausalSpecDraft | None = None,
    retry_feedback: str | None = None,
) -> CausalSpecDraft:
    """
    Compile causal specs from causal drafht
    """

def _validate
(
    causal specs, 
    causal specs dradt
    data and
    adata set sumart
    
)
""""calidate ""    
     valdiate after cleaning 