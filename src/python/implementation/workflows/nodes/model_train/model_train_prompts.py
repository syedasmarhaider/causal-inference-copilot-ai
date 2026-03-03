def get_model_train_node_info() -> str:
    return """        "ModelTrainNode: trains the selected causal model using the cleaned dataset and compiled protocol. "
        "Returns state with trained_model_id, column_transformation_plan, and any training warnings."
        "It does data transformation but requires user input to have a suggestion on transformation if transformation is failed"
        "It runs the training automatically and does not requires user input on that"
    """


ENCODING_PLAN_SYSTEM_PROMPT = """
You are a Clinical Causal Copilot that helps to create a data preprocessing/encoding plan for causal inference modeling based on the selected model and dataset characteristics after model selection.
""".strip()


ENCODING_PLAN_USER_PROMPT_TEMPLATE = """
Task:
Create a preprocessing/encoding plan for causal inference modeling. The plan will be compiled into sklearn transformers.

Critical clinical safety rules:
- DO NOT create encodings for treatment or outcome columns. They are handled separately and must not be transformed here.
- Pay extreme caution when handling missing values. Dropping rows due to missing values is clinically risky and can bias results and if there is prev error about it.

Interpretation rules:
- X = covariates (confounders / adjustment features).
- W = effect modifiers (features where treatment effect may differ across subgroups).
You are given:
1) Model selection output (chosen estimator family + rationale)
2) Compiled protocol information (treatment, outcome, covariates X, effect_modifiers W)
3) Cleaned dataset summary (column types, unique counts, missingness, examples)
4) Supported encoding presets and their parameter schema

Your job:
- Build a TransformPlan for ONLY the eligible columns:
  Eligible columns = (covariates + effect_modifiers) MINUS (treatment column, outcome column).
- Use best practices for the selected estimator:
- If X has missing values consider encoding and mutation strageties that are robust to missingness such as missing indicators, and consider avoiding encodings that would drop rows with missing values. If there are critical errors about missing values please consider strategies to handle them such as imputation or dropping with caution and explain in the message. 
  - Consider Model name and characteristics when making encoding choices. For example like intercepts are always true so encoding needs to be adjusted such as baseline.
  
- Output (JSON):
   needs_user_input: bool true when you have to involve user in making decision
   user_message: str clinician-friendly explanation of the encoding plan and any issues/warnings that require user attention. Be specific about what the issue is, which columns it affects, and what the potential implications are for modeling and results interpretation. Ask user action whenever needs_user_input is true otherwise just tell.
   plan: dict the actual encoding plan for the eligible columns it can be null when you need clarification and needs_user_input is true.
    
  
  
 
Supported presets (JSON):


Chosen causal estimator (JSON):
{selected_model_json}

Protocol (JSON):
{protocol_json}

Dataset summary (JSON):
{dataset_summary_json}

Previous training errors Optional(String):
{prev_training_errors_string}

documentation of model Optional(String):
{documentation_string}

Return JSON only.
""".strip()



FIT_SUCCESS_FAILURE_SYSTEM_PROMPT = """
You are a Clinical Causal Copilot that helps to train causal inference models based on the selected model, compiled protocol, and cleaned dataset.
Explain to the user in a clinician-friendly way the result of the training command execution, including any warnings or errors
warnings or errors if it make sense to present to clinicians and their implications regarding training and reliablity not internal errors etc.
"""