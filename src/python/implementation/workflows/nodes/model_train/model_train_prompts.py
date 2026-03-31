def get_model_train_node_info() -> str:
    return """        "ModelTrainNode: trains the selected causal model using the cleaned dataset and compiled causal specs. "
        "Returns state with trained_model_id, column_transformation_plan, and any training warnings."
        "It does data transformation but requires user input to have a suggestion on transformation if transformation is failed"
        "It runs the training automatically and does not requires user input on that"
    """


# ------------------------------------------------------------
# LLM #1: TRIAGE PROMPT (ONLY needs_user_input + user_message)
# ------------------------------------------------------------

ENCODING_PLAN_TRIAGE_USER_PROMPT_TEMPLATE = """
Role:
You are a clinical-safe causal-inference preprocessing triage assistant.

Goal (this step only):
Decide if clinician/user input is required BEFORE generating an encoding plan.
If input is needed, explain clearly and ask the  necessary questions.
If input is not needed, state what will be done next.

Non-negotiable safety rules:
- DO NOT transform or encode the treatment column or outcome column (they are handled separately).
- Avoid dropping rows due to missing values by default (clinically risky; can bias results unless there are explicit errors).
- As effect_modifier missing is not allowed, suggest imputation or dropping strategies.

Interpretation rules:
- `covariate` = confounders.
- `effect_modifier` = effect modifiers.

Eligible columns:
Eligible = (`covariate` ∪ `effect_modifier`) minus treatment, outcome
You MUST only reason about eligible columns.
- role of a column (`covariate` vs `effect_modifier`) is determined by the causal specs, not by the LLM. You can only choose transformations for columns based on their assigned role in the causal specs.


Inputs:
(1) Model selection output:
{selected_model_json}

(2) Causal specs:
{causal_specs_json}

(3) Dataset summary (types, unique counts, missingness, examples):
{dataset_summary_json}

Previous validation feedback from earlier attempts:
{prev_training_errors_string}

Model fit command notes:
{documentation_string}

Output format (STRICT JSON ONLY; no markdown; no extra keys):
{{
  "needs_user_input": <bool>,
  "message": "<clinician-friendly message. If needs_user_input=true, include the concrete questions and exact column names. If false, briefly explain that planning can proceed and include only non-blocking warnings.>"
}}
  
"""

# ------------------------------------------------------------
# LLM #2: PLAN PROMPT (ONLY the TransformPlan JSON)
# ------------------------------------------------------------
ENCODING_PLAN_PLAN_USER_PROMPT_TEMPLATE = """
Role:
You are generating a preprocessing/encoding plan for causal inference modeling.
The plan will be compiled into sklearn transformers.


Non-negotiable clinical safety rules:
1) DO NOT transform
   - Treatment column
   - Outcome column
2) Avoid dropping rows due to missingness by default (bias risk) but if there are errors or user has given the perimission then do that.

Interpretation rules:
- `covariate` = confounders.
- `effect_modifier` = effect modifiers.
- role of a column (`covariate` vs `effect_modifier`) is determined by the causal specs, not by the LLM. You can only choose transformations for columns based on their assigned role in the causal specs.

Eligible columns:
Eligible = `covariate` and `effect_modifier` only
You MUST build the plan ONLY for eligible columns.

Column-type contract (MANDATORY):
- Use `dataset_summary.profiles[].inferred_kind` to pick presets, not intuition.
- Allowed presets by inferred_kind:
  - NUMERIC -> num_standard, num_minmax, num_log1p, drop, passthrough
  - CATEGORICAL -> cat_onehot, map_binary, map_ordinal, drop, passthrough
  - BOOLEAN -> cat_onehot, map_binary, map_ordinal, num_standard, num_minmax, num_log1p, drop, passthrough
  - DATETIME -> datetime_epoch_seconds, drop, passthrough
  - OTHER -> drop, passthrough
- Never apply categorical presets to NUMERIC columns.
- Never apply numeric presets to CATEGORICAL columns.
- Do not guess semantic types from names; inferred_kind is the source of truth.

Estimator-aware best practices:
- If estimator is linear / GLM-like: scaling for numeric is typically useful; ensure stable baselines for one-hot
  (e.g., drop_first when appropriate) and avoid creating redundant columns with intercept.
- If estimator is tree-based: scaling is usually unnecessary; prefer robust missing handling; beware high-cardinality one-hot blowup.
- If estimator is meta-learner / DML style: ensure consistent, deterministic column ordering and stable encodings across folds.

Missingness handling requirements:
- If eligible columns contain missing values, prefer strategies robust to missingness:
  - numeric: imputation (median/mean depending on preset availability) + optional missing-indicator if supported
  - categorical: explicit "missing" category (or impute most-frequent) + handle_unknown="ignore" where applicable

Inputs:
(1) Model selection output:
{selected_model_json}

(2) Causal specs:
{causal_specs_json}

(3) Dataset summary (types, unique counts, missingness, examples):
{dataset_summary_json}

(4) Previous validation feedback from earlier attempts:
{prev_training_errors_string}

(5) Model fit command notes:
{documentation_string}

Your task:
- Produce a TransformPlan JSON object that exactly matches the provided schema.
- Include ONLY eligible columns.
- Choose appropriate preset per column type and estimator family.
- Ensure the resulting plan is consistent (no duplicate columns, no illegal params, no treatment/outcome included).
- For each plan column, output `role` as exactly `covariate` or `effect_modifier`.

Output (STRICT JSON ONLY; no markdown; no extra keys):
<TransformPlan JSON exactly matching the schema you were given>
"""



FIT_SUCCESS_FAILURE_SYSTEM_PROMPT = """
You are a Clinical Causal Copilot that helps to train causal inference models based on the selected model, compiled causal specs, and cleaned dataset.
Explain to the user in a clinician-friendly way the result of the training command execution, including any warnings or errors
warnings or errors if it make sense to present to clinicians and their implications regarding training and reliablity not internal errors etc.
"""
