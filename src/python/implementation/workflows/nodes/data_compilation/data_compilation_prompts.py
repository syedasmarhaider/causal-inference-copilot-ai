from __future__ import annotations


def data_compilation_node_info() -> str:
    return (
        "Draft-driven compilation and validation stage. It cleans the active dataset "
        "against the locked causal draft, builds a compiled causal spec, plans baseline "
        "transformations, validates the result, publishes a compiled dataset preview, "
        "and asks the user to confirm before freezing downstream causal outputs."
    )


def data_compilation_filter_plan_prompt() -> str:
    return """
You are devising a target-population filtering plan for a tabular causal-analysis dataset.

Inputs:
- target_population text from the locked causal draft
- locked causal draft
- authoritative dataset summary
- current dataframe columns
- revised_instructions text from review feedback or validation retry feedback, when present

Task:
- Write one complete natural-language instruction for a data manipulation tool to filter
  the dataframe to the draft target population.

Rules:
- Use the dataset summary to ground the filter plan in actual available columns, kinds,
  missingness, and observed values.
- Do not invent columns, values, timing rules, or cohort criteria that are not supported
  by the target population text and dataset summary.
- Preserve all draft-selected causal columns: treatment, outcome, negative-control outcome
  when present, covariates, effect modifiers, and an existing identifier column.
- If revised_instructions are present, incorporate them when compatible with the locked
  draft and dataset evidence.
- The instruction is for a data manipulation tool, not raw SQL.

Output policy:
- Output text only.
- Do not output JSON.
- Do not output SQL.
- Do not use markdown or code fences.
""".strip()


def data_compilation_data_type_plan_prompt() -> str:
    return """
You are devising a datatype normalization plan for a tabular causal-analysis dataset.

Inputs:
- locked causal draft
- authoritative dataset summary
- current dataframe columns and pandas dtypes
- required columns and their causal roles
- revised_instructions text, when present

Task:
- Write one complete natural-language instruction for a data manipulation tool to convert
  datatype representations needed for robust causal inference and downstream machine
  learning preprocessing.

Rules:
- Use the dataset summary and current dtypes to ground the datatype plan.
- Preserve all required draft-selected columns and their names.
- Do not rename, drop, duplicate, or regenerate treatment, outcome, negative-control
  outcome, covariate, effect-modifier, or existing identifier columns.
- Convert numeric-coded categories to categorical/string representations when the
  dataset evidence clearly indicates categorical meaning.
- Do not perform missingness imputation, row filtering, or causal role changes here.
- If revised_instructions are present, incorporate them when compatible with the locked
  draft and dataset evidence.
- The instruction is for a data manipulation tool, not raw SQL.

Output policy:
- Output text only.
- Do not output JSON.
- Do not output SQL.
- Do not use markdown or code fences.
""".strip()


def data_compilation_imputation_plan_prompt() -> str:
    return """
You are devising a missing-value imputation or missingness-resolution plan for a tabular
causal-analysis dataset.

Inputs:
- locked causal draft
- authoritative dataset summary
- current dataframe columns
- required columns and their causal roles
- columns_to_impute_this_batch
- missing_count_by_column for this batch
- batch_number and total_batches
- revised_instructions text, when present

Task:
- Write one complete natural-language instruction for a data manipulation tool to resolve
  missing values in the listed batch while preserving every required causal column.

Rules:
- Use the dataset summary and missing counts to ground the plan.
- Treatment, outcome, and negative-control outcome missingness should be resolved
  conservatively, usually by dropping rows or applying only a user-grounded rule.
- Covariate and effect-modifier missingness may be imputed using conservative ML and
  causal-inference practice when grounded by the dataset state.
- Preserve all required draft-selected columns and their names.
- Do not rename, drop, duplicate, or regenerate required columns.
- Do not perform target-population filtering, datatype normalization unrelated to
  missingness, or causal role changes here.
- If revised_instructions are present, incorporate them when compatible with the locked
  draft and dataset evidence.
- The instruction is for a data manipulation tool, not raw SQL.

Output policy:
- Output text only.
- Do not output JSON.
- Do not output SQL.
- Do not use markdown or code fences.
""".strip()


def data_compilation_transformation_retry_guidance_prompt() -> str:
    return """
Additional grounded retry guidance for the next full clean-transform attempt:
- Use the following repair text to revise cleaning choices and baseline transformations
  without changing locked column identities or roles.
- Keep treatment, outcome, covariates, and effect modifiers locked to the causal draft.
- Revise only same-column values, datatype handling, missingness handling, or
  covariate/effect-modifier encoding choices grounded by the repair text.
- Keep the applied transformation aligned to the column's current dataset kind.
- Treat any preferred future raw type as advisory only; do not let it override the
  applied preset rules.
""".strip()


def batch_transform_prompt() -> str:
    return """
You are compiling a strict datatype-driven transformation draft for covariates and effect modifiers only.

Inputs:
- transformation_instructions
- compiled_causal_specification
- scoped_dataset_summary
- eligible_columns
- expected_role_by_column
- required_plan_column_count
- allowed_presets_by_kind
- optional retry_note

Task:
- Produce one JSON object with a `columns` array.
- Include every eligible column exactly once.
- Never include treatment or outcome.
- Never include negative-control outcome.
- Each `columns` entry must contain exactly:
  - `column`
  - `role`
  - `preset`
  - `preferred_type`
  - `preferred_type_reason`

Planning rules:
- The applied `preset` must follow the column's current `kind` from the scoped dataset summary.
- Do not reinterpret the current stored kind for the applied preset.
- Use `allowed_presets_by_kind` as a hard constraint.
- Use `preferred_type` only as a non-blocking recommendation for the ideal future raw type of the column.
- `preferred_type` must not override or change the applied `preset`.
- Prefer minimal transformation when allowed for the current kind.
- Current-kind defaults:
  - `NUMERIC` usually defaults to `passthrough`
  - `BOOLEAN` usually defaults to `passthrough`
  - `CATEGORICAL` defaults to `cat_onehot`
  - `DATETIME` defaults to `datetime_epoch_seconds`
  - `OTHER` defaults to `drop`
- Only choose a non-default preset when `transformation_instructions` or retry guidance
  make it necessary and the preset is still allowed for the current kind.
- Do not emit dataset-change actions.
- Do not emit missing-data logic.
- If `retry_note` is present, use it only to correct malformed, incomplete, or
  previously incompatible draft output while keeping the same eligible columns and roles.

Output policy:
- Output JSON only.
- No markdown.
- No explanatory prose outside the JSON object.
""".strip()


def data_compilation_review_summary_prompt() -> str:
    return """
You are preparing the user-facing review message after draft-driven data compilation.

Inputs:
- compiled causal specification
- compiled dataset summary
- cleaning_summary
- compiled transformation plan
- transformation_suggestions
- compilation_actions
- compilation_warnings
- validation_status
- validation_issues

Task:
- Write a specific review message for the user.
- This is a review step before final confirmation. The compiled dataset is already the
  active preview dataset, but causal outputs are not frozen until the user confirms.

Content rules:
- Write for a clinician, not a data scientist. Prefer intuitive and practical language.
- Start with what changed in the dataset preview and why those changes matter.
- Explain row changes, columns kept/removed/added, datatype changes, missingness changes,
  and any missingness indicator columns using the cleaning_summary.
- Explain the planned baseline transformations column by column when helpful.
- When a column has a preferred future raw type that differs from its current stored type,
  mention that as a non-blocking recommendation and explain why.
- Explain what validation checked in practical terms.
- Surface non-blocking warnings clearly and explain what each warning means.
- State explicitly whether you recommend accepting the setup now or revising it.
- Summarize treatment, outcome, negative-control outcome status, covariates, effect
  modifiers, and compiled dataset shape.
- End by asking the user to confirm the compiled dataset preview, transformation plan,
  and validation result or say exactly what should change.
- Do not mention internal JSON, validators, or workflow implementation details.

Output JSON exactly:
{
  "assistant_message": "<specific clinician-facing review message that asks for confirmation or revision>"
}
""".strip()


def data_compilation_review_decision_prompt() -> str:
    return """
You are reviewing a compiled dataset preview and transformation plan with the user.

Task:
- Interpret the latest user reply to decide whether to accept the compiled setup, answer
  a review-time question, recompile from the original source dataset, reject and go back,
  or ask for clarification.

Decision rules:
- Choose `confirm` only when the user is clearly accepting the compiled dataset preview,
  transformation plan, and validation result as-is.
- Choose `answer_query` when the user is asking an explanatory question about the cached
  compiled setup and is not asking to change it.
- Choose `recompile` when the user wants same-column preprocessing, cleaning, missingness
  handling, row filtering, normalization, or other same-column compilation changes while
  keeping treatment, outcome, covariates, effect modifiers, and their roles unchanged.
- Choose `reject` when the user explicitly does not accept the setup and wants to go back,
  or when the requested change would alter locked column identities or roles.
- Choose `clarify` when the reply is ambiguous, incomplete, or not enough to confirm,
  reject, or recompile safely.

Style rules:
- Keep the assistant message plain, direct, and user-facing.
- If `clarify`, ask one focused follow-up question.
- If `recompile`, summarize the requested same-column changes briefly in `recompile_request`.
- If the reply is a question, do not choose `clarify`; choose `answer_query`.
- Do not invent new draft content.

Output JSON exactly:
{
  "action": "confirm" | "recompile" | "answer_query" | "reject" | "clarify",
  "assistant_message": "<short user-facing message>",
  "recompile_request": "<brief grounded recompilation request or null>"
}
""".strip()


def data_compilation_review_query_prompt() -> str:
    return """
You are answering a user question during the compilation review step.

Inputs:
- compiled causal specification
- compiled dataset summary
- cleaning_summary
- compiled transformation plan
- transformation_suggestions
- compilation_actions
- compilation_warnings
- validation_status
- validation_issues
- latest_user_message

Task:
- Answer the user's question using only the cached compiled review context.
- Do not confirm, reject, or modify the compiled setup unless the user explicitly asks for a change.

Style rules:
- Keep the answer user-facing and specific.
- Explain current compiled facts only; do not invent new draft content.
- If the question asks what changed, highlight row changes, retained/removed/added
  columns, missingness handling, transformations, and validation in plain language.

Output JSON exactly:
{
  "assistant_message": "<specific answer to the user's review-time question>"
}
""".strip()
