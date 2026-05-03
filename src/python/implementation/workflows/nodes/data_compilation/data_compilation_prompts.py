from __future__ import annotations


def data_compilation_node_info() -> str:
    return (
        "Merged compilation and validation stage. It turns a confirmed protocol discussion "
        "and causal draft into a compiled causal spec, a cleaned protocol-scope dataset, "
        "a baseline transformation plan, and a validation result. Repairable downstream "
        "issues trigger one automatic retry before the node presents a detailed clinician-"
        "facing review for confirmation."
    )


_SPEC_GUARDRAILS = """
Grounding and safety rules:
- Use only grounded information from the confirmed protocol discussion and dataset metadata.
- The source dataset summary is authoritative for column names, kinds, and discrete values.
- Do not invent columns, values, timing rules, horizons, or adjustment variables.
- Treatment must be binary.
- Outcome must be binary or continuous only.
- Covariates and effect modifiers must stay distinct.
- Treatment and outcome must not appear in covariates or effect modifiers.
""".strip()


def data_compilation_causal_spec_prompt() -> str:
    return f"""
You are compiling a confirmed target-trial style protocol discussion into a causal specification.

Inputs:
- confirmed protocol discussion text
- authoritative source dataset summary

Task:
- Produce one causal specification JSON object that matches the provided schema exactly.

Rules:
- Preserve the confirmed protocol semantics.
- Use exact dataset column names and exact grounded discrete literals from the source dataset summary.
- Do not invent new protocol assumptions.
- If the discussion identifies post-treatment variables, do not place them into baseline covariates or effect modifiers.

{_SPEC_GUARDRAILS}

Output policy:
- Output JSON only.
- No markdown.
- No explanatory prose outside the JSON object.
""".strip()


def data_compilation_causal_semantics_prompt() -> str:
    return """
You are resolving only the remaining treatment/outcome semantics for a locked causal draft.

Inputs:
- confirmed protocol discussion
- locked causal draft
- treatment column profile
- outcome column profile
- optional compile_feedback

Task:
- Return only the unresolved semantic details needed to build the final causal specification.

Rules:
- Do not change any locked column identities from the draft.
- Treatment is always binary.
- Resolve only:
  - treatment treated/control labels
  - outcome kind (`binary` or `continuous`)
  - binary outcome event/non_event labels or continuous outcome metadata
  - experiment_type
- Use exact grounded values from the protocol discussion and the provided treatment/outcome profiles.
- Do not invent new columns, covariates, effect modifiers, timing rules, or post-treatment assumptions.
- If compile_feedback is present, fix that specific semantic problem directly.
- If the outcome is continuous, only choose `continuous` when the outcome profile is numeric and the protocol supports it.

Output policy:
- Output JSON only.
- No markdown.
- No explanatory prose outside the JSON object.
""".strip()


def data_compilation_cleaning_instructions_prompt() -> str:
    return """
You are planning how protocol-scope missingness should be resolved before protocol compilation.

Inputs:
- confirmed protocol discussion
- confirmed protocol cleaning instructions
- scoped dataset summary
- expected_role_by_column
- optional review_recompile_request

Rules:
- Output one decision for every scoped column.
- The compiled dataset must end with no missing values in treatment, outcome, covariates, or effect modifiers.
- Prefer grounded imputation when the protocol or data type supports it cleanly.
- Prefer row removal only when imputation would be clinically or statistically misleading.
- Keep all column identities and roles locked.
- Do not invent new columns, values, or unsupported mappings.
- If review_recompile_request is present, use it only when it preserves the same locked columns and roles.
- Return JSON only.
- Each decision must contain:
  - `column`
  - `role`
  - `resolution` as `none_needed` | `drop_rows` | `impute`
  - `reason`
  - `instruction`
""".strip()


def data_compilation_simple_transform_prompt() -> str:
    return """
You are planning deterministic same-column dataframe transformations before SQL cleaning.

Inputs:
- confirmed protocol discussion
- confirmed protocol cleaning instructions
- optional review_recompile_request
- scoped dataset summary
- expected_role_by_column
- missingness_plan

Task:
- Return a JSON object with a `columns` array.
- Use an empty `columns` array when no simple deterministic transformation is necessary.

Simple transformation tool scope:
- Literal same-column replacements.
- Static same-column value assignment.
- Scalar missing-value fill.
- Simple dtype casts.

Rules:
- Use only existing scoped columns.
- Keep all column identities and roles locked.
- Do not create, rename, remove, or reorder columns.
- Do not emit row filters, joins, aggregations, window logic, parsing logic, or complex conditional cleaning.
- Do not emit drop-column work; final protocol-scope column dropping is handled by the runtime.
- Do not emit SQL work; a SQL tool runs later for complex cleaning and row filtering.
- Missingness `drop_rows` decisions must be left for the later SQL tool.
- If an imputation cannot be expressed as a scalar `fill_value`, leave it for the later SQL tool.
- Return JSON only.
""".strip()


def data_compilation_discrepancy_repair_prompt() -> str:
    return """
You are preparing one corrective SQL-oriented cleaning pass after protocol compilation and validation found grounded data discrepancies.

Inputs:
- confirmed protocol discussion
- confirmed protocol cleaning instructions
- compiled causal specification
- compiled dataset summary
- validation issues
- user-requested repair direction

Rules:
- Return only user-intent text for a SQL data manipulation tool.
- Keep the dataset inside the compiled protocol scope.
- Keep the locked treatment, outcome, covariate, and effect-modifier columns unchanged.
- Only fix discrepancies that are explicitly grounded by the compiled causal specification, compiled dataset summary, listed validation issues, and user-requested repair direction.
- Prefer safe type corrections, grounded recoding, and removal of invalid treatment or outcome rows over speculative remapping.
- Never add treatment or outcome transformations to the transform plan; fix the data instead when needed.
- Do not invent new cohort rules, new columns, or unsupported category merges.
- Do not include validation, modeling, plotting, or non-SQL tasks.
""".strip()


def data_compilation_transformation_retry_guidance_prompt() -> str:
    return """
Additional grounded retry guidance for the next causal-spec and transformation attempt:
- Use the following repair text to revise the compiled causal specification details and baseline transformations without changing locked column identities or roles.
- Keep treatment, outcome, covariates, and effect modifiers locked to the confirmed draft columns.
- Revise only same-column causal-spec details or datatype-constrained covariate/effect-modifier encoding choices that are grounded by the repair text.
- Keep the applied transformation aligned to the column's current dataset kind.
- Treat any preferred future raw type as advisory only; do not let it override the applied preset rules.
""".strip()


_PLAN_GUARDRAILS = """
Transformation-plan safety rules:
- Build the plan only for covariates and effect modifiers from the compiled causal specification.
- Never include treatment or outcome in the transformation plan.
- Use the compiled dataset summary as the source of truth for the current stored kind of each column.
- The applied preset must stay compatible with the current stored kind; do not reinterpret the current type.
- Keep the plan conservative and type-driven.
- Keep preferred future raw type suggestions separate from the applied preset.
""".strip()


def data_compilation_transformation_plan_prompt() -> str:
    return f"""
You are compiling a baseline transformation plan for a compiled protocol-scope dataset.

Inputs:
- confirmed protocol discussion
- compiled causal specification
- compiled dataset summary
- eligible_columns
- expected_role_by_column
- required_plan_column_count
- allowed_presets_by_kind
- optional retry_note
- optional validation_issues

Task:
- Produce one compact transformation-plan draft JSON object that matches the provided schema exactly.
- The runtime will expand the draft into the final TransformPlan using safe defaults.
- For each eligible column, output:
  - `column`
  - `role`
  - `preset`
  - `preferred_type`
  - `preferred_type_reason`

Defaults:
- NUMERIC columns usually default to `passthrough`; only choose `num_standard`, `num_minmax`, or `num_log1p` when the instructions make that useful.
- BOOLEAN columns usually default to `passthrough`.
- CATEGORICAL columns default to `cat_onehot`.
- DATETIME columns default to `datetime_epoch_seconds`.
- OTHER columns default to `drop`.

Role rules:
- Use `role="covariate"` exactly for compiled causal-spec covariates.
- Use `role="effect_modifier"` exactly for compiled causal-spec effect modifiers.
- Do not infer new roles.
- Include every eligible column exactly once.
- Do not include any column outside `eligible_columns`.
- The number of entries in `columns` must equal `required_plan_column_count`.
- For each entry, `role` must exactly match `expected_role_by_column[column]`.
- The applied `preset` must be chosen only from `allowed_presets_by_kind[column.kind]`.
- Do not reinterpret a numeric-coded category as categorical for the applied preset; keep the applied preset aligned to the current stored kind.
- Use `preferred_type` only to express the ideal future raw type of the column.
- If `retry_note` or `validation_issues` are provided, revise the draft to correct the same-column choice while keeping the same eligible columns and roles.

{_PLAN_GUARDRAILS}

Output policy:
- Output JSON only.
- No markdown.
- No explanatory prose outside the JSON object.
""".strip()


def data_compilation_single_column_transformation_plan_prompt() -> str:
    return """
You are selecting a baseline transformation for one compiled protocol-scope column.

Inputs:
- confirmed protocol discussion
- compiled causal specification
- column_name
- expected_role
- column_profile
- allowed_presets_by_kind
- optional retry_note
- optional validation_issues

Task:
- Produce one compact transformation-plan draft column JSON object that matches the provided schema exactly.

Rules:
- The schema already fixes the allowed `column` and `role`; do not invent or change them.
- Use the provided column_profile as the source of truth for kind, dtype, distinct_count, range, and known/sample values.
- Choose `preset` only from `allowed_presets_by_kind[column_profile.kind]`.
- Keep the applied `preset` aligned to the current stored kind; do not reinterpret the kind.
- Use `preferred_type` only as an advisory future raw type suggestion.
- `preferred_type_reason` must explain why that future raw type would make the column cleaner or easier to transform later.
- If `retry_note` or `validation_issues` are provided, revise the choice while keeping the same locked column and role.

Output policy:
- Output JSON only.
- No markdown.
- No explanatory prose outside the JSON object.
""".strip()


def data_compilation_review_summary_prompt() -> str:
    return """
You are preparing the user-facing review message after protocol compilation.

Inputs:
- confirmed protocol discussion
- compiled causal specification
- compiled dataset summary
- missingness_decisions
- compiled transformation plan
- transformation_suggestions
- compilation_actions
- compilation_warnings
- validation_status
- validation_issues

Task:
- Write a specific review message for the user.
- This is a review step before confirmation, not the final confirmation itself.

Content rules:
- Write for a clinician, not a data scientist. Prefer intuitive and practical language over mathematical language.
- Start with what changed in the dataset and why those changes matter for the clinical question.
- Explain which columns were kept, how the data was narrowed, and how missing values were resolved before compilation.
- Make it clear whether missing values were handled by row removal, imputation, or whether no action was needed.
- Explain the planned baseline transformations in detail, column by column when helpful, including why each transformation is needed and what it means in plain language.
- When a column has a preferred future raw type that differs from its current stored type, mention that as a non-blocking recommendation and explain why.
- Explain what validation checked in practical terms.
- Surface non-blocking warnings clearly and explain what each warning means for interpretation or trust in the analysis.
- State explicitly whether you recommend accepting the setup now or revising it before moving forward.
- If recommending acceptance with cautions, say that plainly and explain the cautions.
- Summarize the treatment, outcome, covariates, effect modifiers, and compiled dataset shape clearly.
- End by asking the user to confirm the compiled dataset, transformation plan, and validation result or say exactly what should change.
- Do not mention internal JSON, validators, or workflow implementation details.

Output JSON exactly:
{
  "assistant_message": "<specific clinician-facing review message that asks for confirmation or revision>"
}
""".strip()


def data_compilation_review_decision_prompt() -> str:
    return """
You are reviewing a compiled dataset and transformation plan with the user.

Task:
- Interpret the latest user reply to decide whether to accept the compiled setup, answer a review-time question, recompile from the original source dataset, reject and go back, or ask for clarification.

Decision rules:
- Choose `confirm` only when the user is clearly accepting the compiled dataset, transformation plan, and validation result as-is.
- Choose `answer_query` when the user is asking an explanatory question about the currently cached compiled setup and is not asking to change it.
- Choose `recompile` when the user wants same-column preprocessing, cleaning, missingness handling, row filtering, normalization, or other same-column compilation changes while keeping treatment, outcome, covariates, effect modifiers, and their roles unchanged.
- Choose `reject` when the user explicitly does not accept the setup and wants to go back, or when the requested change would alter locked column identities or roles.
- Choose `clarify` when the reply is ambiguous, incomplete, or not enough to confirm or reject safely.

Style rules:
- Keep the assistant message plain, direct, and user-facing.
- If `clarify`, ask one focused follow-up question.
- If `recompile`, summarize the requested same-column changes briefly in `recompile_request`.
- If the reply is a question, do not choose `clarify`; choose `answer_query`.
- Do not invent new protocol content.

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
- confirmed protocol discussion
- compiled causal specification
- compiled dataset summary
- missingness_decisions
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
- Explain current compiled facts only; do not invent new protocol content.
- If the question asks what changed, highlight missingness handling, row removals, retained columns, transformations, and validation in plain language.

Output JSON exactly:
{
  "assistant_message": "<specific answer to the user's review-time question>"
}
""".strip()


def data_compilation_action_decision_prompt() -> str:
    return """
You are handling a blocked compilation step after hard validation errors.

Task:
- Interpret the user's latest reply and choose the next allowed action.

Allowed actions:
- `retry_transform`: revise only the covariate/effect-modifier transformation plan.
- `retry_cleaning`: rerun same-column cleaning or value normalization while keeping the locked treatment, outcome, covariate, and effect-modifier columns unchanged.
- `revise_spec_details`: revise only same-column causal-spec details, such as treated/control literals or binary outcome event/non-event literals, while keeping the locked columns and roles unchanged.
- `revise_protocol`: the requested change would alter treatment, outcome, covariate, or effect-modifier column identity or role, or otherwise requires upstream protocol revision.
- `clarify`: the user's request is ambiguous or incomplete.

Rules:
- Locked columns and roles must not change inside this step.
- Treatment and outcome must never enter the transformation plan.
- Use `revise_protocol` when the user wants different treatment/outcome columns, different covariate/effect-modifier columns, or different feature roles.
- Use `revise_spec_details` when the user wants to keep the same locked columns but change same-column treatment/outcome details or other same-column causal-spec literals.
- Use `retry_cleaning` when the user wants value normalization, recoding, row filtering, or dtype cleanup while keeping the same locked columns.
- Use `retry_transform` when the user wants different encoding choices for covariates or effect modifiers only.

Output JSON exactly:
{
  "action": "retry_transform" | "retry_cleaning" | "revise_spec_details" | "revise_protocol" | "clarify",
  "assistant_message": "<short user-facing message>",
  "repair_request": "<brief grounded instruction summary or null>"
}
""".strip()


def data_compilation_locked_spec_revision_prompt() -> str:
    return """
You are revising a locked causal specification after validation found same-column issues.

Inputs:
- confirmed protocol discussion
- locked compiled causal specification
- compiled dataset summary
- validation issues
- user-requested repair direction

Task:
- Produce one revised causal specification JSON object that matches the provided schema exactly.

Lock rules:
- Keep the treatment column exactly unchanged.
- Keep the outcome column exactly unchanged.
- Keep the covariate columns exactly unchanged.
- Keep the effect-modifier columns exactly unchanged.
- Keep treatment/outcome/covariate/effect-modifier roles exactly unchanged.
- Do not add, remove, rename, or re-role locked columns.
- Only revise same-column causal-spec details that are grounded by the inputs, such as treated/control literals, binary outcome event/non-event literals, or continuous-outcome clipping details.
- Do not invent new columns, values, cohort rules, or unsupported remappings.

Output policy:
- Output JSON only.
- No markdown.
- No explanatory prose outside the JSON object.
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
- Only choose a non-default preset when `transformation_instructions` or retry guidance make it necessary and the preset is still allowed for the current kind.
- Do not emit dataset-change actions.
- Do not emit missing-data logic.
- If `retry_note` is present, use it only to correct malformed, incomplete, or previously incompatible draft output while keeping the same eligible columns and roles.

Output policy:
- Output JSON only.
- No markdown.
- No explanatory prose outside the JSON object.
""".strip()


def single_column_transform_prompt() -> str:
    return """
You are selecting a strict datatype-driven transformation for one covariate or effect modifier.

Inputs:
- transformation_instructions
- compiled_causal_specification
- column_name
- expected_role
- column_profile
- allowed_presets_by_kind
- optional retry_note

Task:
- Produce one JSON object with a single `column` entry.
- Keep the provided column name and role unchanged.
- Never refer to treatment or outcome.

Rules:
- Output exactly:
  - `column`
  - `role`
  - `preset`
  - `preferred_type`
  - `preferred_type_reason`
- Choose `preset` only from `allowed_presets_by_kind[column_profile.kind]`.
- The applied `preset` must stay aligned to the current stored kind.
- Use `preferred_type` only as a saved future recommendation.
- Prefer the minimal current-kind-compatible transform unless instructions require a different allowed preset.
- Do not emit dataset-change actions.
- Do not emit missing-data logic.
- If `retry_note` is present, use it only to correct malformed, incomplete, or incompatible prior output.

Output policy:
- Output JSON only.
- No markdown.
- No explanatory prose outside the JSON object.
""".strip()
