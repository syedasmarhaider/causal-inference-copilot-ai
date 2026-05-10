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
- Negative-control outcome, when present, must be binary or continuous only and must
  remain distinct from treatment, primary outcome, identifier, covariates, and effect modifiers.
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
- optional negative_control_outcome column profile
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
  - negative_control_outcome kind/event/non_event/continuous metadata only when the
    locked causal draft includes a negative_control_outcome column
  - experiment_type
- Use exact grounded values from the protocol discussion and the provided treatment/outcome profiles.
- Do not invent new columns, covariates, effect modifiers, timing rules, or post-treatment assumptions.
- Do not invent a negative-control outcome. If the locked causal draft has
  negative_control_outcome: null, return null for negative_control_outcome semantics.
- If the locked causal draft has a negative_control_outcome column, resolve it with the
  same binary/continuous outcome rules as the primary outcome using its provided profile.
- If compile_feedback is present, fix that specific semantic problem directly.
- If the outcome is continuous, only choose `continuous` when the outcome profile is numeric and the protocol supports it.

Output policy:
- Output JSON only.
- No markdown.
- No explanatory prose outside the JSON object.
""".strip()


def data_compilation_filter_plan_prompt() -> str:
    return """
You are devising a target-population filtering plan for a tabular causal-analysis dataset.

Inputs:
- target_population text from the locked causal draft
- locked causal draft
- authoritative dataset summary
- current dataframe columns
- revised_instructions text from the latest user review feedback, when present

Task:
- Write one complete natural-language instruction for a data manipulation tool to filter
  the dataframe to the target population.

Rules:
- Use the dataset summary to ground the filter plan in actual available columns, kinds,
  missingness, and observed values.
- Do not invent columns, values, timing rules, or cohort criteria that are not supported
  by the target population text and dataset summary.
- Preserve all draft-selected causal columns: treatment, outcome, negative-control outcome
  when present, covariates, effect modifiers, and an existing identifier column.
- If revised_instructions are present, incorporate them into the filtering plan and give them priority.
- The instruction is for a data manipulation tool, not raw SQL.

Output policy:
- Output text only.
- Do not output JSON.
- Do not output SQL.
- Do not use markdown or code fences.
""".strip()


def _data_compilation_cleaning_instruction_output_policy() -> str:
    return """
Output JSON exactly:
{
  "action": "run_instruction" | "done",
  "instruction": "<one clear natural-language instruction for the data manipulation tool, or null when done>",
  "reason": "<short reason grounded in the current protocol and dataset state>"
}
""".strip()


def data_compilation_transformation_instruction_prompt() -> str:
    return f"""
You are planning the first adaptive data-manipulation instruction for protocol compilation.

Inputs:
- confirmed protocol discussion
- confirmed protocol cleaning instructions
- optional high_priority_review_recompile_request
- locked causal draft
- effective identifier column
- required final columns
- expected_role_by_column
- current table name
- compact current dataset summary
- prior validation feedback, when present

Task:
- Return one natural-language instruction for the data manipulation tool, or `done`.
- This first instruction should handle protocol-grounded transformations before missingness:
  type normalization, boolean/category recoding, value normalization, same-column replacements,
  and conditional recoding needed by the protocol.

Rules:
- The instruction is for a data manipulation tool, not raw SQL output.
- Ask for one coherent transformation operation only.
- If high_priority_review_recompile_request is present, treat it as the most important
  user correction for this recompilation pass while staying within this tool scope.
- If protocol discussion, cleaning instructions, draft roles, or dataset evidence conflict,
  follow the highest-priority user requirement that is still compatible with the locked
  causal draft and dataset evidence. If no compatible interpretation is explicit, choose
  the most conservative protocol-safe interpretation and name the contradiction in `reason`.
- If the user did not explicitly describe a needed transformation, still perform
  conservative protocol-safe normalization when the current dataset state clearly needs it,
  and explain that dataset-grounded decision in `reason`.
- Keep all locked causal draft columns and roles available.
- Do not drop, null, duplicate, or regenerate the effective identifier column.
- Do not perform missingness handling here unless the high-priority request explicitly
  requires a transformation that also resolves a missing-code representation.
- Return `done` if no protocol-grounded transformation is needed.

{_data_compilation_cleaning_instruction_output_policy()}
""".strip()


def data_compilation_missingness_instruction_prompt() -> str:
    return f"""
You are planning the mandatory missingness data-manipulation instruction for protocol compilation.

Inputs:
- confirmed protocol discussion
- confirmed protocol cleaning instructions
- optional high_priority_review_recompile_request
- locked causal draft
- effective identifier column
- required final columns
- expected_role_by_column
- current table name
- compact current dataset summary
- required-column missing counts
- executed cleaning instructions
- prior validation feedback, when present

Task:
- Return one natural-language instruction for the data manipulation tool, or `done`.
- This instruction should resolve protocol-scope missingness for treatment, outcome,
  negative-control outcome, covariates, and effect modifiers.

Rules:
- The instruction is for a data manipulation tool, not raw SQL output.
- If required-column missing counts are all zero, return `done`.
- If missingness remains, prefer the user-intended handling from the protocol discussion,
  cleaning instructions, and high-priority review request when present.
- If protocol discussion, cleaning instructions, draft roles, or dataset evidence conflict,
  follow the highest-priority user requirement that is still compatible with the locked
  causal draft and dataset evidence. If no compatible interpretation is explicit, choose
  the most conservative protocol-safe missingness handling and name the contradiction in
  `reason`.
- If missingness remains and the user gave no explicit missingness rule, choose the
  conservative protocol-safe handling from the column role and dataset state, and explain
  that default decision in `reason` so it can be reported to the user.
- Ask for one coherent missingness operation only.
- Keep all locked causal draft columns and roles available.
- Do not drop, null, duplicate, or regenerate the effective identifier column.
- Do not invent imputation values, row filters, or missingness rules not grounded in the inputs.

{_data_compilation_cleaning_instruction_output_policy()}
""".strip()


def data_compilation_adaptive_cleaning_instruction_prompt() -> str:
    return f"""
You are planning one adaptive cleanup data-manipulation instruction for protocol compilation.

Inputs:
- confirmed protocol discussion
- confirmed protocol cleaning instructions
- optional high_priority_review_recompile_request
- locked causal draft
- effective identifier column
- required final columns
- expected_role_by_column
- current table name
- compact current dataset summary
- required-column missing counts
- executed cleaning instructions
- prior validation feedback, when present

Task:
- Return one natural-language instruction for the data manipulation tool, or `done`.
- This cleanup instruction should perform the next user-intended cleaning correction that
  remains after earlier transformation and missingness steps.

Rules:
- The instruction is for a data manipulation tool, not raw SQL output.
- Ask for one coherent cleanup operation only.
- Do not repeat any executed cleaning instruction.
- If high_priority_review_recompile_request is present, prioritize that correction while
  preserving the locked protocol scope.
- If protocol discussion, cleaning instructions, draft roles, or dataset evidence conflict,
  follow the highest-priority user requirement that is still compatible with the locked
  causal draft and dataset evidence. If no compatible interpretation is explicit, choose
  the most conservative protocol-safe cleanup and name the contradiction in `reason`.
- If the user gave no explicit cleanup instruction, still perform conservative
  protocol-safe cleanup when the current dataset state clearly needs it, and explain that
  default decision in `reason` so it can be reported to the user.
- Keep all locked causal draft columns and roles available.
- Do not drop, null, duplicate, or regenerate the effective identifier column.
- Do not invent protocol rules, causal roles, columns, timing assumptions, or unsupported
  value mappings.
- Return `done` if no further grounded cleanup is needed.

{_data_compilation_cleaning_instruction_output_policy()}
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
- Never include treatment, primary outcome, or negative-control outcome in the transformation plan.
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
- Summarize the treatment, outcome, negative-control outcome status, covariates, effect modifiers, and compiled dataset shape clearly.
- If no valid negative-control outcome was provided or identified, surface that warning and state that CATE negative-control refutation will not be performed.
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
- Never refer to negative-control outcome.

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
