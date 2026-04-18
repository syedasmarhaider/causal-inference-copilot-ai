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


def data_compilation_cleaning_instructions_prompt() -> str:
    return """
You are preparing first-pass SQL-oriented cleaning instructions before protocol compilation.

Inputs:
- confirmed protocol discussion
- confirmed protocol cleaning instructions
- authoritative source dataset summary

Rules:
- Return only user-intent text for a SQL data manipulation tool.
- Keep it executable and concrete.
- Include only grounded row filters, normalization steps, type fixes, and column handling derived from the confirmed protocol and confirmed protocol cleaning instructions.
- Do not compile the causal spec yourself; only prepare the dataset so later compilation can use it.
- Preserve any columns that may still be needed for later protocol compilation unless the confirmed cleaning instructions explicitly narrow them already.
- If treatment is binary, normalize its labels only when grounded by the confirmed protocol.
- If outcome is binary, normalize its labels only when grounded by the confirmed protocol.
- Do not invent filters, columns, or value mappings.
- Do not include validation, modeling, plotting, or non-SQL tasks.
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


def data_compilation_compile_retry_guidance_prompt() -> str:
    return """
Additional grounded repair guidance for the next full compilation attempt:
- Treat the following feedback as required source-dataset fixes before a safe transformation plan can be produced.
- Apply only changes that are explicitly grounded by the confirmed protocol, the causal draft, and the provided repair text.
- Keep the dataset within the confirmed protocol scope and preserve all draft columns.
- Prefer concrete dtype cleanup, value normalization, recoding, and missingness handling over speculative changes.
""".strip()


def data_compilation_transformation_retry_guidance_prompt() -> str:
    return """
Additional grounded retry guidance for the next causal-spec and transformation attempt:
- Use the following repair text to revise the compiled causal specification details and baseline transformations without changing locked column identities or roles.
- Keep treatment, outcome, covariates, and effect modifiers locked to the confirmed draft columns.
- Revise only same-column causal-spec details or covariate/effect-modifier encoding choices that are grounded by the repair text.
""".strip()


_PLAN_GUARDRAILS = """
Transformation-plan safety rules:
- Build the plan only for covariates and effect modifiers from the compiled causal specification.
- Never include treatment or outcome in the transformation plan.
- Use the compiled dataset summary as the source of truth for column types.
- Prefer conservative, broadly safe encodings.
- Avoid row dropping unless it is clearly grounded and necessary.
- Prefer explicit missingness handling when possible.
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
- optional repair_request
- optional validation_issues

Task:
- Produce one compact transformation-plan draft JSON object that matches the provided schema exactly.
- The runtime will expand the draft into the final TransformPlan using safe defaults.

Defaults:
- NUMERIC baseline features usually default to `num_standard`.
- NUMERIC columns with very small distinct_count can be numeric-coded binary, categorical, or ordinal fields; choose a discrete encoding when the profile supports that interpretation.
- Binary categorical or boolean baseline features usually default to `map_binary` when the mapping is grounded.
- Multi-category baseline features usually default to `cat_onehot`.
- DATETIME baseline features may use `datetime_epoch_seconds` when the timestamp itself is intended as a baseline feature; otherwise prefer `drop`.
- OTHER features should usually be `drop` unless there is a grounded reason to keep them.

Role rules:
- Use `role="covariate"` exactly for compiled causal-spec covariates.
- Use `role="effect_modifier"` exactly for compiled causal-spec effect modifiers.
- Do not infer new roles.
- Include every eligible column exactly once.
- Do not include any column outside `eligible_columns`.
- The number of entries in `columns` must equal `required_plan_column_count`.
- For each entry, `role` must exactly match `expected_role_by_column[column]`.
- Use `map_binary` only when you can provide a grounded `mapping`.
- Use `map_ordinal` only when you can provide a grounded `order`.
- Use the confirmed protocol discussion and compiled causal specification to resolve semantic ambiguity, especially for low-cardinality numeric codes, ordinal categories, baseline datetime fields, and columns that could be either dropped or retained.
- If repair_request or validation_issues are provided, revise the plan to address them while keeping the same eligible columns and roles.

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
- optional repair_request
- optional validation_issues

Task:
- Produce one compact transformation-plan draft column JSON object that matches the provided schema exactly.

Rules:
- The schema already fixes the allowed `column` and `role`; do not invent or change them.
- Use the confirmed protocol discussion and compiled causal specification to understand the semantic role of the column within the study design.
- Use the provided column_profile as the source of truth for kind, dtype, missingness, distinct_count, range, and known/sample values.
- NUMERIC columns with very small distinct_count can represent binary, categorical, or ordinal coded fields; choose a discrete encoding when grounded by the profile.
- Use `map_binary` only when you can provide a grounded `mapping`.
- Use `map_ordinal` only when you can provide a grounded `order`.
- Prefer conservative, broadly safe encodings.
- Avoid `drop` unless there is a grounded reason to exclude the feature.
- If repair_request or validation_issues are provided, revise the encoding choice to address them while keeping the same locked column and role.

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
- compiled transformation plan
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
- Explain which columns were kept, how the data was narrowed, and any grounded row removals or corrective cleaning that happened.
- Explain the planned baseline transformations in detail, column by column when helpful, including why each transformation is needed and what it means in plain language.
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
- Interpret the latest user reply to decide whether the compiled and validated setup is accepted, rejected for revision, or still unclear.

Decision rules:
- Choose `confirm` only when the user is clearly accepting the compiled dataset, transformation plan, and validation result as-is.
- Choose `revise` when the user is asking to change the compiled dataset scope, treatment, outcome, covariates, effect modifiers, protocol filters, normalization, planned encodings, or validation outcome.
- Choose `clarify` when the reply is ambiguous, incomplete, or not enough to confirm or reject safely.

Style rules:
- Keep the assistant message plain, direct, and user-facing.
- If `clarify`, ask one focused follow-up question.
- Do not invent new protocol content.

Output JSON exactly:
{
  "action": "confirm" | "revise" | "clarify",
  "assistant_message": "<short user-facing message>"
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
You are compiling a minimal transformation draft for covariates and effect modifiers only.

Inputs:
- transformation_instructions
- compiled_causal_specification
- scoped_dataset_summary
- eligible_columns
- expected_role_by_column
- required_plan_column_count
- optional repair_request

Task:
- Produce one JSON object with a `columns` array.
- Include every eligible column exactly once.
- Never include treatment or outcome.
- Use `decision="plan"` when the current summarized data can be transformed safely as-is.
- Use `decision="dataset_change"` when the source dataset must be changed before a safe grounded encoding can be chosen.

Planning rules:
- Prefer `passthrough` when the column is already analysis-ready.
- Only choose a real preset when the summary or transformation_instructions make it necessary.
- Use the scoped dataset summary as the source of truth for data kind, missingness, distinct_count, and grounded known values.
- Never guess label mappings or ordinal order for numeric-coded categories.
- If categorical or ordinal semantics are required but not grounded by the summary, emit `decision="dataset_change"` instead of inventing mappings.
- Only use `map_binary` when you can provide a grounded `mapping`.
- Only use `map_ordinal` when you can provide a grounded `order`.

Dataset-change rules:
- `strict_requirement` must clearly say that the source dataset must change first.
- `required_dataset_change` must say exactly what needs to be changed in the dataset.
- `addtional_suggestions_to_user` should give a friendly next step for the user.

Output policy:
- Output JSON only.
- No markdown.
- No explanatory prose outside the JSON object.
""".strip()


def single_column_transform_prompt() -> str:
    return """
You are selecting a minimal transformation for one covariate or effect modifier.

Inputs:
- transformation_instructions
- compiled_causal_specification
- column_name
- expected_role
- column_profile

Task:
- Produce one JSON object with a single `column` entry.
- Keep the provided column name and role unchanged.
- Never refer to treatment or outcome.

Rules:
- Prefer `decision="plan"` with `preset="passthrough"` when the column is already analysis-ready.
- Choose a real preset only when the column actually needs transformation.
- Use the column profile as the source of truth for dtype, inferred kind, missingness, distinct_count, and grounded known values.
- Never guess label mappings or ordinal order for numeric-coded categories.
- If safe categorical or ordinal semantics cannot be grounded from the profile and instructions, emit `decision="dataset_change"`.

Dataset-change rules:
- `strict_requirement` must clearly say that the source dataset must change first.
- `required_dataset_change` must say exactly what needs to be changed in the dataset.
- `addtional_suggestions_to_user` should give a friendly next step for the user.

Output policy:
- Output JSON only.
- No markdown.
- No explanatory prose outside the JSON object.
""".strip()
