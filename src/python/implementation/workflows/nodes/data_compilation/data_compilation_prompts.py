from __future__ import annotations


def data_compilation_node_info() -> str:
    return (
        "Compile-only data preparation stage. It turns a confirmed protocol discussion into "
        "a compiled causal spec, a protocol-scope cleaned dataset, and a baseline "
        "transformation plan, then asks the user to confirm before publishing those outputs "
        "to orchestrator state."
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
You are preparing one corrective SQL-oriented cleaning pass after protocol compilation found grounded data discrepancies.

Inputs:
- confirmed protocol discussion
- confirmed protocol cleaning instructions
- compiled causal specification
- compiled dataset summary
- transformation validation issues

Rules:
- Return only user-intent text for a SQL data manipulation tool.
- Keep the dataset inside the compiled protocol scope.
- Only fix discrepancies that are explicitly grounded by the compiled causal specification, compiled dataset summary, and listed validation issues.
- Prefer safe type corrections, grounded recoding, and removal of invalid treatment or outcome rows over speculative remapping.
- Never add treatment or outcome transformations to the transform plan; fix the data instead when needed.
- Do not invent new cohort rules, new columns, or unsupported category merges.
- Do not include validation, modeling, plotting, or non-SQL tasks.
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
- column_name
- expected_role
- column_profile

Task:
- Produce one compact transformation-plan draft column JSON object that matches the provided schema exactly.

Rules:
- The schema already fixes the allowed `column` and `role`; do not invent or change them.
- Use the provided column_profile as the source of truth for kind, dtype, missingness, distinct_count, range, and known/sample values.
- NUMERIC columns with very small distinct_count can represent binary, categorical, or ordinal coded fields; choose a discrete encoding when grounded by the profile.
- Use `map_binary` only when you can provide a grounded `mapping`.
- Use `map_ordinal` only when you can provide a grounded `order`.
- Prefer conservative, broadly safe encodings.
- Avoid `drop` unless there is a grounded reason to exclude the feature.

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

Task:
- Write a specific review message for the user.
- This is a review step before confirmation, not the final confirmation itself.

Content rules:
- Focus first on the data preparation changes made for causal modeling.
- Explain which protocol-scope columns were retained and how the dataset was narrowed for modeling.
- Explicitly mention grounded row removals, dropped invalid treatment/outcome values, and any corrective cleaning that was applied.
- Surface non-blocking warnings clearly as warnings.
- Summarize the planned baseline transformations in readable language with more detail than the treatment/outcome recap.
- Summarize the treatment, outcome, covariates, and effect modifiers clearly.
- Summarize the compiled dataset shape in readable language.
- Ask the user to confirm the compiled dataset and transformation plan or say exactly what should change.
- Do not mention internal JSON, validators, or workflow implementation details.

Output JSON exactly:
{
  "assistant_message": "<specific review message that asks for confirmation or revision>"
}
""".strip()


def data_compilation_review_decision_prompt() -> str:
    return """
You are reviewing a compiled dataset and transformation plan with the user.

Task:
- Interpret the latest user reply to decide whether the compiled setup is accepted, rejected for revision, or still unclear.

Decision rules:
- Choose `confirm` only when the user is clearly accepting the compiled dataset and transformation plan as-is.
- Choose `revise` when the user is asking to change the compiled dataset scope, treatment, outcome, covariates, effect modifiers, protocol filters, normalization, or planned encodings.
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


__all__ = [
    "data_compilation_causal_spec_prompt",
    "data_compilation_cleaning_instructions_prompt",
    "data_compilation_discrepancy_repair_prompt",
    "data_compilation_node_info",
    "data_compilation_review_decision_prompt",
    "data_compilation_review_summary_prompt",
    "data_compilation_single_column_transformation_plan_prompt",
    "data_compilation_transformation_plan_prompt",
]
