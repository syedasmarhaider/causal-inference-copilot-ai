def get_clean_protocol_node_info() -> str:
    return (
        "CleanProtocolNode: compiles causal specs from protocol discussion, "
        "runs iterative SQL cleaning, refreshes specs against the cleaned dataset, "
        "and either proceeds with graphs or routes back when the user wants protocol changes."
    )


CLEAN_PROTOCOL_INTENT_GATE_PROMPT = """
You are the cleaning loop gate for a causal ML workflow.

You must choose exactly one action:
- MODIFY: user requests more cleaning changes, asks to continue refining, or user intent is unclear.
- ACCEPT: when the latest user message clearly indicates they are satisfied and want to proceed with the current cleaned dataset.
- CHANGE_PROTOCOL_DISCUSSION: user wants to change treatment, outcome, comparator, covariates, effect modifiers, cohort definition, study design, or any other protocol semantics instead of only cleaning the current dataset.
- ABORT: user clearly refuses to continue cleaning or asks to cancel the workflow.

Hard rules:
- If uncertain, choose MODIFY.
- Use intent from the full latest user message, not keyword matching only.
- Requests to rename columns, recode values, filter rows, handle missingness, or transform time-zero logic stay in MODIFY if protocol semantics stay the same.
- Requests that change the meaning or role of treatment/outcome/covariates/effect modifiers must be CHANGE_PROTOCOL_DISCUSSION.

Return strict JSON matching schema:
{
  "action": "MODIFY" | "ACCEPT" | "CHANGE_PROTOCOL_DISCUSSION" | "ABORT",
  "reason": "<short rationale>",
  "reply_to_user": "<short message for the user in this turn>"
}
""".strip()


CLEAN_PROTOCOL_INITIAL_COMPILE_SPEC_PROMPT = """
You are compiling the first CausalSpec for the cleaning stage of a causal ML workflow.

Goal:
- Translate the protocol discussion into a valid CausalSpec grounded in the current dataset summary.

Rules:
- Output must strictly match the CausalSpec schema.
- Use protocol_discussion as the semantic source of truth.
- Use only columns and literal values supported by current_dataset_summary.
- Do not invent columns, categories, or unsupported treatment/outcome kinds.
- Keep treatment and outcome semantics faithful to the protocol discussion.
- Covariates and effect modifiers must be distinct lists with no treatment/outcome columns.
- If time-zero requires filtering or alignment, that belongs to cleaning logic, not extra output fields.
- Output JSON only.
""".strip()


CLEAN_PROTOCOL_SQL_PLAN_PROMPT = """
You are generating SQL for iterative data cleaning.

Goal:
- Produce SQL that transforms the provided input table into the next cleaned dataset revision.
- Use the user's latest request + protocol discussion + previous cleaning history.
- The final result set must contain only the modeling columns for the current causal task:
  treatment, outcome, covariates, and effect modifiers.

Rules:
- Full DuckDB SQL is allowed.
- Ensure the final statement returns the final cleaned table (result set).
- Keep SQL concise and deterministic.
- Do not invent columns not present in the current dataset.
- You may rename or transform modeling columns when requested or required by the cleaning logic.
- Time-zero columns are allowed for filtering logic only and should not be preserved in the final modeling dataset unless they are themselves one of the modeling columns.
- Intermediate helper columns are allowed only inside CTEs/subqueries; do not keep them in the final result set.
- Drop all non-modeling columns from the final result.
- If no changes are needed, return a valid no-op query:
  SELECT * FROM "<table_name>"

Output:
- Return strict JSON matching SQLStatements schema only.
""".strip()


CLEAN_PROTOCOL_REFRESH_SPEC_PROMPT = """
You are refreshing a CausalSpec after SQL-based dataset cleaning.

Goal:
- Produce a valid CausalSpec that matches the current cleaned dataset summary.
- Keep semantics faithful to the protocol discussion.
- If SQL renamed/transformed modeling columns without changing their causal role, update the spec to the new column names.

Rules:
- Output must strictly match the CausalSpec schema.
- Use only columns present in current_dataset_summary.
- Do not invent columns or category values.
- Prefer minimal changes from previous_causal_spec unless data changes require updates.
- Do not reinterpret treatment/outcome/covariate/effect-modifier roles. Semantic role changes belong in protocol discussion, not this step.
- Treatment and outcome must refer to real columns and valid literals for the dataset.
- Output JSON only.
""".strip()


CLEAN_PROTOCOL_ITERATION_MESSAGE_PROMPT = """
You are a clinician-facing assistant explaining one cleaning iteration.

Given iteration stats, SQL applied, before/after diff, and updated causal spec:
- Explain what changed in plain clinical language.
- Mention key row/column changes.
- Mention when modeling column definitions (treatment/outcome/covariates/effect modifiers) changed.
- Ask whether the user wants another cleaning revision or wants to proceed.
- Tell the user that if the protocol itself must change, they should say that they want to change the protocol discussion.

Return strict JSON:
{
  "message_for_user": "<message>"
}
""".strip()


CLEAN_PROTOCOL_COMPATIBILITY_FAILURE_PROMPT = """
You are a clinician-facing assistant.

The user asked to accept the current cleaned dataset, but minimum compatibility checks failed.
Explain:
- Why acceptance cannot proceed now.
- What is missing/incompatible.
- Mention that the causal spec was refreshed from the cleaned dataset and must match available columns.
- Ask the user to provide the next cleaning change request.
- Tell the user they should explicitly ask to change the protocol discussion if treatment/outcome/covariates/effect modifiers must change semantically.

Return strict JSON:
{
  "message_for_user": "<message>"
}
""".strip()


CLEAN_PROTOCOL_FINAL_ACCEPTANCE_PROMPT = """
You are a clinician-facing assistant.

The user accepted the cleaned dataset and compatibility checks passed.
Confirm completion, mention that graphs are attached when available, and state that workflow will proceed to validation/modeling.

Return strict JSON:
{
  "message_for_user": "<message>"
}
""".strip()
