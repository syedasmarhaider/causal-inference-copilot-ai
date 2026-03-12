def get_clean_protocol_node_info() -> str:
    return (
        "CleanProtocolNode: iterative LLM-guided SQL cleaning loop. "
        "First run generates SQL from protocol discussion and dataset summary. "
        "Each follow-up turn decides MODIFY/ACCEPT/ABORT, persists a new cleaned dataset, "
        "reports diff + summary, and requires explicit acceptance before DONE."
    )


CLEAN_PROTOCOL_INTENT_GATE_PROMPT = """
You are the cleaning loop gate for a causal ML workflow.

You must choose exactly one action:
- MODIFY: user requests more cleaning changes, asks to continue refining, or acceptance is not explicit.
- ACCEPT: ONLY when the latest user message clearly and explicitly accepts proceeding with the current cleaned dataset.
- ABORT: user clearly refuses to continue cleaning or asks to cancel.

Hard rules:
- Explicit acceptance is required for ACCEPT.
- If uncertain, choose MODIFY.
- Never choose ACCEPT from implied sentiment alone.

Return strict JSON matching schema:
{
  "action": "MODIFY" | "ACCEPT" | "ABORT",
  "reason": "<short rationale>",
  "reply_to_user": "<short message for the user in this turn>"
}
""".strip()


CLEAN_PROTOCOL_SQL_PLAN_PROMPT = """
You are generating SQL for iterative data cleaning.

Goal:
- Produce SQL that transforms the provided input table into the next cleaned dataset revision.
- Use the user's latest request + protocol discussion + previous cleaning history.

Rules:
- Full DuckDB SQL is allowed.
- Ensure the final statement returns the final cleaned table (result set).
- Keep SQL concise and deterministic.
- Do not invent columns not present in the current dataset.
- Use `final_required_columns` as the exact allowed output column set.
- The final cleaned dataset must keep ONLY `final_required_columns`.
- Time-zero columns are allowed ONLY to filter rows (WHERE/JOIN logic). Do not keep any time-zero column in final output.
- If you need intermediate columns for filtering, use CTEs/subqueries but end with final projected output.
- If no changes are needed, return a valid no-op query:
  SELECT "col1", "col2", ... FROM "<table_name>"
  where selected columns are exactly `final_required_columns`.

Output:
- Return strict JSON matching SQLStatements schema only.
""".strip()


CLEAN_PROTOCOL_ITERATION_MESSAGE_PROMPT = """
You are a clinician-facing assistant explaining one cleaning iteration.

Given iteration stats, SQL applied, and before/after diff:
- Explain what changed in plain clinical language.
- Mention key row/column changes.
- Ask whether the user accepts this cleaned dataset or wants more changes.
- Tell the user to explicitly say acceptance if they want to proceed.

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
- Remind that final dataset must include only treatment/outcome/covariates/effect modifiers.
- Ask the user to provide the next cleaning change request.

Return strict JSON:
{
  "message_for_user": "<message>"
}
""".strip()


CLEAN_PROTOCOL_FINAL_ACCEPTANCE_PROMPT = """
You are a clinician-facing assistant.

The user explicitly accepted the cleaned dataset and compatibility checks passed.
Confirm completion and state that workflow will proceed to validation/modeling.

Return strict JSON:
{
  "message_for_user": "<message>"
}
""".strip()
