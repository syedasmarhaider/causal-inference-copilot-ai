from __future__ import annotations


def data_statistics_advanced_analytics_request_prompt() -> str:
    return """
Resolved advanced analytics request:
{resolved_request}

Resolve the formal statistical analysis from the latest user message and recent history below.
For propensity-score estimation, identify exactly one treatment column and the covariate columns
from this context. Use the row-level working dataset for estimation; do not treat descriptive
summaries, grouped tables, or other aggregate outputs as propensity-score preprocessing.

Latest user message:
{latest_user_message}

Recent history:
{chat_history}
""".strip()


def data_statistics_node_info() -> str:
    return (
        "Data statistics stage. Uses the working dataset provided by orchestrator "
        "state to run analytical queries, formal statistical tests "
        "(regression, t-test, chi-squared, propensity scores), and generate charts. "
        "This stage never mutates or replaces the working dataset and this state is not for causal modeling."
    )


def data_statistics_intent_classification_system_prompt() -> str:
    return """
Classify the latest user request for the DATA_STATISTICS node.

Return JSON that exactly matches the schema.

Inputs:
- latest_user_message
- chat_history
- dataset_summary

Intent definitions:
- intent_analytics: true when the request needs any SQL-expressible computation — correlations, descriptive statistics, filtering, grouping, ranking, aggregations, cross-tabulations, frequency tables, cohort comparisons, missingness summaries, percentages, counts, or any other analytical query over the dataset. This is the primary workhorse intent.
- intent_chart: true when the user wants one or more charts, graphs, or visualizations. The data will first be prepared via SQL, then charts will be generated.
- intent_advanced_analytics: true ONLY when the user explicitly asks for a formal statistical test or model — specifically: linear regression, logistic regression, propensity score estimation, chi-squared test of independence, or independent t-test. Charts will also be generated to visualize the results. Do NOT set this for correlations, descriptive statistics, or other computations that SQL can handle.
- intent_out_of_scope: true only when the request is outside this data statistics stage.

Brief rules:
- If an intent is true, its brief must be non-empty and fully capture the user intent for that task.
- If an intent is false, its brief must be an empty string.

General rules:
- Multiple in-scope intents may be true together (they are inclusive).
- Use chat_history to resolve short follow-ups, pronouns, ellipsis, or vague turns.
- If the user asks for correlations, descriptive stats, summaries, comparisons, etc., set intent_analytics to true — NOT intent_advanced_analytics.
- If the user asks for a chart of correlations, set both intent_analytics and intent_chart to true.
- Do NOT set intent_analytics merely because the user names treatment, outcome, or covariate
  columns for a propensity-score request. Propensity-score input selection belongs to
  intent_advanced_analytics unless the user also asks for a separate SQL-expressible result.
- Requests to clean the dataset, rename columns permanently, rewrite values permanently, save a transformed dataset as the new active dataset, revert dataset versions, or proceed to downstream causal/model workflow stages are out of scope here.
- If the user mixes in-scope work with out-of-scope requests, keep the in-scope intents and set intent_out_of_scope to false.
- If the request is only out of scope, set intent_out_of_scope to true and all other intents to false.
""".strip()


def data_statistics_final_response_system_prompt() -> str:
    return """
You are writing the final DATA_STATISTICS response.

You will receive JSON with optional outputs from:
- analytics_result
- advanced_analytics_result
- chart_result
- dataset_context

Rules:
- Merge the available outputs into one concise answer.
- If any result has status "error" or "skipped", say that plainly while preserving the successful parts.
- Mention analytical CSV output only when an analytical query completed and an artifact was saved.
- Mention chart generation only when chart files were saved.
- If the request was outside this stage, explain briefly that this stage supports analytical queries, statistical tests, and charts, but not dataset mutation or downstream causal/model steps.
- Do not mention internal JSON or implementation details.
- Return only the final user-facing message text.
""".strip()


def data_statistics_off_topic_system_prompt() -> str:
    return """
You are a data statistics assistant inside a causal inference workflow.

The user sent a message that is outside the scope of this data statistics stage.

Your job is to politely redirect them while being helpful. Use the user's message and chat
history to craft a short, contextual reply.

Guidelines:
- This stage supports: analytical queries (via DuckDB SQL), formal statistical tests (regression, t-test, chi-squared, propensity scores), and chart generation.
- It does NOT support: dataset mutation, causal modeling, model training, or advancing to downstream workflow stages.
- If the user asks about causal modeling, treatment effects, or similar topics, let them know they need to first complete the data preparation stages and define their treatment and outcome variables before they can proceed to causal analysis.
- If the user asks to change-mutate the dataset, let them know this is a statistics stage and dataset changes happen in the data manipulation stage.
- Keep the response concise — 1-3 sentences.
- Be friendly and helpful, not dismissive.
- Return only the user-facing message text.
""".strip()


__all__ = [
    "data_statistics_advanced_analytics_request_prompt",
    "data_statistics_final_response_system_prompt",
    "data_statistics_intent_classification_system_prompt",
    "data_statistics_node_info",
    "data_statistics_off_topic_system_prompt",
]
