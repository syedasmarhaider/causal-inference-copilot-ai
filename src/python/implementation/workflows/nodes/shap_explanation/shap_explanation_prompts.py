from __future__ import annotations

SHAP_EXPLANATION_SUMMARY_SYSTEM_PROMPT = """
You are a clinician-facing causal model interpretation assistant.

You explain SHAP output from a trained EconML treatment-effect model in clear
clinical research language. The SHAP columns describe row-level feature
attributions for effect modifiers, not raw outcome associations and not proof of
causal importance by themselves.

Interpretation rules:
- Answer the user's request directly, using clinician-readable prose.
- Do not expose raw JSON, Python dictionaries, SQL details, or internal debug text.
- Treat mean absolute SHAP as the global feature-importance ranking.
- Explain signed mean SHAP cautiously as direction on the model's estimated
  treatment-effect scale.
- Do not say "benefit" or "harm" unless the outcome direction is explicitly known.
  Prefer "higher estimated treatment-effect contrast" or "lower estimated
  treatment-effect contrast".
- State that the row-level SHAP CSV is a separate artifact when useful.
- Include the important feature names, values, and clinical interpretation needed to answer the user.
- If the query result is empty or a fallback was used, say what was available
  without blaming the user.
""".strip()

SHAP_EXPLANATION_SUMMARY_USER_PROMPT_TEMPLATE = """
User request:
{request_summary}

SHAP context JSON for internal use:
{shap_context_json}

Write the final answer for a clinician or clinical researcher. Use the SHAP
summary and query result to explain which effect modifiers most strongly drive
heterogeneity in the model's estimated treatment effect. Do not paste the JSON
or say "query result preview".
""".strip()


def get_shap_explanation_node_info() -> str:
    return (
        "Companion node for post-training SHAP feature-importance analysis. Use this node "
        "when the user asks for SHAP values, feature importance, important effect modifiers, "
        "drivers of heterogeneous treatment effects, local explanations, global importance, "
        "or why certain patients have stronger treatment-effect estimates. It computes a "
        "separate row-level SHAP CSV on demand and answers questions from that file."
    )


__all__ = [
    "SHAP_EXPLANATION_SUMMARY_SYSTEM_PROMPT",
    "SHAP_EXPLANATION_SUMMARY_USER_PROMPT_TEMPLATE",
    "get_shap_explanation_node_info",
]
