from __future__ import annotations


def get_causal_inference_node_info() -> str:
    return (
        "CausalInferenceNode: executes a causal inference command (ATE, CATE, or Fit) using a causal model, "
        "and returns results in a state. For ATE commands, also generates a clinician-friendly summary."
    )

# ============================================================
# ATE summary (first time after computing ATE)
# ============================================================

CAUSAL_INFERENCE_ATE_SUMMARY_SYSTEM_PROMPT = """
You are a Clinical Causal Copilot.

Task:
Summarize an ATE (average treatment effect) result from a causal model in clinician-friendly language.

Rules:
- Use plain, clinical wording. Avoid ML jargon.
- Be explicit about: what outcome, what treatment comparison (baseline vs treated), direction, and uncertainty.
- If confidence intervals or inference objects exist, interpret them cautiously.
- If warnings exist, surface the clinically relevant ones.
- If result is missing key pieces, say what is missing and how it limits interpretation.
- Do NOT claim causality beyond the assumptions of observational causal inference.

Output:
Return JSON only with:
{
  "summary": "string",
}
""".strip()


CAUSAL_INFERENCE_ATE_SUMMARY_USER_PROMPT_TEMPLATE = """
Context (JSON):
{context_json}

Raw ATE result (JSON):
{ate_result_json}

Warnings (JSON):
{warnings_json}

Now write the JSON output.
""".strip()