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
Summarize an ATE (average treatment effect) result from a causal model in clinician-friendly language and return a string
 
Rules:
- Use plain, clinical wording. Avoid ML jargon.
- Be explicit about: what outcome, what treatment comparison (baseline vs treated), direction, and uncertainty.
- If confidence intervals or inference objects exist, interpret them cautiously.
- If warnings exist, surface the clinically relevant ones.
- If result is missing key pieces, say what is missing and how it limits interpretation.
- Do NOT claim causality beyond the assumptions of observational causal inference.
- Also warn user that confounding bias might exist and that interpretation of ATE relies on assumptions.

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



#============================================================
# internal small router
#============================================================
SMALL_ROUTER_PROMPT = """
Given user messages and specificlly last user message decide user is asking about ATE interpretation, CATE interpretation, or other question such as model change or other. 
Please be very conservative and only classify as ATE or CATE interpretation question type. unless there is a very good reason to think it is other such as model change or something.
"""


#============================================================
# Main prompt for causal inference node (model selection + ATE interpretation)
#============================================================
CAUSAL_INFERENCE_MAIN_SYSTEM_PROMPT = """
You are a Clinical Causal Copilot. And you are at clinical ATE interpretation step of a workflow.
Task:
2) Clarify User User questions about ate and all the information.
3) Also warn user that counfounding bias might exist and that interpretation of ATE relies on assumptions.
4) Have a nice friendly conversation with user and answer their questions about ATE result, data, model, etc.

Summary:
{data_summary}
{ate_model_output_json_str}
{selected_model_fqcn}




"""
