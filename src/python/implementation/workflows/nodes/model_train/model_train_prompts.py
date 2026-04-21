from __future__ import annotations


def get_model_train_node_info() -> str:
    return (
        "Training stage for fitting the confirmed selected model against the compiled "
        "dataset using the confirmed causal specification and transformation plan. It "
        "runs directly from the confirmed setup and publishes the fitted model id after "
        "successful training."
    )


def get_model_failure_summary_prompt() -> str:
    return """
You are summarizing a causal-model execution failure for an end user.

Task:
- Explain briefly why the operation failed using only the provided error details.
- Mention concrete likely causes when they are directly supported by the error text.
- Keep the explanation short, direct, and actionable.

Rules:
- Do not mention stack traces.
- Avoid unnecessary internal library jargon unless it is needed to identify the failure.
- Do not invent fixes that are not grounded in the error details.
- Keep the answer to at most 3 sentences.
""".strip()
