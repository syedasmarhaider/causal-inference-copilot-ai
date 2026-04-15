from __future__ import annotations


def data_validation_node_info() -> str:
    return (
        "Validation review stage. It validates the compiled causal specification and "
        "transformation plan against the active scoped dataset, blocks on hard failures, "
        "and asks for confirmation before publishing accepted validation results."
    )


def data_validation_review_summary_prompt() -> str:
    return """
You are preparing the user-facing review message after causal validation completed without hard failures.

Inputs:
- compiled causal specification
- compiled transformation plan
- validation status
- validation issues

Task:
- Write a concise review message for the user.
- If warnings exist, explain the practical review points clearly.
- If no warnings exist, say validation passed without blocking issues or additional warnings.
- End by asking the user to confirm the validation result or say exactly what should change.

Style rules:
- Be direct and user-facing.
- Do not mention internal JSON, validators, or workflow phases.
- Do not say the result is already confirmed.

Output JSON exactly:
{
  "assistant_message": "<short validation review message>"
}
""".strip()


def data_validation_review_decision_prompt() -> str:
    return """
You are reviewing a validation result with the user.

Task:
- Interpret the latest user reply to decide whether the validation result is accepted, rejected for revision, or still unclear.

Decision rules:
- Choose `confirm` only when the user is clearly accepting the validation result as-is.
- Choose `revise` when the user is rejecting the validation result or asking to change the dataset, causal specification, transformation plan, or validation assumptions.
- Choose `clarify` when the reply is ambiguous, incomplete, or not enough to confirm or reject safely.

Style rules:
- Keep the assistant message plain, direct, and user-facing.
- If `clarify`, ask one focused follow-up question.
- Do not invent new technical facts.

Output JSON exactly:
{
  "action": "confirm" | "revise" | "clarify",
  "assistant_message": "<short user-facing message>"
}
""".strip()


__all__ = [
    "data_validation_node_info",
    "data_validation_review_decision_prompt",
    "data_validation_review_summary_prompt",
]
