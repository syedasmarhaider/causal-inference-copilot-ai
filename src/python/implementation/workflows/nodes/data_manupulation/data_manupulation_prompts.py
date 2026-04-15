from __future__ import annotations


def data_manupulation_node_info() -> str:
    return (
        "Dataset-changing manipulation stage. Uses the current working dataset from "
        "orchestrator state to apply cleaning, filtering, derivations, renaming, reshaping, "
        "and similar transformations, then saves a new dataset version and updates the "
        "orchestrator with the new dataset id and summary."
    )


def data_manupulation_intent_classification_system_prompt() -> str:
    return """
Classify the latest user request for the DATA_MANUPULATION node.

Return JSON that exactly matches the schema.

Inputs:
- latest_user_message
- chat_history
- dataset_summary

Intent definitions:
- intent_dataset_mutation: true only when the request asks to change the dataset itself.
- intent_out_of_scope: true only when the request does not belong to dataset-changing manipulation.

Brief rules:
- If intent_dataset_mutation is true, intent_dataset_mutation_brief must be non-empty and fully describe the requested change.
- If intent_dataset_mutation is false, intent_dataset_mutation_brief must be an empty string.

General rules:
- Dataset-changing manipulation includes filtering rows, cleaning values, imputing, renaming columns, recoding values, deriving columns, dropping columns, reshaping, deduplicating, and other persistent changes to the working dataset.
- Read-only summaries, analytical queries, statistical analyses, and chart requests are out of scope here because they do not update the dataset.
- Requests for downstream causal inference, model selection, training, or estimation are out of scope here.
- If the request mixes dataset-changing work with out-of-scope requests, keep only the dataset-changing manipulation intent.
- If the request is only out of scope, set intent_out_of_scope to true and intent_dataset_mutation to false.
- Use chat_history to resolve short follow-ups, pronouns, and ellipsis.
""".strip()


def data_manupulation_final_response_system_prompt() -> str:
    return """
You are writing the final DATA_MANUPULATION response.

You will receive JSON with:
- manipulation_result
- dataset_context

Rules:
- This node updates the working dataset.
- Summarize what changed in concise user-facing language.
- Mention the new dataset only when the update succeeded.
- If the update failed, say that plainly and do not claim the dataset changed.
- Do not mention internal JSON or implementation details.
- Return only the final user-facing message text.
""".strip()


__all__ = [
    "data_manupulation_final_response_system_prompt",
    "data_manupulation_intent_classification_system_prompt",
    "data_manupulation_node_info",
]
