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


def data_manupulation_out_of_scope_system_prompt() -> str:
    return """
You are assisting a user in a DATA_MANIPULATION stage that only applies persistent changes to a dataset.

The user sent a message that does not belong to dataset-changing manipulation (e.g. a statistical question, chart request, or general query).

You will receive:
- user_message: what the user asked
- dataset_summary: a JSON summary of the current working dataset

Your task:
1. Briefly acknowledge that this stage only handles dataset-changing operations (filtering, cleaning, renaming, recoding, deriving columns, deduplication, reshaping, etc.).
2. Summarise the current dataset in plain language (number of rows, columns, a few notable column names and types).
3. Invite the user to request a dataset change, or let them know they can move to a different stage if they want statistics, charts, or analysis.

Rules:
- Be concise and friendly.
- Do not mention internal JSON, implementation details, or tool names.
- Return only the user-facing message text.
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


def data_manupulation_protocol_driven_cleaning_appendix() -> str:
    return """
Additional protocol-driven cleaning rules:
- Fix protocol-scope column data types whenever the stored type does not match the grounded semantic meaning of the variable. This applies to all relevant conversions, not just one pattern. For example, categorical codes stored as numbers should be treated as categories, numeric values stored as strings should be cast to numeric, boolean-like values should be normalized to a stable boolean/binary representation, and date/time values stored as strings should be parsed into an appropriate temporal representation when the protocol depends on time.
- If a protocol-scope column behaves like a discrete category but is stored with another type (for example numeric or free-text codes), normalize it to a stable categorical representation instead of treating it like a continuous measurement.
- For the confirmed binary treatment column, keep only the two grounded treatment values from the confirmed protocol. If the raw data contains extra treatment values and the protocol does not explicitly instruct a grounded merge or remapping, remove those rows instead of guessing.
- For a confirmed binary outcome column, keep only the two grounded outcome values from the confirmed protocol. If the raw data contains extra outcome values and the protocol does not explicitly instruct a grounded merge or remapping, remove those rows instead of guessing.
- Do not invent category merges such as collapsing values 2 and 3 together unless the confirmed protocol or an explicit user request clearly says to do that.
- Keep all normalization grounded in the confirmed protocol discussion and the observed data.
""".strip()


__all__ = [
    "data_manupulation_final_response_system_prompt",
    "data_manupulation_intent_classification_system_prompt",
    "data_manupulation_node_info",
    "data_manupulation_out_of_scope_system_prompt",
    "data_manupulation_protocol_driven_cleaning_appendix",
]
