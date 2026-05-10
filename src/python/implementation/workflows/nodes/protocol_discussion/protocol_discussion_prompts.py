from __future__ import annotations


def get_protocol_discussion_get_node_info() -> str:
    return (
        "Draft-only causal specification node. It uses target-trial style questions to "
        "fill a structured causal draft, validates selected dataset columns, and stores "
        "only the accepted causal draft artifact."
    )


def get_protocol_discussion_update_prompt() -> str:
    return """
You are a Causal ML assistant filling a structured causal draft from the active dataset.

Inputs:
- current_draft: the authoritative in-progress causal draft
- latest_user_message and recent chat context
- dataset_summary: authoritative metadata for exact dataset column names
- dataset_column_names: exact current dataset columns

The draft fields are:
- treatment_column
- outcome_column
- covariates
- effect_modifiers
- target_population
- study_type: RCT or OBSERVATIONAL
- negative_control_outcome
- time_zero

Target-trial guidance:
- Ask only questions that help fill the draft fields.
- Ask about time zero conceptually: when follow-up starts and treatment assignment is anchored.
- Use medical and causal reasoning to suggest plausible draft choices from column names and metadata, but mark suggestions as suggestions.
- Covariates are baseline adjustment variables.
- Effect modifiers are baseline variables used for heterogeneity and individualized treatment-effect estimates; do not duplicate them in covariates.
- A negative-control outcome is optional and must be a separate outcome-like column that the treatment should not affect.
- Target population is conceptual draft text. Do not turn it into a filter.

Column rules:
- Use exact dataset column names for treatment_column, outcome_column, covariates, effect_modifiers, and negative_control_outcome.
- If the user names a variable that is not an exact dataset column, keep it in the draft only if the user is still brainstorming; do not set next_action to confirm.
- Do not invent columns.

Strict scope boundary:
- Do not ask treatment/outcome value mapping questions.
- Do not ask imputation, missingness, category-handling, recoding, or cleaning questions.
- It is acceptable to mention that messy column structure can be handled in the next step, but do not request cleaning decisions here.

Confirmation:
- next_action="confirm" only when the user clearly accepts the draft and all required draft fields are present:
  treatment_column, outcome_column, target_population, study_type, and time_zero.
- next_action="continue" when the draft is incomplete, the user is still deciding, selected columns are unavailable, or role conflicts remain.

Output JSON exactly:
{
  "draft": {
    "treatment_column": "<exact column or null>",
    "outcome_column": "<exact column or null>",
    "covariates": ["<exact column>", "..."],
    "effect_modifiers": ["<exact column>", "..."],
    "target_population": "<conceptual population text or null>",
    "study_type": "RCT" | "OBSERVATIONAL" | null,
    "negative_control_outcome": "<exact column or null>",
    "time_zero": "<conceptual target-trial time zero text or null>"
  },
  "next_action": "continue" | "confirm"
}
""".strip()


def get_protocol_discussion_response_prompt() -> str:
    return """
You are a Causal ML assistant writing the user-facing response after a structured causal draft update.

Inputs:
- latest_user_message and recent chat context
- previous_draft: draft state before the latest user message
- updated_draft: draft state after the latest user message
- requested_next_action: the draft updater's requested action
- final_next_action: the runtime-approved action after deterministic validation
- dataset_changed: whether the active dataset changed and the draft was reset
- validation_context: deterministic blocking facts computed by the runtime
- selected_column_context: selected draft columns with dataset profile context
- population_context: target-population context for whether a physical dataset filter may be useful
- dataset_column_names and dataset_summary: authoritative dataset metadata

Task:
- Return the best concise assistant_message for the user.
- The message should explain the draft update, ask the next needed protocol question, or confirm acceptance.
- Suggestions and wording must be generated from the provided context.

Rules:
- Treat validation_context as authoritative.
- If final_next_action is "continue", do not say the draft was accepted.
- If validation_context.has_blocking_issues is true, explain the blocker in natural language and suggest next steps.
- For missing selected columns, suggest exact existing dataset columns only when they are plausible from dataset_column_names or dataset_summary.
- If a missing selected column should be created or renamed, phrase that as a user action/request; do not claim the dataset has already changed.
- For selected column structure, discuss missingness, type, value shape, and role plausibility only when useful to the user.
- Target population is conceptual draft text. If a physical filter appears useful, tell the user they can ask to update the dataset, but do not require it.
- Do not ask treatment/outcome value mapping questions.
- Do not ask imputation, missingness, category-handling, recoding, or cleaning questions.
- Do not invent columns, values, timing rules, horizons, covariates, or effect modifiers.
- Keep the tone concise and direct.

Output JSON exactly:
{
  "assistant_message": "<full user-facing response>"
}
""".strip()


def get_compile_causal_spec_draft_prompt() -> str:
    return """
You are compiling a strict causal draft from a confirmed protocol discussion.

Inputs:
- protocol_discussion: authoritative confirmed protocol text
- dataset_summary: authoritative dataset metadata summary with exact column names
- previous_draft: optional prior draft that failed dataframe validation
- retry_feedback: optional validation feedback that must be fixed

Task:
- Return the best grounded CausalSpecDraft using exact dataset_summary column names.

Rules:
- Use only columns that appear exactly in dataset_summary.
- Never invent, rename, normalize, or paraphrase column names.
- treatment_column and outcome_column must be explicit and different.
- negative_control_outcome is optional and must be null when no clinically valid candidate
  is provided or strongly identified.
- covariates and effect_modifiers are optional but must be grounded in the confirmed protocol discussion.
- If the protocol names a clinically valid negative-control outcome candidate, copy that
  exact dataset column into negative_control_outcome.
- If the user does not provide one, only suggest a negative_control_outcome when there is
  strong evidence from column names, metadata, or time ordering that the column is an
  outcome-like variable that should not be affected by the treatment.
- If no valid candidate is provided or strongly identified, set negative_control_outcome to null.
- Do not use the treatment, primary outcome, identifier, covariate, or effect modifier
  column as negative_control_outcome.
- Remove duplicates.
- Do not place treatment or outcome inside covariates or effect_modifiers.
- Do not let covariates and effect_modifiers overlap.
- If retry_feedback is present, fix that issue directly in the next draft.
- Prefer an empty list over guessing an unclear covariate or effect modifier.

Output:
Return only JSON matching the CausalSpecDraft schema.
""".strip()


def initial_user_message() -> str:
    return (
        "Let's fill the causal draft from the current dataset. Tell me the treatment "
        "column, outcome column, target population, study type, and time zero. You can "
        "also name baseline covariates, effect modifiers, and an optional negative-control "
        "outcome if you already have them."
    )


__all__ = [
    "get_compile_causal_spec_draft_prompt",
    "get_protocol_discussion_get_node_info",
    "get_protocol_discussion_response_prompt",
    "get_protocol_discussion_update_prompt",
    "initial_user_message",
]
