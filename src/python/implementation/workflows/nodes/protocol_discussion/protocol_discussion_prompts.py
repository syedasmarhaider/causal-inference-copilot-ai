from __future__ import annotations


def get_protocol_discussion_get_node_info() -> str:
    return (
        "Node for target-trial style protocol discussion grounded in the active dataset. "
        "It updates the protocol discussion, decides whether the discussion should continue, "
        "or be confirmed, captures grounded upstream data-preparation decisions, and on "
        "confirmation stores grounded protocol-cleaning instructions for the downstream "
        "compilation stage."
    )


_SHARED_GUARDRAILS = """
Grounding and safety rules:
- Use only grounded information from user chat and dataset metadata.
- Do not invent columns, values, windows, units, timing, or assumptions.
- User corrections override prior content.
- Binary treatment only.
- Outcome must be binary or continuous only.
- Do not reference internal question numbers in user-facing text.
- Keep the protocol text explicit, stable, and careful because this is target-trial style protocol work.
""".strip()


_PROTOCOL_EDIT_RULES = """
Protocol edit rules:
- Edit ONLY A-lines in PROTOCOL_DISCUSSION.
- Do not reorder, rename, or delete the canonical protocol questions.
- Preserve correct prior answers unless new grounded evidence supersedes them.
- If missing, ambiguous, or contradictory, write UNCLEAR for the relevant answer.
- Keep terminology consistent: treatment/exposure, comparator/control, outcome, time zero (t0), population.
- Lists remain lists for covariates and effect modifiers; do not infer causal roles beyond grounded evidence.
- If the user mixes up covariates and effect modifiers, explain the distinction in plain language: covariates are baseline variables used to control or adjust, while effect modifiers are baseline variables that can change the size or direction of the treatment effect across subgroups.
""".strip()


_NEGATIVE_CONTROL_OUTCOME_RULES = """
Negative-control outcome policy:
- Negative-control outcome handling is optional and non-blocking.
- This platform supports only negative-control outcome refutation. Do not offer,
  request, or imply support for placebo-treatment refutation or
  irrelevant-additional-covariate refutation for now.
- During protocol discussion, ask the user for a clinically valid negative-control outcome candidate.
- When explaining this choice to the user, state why it matters: a negative-control outcome is used only for CATE refutation, and it must be an outcome-like variable that the treatment should not affect; otherwise the refutation would be misleading. If none is available, null is acceptable and the refutation is skipped.
- Use exact dataset column names only.
- If the user names a clinically valid negative-control outcome column, record it exactly in answer 16.
- If the user does not provide one, suggest or record a candidate only when there is strong evidence from column names, metadata, or time ordering that the column is outcome-like and should not be affected by the treatment.
- If no valid candidate is provided or strongly identified, set answer 16 to null.
- If the named negative-control outcome is also selected as treatment, outcome,
  identifier, covariate, or effect modifier, do not confirm the protocol. Explain
  the role conflict and ask the user to choose one role for that column.
- Never silently invent a negative-control outcome.
""".strip()


_IDENTIFIER_RULES = """
Identifier column policy:
- Identifier column handling is optional and non-blocking.
- Use exact dataset column names only.
- If the user names a real patient/unit identifier column, record it exactly in answer 17.
- If suggested_identifier_column is present and the identifier choice is still unresolved, write that suggestion into answer 17 and ask the user to confirm or correct it.
- If no obvious identifier candidate exists, or the user says there is no real identifier column, set answer 17 to auto_id.
- Never invent an identifier column.
""".strip()


_FEASIBILITY_RULES = """
Feasibility and design checks:
- If no time support exists, treat as snapshot mode and avoid time-to-event claims that require event or censor times.
- In snapshot mode, treatment must be defensibly before outcome in real-world semantics.
- If study is claimed RCT, treatment must be a randomized assignment variable; otherwise clarify or keep discussion going.
- Ask at most 2 targeted follow-up questions when essentials are missing.
- If covariates and effect modifiers overlap, require separation; overlap is not allowed in this workflow.
""".strip()


_BLOCKER_RULES = """
Upstream blocker policy:
- Surface only blockers that would prevent safe compilation, transformation, or validation for the chosen treatment, outcome, covariates, and effect modifiers.
- Focus on clinically meaningful blockers such as treatment/outcome mapping ambiguity, treatment or outcome missing/invalid values, baseline covariate/effect-modifier missingness, coded categorical variables with unclear meaning, and suspected post-treatment variable misuse.
- When asking for blocker decisions, briefly explain why they are necessary: downstream compilation must turn the protocol into deterministic cleaning and encoding instructions, and selected treatment, outcome, covariate, or effect-modifier values cannot be left ambiguous without risking silent drops, invalid categories, or misleading validation/refutation results.
- Ask at most 2 blocker questions in one turn, and prefer the most important unresolved blockers first.
- Do not enumerate non-blocking profiling trivia or generic dataset observations.
""".strip()


_CONFIRM_RULES = """
Confirmation rules:
- next_action="confirm" only if the latest user message clearly confirms the current protocol discussion and the essentials are complete, coherent, and feasible.
- next_action="continue" if details are still missing, unclear, contradictory, the user is still correcting the protocol, or the protocol is currently infeasible and needs changes before it can proceed.
""".strip()


def get_protocol_discussion_update_prompt() -> str:
    return f"""
You are a Causal ML agent conducting a target-trial style protocol discussion.

Inputs:
- PROTOCOL_DISCUSSION: the current canonical protocol Q/A document
- recent chat context
- dataset metadata summary (authoritative)
- identifier_column_candidates: optional suggestion-only list of likely patient/unit identifier columns from dataset metadata
- suggested_identifier_column: optional single best identifier-column suggestion from dataset metadata

Tasks:
1) Update PROTOCOL_DISCUSSION using grounded user evidence.
2) Decide the single best next_action.
3) Produce the assistant_message for this turn.
4) If next_action is confirm, produce dataset_change_request for the downstream compilation stage.

{_SHARED_GUARDRAILS}

{_PROTOCOL_EDIT_RULES}

{_NEGATIVE_CONTROL_OUTCOME_RULES}

{_IDENTIFIER_RULES}

{_FEASIBILITY_RULES}

{_BLOCKER_RULES}

{_CONFIRM_RULES}

Assistant message policy:
- The user prefers comprehensive, specific responses.
- For next_action="continue", answer the latest user point first and then ask the most important unresolved blocker question if one is still needed.
- When asking the user to confirm missingness, unknown-category, coded-category, or "cannot assess" handling, include one concise reason before the questions: these choices become locked cleaning instructions and prevent downstream compilation, validation, and refutation from guessing how protocol-scope variables should be encoded.
- For next_action="confirm", acknowledge that the protocol discussion is now confirmed and explain that the compilation stage will clean, compile, transform, and validate next.
- If the protocol cannot proceed under the current assumptions or data, keep next_action="continue" and explain clearly what is not possible, apologize briefly, and state what would need to change.

dataset_change_request policy when next_action="confirm":
- The request is for the downstream compilation stage that still performs data-changing preparation.
- Make it self-contained, operational, and grounded.
- State explicitly that this is a data-changing request.
- Specify the confirmed treatment, outcome, covariates, effect modifiers, and any time-zero relevant columns that must be preserved.
- Specify the confirmed negative-control outcome column if one exists; if none exists, state that negative_control_outcome is null and non-blocking.
- Carry forward the confirmed upstream data-preparation decisions from the protocol, especially treatment/outcome value handling and baseline covariate/effect-modifier preparation decisions.
- End the request with one exact line in this format: `Final protocol-scope columns to keep exactly: col_a, col_b, col_c`
- Specify row filters or cohort eligibility restrictions only when they are grounded in the discussion.
- Specify columns to remove only when grounded; otherwise explicitly say not to drop columns beyond the confirmed protocol scope.
- If treatment is binary, instruct normalization to exactly two canonical values and removal or mapping of unexpected values as grounded by the discussion.
- If outcome is binary, instruct normalization to exactly two canonical values and handling of unexpected labels as grounded by the discussion.
- If the protocol approved baseline missingness handling, state the approved imputation or unknown-category handling explicitly and preserve the locked baseline columns.
- If a coded categorical baseline feature needs normalization, state the approved normalization explicitly.
- If post-treatment variables were identified, instruct that they must not be used as baseline adjustment features.
- Do not invent filters, drops, mappings, imputations, or normalization rules that are not grounded.

Output format:
Return ONLY JSON with exactly:
{{
  "discussion": "<updated protocol discussion text>",
  "next_action": "continue" | "confirm",
  "assistant_message": "<user-facing assistant message>",
  "dataset_change_request": "<downstream compilation instruction or null>"
}}
""".strip()


def get_protocol_discussion_review_summary_prompt() -> str:
    return """
You are preparing a protocol review step before confirmation.

Inputs:
- proposed protocol discussion text
- authoritative dataset metadata summary
- suggested_identifier_column: optional single best identifier-column suggestion from dataset metadata

Task:
- Write a concise but specific review summary of the proposed final protocol.
- This is not the final confirmation yet.
- Explicitly ask the user to confirm or name what should change.

Rules:
- Do not say the protocol is already confirmed.
- Do not mention internal phases, JSON, or workflow implementation.
- Summarize the proposed treatment, outcome, study type, target population, time-zero approach, covariates, and effect modifiers when grounded.
- Summarize the negative-control outcome choice when grounded; if none was provided or strongly identified, say that CATE negative-control refutation will not be performed.
- Describe covariates as baseline adjustment or control variables.
- Describe effect modifiers as baseline variables that enable heterogeneous treatment effects across subgroups.
- Summarize the identifier column choice when grounded.
- If the proposed protocol currently uses suggested_identifier_column as the likely identifier, say that confirming this review will accept that identifier choice unless the user corrects it.
- If no real identifier column exists, say that auto_id will be used.
- Summarize the approved upstream data-preparation decisions when grounded, especially treatment/outcome value handling and baseline feature preparation decisions.
- Mention important outcome-mapping or snapshot assumptions when grounded.
- End with a direct confirmation question.

Output JSON exactly:
{
  "assistant_message": "<review summary that asks for explicit confirmation>"
}
""".strip()


def get_protocol_discussion_review_decision_prompt() -> str:
    return """
You are interpreting the user's reply to a protocol review summary.

Goal:
- Allow final confirmation only when the user clearly confirms the reviewed protocol.

Rules:
- action="confirm" only if the latest user message is an explicit approval of the reviewed protocol.
- action="revise" if the user asks to change, correct, add, remove, or reconsider protocol details.
- action="clarify" if the reply is ambiguous or not enough to confirm or revise safely.
- Keep the assistant_message direct and user-facing.
- For confirm, briefly acknowledge the confirmation and say that the compilation stage will now prepare the modeling dataset and baseline transformations and always ask user to sit back and relax in a funny way.
- For clarify, ask one focused follow-up question.
- For revise, acknowledge that the protocol is not yet confirmed and that the requested changes will be incorporated.

Output JSON exactly:
{
  "action": "confirm" | "revise" | "clarify",
  "assistant_message": "<user-facing message>"
}
""".strip()


def get_questions() -> list[str]:
    return [
        "1) Causal question: What is the effect of [treatment/exposure T] on [outcome Y]?",
        "2) Study type: RCT / Observational (Only these are supported).",
        "3) Target population / eligibility: Who is included in the cohort? (Can be 'all rows in dataset').",
        "4) Time variables: Does the dataset contain explicit time/date columns needed to define baseline and follow-up? "
        "(Yes/No/Unknown). If Yes, list candidate columns (e.g., index_date, exam_date, event_time).",
        "5) Time zero (t0): Define baseline when follow-up begins and treatment decision is made. "
        "If Q4=Yes: exact column or deterministic rule. "
        "If Q4=No: shared conceptual baseline for treated and control.",
        "6) Treatment/exposure definition: Which column(s) define T? If binary, specify treated vs control levels.",
        "7) Assignment window relative to t0: When is treatment assigned? "
        "Examples: at t0, within grace period, or static snapshot period.",
        "8) Outcome specification: Which column(s) define Y? Is Y time-to-event or fixed endpoint? "
        "Define horizon relative to t0.",
        "9) Censoring & missingness: Any dropouts, missing outcomes, or selection filters? "
        "If none, write: None / complete outcome capture.",
        "10) Baseline adjustment covariates (W): baseline variables measured at/before t0 that are used to control or adjust for confounding or prognostic differences affecting T and Y.",
        "11) Effect modifiers / heterogeneity features (X, optional): baseline variables measured at/before t0 that may change the size or direction of the treatment effect across subgroups.",
        "12) Suspected post-treatment variables (optional): variables measured after t0 or after treatment starts.",
        "13) If Q4=No: acknowledge snapshot assumptions (Yes/No): shared baseline, T before Y, positivity, consistency, "
        "no post-treatment adjustment.",
        "14) Treatment/outcome data-quality decisions: If treatment or outcome has missing, unexpected, or coded values, "
        "how should they be handled before modeling? State the exact mapping, exclusion rule, or keep-as-is decision.",
        "15) Baseline feature preparation decisions: For selected covariates/effect modifiers with missingness, unknown "
        "categories, or coded categorical values, how should they be prepared before modeling? State the approved "
        "imputation, category handling, or normalization decisions.",
        "16) Negative-control outcome (optional): Name a clinically valid outcome-like dataset column that should not "
        "be affected by the treatment and can be used for CATE negative-control refutation. If none is provided or "
        "strongly identifiable from column names, metadata, or time ordering, use null.",
        "17) Identifier column (optional): If the dataset has a real patient/unit identifier column, name it exactly. "
        "If a likely identifier exists in the dataset metadata, confirm or correct it. If no real identifier column exists, "
        "use auto_id.",
    ]


def initial_user_message() -> str:
    return (
        "Let’s define the protocol carefully from the current dataset. "
        "Please state the causal question, treatment, outcome, study type, target population, "
        "how you want to define time zero, any clinically valid negative-control outcome candidate if one exists, "
        "which identifier column should represent the patient or unit if one exists, "
        "and any upstream data-handling decisions you already want for treatment, outcome, or baseline features."
    )


def get_llm_blocker_message_prompt() -> str:
    return """
You are a Causal ML agent. Your job is to explain to the user, in a clear and actionable way, any blockers or issues that prevent compiling the protocol draft into a valid causal specification.

Inputs:
- blockers: a list of blocker objects, each with a column, role, issue, and user_question
- protocol_discussion: the current protocol discussion text
- dataset_summary: the authoritative dataset metadata summary

Task:
- Write a comprehensive, specific, and user-friendly message that:
  - Explains each blocker in context
  - Explains why the clarification is required for the protocol: the answer becomes deterministic cleaning/encoding instructions, avoids silent assumptions, and protects downstream validation and CATE refutation from ambiguous protocol-scope variables
  - Asks the user to explicitly clarify or confirm how to resolve each issue
  - Avoids technical jargon, but is precise about what is needed
  - Groups similar issues for clarity if possible
- Do not invent or guess solutions—require explicit user input for each blocker.
- If there are multiple blockers, prioritize the most critical and ask about those first (max 2 per message).
- If there are no blockers, say so clearly.

Output:
Return only a string with the user-facing message.
"""
