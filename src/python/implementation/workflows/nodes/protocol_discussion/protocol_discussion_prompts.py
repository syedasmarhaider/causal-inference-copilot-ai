from __future__ import annotations


def get_protocol_discussion_get_node_info() -> str:
    return (
        "Node for target-trial style protocol discussion grounded in the active dataset. "
        "It updates the protocol discussion, decides whether the discussion should continue, "
        "or be confirmed, and on confirmation stores grounded protocol-cleaning "
        "instructions for the downstream compilation stage."
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
""".strip()


_FEASIBILITY_RULES = """
Feasibility and design checks:
- If no time support exists, treat as snapshot mode and avoid time-to-event claims that require event or censor times.
- In snapshot mode, treatment must be defensibly before outcome in real-world semantics.
- If study is claimed RCT, treatment must be a randomized assignment variable; otherwise clarify or keep discussion going.
- Ask at most 2 targeted follow-up questions when essentials are missing.
- If covariates and effect modifiers overlap, require separation; overlap is not allowed in this workflow.
""".strip()


_CONFIRM_RULES = """
Confirmation rules:
- next_action="confirm" only if the latest user message clearly confirms the current protocol discussion and the essentials are complete, coherent, and feasible.
- next_action="continue" if details are still missing, unclear, contradictory, the user is still correcting the protocol, or the protocol is currently infeasible and needs changes before it can proceed.
""".strip()


def get_protocol_discussion_update_prompt() -> str:
    return f"""
You are a Causal ML Copilot conducting a target-trial style protocol discussion.

Inputs:
- PROTOCOL_DISCUSSION: the current canonical protocol Q/A document
- recent chat context
- dataset metadata summary (authoritative)

Tasks:
1) Update PROTOCOL_DISCUSSION using grounded user evidence.
2) Decide the single best next_action.
3) Produce the assistant_message for this turn.
4) If next_action is confirm, produce dataset_change_request for the downstream compilation stage.

{_SHARED_GUARDRAILS}

{_PROTOCOL_EDIT_RULES}

{_FEASIBILITY_RULES}

{_CONFIRM_RULES}

Assistant message policy:
- Do not be terse. The user prefers comprehensive, specific responses.
- For next_action="continue", answer the latest user point first and then ask the most important missing follow-up question if one is still needed.
- For next_action="confirm", acknowledge that the protocol discussion is now confirmed and explain that the compilation stage will clean, compile, transform, and validate next.
- If the protocol cannot proceed under the current assumptions or data, keep next_action="continue" and explain clearly what is not possible, apologize briefly, and state what would need to change.

dataset_change_request policy when next_action="confirm":
- The request is for the downstream compilation stage that still performs data-changing preparation.
- Make it self-contained, operational, and grounded.
- State explicitly that this is a data-changing request.
- Specify the confirmed treatment, outcome, covariates, effect modifiers, and any time-zero relevant columns that must be preserved.
- End the request with one exact line in this format: `Final protocol-scope columns to keep exactly: col_a, col_b, col_c`
- Specify row filters or cohort eligibility restrictions only when they are grounded in the discussion.
- Specify columns to remove only when grounded; otherwise explicitly say not to drop columns beyond the confirmed protocol scope.
- If treatment is binary, instruct normalization to exactly two canonical values and removal or mapping of unexpected values as grounded by the discussion.
- If outcome is binary, instruct normalization to exactly two canonical values and handling of unexpected labels as grounded by the discussion.
- If post-treatment variables were identified, instruct that they must not be used as baseline adjustment features.
- Do not invent filters, drops, or mappings that are not grounded.

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

Task:
- Write a concise but specific review summary of the proposed final protocol.
- This is not the final confirmation yet.
- Explicitly ask the user to confirm or name what should change.

Rules:
- Do not say the protocol is already confirmed.
- Do not mention internal phases, JSON, or workflow implementation.
- Summarize the proposed treatment, outcome, study type, target population, time-zero approach, covariates, and effect modifiers when grounded.
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
- For confirm, briefly acknowledge the confirmation and say that the compilation stage will now prepare the modeling dataset and baseline transformations.
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
        "10) Baseline adjustment covariates (W): variables measured at/before t0 that affect both T and Y.",
        "11) Effect modifiers / heterogeneity features (X, optional): baseline variables for subgroup effects.",
        "12) Suspected post-treatment variables (optional): variables measured after t0 or after treatment starts.",
        "13) If Q4=No: acknowledge snapshot assumptions (Yes/No): shared baseline, T before Y, positivity, consistency, "
        "no post-treatment adjustment.",
    ]


def initial_user_message() -> str:
    return (
        "Let’s define the protocol carefully from the current dataset. "
        "Please state the causal question, treatment, outcome, study type, target population, "
        "and how you want to define time zero."
    )
