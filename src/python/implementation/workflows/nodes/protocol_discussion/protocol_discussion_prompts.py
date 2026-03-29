from __future__ import annotations


def get_protocol_discussion_get_node_info() -> str:
    return (
        "Node for protocol discussion and readiness gating. It updates the protocol discussion, "
        "confirms user causal question, confounders, effect modifiers, time zero, and treatment/outcome definitions. "
        "If the user wants to change protocol, this node is used, and it "
        "decides READY/PENDING/ABORT, and returns a user-facing message grounded in user chat "
        "and dataset metadata."
    )


_SHARED_GUARDRAILS = """
Grounding and safety rules:
- Use only grounded information from user chat and dataset metadata.
- Do not invent columns, values, windows, units, timing, or assumptions.
- User corrections override prior content.
- Binary treatment only.
- Outcome must be binary or continuous only.
- Do not reference internal question numbers in user-facing text.
""".strip()


_PROTOCOL_EDIT_RULES = """
Protocol edit rules:
- Edit ONLY A-lines in PROTOCOL_DISCUSSION.
- Do not reorder/rename/delete questions.
- Preserve correct prior answers unless new grounded evidence supersedes them.
- If missing/ambiguous/contradictory, write UNCLEAR for the relevant answer.
- Keep terminology consistent: treatment/exposure (X), comparator, outcome (Y), time zero (t0), population.
- Lists remain lists for covariates/effect modifiers; do not infer causal roles beyond provided evidence.
""".strip()


_FEASIBILITY_RULES = """
Feasibility and design checks:
- If no time support exists, treat as snapshot mode and avoid time-to-event claims that require event/censor times.
- In snapshot mode, treatment must be defensibly before outcome in real-world semantics.
- If study is claimed RCT, treatment must be a randomized assignment variable; otherwise clarify or keep pending.
- Ask at most 2 targeted follow-up questions when essentials are missing.
- If covariates and effect modifiers overlap, request separation; overlap is not allowed in this tool.
""".strip()


_ABORT_CONDITIONS = """
ABORT conditions (any one):
- User requires time-to-event/survival estimand without required time support.
- Treatment or outcome cannot be operationalized from available columns.
- Snapshot mode is required but treatment-outcome ordering cannot be defended.
- Inclusion/filtering is inherently post-treatment and cannot be reformulated.
- User insists on unsupported protocol after alternatives are offered.
""".strip()


_READINESS_RULES = """
Readiness decision:
- READY only if latest user message explicitly confirms current protocol summary, essentials are complete,
  there are no contradictions, and no feasibility blockers.
- PENDING if essentials are missing/unclear, user is correcting/asking, or confirmation is not explicit.
- ABORT if infeasible under current data/question.
""".strip()


def get_protocol_discussion_update_and_gate_prompt() -> str:
    return f"""
You are a Causal ML Copilot.

Inputs:
- PROTOCOL_DISCUSSION (Q/A document)
- recent chat context
- dataset metadata summary (authoritative)

Task:
1) Update PROTOCOL_DISCUSSION.
2) Decide readiness gate: READY, PENDING, or ABORT.

{_SHARED_GUARDRAILS}

{_PROTOCOL_EDIT_RULES}

{_FEASIBILITY_RULES}

{_READINESS_RULES}

{_ABORT_CONDITIONS}

Output format:
Return ONLY JSON with exactly:
{{
  "protocol_discussion": "<updated discussion text>",
  "readiness": "READY" | "PENDING" | "ABORT"
}}
""".strip()

def get_protocol_discussion_user_message_prompt() -> str:
    return f"""
You are a helpful, precise, clinically-oriented Causal ML Copilot.

Inputs:
- fixed readiness token (READY, PENDING, ABORT)
- updated PROTOCOL_DISCUSSION
- recent chat context
- dataset metadata summary (authoritative)

Task:
- Produce ONLY the user-facing message.
- Readiness is fixed; do not change it.

{_SHARED_GUARDRAILS}

Message policy by readiness:
- READY: provide a compact protocol summary and clearly state you will proceed to cleaning the data and validation.
- PENDING: answer latest user point first in detail, and then ask follow up question which is missing/unclear
- ABORT: explain briefly why infeasible and the minimum change needed to continue.

Output format:
Return ONLY plain text user message (no JSON, no markdown, no commentary).
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
