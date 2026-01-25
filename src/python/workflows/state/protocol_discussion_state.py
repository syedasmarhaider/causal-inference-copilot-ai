from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProtocolDiscussionState:
    discussion: str = ""

    @staticmethod
    def get_questions() -> List[str]:
        return [
            # Core causal intent
            "1) What is the causal question (one sentence: effect of [treatment] on [outcome])?",
            "2) Is this randomized/experiment (RCT) or observational? (Write: RCT / Observational / Unknown)",
            "3) Define the target population/cohort (eligibility in plain text; can be 'all rows in dataset').",

            # Data support / feasibility gate (prevents impossible time zero definitions)
            "4) Does the dataset contain any explicit time or date columns needed to define follow-up (e.g., start_date, exam_date, event_time)? (Yes/No/Unknown). If Yes, list candidate column names.",

            # Time zero (strict if possible, conceptual fallback if not)
            "5) Define Time Zero: the event/timepoint when follow-up begins. "
            "If Q4=Yes, specify the exact time zero column OR how to construct it from columns. "
            "If Q4=No, choose a shared conceptual baseline (e.g., 'start of term') that applies to BOTH treated and control units.",

            # Treatment definition (must be implementable)
            "6) Define the treatment/exposure using exact column name(s) when possible. "
            "If treatment is binary, state the treated level (e.g., uses_ai=1) and the control level (uses_ai=0).",

            # Treatment timing/window (supports static exposures)
            "7) Define the treatment window relative to Time Zero. "
            "If timing is observed: specify assignment at Time Zero or within a window. "
            "If snapshot data: specify it as a STATIC exposure over a period (e.g., 'used AI at any time during the term').",

            # Comparator (must be concrete)
            "8) Define the comparator/control strategy (must be concrete and feasible in the dataset; e.g., 'no AI usage').",

            # Outcome definition + measurement timing
            "9) Define the outcome using exact column name(s) when possible (e.g., final_score).",
            "10) Is the outcome time-to-event / duration? (Yes/No/Unknown)",

            # Outcome window/horizon (snapshot safe)
            "11) Define the outcome window/horizon relative to Time Zero. "
            "If timing is observed: specify end of follow-up / horizon. "
            "If snapshot data: specify outcome as end-of-period measurement (e.g., 'final score at end of term').",

            # Confounding control (must be pre-treatment)
            "12) List covariates to adjust for (comma-separated). These should be PRE-TREATMENT variables. "
            "Do NOT include treatment itself, outcome, or likely mediators caused by treatment.",

            # Mediator / post-treatment exclusion (critical for causal validity)
            "13) List variables that are likely POST-TREATMENT or mediators (comma-separated). "
            "These should NOT be adjusted for in the main causal effect estimation.",

            # Heterogeneity (optional)
            "14) List effect modifiers/subgroups for heterogeneity analysis (comma-separated). If none, write: None.",

            # Censoring / missingness
            "15) Censoring / follow-up end rules / loss-to-follow-up notes. "
            "If outcome is fully observed for all rows, write: 'None / complete outcome capture'.",

            # Design-specific details
            "16) If observational: briefly describe major confounding risks and sources of selection bias you expect (plain text).",

            # Assumptions (forces honesty when time is not observed)
            "17) Snapshot-data assumptions (required if Q4=No): explicitly state the stronger assumptions you rely on, including: "
            "(i) shared conceptual baseline, (ii) treatment precedes outcome, (iii) no unmeasured confounding after adjustment, "
            "(iv) positivity, (v) consistency, (vi) no post-treatment adjustment."
        ]

def _format_qa_block(questions: List[str], answers: Dict[str, str]) -> str:
    """
    Formats a deterministic Q/A transcript.

    - Uses the provided questions list for ordering.
    - If an answer is missing, fills with an explicit placeholder to avoid silent gaps.
    """
    lines: List[str] = ["TEST PROTOCOL DISCUSSION (auto-filled)", ""]
    for i, q in enumerate(questions, start=1):
        a = answers.get(q, "[MISSING TEST ANSWER]")
        lines.append(f"Q{i}. {q}")
        lines.append(f"A{i}. {a}")
        lines.append("")  # blank line between Q/A pairs
    return "\n".join(lines).rstrip() + "\n"


def get_test_protocol_discussion_state_pdl1_os() -> ProtocolDiscussionState:
    """
    Prefilled Q/A-style discussion aligned to ProtocolDiscussionState.get_questions().
    The content is intentionally realistic and matches the test ProtocolState for:
      - Treatment: Sample PD-L1 Positive (NLP)
      - Outcome: Overall Survival (Months) with event indicator Overall Survival Status
    """
    questions = ProtocolDiscussionState.get_questions()

    # Map EXACT question strings to answers to keep the coupling explicit and stable.
    answers: Dict[str, str] = {
        "What is the causal question (one sentence, effect of treatment on outcome)?": (
            "What is the causal effect of PD-L1 positivity on overall survival in cancer patients?"
        ),
        "Is this randomized/experiment (RCT) or observational? (Write: RCT / Observational / Unknown)": (
            "Observational"
        ),
        "Define the target population/cohort (eligibility in plain text, population can be all the data).": (
            "Cancer patients in the dataset with observed PD-L1 status "
            "in 'Sample PD-L1 Positive (NLP)' and observed survival outcome."
        ),
        "Define Time Zero in plain text (the moment follow-up starts).": (
            "Time zero is the time of PD-L1 status ascertainment (baseline) for each patient."
        ),
        "If Time Zero is a dataset column, write the column name; otherwise describe how it is constructed.": (
            "CONCEPTUAL: the dataset does not provide an explicit PD-L1 assessment date column; "
            "we treat PD-L1 as a baseline exposure at the start of follow-up for patients with recorded status."
        ),
        "What is the treatment/exposure (plain text or column name if known)?": (
            "Sample PD-L1 Positive (NLP) (Yes vs No)"
        ),
        "Define the treatment window relative to Time Zero (plain text).": (
            "At time zero (baseline measurement). No exposure window beyond baseline assignment."
        ),
        "What is the comparator/control condition (plain text)?": (
            "PD-L1 negative (Sample PD-L1 Positive (NLP) == No)."
        ),
        "What is the outcome (plain text or column name if known)?": (
            "Overall survival time in months: 'Overall Survival (Months)'."
        ),
        "Is the outcome time-to-event / duration? (Yes/No/Unknown)": (
            "Yes"
        ),
        "Define the outcome window/horizon relative to Time Zero (plain text).": (
            "From time zero until death or censoring (end of follow-up)."
        ),
        "List covariates to adjust for (comma-separated, plain text).": (
            "Current Age, Sex, Race, Ethnicity, Cancer Type, Stage (Highest Recorded)"
        ),
        "List effect modifiers/subgroups of interest (comma-separated, plain text).": (
            "Cancer Type, Stage (Highest Recorded), Sex"
        ),
        "List censoring rules / follow-up end rules / loss-to-follow-up notes (comma-separated, plain text).": (
            "Use 'Overall Survival Status' as event indicator (1:DECEASED=event, 0:LIVING=censored); "
            "time scale is 'Overall Survival (Months)'."
        ),
        "If RCT: briefly describe randomization unit + mechanism (patient/site/block/etc.).": (
            "N/A (observational study)."
        ),
        "If observational: briefly describe major confounding risks you expect (plain text).": (
            "Confounding by indication and disease severity: PD-L1 status may correlate with "
            "tumor biology, stage, cancer type, and patient characteristics that also affect survival."
        ),
    }
    discussion = _format_qa_block(questions, answers)
    return ProtocolDiscussionState(discussion=discussion)