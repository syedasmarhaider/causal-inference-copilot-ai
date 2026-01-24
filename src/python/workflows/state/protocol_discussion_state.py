from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProtocolDiscussionState:
    discussion: str = ""

    @staticmethod
    def get_questions() -> List[str]:
        return [
            "What is the causal question (one sentence, effect of treatment on outcome)?",
            "Is this randomized/experiment (RCT) or observational? (Write: RCT / Observational / Unknown)",
            "Define the target population/cohort (eligibility in plain text, population can be all the data).",
            "Define Time Zero in plain text (the moment follow-up starts).",
            "If Time Zero is a dataset column, write the column name; otherwise describe how it is constructed.",
            "What is the treatment/exposure (plain text or column name if known)?",
            "Define the treatment window relative to Time Zero (plain text).",
            "What is the comparator/control condition (plain text)?",
            "What is the outcome (plain text or column name if known)?",
            "Is the outcome time-to-event / duration? (Yes/No/Unknown)",
            "Define the outcome window/horizon relative to Time Zero (plain text).",
            "List covariates to adjust for (comma-separated, plain text).",
            "List effect modifiers/subgroups of interest (comma-separated, plain text).",
            "List censoring rules / follow-up end rules / loss-to-follow-up notes (comma-separated, plain text).",
            "If RCT: briefly describe randomization unit + mechanism (patient/site/block/etc.).",
            "If observational: briefly describe major confounding risks you expect (plain text).",
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