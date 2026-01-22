from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ProtocolDiscussionState:
    discussion: str = ""

    @staticmethod
    def get_questions() -> List[str]:
        return [
            "What is the causal question (one sentence, effect of treatment on outcome)?",
            "Is this randomized/experiment (RCT) or observational? (Write: RCT / Observational / Unknown)",
            "Define the target population/cohort (eligibility in plain text).",
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
