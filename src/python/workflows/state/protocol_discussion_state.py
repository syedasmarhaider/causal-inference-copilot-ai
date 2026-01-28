from __future__ import annotations

from dataclasses import dataclass
from typing import List


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