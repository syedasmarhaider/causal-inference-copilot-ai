# src/python/workflows/state/protocol_state.py
from __future__ import annotations

from typing import List, Literal, TypedDict, NotRequired

TimeZeroType = Literal["COLUMN", "CONCEPTUAL"]
WindowUnit = Literal["minutes", "hours", "days", "weeks", "months", "years"]


class ProtocolState(TypedDict):
    accepted: bool
    clarified: List[str]
    open_questions: List[str]

    population: str

    time_zero_type: TimeZeroType
    time_zero: str
    time_zero_definition: str

    treatment: str
    treatment_window_start: str
    treatment_window_end: str
    treatment_window_unit: WindowUnit

    comparator: str

    outcome: str
    outcome_is_duration: bool
    outcome_window: str
    outcome_window_unit: WindowUnit

    covariates: List[str]
    effect_modifiers: List[str]
    censoring_rules: List[str]

    # Optional: internal-only bookkeeping (still ONLY inside protocol)
    user_log: NotRequired[List[str]]
    assistant_log: NotRequired[List[str]]
    revision: NotRequired[int]


def get_string_protocol_state(protocol: ProtocolState | None) -> str:
    if protocol is None:
        return "Protocol: None"
    return (
        f"Accepted: {protocol['accepted']} | "
        f"Population: {protocol['population']} | "
        f"Time Zero Type: {protocol['time_zero_type']} | "
        f"Time Zero Definition: {protocol['time_zero_definition']} | "
        f"Treatment: {protocol['treatment']} | "
        f"Comparator: {protocol['comparator']} | "
        f"Outcome: {protocol['outcome']} | "
        f"Covariates: {', '.join(protocol['covariates'])} | "
        f"Effect Modifiers: {', '.join(protocol['effect_modifiers'])}"
    )