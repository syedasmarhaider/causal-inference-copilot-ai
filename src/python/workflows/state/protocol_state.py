from __future__ import annotations

from typing import List, Literal, TypedDict


TimeZeroType = Literal["COLUMN", "CONCEPTUAL"]
WindowUnit = Literal["minutes", "hours", "days", "weeks", "months", "years"]

FilterOp = Literal["==", "!=", "in", "not_in", ">=", "<=", ">", "<", "is_null", "not_null"]
REQUIRED_KEYS = [
    "population",
    "exclusions",
    "time_zero_type",
    "time_zero",
    "time_zero_definition",
    "treatment",
    "treatment_window_start",
    "treatment_window_end",
    "treatment_window_unit",
    "outcome",
    "outcome_is_duration",
    "outcome_window",
    "outcome_window_unit",
    "covariates",
    "effect_modifiers",
    "censoring_rules",
    "experiment_type",
]

class ExclusionRule(TypedDict):
    column: str
    op: FilterOp
    values: List[str]
    reason: str


class ProtocolState(TypedDict):
    # Cohort
    population: str
    exclusions: List[ExclusionRule]  # NEW

    # Time / design
    time_zero_type: TimeZeroType
    time_zero: str
    time_zero_definition: str

    # Treatment
    treatment: str
    treatment_window_start: str
    treatment_window_end: str
    treatment_window_unit: WindowUnit

    # Outcome
    outcome: str
    outcome_is_duration: bool
    outcome_window: str
    outcome_window_unit: WindowUnit

    # Covariates / modifiers
    covariates: List[str]
    effect_modifiers: List[str]

    # Missingness / censoring rules (plain English is OK)
    censoring_rules: List[str]

    # "RCT" or "Observational"
    experiment_type: str


def get_string_protocol_state(protocol: ProtocolState | None) -> str:
    if protocol is None:
        return "Protocol: None"

    excl = protocol.get("exclusions", [])
    excl_s = "; ".join(
        [f"{e['column']} {e['op']} {e['values']}" for e in excl]
    ) if excl else "None"

    return (
        f"Population: {protocol['population']} | "
        f"Exclusions: {excl_s} | "
        f"Time Zero Type: {protocol['time_zero_type']} | "
        f"Time Zero Definition: {protocol['time_zero_definition']} | "
        f"Treatment: {protocol['treatment']} | "
        f"Outcome: {protocol['outcome']} | "
        f"Covariates: {', '.join(protocol['covariates'])} | "
        f"Effect Modifiers: {', '.join(protocol['effect_modifiers'])}"
    )
