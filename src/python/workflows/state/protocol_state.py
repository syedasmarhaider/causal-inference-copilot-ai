from __future__ import annotations

from typing import Final, List, Literal, Sequence, TypedDict, cast, get_args

from typing import NotRequired


# -----------------------------
# Core enums
# -----------------------------
TimeZeroType = Literal["COLUMN", "CONCEPTUAL"]
WindowUnit = Literal["minutes", "hours", "days", "weeks", "months", "years"]

FilterOp = Literal[
    "==", "!=", "in", "not_in",
    ">=", "<=", ">", "<",
    "is_null", "not_null",
]


# -----------------------------
# Exclusions
# -----------------------------
class ExclusionRule(TypedDict):
    column: str
    op: FilterOp
    values: List[str]
    reason: str


# -----------------------------
# Treatment specs
# -----------------------------
TreatmentKind = Literal["binary", "continuous", "categorical"]

NumericTransform = Literal["none", "log", "standardize", "minmax"]


class BinaryTreatmentSpec(TypedDict):
    kind: Literal["binary"]
    column: str
    treated: str
    control: str
    # optional: allow coercion of raw values
    treated_aliases: NotRequired[List[str]]
    control_aliases: NotRequired[List[str]]


class ContinuousTreatmentSpec(TypedDict):
    kind: Literal["continuous"]
    column: str
    # optional: metadata (used for normalization / sanity checks)
    unit: NotRequired[str]
    transform: NotRequired[NumericTransform]
    clip_min: NotRequired[float]
    clip_max: NotRequired[float]


class CategoricalTreatmentSpec(TypedDict):
    kind: Literal["categorical"]
    column: str
    levels: List[str]
    # optional: choose a baseline; some estimators need/assume one
    baseline: NotRequired[str]


TreatmentSpec = BinaryTreatmentSpec | ContinuousTreatmentSpec | CategoricalTreatmentSpec


# -----------------------------
# Outcome specs
# -----------------------------
OutcomeKind = Literal["binary", "continuous", "categorical", "duration"]


class BinaryOutcomeSpec(TypedDict):
    kind: Literal["binary"]
    column: str
    event: str
    non_event: str
    event_aliases: NotRequired[List[str]]
    non_event_aliases: NotRequired[List[str]]


class ContinuousOutcomeSpec(TypedDict):
    kind: Literal["continuous"]
    column: str
    unit: NotRequired[str]
    transform: NotRequired[NumericTransform]
    clip_min: NotRequired[float]
    clip_max: NotRequired[float]


class CategoricalOutcomeSpec(TypedDict):
    kind: Literal["categorical"]
    column: str
    levels: List[str]
    baseline: NotRequired[str]


class DurationOutcomeSpec(TypedDict):
    """
    Time-to-event without requiring explicit date columns.
    - duration_column: e.g. "Overall Survival (Months)"
    - event_column: e.g. "Overall Survival Status"
    """
    kind: Literal["duration"]
    duration_column: str
    event_column: str
    event_value: str
    censor_value: str
    # optional: handle different censor encodings
    event_aliases: NotRequired[List[str]]
    censor_aliases: NotRequired[List[str]]


OutcomeSpec = BinaryOutcomeSpec | ContinuousOutcomeSpec | CategoricalOutcomeSpec | DurationOutcomeSpec


# -----------------------------
# ProtocolState (v2)
# -----------------------------
REQUIRED_KEYS = [
    # cohort
    "population",
    "exclusions",

    # time / design
    "time_zero_type",
    "time_zero",
    "time_zero_definition",

    # treatment
    "treatment_spec",
    "treatment_window_start",
    "treatment_window_end",
    "treatment_window_unit",

    # outcome
    "outcome_spec",
    "outcome_window",
    "outcome_window_unit",

    # adjustment / heterogeneity
    "covariates",
    "effect_modifiers",

    # missingness / censoring narrative
    "censoring_rules",

    # RCT vs observational
    "experiment_type",
]

ALLOWED_TIME_ZERO: Final[set[str]] = set(cast(Sequence[str], get_args(TimeZeroType)))
ALLOWED_UNITS: Final[set[str]] = set(cast(Sequence[str], get_args(WindowUnit)))
ALLOWED_OPS: Final[set[str]] = set(cast(Sequence[str], get_args(FilterOp)))

ALLOWED_TREAT_KINDS: Final[set[str]] = {"binary", "continuous", "categorical"}
ALLOWED_OUT_KINDS: Final[set[str]] = {"binary", "continuous", "categorical", "duration"}


class ProtocolState(TypedDict):
    # Cohort
    population: str
    exclusions: List[ExclusionRule]

    # Time / design
    time_zero_type: TimeZeroType
    time_zero: str
    time_zero_definition: str

    # Treatment
    treatment_spec: TreatmentSpec
    treatment_window_start: str
    treatment_window_end: str
    treatment_window_unit: WindowUnit

    # Outcome
    outcome_spec: OutcomeSpec
    outcome_window: str
    outcome_window_unit: WindowUnit

    # Covariates / modifiers
    covariates: List[str]         # W (adjustment)
    effect_modifiers: List[str]   # X (heterogeneity features)

    # Missingness / censoring rules (plain English OK)
    censoring_rules: List[str]

    # "RCT" or "Observational"
    experiment_type: str

    # Optional human-readable text (UI/logging only)
    treatment_text: NotRequired[str]
    outcome_text: NotRequired[str]


# -----------------------------
# Pretty printer
# -----------------------------
def get_string_protocol_state(protocol: ProtocolState | None) -> str:
    if protocol is None:
        return "Protocol: None"

    excl = protocol.get("exclusions", [])
    excl_s = "; ".join([f"{e['column']} {e['op']} {e['values']}" for e in excl]) if excl else "None"

    t = protocol["treatment_spec"]
    if t["kind"] == "binary":
        t_s = f"`{t['column']}`: '{t['treated']}' vs '{t['control']}'"
    elif t["kind"] == "continuous":
        t_s = f"`{t['column']}`: continuous"
    else:
        t_s = f"`{t['column']}`: categorical levels={t['levels']}"

    y = protocol["outcome_spec"]
    if y["kind"] == "binary":
        y_s = f"`{y['column']}`: '{y['event']}' vs '{y['non_event']}'"
    elif y["kind"] == "continuous":
        y_s = f"`{y['column']}`: continuous"
    elif y["kind"] == "categorical":
        y_s = f"`{y['column']}`: categorical levels={y['levels']}"
    else:
        y_s = f"`{y['duration_column']}` + `{y['event_column']}`: duration/event"

    return (
        f"Population: {protocol['population']} | "
        f"Exclusions: {excl_s} | "
        f"Time Zero Type: {protocol['time_zero_type']} | "
        f"Time Zero Definition: {protocol['time_zero_definition']} | "
        f"Treatment: {t_s} | "
        f"Outcome: {y_s} | "
        f"Covariates(W): {', '.join(protocol['covariates'])} | "
        f"Effect Modifiers(X): {', '.join(protocol['effect_modifiers'])}"
    )
