from __future__ import annotations

from typing import Final, List, Literal, Sequence, TypedDict, cast, get_args
from typing import NotRequired

# -----------------------------
# Enums
# -----------------------------
TimeZeroType = Literal["COLUMN", "CONCEPTUAL"]
WindowUnit = Literal["minutes", "hours", "days", "weeks", "months", "years"]
FilterOp = Literal["==", "!=", "in", "not_in", ">=", "<=", ">", "<"]
StudyDesignType = Literal["RCT", "Observational"]


# -----------------------------
# External cohort exclusions only
# -----------------------------
class ExclusionRule(TypedDict):
    column: str
    op: FilterOp
    values: List[str]
    reason: str


# -----------------------------
# Treatment spec (single-treatment)
# -----------------------------
class BinaryTreatmentSpec(TypedDict):
    kind: Literal["binary"]
    column: str
    treated_values: List[str]
    control_values: List[str]


class ContinuousTreatmentSpec(TypedDict):
    kind: Literal["continuous"]
    column: str
    valid_min: NotRequired[float]
    valid_max: NotRequired[float]


class CategoricalTreatmentSpec(TypedDict):
    kind: Literal["categorical"]
    column: str
    included_levels: List[str]


TreatmentSpec = BinaryTreatmentSpec | ContinuousTreatmentSpec | CategoricalTreatmentSpec


# -----------------------------
# Outcome spec
# -----------------------------
class BinaryOutcomeSpec(TypedDict):
    kind: Literal["binary"]
    column: str
    event_values: List[str]
    non_event_values: List[str]


class ContinuousOutcomeSpec(TypedDict):
    kind: Literal["continuous"]
    column: str
    valid_min: NotRequired[float]
    valid_max: NotRequired[float]


class CategoricalOutcomeSpec(TypedDict):
    kind: Literal["categorical"]
    column: str
    included_levels: List[str]


OutcomeSpec = BinaryOutcomeSpec | ContinuousOutcomeSpec | CategoricalOutcomeSpec


# -----------------------------
# ProtocolState (v3 minimal, executable)
# -----------------------------
REQUIRED_KEYS: Final[List[str]] = [
    "exclusions",
    "time_zero_type",
    "time_zero",
    "time_zero_definition",
    "treatment_spec",
    "treatment_window_start",
    "treatment_window_end",
    "treatment_window_unit",
    "outcome_spec",
    "outcome_window",
    "outcome_window_unit",
    "covariates",
    "effect_modifiers",
    "experiment_type",
]

ALLOWED_TIME_ZERO: Final[set[str]] = set(cast(Sequence[str], get_args(TimeZeroType)))
ALLOWED_UNITS: Final[set[str]] = set(cast(Sequence[str], get_args(WindowUnit)))
ALLOWED_OPS: Final[set[str]] = set(cast(Sequence[str], get_args(FilterOp)))
ALLOWED_STUDY_DESIGN: Final[set[str]] = set(cast(Sequence[str], get_args(StudyDesignType)))


class ProtocolState(TypedDict):
    exclusions: List[ExclusionRule]

    time_zero_type: TimeZeroType
    time_zero: str
    time_zero_definition: str

    # Inclusion is defined by this spec (all other values excluded by default)
    treatment_spec: TreatmentSpec
    treatment_window_start: str
    treatment_window_end: str
    treatment_window_unit: WindowUnit

    # Inclusion is defined by this spec (all other values excluded by default)
    outcome_spec: OutcomeSpec
    outcome_window: str
    outcome_window_unit: WindowUnit

    covariates: List[str]
    effect_modifiers: List[str]

    experiment_type: StudyDesignType
