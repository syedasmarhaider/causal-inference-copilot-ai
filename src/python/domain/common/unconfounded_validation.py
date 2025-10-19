from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Any


# -------------------------------
# Helpers: typed default factories (make Pylance happy)
# -------------------------------

def _severity_override_default() -> Dict["IssueCode", "Severity"]:
    return {}

def _str_float_dict() -> Dict[str, float]:
    return {}

def _str_any_dict() -> Dict[str, Any]:
    return {}

def _issues_list_default() -> List["ValidationIssue"]:
    return []

def _str_list_default() -> List[str]:
    return []


# -------------------------------
# Severity & Code for clear action
# -------------------------------

class Severity(Enum):
    """How urgently the pipeline must react."""
    HARD = auto()     # -> Stop estimation. Life-safety / correctness risk if ignored.
    WARNING = auto()  # -> Proceed with caution; review and possibly adjust.
    INFO = auto()     # -> For visibility only (book-keeping / audit).


class IssueCode(Enum):
    """Standardized issue IDs (stable; log & test against these)."""
    TYPE_MISMATCH = auto()               # dtype in data != declared dtype in metadata
    POST_TREAT_FEATURE = auto()          # feature timestamp occurs after treatment time
    IMMORTAL_TIME = auto()               # outcome window starts before treatment in as-treated design
    MISSINGNESS_EXCESS = auto()          # feature missingness > allowed threshold
    OVERLAP_TAIL_MASS_HIGH = auto()      # too many rows with propensity outside [lo, hi]
    OVERLAP_ESS_TOO_LOW = auto()         # effective sample size (diagnostic) too small
    FEW_EVENTS_PER_FOLD = auto()         # binary outcome: too few events per fold/arm
    CLASS_IMBALANCE_EXTREME = auto()     # treated share or outcome prevalence extreme
    PREPROC_OUTSIDE_FOLD = auto()        # leakage: fit/transform not confined to folds
    CLUSTER_LEAKAGE = auto()             # same group_id appears in train and test
    SURVIVAL_COLUMNS_MISSING = auto()    # survival outcome selected but time/event cols absent
    CENSORING_GAP_LARGE = auto()         # censoring differs a lot across arms (diagnostic)
    TIME_SPLIT_VIOLATION = auto()        # time-aware split requested but violated
    TRANSPORT_SHIFT_LARGE = auto()       # big source→target covariate shift (diagnostic)
    POLICY_SPEC_MISSING = auto()         # policy evaluation requested but utility/budget undefined


# -------------------------------
# Spec: the rulebook (validation-only)
# -------------------------------

@dataclass
class ValidationSpec:
    """
    Validation-only thresholds & toggles (read-only checks).
    If a rule is violated, the validator *reports* an Issue; it does NOT mutate data.
    """

    # ---- Missingness ----
    missingness_max: float = 0.40  # >40% missing → unstable; report MISSINGNESS_EXCESS

    # ---- Overlap diagnostics (propensity support) ----
    trim_lo: float = 0.02          # hypothetical lower support bound
    trim_hi: float = 0.98          # hypothetical upper support bound
    max_tail_mass: float = 0.15    # if >15% would be off-support, flag instability
    weight_clip_for_diagnostic: float = 10.0  # cap for ESS diagnostic only
    ess_min_frac: float = 0.60     # ESS/n must be ≥ 0.60 or report OVERLAP_ESS_TOO_LOW

    # ---- Outcome prevalence / events per fold ----
    min_events_per_fold: int = 30
    min_treated_per_fold: int = 50
    min_control_per_fold: int = 50

    # ---- Splitting hygiene ----
    n_folds: int = 5
    time_aware: bool = True
    group_col: Optional[str] = None

    # ---- Temporal order (anti-leakage) ----
    forbid_post_treatment_features: bool = True
    require_outcome_window_after_treatment: bool = True

    # ---- Survival/censoring ----
    survival_requires: Optional[Tuple[str, str]] = None  # (time_to_event_col, event_col)
    censoring_gap_tol: float = 0.10

    # ---- Transportability (optional diagnostic) ----
    check_transport: bool = False
    transport_shift_tol: float = 0.10

    # ---- Policy evaluation (optional) ----
    require_policy_spec: bool = False

    # ---- Per-issue severity overrides (typed default) ----
    severity_override: Dict[IssueCode, Severity] = field(default_factory=_severity_override_default)
    # Example: {IssueCode.OVERLAP_TAIL_MASS_HIGH: Severity.HARD}


# -------------------------------
# Report: the receipt (no mutation)
# -------------------------------

@dataclass
class ValidationIssue:
    """
    One concrete problem found by the validator.
    - 'code' lets automation decide block/allow.
    - 'severity' states default action urgency (HARD/WARNING/INFO).
    - 'message' is human-readable; concise and actionable.
    - 'details' carries numbers (e.g., tail_mass=0.18, ess_frac=0.52).
    - 'columns' lists implicated columns (for quick fixes).
    """
    code: IssueCode
    severity: Severity
    message: str
    details: Dict[str, Any] = field(default_factory=_str_any_dict)
    columns: List[str] = field(default_factory=_str_list_default)


@dataclass
class ValidationReport:
    """
    The validator’s output. Purely descriptive/diagnostic; it does NOT alter data.

    Key design choices:
    - 'issues' gives a structured list you can filter by severity to decide proceed/block.
    - 'overlap_stats' summarizes propensity support so later stages can limit to in-support.
    - 'missingness_by_feature' informs which columns would likely need action in sanitization.
    - 'split_plan_summary' documents intended cross-fitting hygiene (auditability).
    """
    issues: List[ValidationIssue] = field(default_factory=_issues_list_default)

    # Diagnostics only (no operations performed)
    overlap_stats: Dict[str, float] = field(default_factory=_str_float_dict)
    # Expected keys:
    #   "prop_min","prop_max","q1","median","q3","tail_mass","ess_raw","ess_after_clip","ess_frac"

    potential_trim_fraction: Optional[float] = None  # hypothetical: how much would be trimmed at [lo,hi]
    missingness_by_feature: Dict[str, float] = field(default_factory=_str_float_dict)
    split_plan_summary: Dict[str, Any] = field(default_factory=_str_any_dict)

    notes: str = ""  # free-text audit trail

    def has_hard_fail(self) -> bool:
        """True if any HARD issue is present (use to block estimation)."""
        return any(iss.severity == Severity.HARD for iss in self.issues)

    def summarize(self) -> Dict[str, Any]:
        """Small machine-readable digest for logs/GUI."""
        counts = {"HARD": 0, "WARNING": 0, "INFO": 0}
        for iss in self.issues:
            counts[iss.severity.name] += 1
        return {
            "counts": counts,
            "overlap": self.overlap_stats,
            "potential_trim_fraction": self.potential_trim_fraction,
            "n_problem_features": sum(
                1 for _, m in self.missingness_by_feature.items()
                if m > 0
            ),
            "split_plan": self.split_plan_summary,
        }
