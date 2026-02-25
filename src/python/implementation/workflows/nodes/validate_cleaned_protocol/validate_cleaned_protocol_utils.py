from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, TypedDict

import pandas as pd
import pandas.api.types as ptypes
from typing import cast
from python.implementation.workflows.nodes.compile_protocol.protocol_specs import (
    BinaryOutcomeSpecModel,
    BinaryTreatmentSpecModel,
    CategoricalTreatmentSpecModel,
    ContinuousOutcomeSpecModel,
)

from python.implementation.workflows.nodes.compile_protocol.protocol_specs import (
    ProtocolSpec,
)
from python.implementation.workflows.utils.utils import BOOL_FALSE, BOOL_TRUE
# =============================================================================
# 5) Covariates / Effect Modifiers validations (pre-transform, raw df)
# =============================================================================

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, cast

import numpy as np

from python.implementation.workflows.utils.validation import ValidationSeverity

# TODO: move to tools
class ValidationIssue(TypedDict):
    severity: ValidationSeverity
    message: str
    evidence: Dict[str, Any]
    fix_hint: str | None

# =============================================================================
# 2) Structural invariants (protocol + df)
# =============================================================================

def validate_min_rows(
    df: pd.DataFrame,
    *,
    min_rows_fail: int = 20,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    """
    Hard structural check: we need rows to even talk about treatment/outcome variation later.
    """
    n = int(df.shape[0])
    metrics = {"n_rows": n, "min_rows_fail": int(min_rows_fail)}
    if n < int(min_rows_fail):
        return (
            [
                _issue(
                    severity="FAIL",
                    message="Dataset has too few rows for validation.",
                    evidence=metrics,
                    fix_hint="Verify filtering/whitelisting steps did not remove all rows.",
                )
            ],
            metrics,
        )
    return [], metrics


def validate_protocol_role_columns_invariants(protocol: ProtocolSpec) -> List[ValidationIssue]:
    """
    Pure protocol invariants about referenced column names (no DataFrame needed).

    FAIL:
      - duplicates within: outcome columns, covariates, effect_modifiers
      - covariates/effect_modifiers include treatment column or outcome columns
      - duration outcome: duration_column == event_column

    WARN:
      - covariates and effect_modifiers overlap
      - time_zero overlaps with any other role column when time_zero_type == "COLUMN"
    """
    issues: List[ValidationIssue] = []

    # -------------------------
    # Treatment column
    # -------------------------
    treatment_col: str = protocol.treatment_spec.column

    # -------------------------
    # Outcome columns (duration has two)
    # -------------------------

    outcome_cols = [protocol.outcome_spec.column]

    # -------------------------
    # Covariates / effect modifiers
    # -------------------------
    covariates: List[str] = list(protocol.covariates or [])
    effect_modifiers: List[str] = list(protocol.effect_modifiers or [])

    # -------------------------
    # time_zero is a real column only when time_zero_type == "COLUMN"
    # -------------------------
    time_zero_col = protocol.time_zero if protocol.time_zero_type == "COLUMN" else None

    # -------------------------
    # 1) Duplicates within lists (FAIL)
    # -------------------------
    for label, cols in (
        ("outcome_spec columns", outcome_cols),
        ("covariates", covariates),
        ("effect_modifiers", effect_modifiers),
    ):
        dups = _duplicates(cols)
        if dups:
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"Duplicate column names found in {label}.",
                    evidence={"duplicates": dups, "all": cols},
                    fix_hint="Remove duplicates in compiled protocol; each list should contain unique column names.",
                )
            )

    # duration outcome: duration_column and event_column must differ (FAIL)
    if len(outcome_cols) == 2 and outcome_cols[0] == outcome_cols[1]:
        issues.append(
            _issue(
                severity="FAIL",
                message="Duration outcome requires distinct duration_column and event_column.",
                evidence={"duration_column": outcome_cols[0], "event_column": outcome_cols[1]},
                fix_hint="Set outcome_spec.duration_column and outcome_spec.event_column to different columns.",
            )
        )

    # -------------------------
    # 2) covariates/effect_modifiers must not include treatment/outcome columns (FAIL)
    # -------------------------
    forbidden: Set[str] = {treatment_col, *outcome_cols}

    cov_bad = sorted([c for c in covariates if c in forbidden])
    em_bad = sorted([c for c in effect_modifiers if c in forbidden])

    if cov_bad:
        issues.append(
            _issue(
                severity="FAIL",
                message="covariates include treatment/outcome columns.",
                evidence={"overlap": cov_bad, "treatment_col": treatment_col, "outcome_cols": outcome_cols},
                fix_hint="Remove treatment/outcome columns from covariates; keep them only in treatment_spec/outcome_spec.",
            )
        )

    if em_bad:
        issues.append(
            _issue(
                severity="FAIL",
                message="effect_modifiers include treatment/outcome columns.",
                evidence={"overlap": em_bad, "treatment_col": treatment_col, "outcome_cols": outcome_cols},
                fix_hint="Remove treatment/outcome columns from effect_modifiers; keep them only in treatment_spec/outcome_spec.",
            )
        )

    # -------------------------
    # 3) covariates vs effect_modifiers overlap (WARN)
    # -------------------------
    overlap = sorted(set(covariates).intersection(set(effect_modifiers)))
    if overlap:
        issues.append(
            _issue(
                severity="WARN",
                message="covariates and effect_modifiers overlap: some columns appear in both lists.",
                evidence={"overlap_cols": overlap, "n_overlap": len(overlap)},
                fix_hint="Allowed, but redundant. Downstream code should dedupe combined feature lists.",
            )
        )

    # -------------------------
    # 4) time_zero overlaps with other role columns (WARN)
    # -------------------------
    if isinstance(time_zero_col, str) and time_zero_col.strip():
        tz_overlap = sorted({time_zero_col}.intersection({treatment_col, *outcome_cols, *covariates, *effect_modifiers}))
        if tz_overlap:
            issues.append(
                _issue(
                    severity="WARN",
                    message="time_zero column overlaps with another protocol role column.",
                    evidence={"time_zero_col": time_zero_col, "overlaps_with": tz_overlap},
                    fix_hint="If intentional, keep it. Otherwise pick a dedicated baseline/time column.",
                )
            )

    return issues


def validate_time_zero_semantics_protocol(
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    *,
    sample_n: int = 2000,
    parse_fail_rate_warn: float = 0.10,
    parse_fail_rate_fail: float = 0.50,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    """
    Lightweight time_zero validation for ProtocolSpec.

    Only applies when protocol.time_zero_type == "COLUMN".
    We do NOT enforce strict datetime dtype, but we flag clearly unparseable values.
    """
    issues: List[ValidationIssue] = []

    tz_type = protocol.time_zero_type
    if tz_type != "COLUMN":
        return issues, {"time_zero_type": tz_type}

    tz = protocol.time_zero
    if not tz.strip():
        issues.append(
            _issue(
                severity="FAIL",
                message="time_zero_type is 'COLUMN' but protocol.time_zero is missing/empty.",
                evidence={"time_zero": tz},
                fix_hint="Set protocol.time_zero to a non-empty dataset column name when time_zero_type='COLUMN'.",
            )
        )
        return issues, {"time_zero_col": tz}

    if tz not in df.columns:
        issues.append(
            _issue(
                severity="FAIL",
                message="time_zero column not found in dataframe.",
                evidence={"time_zero_col": tz, "n_df_cols": int(df.shape[1])},
                fix_hint="Ensure the time_zero column is retained after filtering and spelled exactly as in the dataset.",
            )
        )
        return issues, {"time_zero_col": tz, "present": False}

    s = df[tz]
    n = int(s.shape[0])
    miss_rate = float(s.isna().mean()) if n > 0 else 0.0

    metrics: Dict[str, Any] = {
        "time_zero_type": tz_type,
        "time_zero_col": tz,
        "dtype": str(s.dtype),
        "n_rows": n,
        "missing_rate": miss_rate,
    }

    # If already datetime-like => OK
    if ptypes.is_datetime64_any_dtype(s.dtype):
        return issues, metrics

    # Numeric times are possible but ambiguous (timestamp vs offset)
    if ptypes.is_numeric_dtype(s.dtype):
        issues.append(
            _issue(
                severity="WARN",
                message="time_zero column is numeric; ensure downstream logic interprets it correctly (timestamp vs offset).",
                evidence=metrics,
                fix_hint="If this is a timestamp, consider converting to datetime for clarity.",
            )
        )
        return issues, metrics

    # Try parsing strings/objects/categories to datetime
    ss = s.dropna()
    if ss.empty:
        issues.append(
            _issue(
                severity="FAIL",
                message="time_zero column has only missing values after filtering.",
                evidence=metrics,
                fix_hint="Fix upstream filtering/null purge or choose a valid time_zero column.",
            )
        )
        return issues, metrics

    # Deterministic sampling for speed
    if int(ss.shape[0]) > int(sample_n) and int(sample_n) > 0:
        ss = ss.sample(n=int(sample_n), random_state=0)

    parsed = pd.to_datetime(ss, errors="coerce", utc=False)
    fail_rate = float(parsed.isna().mean())

    metrics["parse_fail_rate_sample"] = fail_rate
    metrics["sample_n"] = int(ss.shape[0])

    if fail_rate >= float(parse_fail_rate_fail):
        issues.append(
            _issue(
                severity="FAIL",
                message="time_zero values are largely unparseable as datetimes (sample-based).",
                evidence=metrics,
                fix_hint="Ensure time_zero contains ISO-like datetimes or a consistent parseable format.",
            )
        )
    elif fail_rate >= float(parse_fail_rate_warn):
        issues.append(
            _issue(
                severity="WARN",
                message="Some time_zero values are unparseable as datetimes (sample-based).",
                evidence=metrics,
                fix_hint="Standardize time_zero format (e.g., ISO-8601) to avoid downstream windowing issues.",
            )
        )

    return issues, metrics

# =============================================================================
# 3) Treatment validations (pre-transform; whitelist already applied upstream)
# =============================================================================
def _treatment_allowed_literals(protocol: ProtocolSpec) -> List[str]:
    """
    Returns the allowed literal domain for treatment_spec.
    - binary -> [treated, control]
    - categorical -> levels
    - continuous -> None (no finite domain)
    """
    ts = protocol.treatment_spec
    if isinstance(ts, BinaryTreatmentSpecModel):
        return [ts.treated, ts.control]
    if isinstance(ts, CategoricalTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        return list(ts.levels)
    # Should not happen if schema is enforced
    raise ValueError(f"Unknown treatment_spec type: {type(ts)}")

def validate_treatment(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    min_count_per_literal_fail: int = 15,
    # imbalance gates (counts-only)
    min_share_fail: float = 0.05,
    max_ratio_fail: float = 20.0,
    min_neff_fail: float = 100.0,  # binary only; set 0 to disable
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    """
    STRICT (FAIL-only) treatment validation, step-by-step.

    Step 0: Column presence.
    Step 1: Hard missingness gate (missing_rate must be 0).
    Step 2: If continuous treatment => stop after missingness (no finite domain).
    Step 3: Allowed literals from protocol (source of truth).
    Step 4: Observed tokens from df (exact, no normalization).
    Step 5: Domain equality gates:
              - FAIL if unexpected values exist (observed \\ allowed)
              - FAIL if allowed values are missing (allowed \\ observed)
    Step 6: Minimum count per literal gate.
    Step 7: Imbalance gates (counts-only):
              - FAIL if min_share < min_share_fail
              - FAIL if max/min ratio > max_ratio_fail
              - binary only: FAIL if harmonic-mean effective sample size < min_neff_fail
    """
    issues: List[ValidationIssue] = []

    # -------------------------
    # Step 0: Column presence
    # -------------------------
    tcol = protocol.treatment_spec.column
    if tcol not in df.columns:
        logging.warning(f"FUCK Treatment column '{tcol}' not found in dataframe columns: {df.columns.tolist()}")
        metrics = {"treatment_col": tcol, "present": False}
        issues.append(
            _issue(
                severity="FAIL",
                message="treatment_spec.column not found in dataframe.",
                evidence=metrics,
                fix_hint="Ensure the treatment column is retained after filtering and matches exactly.",
            )
        )
        return issues, metrics

    s = df[tcol]
    n_rows = int(s.shape[0])
    missing_rate = float(s.isna().mean()) if n_rows > 0 else 0.0

    metrics: Dict[str, Any] = {
        "treatment_col": tcol,
        "present": True,
        "treatment_kind": getattr(protocol.treatment_spec, "kind", None),
        "dtype": str(s.dtype),
        "n_rows": n_rows,
        "missing_rate": missing_rate,
        "min_count_per_literal_fail": int(min_count_per_literal_fail),
        "min_share_fail": float(min_share_fail),
        "max_ratio_fail": float(max_ratio_fail),
        "min_neff_fail": float(min_neff_fail),
    }

    # -------------------------
    # Step 1: Hard missingness gate
    # -------------------------
    if n_rows == 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="No rows available to validate treatment.",
                evidence=metrics,
                fix_hint="Fix upstream filtering that removed all rows.",
            )
        )
        return issues, metrics

    if missing_rate > 0.0:
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment must not contain any missing values.",
                evidence=metrics,
                fix_hint="Exclude rows with missing treatment values upstream.",
            )
        )
        return issues, metrics

    # -------------------------
    # Step 3: Allowed literals (protocol truth)
    # -------------------------
    allowed_literals = _treatment_allowed_literals(protocol)
    allowed_unique: List[str] = list(dict.fromkeys(allowed_literals))  # stable unique
    allowed_set = set(allowed_unique)

    metrics["allowed_literals"] = list(allowed_literals)
    metrics["allowed_unique"] = list(allowed_unique)

    # -------------------------
    # Step 4: Observed tokens (exact, no normalization)
    # -------------------------
    # missing_rate==0, but keep it explicit
    s_nonnull = s.dropna()
    if s_nonnull.empty:
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment has no observed values after filtering.",
                evidence={**metrics, "domain_check": "no_nonnull_values", "n_unique_observed": 0},
                fix_hint="Fix upstream filtering/mapping so treatment values remain.",
            )
        )
        return issues, metrics

    obs_values = s_nonnull.unique().tolist()
    obs_set = set(obs_values)

    # counts keyed by allowed literals
    vc = s_nonnull.value_counts(dropna=True)
    counts_by_allowed: Dict[str, int] = {a: int(vc.get(a, 0)) for a in allowed_unique}

    metrics.update(
        {
            "n_unique_observed": int(len(obs_set)),
            "counts_by_allowed": counts_by_allowed,
        }
    )

    # -------------------------
    # Step 5: Strict domain equality gates
    # -------------------------
    unexpected = sorted(list(obs_set - allowed_set), key=lambda x: str(x))
    missing_allowed = sorted(list(allowed_set - obs_set), key=lambda x: str(x))

    metrics.update(
        {
            "n_unexpected": int(len(unexpected)),
            "unexpected": [{"value": _safe_display(v), "count": int(vc.get(v, 0))} for v in unexpected[:50]],
            "n_missing_allowed": int(len(missing_allowed)),
            "missing_allowed": [_safe_display(v) for v in missing_allowed[:50]],
        }
    )

    if unexpected:
        issues.append(
            _issue(
                severity="FAIL",
                message="Strict protocol violation: treatment contains values outside protocol literals (data not normalized).",
                evidence=metrics,
                fix_hint="Map/filter upstream so treatment values are exactly the protocol literals (treated/control or levels).",
            )
        )
        return issues, metrics

    if missing_allowed:
        issues.append(
            _issue(
                severity="FAIL",
                message="Strict protocol violation: not all protocol treatment literals are present after filtering.",
                evidence=metrics,
                fix_hint="Relax filtering or fix mapping so every allowed literal appears at least once (or update the protocol literals).",
            )
        )
        return issues, metrics

    # -------------------------
    # Step 6: Minimum count per literal gate
    # -------------------------
    low_counts: List[Dict[str, Any]] = [ 
        {"literal": _safe_display(a), "count": counts_by_allowed[a]}
        for a in allowed_unique
        if counts_by_allowed[a] < int(min_count_per_literal_fail)
    ]
    metrics["n_low_count"] = int(len(low_counts))
    metrics["low_count_literals"] = low_counts[:50]

    if low_counts:
        issues.append(
            _issue(
                severity="FAIL",
                message="Strict protocol violation: some treatment literals do not meet the minimum count threshold.",
                evidence=metrics,
                fix_hint="Increase cohort size, relax filters, or redefine mapping so each literal has enough rows.",
            )
        )
        return issues, metrics

    # -------------------------
    # Step 7: Imbalance gates (counts-only)
    # -------------------------
    total = sum(counts_by_allowed.values())
    if total <= 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment total count is zero across allowed literals.",
                evidence=metrics,
                fix_hint="Fix upstream filtering/mapping so treatment values remain.",
            )
        )
        return issues, metrics

    shares_by_allowed = {a: counts_by_allowed[a] / total for a in allowed_unique}
    min_share = min(shares_by_allowed.values())
    min_count = min(counts_by_allowed.values())
    max_count = max(counts_by_allowed.values())
    ratio = (max_count / min_count) if min_count > 0 else float("inf")

    # concentration diagnostics (mostly useful for multi-arm)
    hhi = sum(p * p for p in shares_by_allowed.values())
    entropy = -sum(p * math.log(p) for p in shares_by_allowed.values() if p > 0.0)

    metrics.update(
        {
            "total_in_allowed": int(total),
            "shares_by_allowed": shares_by_allowed,
            "min_share": float(min_share),
            "imbalance_ratio_max_over_min": float(ratio),
            "hhi": float(hhi),
            "entropy": float(entropy),
        }
    )

    if float(min_share) < float(min_share_fail):
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment arm imbalance: minimum arm share is below threshold (positivity risk).",
                evidence=metrics,
                fix_hint="Relax filtering / broaden cohort / redefine treatment so each arm has sufficient support.",
            )
        )
        return issues, metrics

    if float(ratio) > float(max_ratio_fail):
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment arm imbalance: max/min count ratio exceeds threshold.",
                evidence=metrics,
                fix_hint="Relax filtering or collapse levels; extreme imbalance makes estimates unstable.",
            )
        )
        return issues, metrics

    # Binary-only effective sample size gate (harmonic mean)
    if len(allowed_unique) == 2 and float(min_neff_fail) > 0.0:
        a0, a1 = allowed_unique[0], allowed_unique[1]
        n0, n1 = counts_by_allowed[a0], counts_by_allowed[a1]
        neff = (2.0 * n0 * n1) / (n0 + n1) if (n0 + n1) > 0 else 0.0
        metrics["n_eff_harmonic_mean"] = float(neff)

        if float(neff) < float(min_neff_fail):
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Treatment arm imbalance: effective sample size (harmonic mean) is too small.",
                    evidence=metrics,
                    fix_hint="Increase cohort size or redefine treatment/filters to raise effective overlap.",
                )
            )
            return issues, metrics

    return issues, metrics



# =============================================================================
# 4) Outcome validations (pre-transform; whitelist already applied upstream)
# =============================================================================
def _outcome_allowed_literals(protocol: ProtocolSpec) -> List[str] | None:
    """
    Protocol is source of truth for DISCRETE outcome domains.

    Returns allowed literals:
      - binary      -> [event, non_event]
      - categorical -> levels
      - continuous  -> None (no finite literal domain)
    """
    ys = protocol.outcome_spec
    if isinstance(ys, BinaryOutcomeSpecModel):
        return [ys.event, ys.non_event]
    if isinstance(ys, ContinuousOutcomeSpecModel):  # pyright: ignore[reportUnnecessaryIsInstance]
        return None
    raise ValueError(f"Unknown outcome_spec type: {type(ys)}")


def validate_outcome(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    # outcome missingness policy (NEW)
    allow_missing_outcome: bool = True,
    missing_rate_warn: float = 0.01,
    missing_rate_fail: float = 0.20,
    # if outcome missingness differs a lot by treatment arm, that's a red flag (selection)
    arm_missing_rate_diff_warn: float = 0.05,
    arm_missing_rate_diff_fail: float = 0.15,
    min_arm_n_for_missingness_gates: int = 50,
    # count gates (discrete outcomes; applied on NON-MISSING outcomes only)
    min_count_per_literal_fail: int = 15,
    # continuous outcome gates
    min_unique_numeric_fail: int = 2,
    require_strict_numeric: bool = True,  # FAIL if any non-numeric token among NON-MISSING entries
    # imbalance gates (discrete outcomes; computed on NON-MISSING outcomes only)
    min_share_fail: float = 0.05,
    max_ratio_fail: float = 20.0,
    min_neff_fail: float = 100.0,  # only used when there are exactly 2 observed literals
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    """
    Outcome validation that DOES NOT require outcome missingness == 0.

    Key behavior:
      - If allow_missing_outcome=True, missing outcomes do NOT automatically fail.
      - We still validate domain + counts + imbalance on the NON-MISSING subset.
      - We additionally diagnose outcome missingness overall and by treatment arm (important in medical/EHR).

    Notes:
      - If missingness is high or very different by treatment arm, we FAIL (or WARN) because naive
        complete-case analysis can be biased (selection on being observed).
    """
    import math

    issues: List[ValidationIssue] = []
    ys = protocol.outcome_spec
    ycol = ys.column
    tcol = protocol.treatment_spec.column

    # -------------------------
    # Step 1: presence + empty df
    # -------------------------
    n_rows = int(df.shape[0])
    missing_cols = [c for c in [ycol, tcol] if c not in df.columns]
    if missing_cols:
        metrics = {
            "present": False,
            "outcome_kind": getattr(ys, "kind", None),
            "required_cols": [ycol, tcol],
            "missing_cols": missing_cols[:50],
            "n_missing": int(len(missing_cols)),
            "n_rows": n_rows,
            "n_df_cols": int(df.shape[1]),
        }
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome/treatment column referenced by protocol is missing from dataframe.",
                evidence=metrics,
                fix_hint="Ensure treatment/outcome columns are retained and match exactly.",
            )
        )
        return issues, metrics

    if n_rows == 0:
        metrics = {"present": True, "outcome_kind": getattr(ys, "kind", None), "n_rows": 0}
        issues.append(
            _issue(
                severity="FAIL",
                message="No rows available to validate outcome.",
                evidence=metrics,
                fix_hint="Fix upstream filtering that removed all rows.",
            )
        )
        return issues, metrics

    # -------------------------
    # Step 2: outcome missingness (overall + by treatment arm)
    # -------------------------
    y = df[ycol]
    t = df[tcol]

    n_y_missing = int(y.isna().sum())
    miss_rate = float(n_y_missing / max(1, n_rows))
    n_y_nonmissing = int(n_rows - n_y_missing)

    # compute missingness by treatment arm (only for arms with enough rows)
    # (works for binary or categorical treatments)
    arm_stats: List[Dict[str, Any]] = []
    # include NA treatment rows as their own "arm" for diagnostics
    for arm_val, g in df.groupby(t, dropna=False): # pyright: ignore[reportUnknownMemberType]
        n_arm = int(g.shape[0])
        mr_arm = float(g[ycol].isna().mean())
        arm_stats.append({"arm": _safe_display(arm_val), "n": n_arm, "missing_rate_y": mr_arm})

    # treatment-arm missingness dispersion
    eligible_arm_rates = [a["missing_rate_y"] for a in arm_stats if int(a["n"]) >= int(min_arm_n_for_missingness_gates)]
    arm_diff = float(max(eligible_arm_rates) - min(eligible_arm_rates)) if len(eligible_arm_rates) >= 2 else 0.0

    metrics: Dict[str, Any] = {
        "present": True,
        "outcome_kind": getattr(ys, "kind", None),
        "treatment_col": tcol,
        "outcome_col": ycol,
        "n_rows": n_rows,
        "n_y_missing": n_y_missing,
        "missing_rate_y": miss_rate,
        "n_y_nonmissing": n_y_nonmissing,
        "allow_missing_outcome": bool(allow_missing_outcome),
        "missing_rate_warn": float(missing_rate_warn),
        "missing_rate_fail": float(missing_rate_fail),
        "arm_missing_rate_diff_warn": float(arm_missing_rate_diff_warn),
        "arm_missing_rate_diff_fail": float(arm_missing_rate_diff_fail),
        "min_arm_n_for_missingness_gates": int(min_arm_n_for_missingness_gates),
        "missingness_by_treatment_arm": arm_stats[:50],
        "arm_missing_rate_diff": arm_diff,
        "min_count_per_literal_fail": int(min_count_per_literal_fail),
        "min_unique_numeric_fail": int(min_unique_numeric_fail),
        "require_strict_numeric": bool(require_strict_numeric),
        "min_share_fail": float(min_share_fail),
        "max_ratio_fail": float(max_ratio_fail),
        "min_neff_fail": float(min_neff_fail),
    }

    if not allow_missing_outcome and n_y_missing > 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome contains missing values but allow_missing_outcome=False.",
                evidence=metrics,
                fix_hint="Drop missing outcomes upstream or set allow_missing_outcome=True and handle missingness explicitly.",
            )
        )
        return issues, metrics

    # If missingness exists, add gates (WARN/FAIL) but do not necessarily stop.
    if n_y_missing > 0:
        if miss_rate >= float(missing_rate_fail):
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Outcome missingness is too high; complete-case estimation is likely biased/unreliable.",
                    evidence=metrics,
                    fix_hint="Consider IPW/MI for missing outcomes, redefine outcome window, or improve outcome capture.",
                )
            )
            return issues, metrics
        if miss_rate >= float(missing_rate_warn):
            issues.append(
                _issue(
                    severity="WARN",
                    message="Outcome has non-trivial missingness; complete-case estimation may introduce selection bias.",
                    evidence=metrics,
                    fix_hint="Check missingness by treatment arm; consider IPW/MI if missingness is differential.",
                )
            )

        # differential missingness by treatment arm is a stronger red flag
        if arm_diff >= float(arm_missing_rate_diff_fail) and len(eligible_arm_rates) >= 2:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Outcome missingness differs substantially by treatment arm (high selection-bias risk).",
                    evidence=metrics,
                    fix_hint="Do not naively drop missing outcomes; use IPW/MI or redesign outcome capture/window.",
                )
            )
            return issues, metrics
        if arm_diff >= float(arm_missing_rate_diff_warn) and len(eligible_arm_rates) >= 2:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Outcome missingness differs by treatment arm (selection-bias risk).",
                    evidence=metrics,
                    fix_hint="Prefer IPW/MI for missing outcomes; at minimum report arm-specific missingness.",
                )
            )

    # If everything is missing, we cannot validate domain/counts.
    if n_y_nonmissing == 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome is missing for all rows; cannot validate or estimate effects.",
                evidence=metrics,
                fix_hint="Fix outcome extraction/mapping; ensure outcome is recorded for at least some rows.",
            )
        )
        return issues, metrics

    # Work on NON-MISSING outcomes only from here on
    y_nm = y.dropna()

    # -------------------------
    # Step 3A: Continuous outcome (validate on NON-MISSING only)
    # -------------------------
    if isinstance(ys, ContinuousOutcomeSpecModel):
        v = pd.to_numeric(y_nm, errors="coerce")

        n_nonmissing = int(y_nm.shape[0])
        n_numeric = int(v.notna().sum())
        n_bad = int(n_nonmissing - n_numeric)

        metrics.update(
            {
                "dtype": str(y.dtype),
                "n_nonmissing_used": n_nonmissing,
                "n_numeric_used": n_numeric,
                "n_non_numeric_nonmissing_used": n_bad,
                "numeric_parse_rate_used": float(n_numeric / max(1, n_nonmissing)),
            }
        )

        if require_strict_numeric and n_bad > 0:
            bad_mask = v.isna()
            bad_sample = y_nm.loc[bad_mask].unique().tolist()[:25]
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Continuous outcome contains non-numeric tokens among non-missing entries (data not clean).",
                    evidence={**metrics, "bad_value_sample": [_safe_display(x) for x in bad_sample]},
                    fix_hint="Clean/mapping step required: make the outcome column strictly numeric.",
                )
            )
            return issues, metrics

        n_unique = int(v.nunique(dropna=True))
        metrics["n_unique_numeric_used"] = n_unique

        if n_unique < int(min_unique_numeric_fail):
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Continuous outcome is degenerate (too few unique numeric values among observed outcomes).",
                    evidence=metrics,
                    fix_hint="Use an outcome with variability or adjust cohort/filters that collapsed variation.",
                )
            )
            return issues, metrics

        return issues, metrics

    # -------------------------
    # Step 3B: Binary/Categorical outcome (strict domain on NON-MISSING only)
    # -------------------------
    allowed_literals = _outcome_allowed_literals(protocol)
    if allowed_literals is None:
        raise ValueError("Non-continuous outcome must have finite literal domain; got None.")

    allowed_unique = list(dict.fromkeys(allowed_literals))
    allowed_set = set(allowed_unique)

    obs_values = y_nm.unique().tolist()
    obs_set = set(obs_values)

    unexpected = sorted(list(obs_set - allowed_set), key=lambda x: str(x))

    vc = y_nm.value_counts(dropna=True)
    counts_by_observed = {v: int(vc.get(v, 0)) for v in sorted(list(obs_set), key=lambda x: str(x))}

    metrics.update(
        {
            "dtype": str(y.dtype),
            "allowed_literals": list(allowed_literals),
            "allowed_unique": allowed_unique,
            "n_unique_observed_nonmissing": int(len(obs_set)),
            "unexpected": [{"value": _safe_display(v), "count": int(vc.get(v, 0))} for v in unexpected[:50]],
            "n_unexpected": int(len(unexpected)),
            "counts_by_observed": {str(_safe_display(k)): int(v) for k, v in counts_by_observed.items()},
        }
    )

    if unexpected:
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome contains values outside protocol literals among non-missing entries (data not normalized).",
                evidence=metrics,
                fix_hint="Map/filter upstream so non-missing outcome values are exactly the protocol literals.",
            )
        )
        return issues, metrics
    else:
        # categorical: require at least 2 observed levels among NON-MISSING
        if len(obs_set) < 2:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Categorical outcome has <2 observed levels among non-missing entries; cannot estimate contrasts.",
                    evidence=metrics,
                    fix_hint="Relax filters, increase cohort, or redefine outcome so multiple levels appear.",
                )
            )
            return issues, metrics

    # min count per observed literal (do NOT penalize allowed-but-absent levels)
    low_counts: List[Dict[str, Any]] = [
        {"literal": _safe_display(v), "count": int(vc.get(v, 0))}
        for v in obs_set
        if int(vc.get(v, 0)) < int(min_count_per_literal_fail)
    ]
    if low_counts:
        issues.append(
            _issue(
                severity="FAIL",
                message="Some observed outcome literals do not meet the minimum count threshold (among non-missing).",
                evidence={**metrics, "low_count_literals": low_counts[:50], "n_low_count": int(len(low_counts))},
                fix_hint="Increase cohort size, relax filters, or redefine mapping so each observed class has enough rows.",
            )
        )
        return issues, metrics

    # imbalance gates on observed non-missing outcomes
    total = int(vc.sum())
    shares = {v: float(vc.get(v, 0) / max(1, total)) for v in obs_set}
    min_share = float(min(shares.values()))
    min_count = int(min(int(vc.get(v, 0)) for v in obs_set))
    max_count = int(max(int(vc.get(v, 0)) for v in obs_set))
    ratio = float((max_count / min_count) if min_count > 0 else float("inf"))

    hhi = float(sum(p * p for p in shares.values()))
    entropy = float(-sum(p * math.log(p) for p in shares.values() if p > 0.0))

    metrics.update(
        {
            "total_nonmissing_in_domain": total,
            "shares_by_observed": {str(_safe_display(k)): float(v) for k, v in shares.items()},
            "min_share": min_share,
            "imbalance_ratio_max_over_min": ratio,
            "hhi": hhi,
            "entropy": entropy,
        }
    )

    if min_share < float(min_share_fail):
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome imbalance (non-missing): minimum class share is below threshold.",
                evidence=metrics,
                fix_hint="Adjust cohort/definition so outcome classes have adequate representation.",
            )
        )
        return issues, metrics

    if ratio > float(max_ratio_fail):
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome imbalance (non-missing): max/min count ratio exceeds threshold.",
                evidence=metrics,
                fix_hint="Adjust cohort/definition so classes are not extremely imbalanced.",
            )
        )
        return issues, metrics

    # neff gate only when exactly 2 observed classes (binary or 2-level subset of categorical)
    if len(obs_set) == 2 and float(min_neff_fail) > 0.0:
        vals2 = list(sorted(list(obs_set), key=lambda x: str(x)))
        v0, v1 = vals2[0], vals2[1]
        n0, n1 = int(vc.get(v0, 0)), int(vc.get(v1, 0))
        neff = float((2.0 * n0 * n1) / (n0 + n1) if (n0 + n1) > 0 else 0.0)
        metrics["n_eff_harmonic_mean"] = neff
        if neff < float(min_neff_fail):
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Outcome imbalance (non-missing): effective sample size is too small.",
                    evidence=metrics,
                    fix_hint="Increase cohort size or redefine outcome to raise effective information.",
                )
            )
            return issues, metrics

    return issues, metrics


# =============================================================================
# 5) Covariates / Effect Modifiers validations (pre-transform, raw df)
# =============================================================================
def validate_covariate_and_effect_modifier_presence(
    *,
    df: pd.DataFrame,
    protocol: "ProtocolSpec",
    require_covariates: bool,
) -> Tuple[List["ValidationIssue"], Dict[str, Any]]:
    """
    Protocol-native presence + overlap validation for:
      - covariates (adjustment set)
      - effect_modifiers (heterogeneity drivers)

    HARD FAIL:
      - dataframe has duplicate column labels (ambiguous schema)
      - any referenced covariate/effect_modifier column is missing from df
      - require_covariates=True and no covariates exist in the protocol (or all were filtered away)

    WARN:
      - no effect modifiers available (CATE/heterogeneity limited)
      - covariates and effect_modifiers overlap (redundant / confusing roles)
      - require_covariates=False and covariates are empty (unadjusted / likely biased)

    Returns:
      (issues, metrics) with bounded evidence payloads.
    """
    issues: List["ValidationIssue"] = []

    # -------------------------
    # 0) Global schema sanity: duplicate df column labels are ambiguous in pandas
    # -------------------------
    if not df.columns.is_unique:
        dupes = df.columns[df.columns.duplicated()].tolist()
        counts: Dict[str, int] = {}
        for c in dupes:
            counts[c] = counts.get(c, 0) + 1

        sample = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:25]
        metrics = {
            "n_rows": int(df.shape[0]),
            "n_total_columns": int(df.shape[1]),
            "n_duplicated_labels": int(len(counts)),
            "duplicated_label_sample": [{"name": k, "count": v} for k, v in sample],
        }
        issues.append(
            _issue(
                severity="FAIL",
                message="Dataframe contains duplicate column labels (ambiguous schema).",
                evidence=metrics,
                fix_hint="Fix upstream joins/concats/encoders to guarantee globally unique column names.",
            )
        )
        return issues, metrics

    # -------------------------
    # 1) Pull protocol lists (defensive de-dup, stable order)
    # -------------------------
    def _dedup_keep_order(xs: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for x in xs:
            if  x.strip() and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    covariates = _dedup_keep_order(list(getattr(protocol, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(protocol, "effect_modifiers", []) or []))

    # -------------------------
    # 2) Existence checks
    # -------------------------
    missing_covariates = [c for c in covariates if c not in df.columns]
    missing_effect_modifiers = [c for c in effect_modifiers if c not in df.columns]

    n_covariates_present = int(len(covariates) - len(missing_covariates))
    n_effect_modifiers_present = int(len(effect_modifiers) - len(missing_effect_modifiers))

    overlap_cols = sorted(set(covariates).intersection(set(effect_modifiers)))

    metrics: Dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_df_cols": int(df.shape[1]),
        "require_covariates": bool(require_covariates),
        "n_covariates_requested": int(len(covariates)),
        "n_effect_modifiers_requested": int(len(effect_modifiers)),
        "n_covariates_present": int(n_covariates_present),
        "n_effect_modifiers_present": int(n_effect_modifiers_present),
        "n_missing_covariates": int(len(missing_covariates)),
        "n_missing_effect_modifiers": int(len(missing_effect_modifiers)),
        "missing_covariates": missing_covariates[:200],
        "missing_effect_modifiers": missing_effect_modifiers[:200],
        "n_overlap_covariate_effect_modifier": int(len(overlap_cols)),
        "overlap_cols": overlap_cols[:200],
    }

    # Missing referenced columns is structural: hard fail and stop.
    if missing_covariates or missing_effect_modifiers:
        issues.append(
            _issue(
                severity="FAIL",
                message="Some covariate/effect-modifier columns referenced by the protocol are missing from the dataframe.",
                evidence=metrics,
                fix_hint="Ensure upstream filtering/column-drop keeps all covariates and effect modifiers required by the protocol.",
            )
        )
        return issues, metrics

    # -------------------------
    # 3) Presence requirements / warnings
    # -------------------------
    if require_covariates and n_covariates_present == 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="No covariates available for adjustment.",
                evidence=metrics,
                fix_hint="Add covariates in the protocol or relax filtering that removed them.",
            )
        )
    elif n_covariates_present == 0:
        issues.append(
            _issue(
                severity="WARN",
                message="No covariates available; estimates will be unadjusted and likely biased.",
                evidence=metrics,
                fix_hint="Add covariates in the protocol if you intend adjustment.",
            )
        )

    if n_effect_modifiers_present == 0:
        issues.append(
            _issue(
                severity="WARN",
                message="No effect modifiers available; heterogeneity (CATE) analysis will be limited.",
                evidence=metrics,
                fix_hint="If you want heterogeneous effects, add effect modifiers; otherwise ignore.",
            )
        )

    # -------------------------
    # 4) Overlap warning (redundant role assignment)
    # -------------------------
    if overlap_cols:
        issues.append(
            _issue(
                severity="WARN",
                message="Covariate/effect-modifier overlap detected: some columns appear in both lists.",
                evidence=metrics,
                fix_hint=(
                    "This is allowed but often redundant. Prefer disjoint sets: put heterogeneity drivers in effect_modifiers "
                    "and keep pure confounders/controls in covariates. Ensure downstream code de-dupes combined feature lists."
                ),
            )
        )

    # If literally nothing is available, flag degeneracy.
    if (n_covariates_present + n_effect_modifiers_present) == 0:
        issues.append(
            _issue(
                severity="FAIL" if require_covariates else "WARN",
                message="No covariates or effect modifiers available after filtering; modeling is likely degenerate.",
                evidence=metrics,
                fix_hint="Verify protocol lists and upstream filtering/null purge steps.",
            )
        )

    return issues, metrics


def validate_covariate_and_effect_modifier_missingness(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    missing_rate_warn: float = 0.05,
    missing_rate_fail: float = 0.30,
    ignore_cols: Sequence[str] = (),
    max_cols: int = 500,
) -> Tuple[List["ValidationIssue"], Dict[str, Any]]:
    """
    Protocol-native missingness check for covariates + effect_modifiers (pre-transform).

    FAIL:
      - any referenced column missing from df (structural)
      - any feature has missing_rate >= missing_rate_fail

    WARN:
      - any feature has missing_rate >= missing_rate_warn (and < fail)

    Returns:
      (issues, metrics)
    """
    issues: List["ValidationIssue"] = []
    ignore = {c for c in ignore_cols if c.strip()}

    def _dedup_keep_order(xs: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for x in xs:
            if x.strip() and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    covariates = _dedup_keep_order(list(getattr(protocol, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(protocol, "effect_modifiers", []) or []))

    # Combined list (stable): covariates first
    cols_all = _dedup_keep_order([c for c in (covariates + effect_modifiers) if c not in ignore])[: int(max_cols)]

    missing_cols = [c for c in cols_all if c not in df.columns]
    n_rows = int(df.shape[0])

    metrics: Dict[str, Any] = {
        "n_rows": n_rows,
        "n_covariates_protocol": int(len(covariates)),
        "n_effect_modifiers_protocol": int(len(effect_modifiers)),
        "n_checked": int(len(cols_all)),
        "missing_rate_warn": float(missing_rate_warn),
        "missing_rate_fail": float(missing_rate_fail),
        "n_missing_cols": int(len(missing_cols)),
        "missing_cols": missing_cols[:200],
    }

    if missing_cols:
        issues.append(
            _issue(
                severity="FAIL",
                message="Some covariate/effect-modifier columns are missing from the dataframe.",
                evidence=metrics,
                fix_hint="Fix upstream column-drop/filtering or remove these columns from protocol.covariates/effect_modifiers.",
            )
        )
        return issues, metrics

    warn_off: List[Dict[str, Any]] = []
    fail_off: List[Dict[str, Any]] = []

    for c in cols_all:
        s = df[c]
        mr = float(s.isna().mean()) if n_rows > 0 else 0.0
        row: Dict[str, Any] = {"col": c, "missing_rate": mr, "dtype": str(s.dtype)}
        if mr >= float(missing_rate_fail):
            fail_off.append(row)
        elif mr >= float(missing_rate_warn):
            warn_off.append(row)

    metrics.update(
        {
            "n_warn": int(len(warn_off)),
            "n_fail": int(len(fail_off)),
            "warn_offenders": warn_off[:50],
            "fail_offenders": fail_off[:50],
        }
    )

    if fail_off:
        issues.append(
            _issue(
                severity="FAIL",
                message="Some covariate/effect-modifier features have high missingness.",
                evidence=metrics,
                fix_hint="Drop/impute these columns explicitly in transform, or fix upstream null-handling.",
            )
        )
    if warn_off:
        issues.append(
            _issue(
                severity="WARN",
                message="Some covariate/effect-modifier features have non-trivial missingness.",
                evidence=metrics,
                fix_hint="Consider imputation, missingness indicators, or dropping weak/noisy columns.",
            )
        )

    return issues, metrics


# -----------------------------------------------------------------------------
# Differential missingness by treatment arm (causal-relevant)
# -----------------------------------------------------------------------------

def validate_covariate_and_effect_modifier_missingness_by_treatment(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    delta_warn: float = 0.05,
    delta_fail: float = 0.20,
    ignore_cols: Sequence[str] = (),
    max_cols: int = 300,
    min_arm_n: int = 25,
) -> Tuple[List["ValidationIssue"], Dict[str, Any]]:
    """
    Detect differential missingness across treatment arms for covariates/effect_modifiers.

    Why it matters:
      - If missingness differs by arm, you risk selection bias and broken overlap/positivity.

    Supports:
      - binary treatment
      - categorical treatment (multi-arm)
    Continuous treatment:
      - skipped (WARN) unless you later implement binning-based diagnostics.

    WARN/FAIL:
      - if max missingness gap across arms exceeds thresholds:
          gap = max_arm_missing_rate - min_arm_missing_rate

    Returns:
      (issues, metrics)
    """
    issues: List["ValidationIssue"] = []
    ignore = {c for c in ignore_cols if c.strip()}

    ts = protocol.treatment_spec
    tcol = ts.column

    metrics: Dict[str, Any] = {
        "treatment_col": tcol,
        "treatment_kind": getattr(ts, "kind", None),
        "delta_warn": float(delta_warn),
        "delta_fail": float(delta_fail),
        "min_arm_n": int(min_arm_n),
        "n_rows": int(df.shape[0]),
    }

    if tcol not in df.columns:
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment column missing; cannot compute arm-wise missingness diagnostics.",
                evidence=metrics,
                fix_hint="Ensure treatment_spec.column is retained through filtering steps.",
            )
        )
        return issues, metrics

    # Build arm masks from protocol literals (dtype-aware)
    sT = df[tcol]

    def _mask_equals_literal(series: pd.Series, literal: str) -> pd.Series:
        lit = str(literal).strip()
        if not lit:
            return pd.Series([False] * len(series), index=series.index)

        dt = series.dtype

        if ptypes.is_bool_dtype(dt):
            b = _parse_bool_token(lit)
            if b is None:
                return pd.Series([False] * len(series), index=series.index)
            bool_series = cast(pd.Series, series.astype("boolean").fillna(False))  # type: ignore[call-overload]
            return bool_series.eq(bool(b))

        if ptypes.is_numeric_dtype(dt):
            try:
                thr = float(lit)
            except Exception:
                return pd.Series([False] * len(series), index=series.index)
            v = pd.to_numeric(series, errors="coerce")
            return v.eq(thr)

        if ptypes.is_datetime64_any_dtype(dt):
            ts_ = pd.to_datetime(lit, errors="coerce")
            if pd.isna(ts_):
                return pd.Series([False] * len(series), index=series.index)
            return pd.to_datetime(series, errors="coerce").eq(ts_)

        ss = series.astype("string").str.strip().str.casefold()
        return ss.eq(lit.casefold())

    arm_masks: Dict[str, pd.Series] = {}

    if isinstance(ts, BinaryTreatmentSpecModel):
        arm_masks["treated"] = _mask_equals_literal(sT, ts.treated)
        arm_masks["control"] = _mask_equals_literal(sT, ts.control)

    elif isinstance(ts, CategoricalTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        for lvl in list(ts.levels):
            arm_masks[str(lvl)] = _mask_equals_literal(sT, str(lvl))
            
    else:
        issues.append(
            _issue(
                severity="FAIL",
                message="Unknown treatment kind; cannot compute arm-wise missingness diagnostics.",
                evidence=metrics,
                fix_hint="Ensure protocol emits a supported treatment spec model.",
            )
        )
        return issues, metrics

    # Arm counts + filter tiny arms (avoid noisy rates)
    arm_counts = {k: int(m.sum()) for k, m in arm_masks.items()}
    metrics["arm_counts"] = arm_counts
    eligible_arms = [k for k, n in arm_counts.items() if n >= int(min_arm_n)]

    metrics["eligible_arms"] = eligible_arms
    if len(eligible_arms) < 2:
        issues.append(
            _issue(
                severity="WARN",
                message="Too few eligible treatment arms to assess differential missingness (arms too small).",
                evidence=metrics,
                fix_hint="Increase cohort size or relax filtering; differential missingness diagnostics need adequate arm sizes.",
            )
        )
        return issues, metrics

    def _dedup_keep_order(xs: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for x in xs:
            if  x not in seen:
                seen.add(x)
                out.append(x)
        return out

    covariates = _dedup_keep_order(list(getattr(protocol, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(protocol, "effect_modifiers", []) or []))
    cols_all = _dedup_keep_order([c for c in (covariates + effect_modifiers) if c not in ignore])[: int(max_cols)]

    missing_cols = [c for c in cols_all if c not in df.columns]
    metrics["n_checked"] = int(len(cols_all))
    metrics["missing_feature_cols"] = missing_cols[:200]
    if missing_cols:
        issues.append(
            _issue(
                severity="FAIL",
                message="Some covariate/effect-modifier columns are missing; cannot compute arm-wise missingness.",
                evidence=metrics,
                fix_hint="Run presence validation earlier or fix upstream column retention.",
            )
        )
        return issues, metrics

    offenders_warn: List[Dict[str, Any]] = []
    offenders_fail: List[Dict[str, Any]] = []

    for c in cols_all:
        s = df[c]
        per_arm: Dict[str, float] = {}
        for a in eligible_arms:
            m = arm_masks[a]
            sa = s.loc[m]
            per_arm[a] = float(sa.isna().mean()) if int(sa.shape[0]) > 0 else 0.0

        gap = float(max(per_arm.values()) - min(per_arm.values())) if per_arm else 0.0

        row: Dict[str, Any] = {
            "col": c,
            "dtype": str(s.dtype),
            "missing_rate_by_arm": per_arm,
            "gap": gap,
        }

        if gap >= float(delta_fail):
            offenders_fail.append(row)
        elif gap >= float(delta_warn):
            offenders_warn.append(row)

    metrics["n_warn"] = int(len(offenders_warn))
    metrics["n_fail"] = int(len(offenders_fail))
    metrics["warn_offenders"] = offenders_warn[:50]
    metrics["fail_offenders"] = offenders_fail[:50]

    if offenders_fail:
        issues.append(
            _issue(
                severity="FAIL",
                message="Differential missingness across treatment arms is severe for some covariates/effect modifiers.",
                evidence=metrics,
                fix_hint="Investigate selection mechanisms; consider missingness indicators, arm-specific imputation, or revisiting cohort/treatment definition.",
            )
        )
    elif offenders_warn:
        issues.append(
            _issue(
                severity="WARN",
                message="Differential missingness across treatment arms detected for some covariates/effect modifiers.",
                evidence=metrics,
                fix_hint="Consider missingness indicators or targeted imputation; verify missingness is not post-treatment/selection-driven.",
            )
        )

    return issues, metrics


def validate_covariate_and_effect_modifier_constantness(
    *,
    df: pd.DataFrame,
    protocol: "ProtocolSpec",
    # hard-ish thresholds
    max_constant_frac_warn: float = 0.30,
    max_constant_frac_fail: float = 0.70,
    # numeric near-constant threshold
    min_variance: float = 1e-12,
    # column filtering
    ignore_cols: Sequence[str] = (),
    max_cols: int = 500,
    # treat "all missing" as constant-like (but missingness validator should usually fail earlier)
    treat_all_missing_as_constant: bool = True,
) -> Tuple[List["ValidationIssue"], Dict[str, Any]]:
    """
    Protocol-native constant/near-constant detection for covariates + effect_modifiers (pre-transform).

    Constant-like definition:
      - nunique(dropna=True) <= 1  -> constant
      - numeric variance <= min_variance -> near-constant

    Output policy:
      - WARN/FAIL if fraction of constant-like features exceeds thresholds
      - Always emits per-column samples (bounded) to support debugging.

    Returns:
      (issues, metrics)
    """
    issues: List["ValidationIssue"] = []
    ignore = {c for c in ignore_cols if c.strip()}

    def _dedup_keep_order(xs: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for x in xs:
            if x.strip() and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    covariates = _dedup_keep_order(list(getattr(protocol, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(protocol, "effect_modifiers", []) or []))
    cols_all = _dedup_keep_order([c for c in (covariates + effect_modifiers) if c not in ignore])[: int(max_cols)]

    missing_cols = [c for c in cols_all if c not in df.columns]
    metrics: Dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_checked": int(len(cols_all)),
        "n_missing_cols": int(len(missing_cols)),
        "missing_cols": missing_cols[:200],
        "min_variance": float(min_variance),
        "max_constant_frac_warn": float(max_constant_frac_warn),
        "max_constant_frac_fail": float(max_constant_frac_fail),
    }

    if missing_cols:
        issues.append(
            _issue(
                severity="FAIL",
                message="Some covariate/effect-modifier columns are missing; cannot validate constantness safely.",
                evidence=metrics,
                fix_hint="Run presence validation earlier or fix upstream column retention.",
            )
        )
        return issues, metrics

    constant_like: List[Dict[str, Any]] = []
    numeric_near_constant: List[Dict[str, Any]] = []

    for c in cols_all:
        s = df[c]

        # constant by unique count (dropna=True, unless all missing handling is requested)
        nonnull = s.dropna()
        if nonnull.empty:
            if treat_all_missing_as_constant:
                constant_like.append({"col": c, "reason": "all_missing", "dtype": str(s.dtype), "nunique": 0})
            continue

        nunq = int(nonnull.nunique(dropna=True))
        if nunq <= 1:
            constant_like.append({"col": c, "reason": "nunique<=1", "dtype": str(s.dtype), "nunique": nunq})
            continue

        # near-constant numeric by variance
        if ptypes.is_bool_dtype(s.dtype):
            # bool already handled by nunique<=1; if it has two values, it's fine
            continue

        if ptypes.is_numeric_dtype(s.dtype):
            x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, copy=False)
            x = x[np.isfinite(x)]
            if x.size <= 1:
                # treat as constant-like
                constant_like.append({"col": c, "reason": "insufficient_numeric", "dtype": str(s.dtype), "n": int(x.size)})
                continue
            var = float(np.var(x))
            if var <= float(min_variance):
                numeric_near_constant.append(
                    {"col": c, "reason": "variance<=min_variance", "dtype": str(s.dtype), "variance": var, "n": int(x.size)}
                )

    # aggregate counts
    n_feat = max(1, int(len(cols_all)))
    n_const = int(len(constant_like))
    n_near = int(len(numeric_near_constant))
    # treat both as "bad constant-like" for fraction metrics
    n_bad = int(n_const + n_near)
    frac_bad = float(n_bad / n_feat)

    metrics.update(
        {
            "n_constant": n_const,
            "n_numeric_near_constant": n_near,
            "n_constant_like_total": n_bad,
            "constant_like_frac": frac_bad,
            "constant_sample": constant_like[:50],
            "near_constant_sample": numeric_near_constant[:50],
        }
    )

    if n_bad == 0:
        return issues, metrics

    if frac_bad >= float(max_constant_frac_fail):
        issues.append(
            _issue(
                severity="FAIL",
                message="Too many constant/near-constant covariates/effect modifiers; modeling is likely degenerate.",
                evidence=metrics,
                fix_hint="Drop constant features and verify cohort filtering/feature selection didn't collapse variation.",
            )
        )
    elif frac_bad >= float(max_constant_frac_warn):
        issues.append(
            _issue(
                severity="WARN",
                message="Many covariates/effect modifiers are constant/near-constant; they add little signal.",
                evidence=metrics,
                fix_hint="Drop constant features in transform; verify cohort filtering and feature lists.",
            )
        )
    else:
        # Even if fraction small, it's useful to warn about existence of constant columns
        issues.append(
            _issue(
                severity="WARN",
                message="Some covariates/effect modifiers are constant/near-constant.",
                evidence=metrics,
                fix_hint="Drop constant features; they add no information and can destabilize some estimators.",
            )
        )

    return issues, metrics


def validate_covariate_and_effect_modifier_high_cardinality_and_id_like(
    *,
    df: pd.DataFrame,
    protocol: "ProtocolSpec",
    # cardinality thresholds for categorical/string-ish
    max_levels_warn: int = 50,
    max_levels_fail: int = 200,
    # ID-like heuristic for string-ish cols
    id_like_unique_ratio_warn: float = 0.90,
    id_like_unique_ratio_fail: float = 0.98,
    min_unique_for_id_like: int = 50,
    # optional: allow some flagged columns before failing
    max_id_like_allowed: int = 0,  # 0 => FAIL on any id-like (fail threshold); set >0 to soften
    ignore_cols: Sequence[str] = (),
    max_cols: int = 500,
    sample_n_for_obj_type_scan: int = 200,
) -> Tuple[List["ValidationIssue"], Dict[str, Any]]:
    """
    Protocol-native high-cardinality + ID-like detection for covariates/effect_modifiers (pre-transform).

    Scope (careful by design):
      - Only applies to string/object/categorical columns (NOT continuous numeric floats).
      - Numeric columns are NOT flagged as ID-like based on uniqueness ratio (avoids false positives).

    Flags:
      - High cardinality: nunique >= max_levels_warn / max_levels_fail   (WARN/FAIL)
      - ID-like (string-ish): unique_ratio >= thresholds + nunique >= min_unique_for_id_like

    Returns:
      (issues, metrics)
    """
    issues: List["ValidationIssue"] = []
    ignore = {c for c in ignore_cols if c.strip()}

    def _dedup_keep_order(xs: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for x in xs:
            if x.strip() and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    covariates = _dedup_keep_order(list(getattr(protocol, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(protocol, "effect_modifiers", []) or []))
    cols_all = _dedup_keep_order([c for c in (covariates + effect_modifiers) if c not in ignore])[: int(max_cols)]

    missing_cols = [c for c in cols_all if c not in df.columns]
    n_rows = int(df.shape[0])

    metrics: Dict[str, Any] = {
        "n_rows": n_rows,
        "n_checked": int(len(cols_all)),
        "n_missing_cols": int(len(missing_cols)),
        "missing_cols": missing_cols[:200],
        "max_levels_warn": int(max_levels_warn),
        "max_levels_fail": int(max_levels_fail),
        "id_like_unique_ratio_warn": float(id_like_unique_ratio_warn),
        "id_like_unique_ratio_fail": float(id_like_unique_ratio_fail),
        "min_unique_for_id_like": int(min_unique_for_id_like),
        "max_id_like_allowed": int(max_id_like_allowed),
    }

    if missing_cols:
        issues.append(
            _issue(
                severity="FAIL",
                message="Some covariate/effect-modifier columns are missing; cannot validate cardinality safely.",
                evidence=metrics,
                fix_hint="Run presence validation earlier or fix upstream column retention.",
            )
        )
        return issues, metrics

    hi_warn: List[Dict[str, Any]] = []
    hi_fail: List[Dict[str, Any]] = []
    id_warn: List[Dict[str, Any]] = []
    id_fail: List[Dict[str, Any]] = []

    for c in cols_all:
        s = df[c]
        dt = s.dtype

        # Only string-ish / categorical-ish. Do NOT apply to numeric floats (avoids false positives).
        is_cat = isinstance(dt, pd.CategoricalDtype)
        is_strish = ptypes.is_string_dtype(dt) or ptypes.is_object_dtype(dt) or is_cat
        if not is_strish:
            continue

        # Determine nunique / unique ratio (dropna=False to catch "ID-like" with missing too)
        nunique = int(s.nunique(dropna=False))
        uniq_ratio = float(nunique / max(1, n_rows))

        row: Dict[str, Any] = {
            "col": c,
            "dtype": str(dt),
            "nunique": nunique,
            "unique_ratio": uniq_ratio,
        }

        # High-cardinality categoricals/strings explode one-hot later
        if nunique >= int(max_levels_fail):
            hi_fail.append(row)
        elif nunique >= int(max_levels_warn):
            hi_warn.append(row)

        # ID-like heuristic (string-ish only)
        if nunique >= int(min_unique_for_id_like):
            if uniq_ratio >= float(id_like_unique_ratio_fail):
                id_fail.append(row)
            elif uniq_ratio >= float(id_like_unique_ratio_warn):
                id_warn.append(row)

        # Optional: catch mixed python types in object columns (noise source)
        if ptypes.is_object_dtype(dt):
            ss = s.dropna()
            if not ss.empty:
                if int(ss.shape[0]) > int(sample_n_for_obj_type_scan):
                    ss = ss.sample(n=int(sample_n_for_obj_type_scan), random_state=0)
                type_set = sorted({type(x).__name__ for x in ss.tolist()})
                if len(type_set) > 1:
                    row["mixed_python_types"] = type_set[:20]

    metrics.update(
        {
            "n_hi_warn": int(len(hi_warn)),
            "n_hi_fail": int(len(hi_fail)),
            "n_id_warn": int(len(id_warn)),
            "n_id_fail": int(len(id_fail)),
            "hi_warn_sample": hi_warn[:50],
            "hi_fail_sample": hi_fail[:50],
            "id_warn_sample": id_warn[:50],
            "id_fail_sample": id_fail[:50],
        }
    )

    # High-cardinality: fail if any extreme offenders exist
    if hi_fail:
        issues.append(
            _issue(
                severity="FAIL",
                message="Some string/categorical covariates/effect modifiers have extreme cardinality (encoding will explode).",
                evidence=metrics,
                fix_hint="Collapse/bucket rare levels, cap one-hot levels, or drop these columns before transform.",
            )
        )
    elif hi_warn:
        issues.append(
            _issue(
                severity="WARN",
                message="Some string/categorical covariates/effect modifiers have high cardinality.",
                evidence=metrics,
                fix_hint="Consider bucketing, hashing, frequency encoding, or dropping these columns before transform.",
            )
        )

    # ID-like: fail policy (default: fail on any id_like at fail threshold unless max_id_like_allowed softens)
    if id_fail:
        sev: str = "FAIL"
        if int(max_id_like_allowed) > 0 and len(id_fail) <= int(max_id_like_allowed):
            sev = "WARN"
        issues.append(
            _issue(
                severity=sev,  # type: ignore[arg-type]
                message="ID-like (near-unique) string features detected in covariates/effect modifiers.",
                evidence=metrics,
                fix_hint="Drop/mask identifiers (patient_id, encounter_id, free-text keys). They enable memorization and harm causal estimation.",
            )
        )
    elif id_warn:
        issues.append(
            _issue(
                severity="WARN",
                message="Potentially ID-like string features detected (high uniqueness ratio).",
                evidence=metrics,
                fix_hint="Inspect and likely drop these columns or replace with coarse groupings.",
            )
        )

    return issues, metrics

def validate_covariate_and_effect_modifier_type_risks(
    *,
    df: pd.DataFrame,
    protocol: "ProtocolSpec",
    ignore_cols: Sequence[str] = (),
    max_cols: int = 500,
    # object-type scan is bounded for speed/determinism
    obj_type_scan_n: int = 200,
    # if True, datetime columns are WARN; if False they are INFO-level (but you only support WARN/FAIL)
    warn_on_datetime: bool = True,
    # If you have a strict policy that "object dtype is not allowed" pre-transform, set to True
    fail_on_object_mixed_types: bool = False,
) -> Tuple[List["ValidationIssue"], Dict[str, Any]]:
    """
    Protocol-native type-risk validation for covariates + effect_modifiers (pre-transform).

    What it flags (deterministic):
      1) DATETIME columns present in covariates/effect_modifiers (WARN)
         - Because most causal estimators need numeric encodings: offsets-from-time_zero, components, etc.
      2) OBJECT dtype with mixed python types (WARN or FAIL by policy)
         - E.g., ints + strings in same column -> unstable transforms/coercions.
      3) OBJECT dtype that is "string-ish" but has very high average length (WARN)
         - Often indicates free-text fields; default pipelines will explode or behave poorly.
      4) Category-like stored as object with many unique levels (WARN; complements cardinality validator)

    This validator does not replace:
      - missingness / domain / cardinality checks
    It is a *transform strategy* prompt:
      - "you need to explicitly encode these"

    Returns:
      (issues, metrics)
    """
    issues: List["ValidationIssue"] = []
    ignore = {c for c in ignore_cols if c and c.strip()}

    def _dedup_keep_order(xs: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for x in xs:
            if x.strip() and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    covariates = _dedup_keep_order(list(getattr(protocol, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(protocol, "effect_modifiers", []) or []))
    cols_all = _dedup_keep_order([c for c in (covariates + effect_modifiers) if c not in ignore])[: int(max_cols)]

    missing_cols = [c for c in cols_all if c not in df.columns]
    n_rows = int(df.shape[0])

    metrics: Dict[str, Any] = {
        "n_rows": n_rows,
        "n_checked": int(len(cols_all)),
        "n_missing_cols": int(len(missing_cols)),
        "missing_cols": missing_cols[:200],
        "obj_type_scan_n": int(obj_type_scan_n),
        "warn_on_datetime": bool(warn_on_datetime),
        "fail_on_object_mixed_types": bool(fail_on_object_mixed_types),
    }

    if missing_cols:
        issues.append(
            _issue(
                severity="FAIL",
                message="Some covariate/effect-modifier columns are missing; cannot validate type risks safely.",
                evidence=metrics,
                fix_hint="Run presence validation earlier or fix upstream column retention.",
            )
        )
        return issues, metrics

    datetime_cols: List[Dict[str, Any]] = []
    mixed_object_cols: List[Dict[str, Any]] = []
    long_text_cols: List[Dict[str, Any]] = []
    object_high_card_cols: List[Dict[str, Any]] = []

    for c in cols_all:
        s = df[c]
        dt = s.dtype

        # 1) datetime-like
        if ptypes.is_datetime64_any_dtype(dt):
            datetime_cols.append({"col": c, "dtype": str(dt)})
            continue

        # 2) object dtype: mixed python types / text-ish
        if ptypes.is_object_dtype(dt):
            ss = s.dropna()
            if ss.empty:
                continue

            # bounded scan for python types
            scan = ss
            if int(scan.shape[0]) > int(obj_type_scan_n) and int(obj_type_scan_n) > 0:
                scan = scan.sample(n=int(obj_type_scan_n), random_state=0)

            types = sorted({type(x).__name__ for x in scan.tolist()})
            if len(types) > 1:
                mixed_object_cols.append(
                    {
                        "col": c,
                        "dtype": str(dt),
                        "python_types": types[:20],
                    }
                )

            # text-length heuristics (free-text detection)
            # Only compute if values are mostly strings in the scanned sample.
            str_like = [x for x in scan.tolist() if isinstance(x, str)]
            if str_like:
                lengths = [len(x) for x in str_like]
                avg_len = float(sum(lengths) / max(1, len(lengths)))
                p95_len = float(pd.Series(lengths).quantile(0.95)) if len(lengths) >= 5 else float(max(lengths))
                if avg_len >= 40.0 or p95_len >= 200.0:
                    long_text_cols.append(
                        {
                            "col": c,
                            "dtype": str(dt),
                            "avg_len_sample": avg_len,
                            "p95_len_sample": p95_len,
                            "n_str_in_sample": int(len(str_like)),
                            "sample_n": int(len(scan)),
                        }
                    )

            # object cardinality (coarse hint; dedicated validator handles thresholds)
            nunq = int(s.nunique(dropna=False))
            uniq_ratio = float(nunq / max(1, n_rows))
            if nunq >= 50 and uniq_ratio >= 0.50:
                object_high_card_cols.append(
                    {"col": c, "dtype": str(dt), "nunique": nunq, "unique_ratio": uniq_ratio}
                )

    metrics.update(
        {
            "n_datetime": int(len(datetime_cols)),
            "n_mixed_object": int(len(mixed_object_cols)),
            "n_long_text": int(len(long_text_cols)),
            "n_object_high_card": int(len(object_high_card_cols)),
            "datetime_sample": datetime_cols[:50],
            "mixed_object_sample": mixed_object_cols[:50],
            "long_text_sample": long_text_cols[:25],
            "object_high_card_sample": object_high_card_cols[:50],
        }
    )

    # Emit issues (bounded)
    if datetime_cols and warn_on_datetime:
        issues.append(
            _issue(
                severity="WARN",
                message="Datetime covariates/effect modifiers detected; transform must encode them explicitly.",
                evidence=metrics,
                fix_hint="Convert datetimes to numeric features (e.g., seconds since time_zero, day/week components) in the transform step.",
            )
        )

    if mixed_object_cols:
        sev: str = "FAIL" if fail_on_object_mixed_types else "WARN"
        issues.append(
            _issue(
                severity=sev,  # type: ignore[arg-type]
                message="Object columns with mixed python types detected; transforms/coercions may be unstable.",
                evidence=metrics,
                fix_hint="Normalize these columns upstream (cast to string or numeric) before encoding/transform.",
            )
        )

    if long_text_cols:
        issues.append(
            _issue(
                severity="WARN",
                message="Potential free-text columns detected in covariates/effect modifiers.",
                evidence=metrics,
                fix_hint="Drop free-text by default, or implement explicit text featurization (careful: leakage/post-treatment risk).",
            )
        )

    if object_high_card_cols:
        issues.append(
            _issue(
                severity="WARN",
                message="Object/string columns with high uniqueness ratio detected (likely IDs or high-cardinality categories).",
                evidence=metrics,
                fix_hint="Inspect and likely bucket/drop these columns; if categorical, cap levels or hash before one-hot.",
            )
        )

    return issues, metrics


# =============================================================================
# 6) Overlap / positivity diagnostics (pre-transform, df-backed)
# =============================================================================

ArmKind = Literal["binary", "categorical", "continuous"]


@dataclass(frozen=True)
class ArmMasks:
    kind: ArmKind
    treatment_col: str
    masks: Dict[str, pd.Series]  # arm_name -> bool mask
    counts: Dict[str, int]       # arm_name -> count
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "treatment_col": self.treatment_col,
            "counts": dict(self.counts),
            "arms": list(self.masks.keys()),
            "notes": self.notes,
        }


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _parse_bool_token(raw: str) -> Optional[bool]:
    tok = str(raw).strip().casefold()
    if tok in BOOL_TRUE:
        return True
    if tok in BOOL_FALSE:
        return False
    return None


def _mask_equals_literal(s: pd.Series, literal: str) -> pd.Series:
    """
    Compare series values to a protocol literal in a dtype-aware way:
      - bool: strict bool token parsing
      - numeric: float(literal)
      - datetime: pd.to_datetime(literal)
      - fallback: normalized string compare (strip+casefold)
    """
    lit = str(literal).strip()
    if not lit:
        return pd.Series([False] * len(s), index=s.index)

    if ptypes.is_bool_dtype(s.dtype):
        b = _parse_bool_token(lit)
        if b is None:
            return pd.Series([False] * len(s), index=s.index)
        return cast(pd.Series, s.astype("boolean").fillna(False)).eq(bool(b))  # type: ignore[call-overload]

    if ptypes.is_numeric_dtype(s.dtype):
        try:
            thr = float(lit)
        except Exception:
            return pd.Series([False] * len(s), index=s.index)
        v = pd.to_numeric(s, errors="coerce")
        return v.eq(thr)

    if ptypes.is_datetime64_any_dtype(s.dtype):
        ts = pd.to_datetime(lit, errors="coerce")
        if pd.isna(ts):
            return pd.Series([False] * len(s), index=s.index)
        return pd.to_datetime(s, errors="coerce").eq(ts)

    ss = s.astype("string").str.strip().str.casefold()
    return ss.eq(lit.casefold())


def _dedup_keep_order(xs: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in xs:
        if  x.strip() and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# -----------------------------------------------------------------------------
# 1) Build arm masks from ProtocolSpec.treatment_spec
# -----------------------------------------------------------------------------

def compute_arm_masks_from_protocol(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    max_bins_continuous: int = 5,
) -> ArmMasks:
    tcol = protocol.treatment_spec.column
    if tcol not in df.columns:
        raise KeyError(f"treatment_col not found in df: {tcol!r}")

    ts = protocol.treatment_spec
    s = df[tcol]

    if isinstance(ts, BinaryTreatmentSpecModel):
        m_t = _mask_equals_literal(s, ts.treated)
        m_c = _mask_equals_literal(s, ts.control)
        masks = {"treated": m_t, "control": m_c}
        counts = {k: int(v.sum()) for k, v in masks.items()}
        return ArmMasks(kind="binary", treatment_col=tcol, masks=masks, counts=counts)

    if isinstance(ts, CategoricalTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        masks: Dict[str, pd.Series] = {}
        for lvl in list(ts.levels):
            masks[str(lvl)] = _mask_equals_literal(s, str(lvl))
        counts = {k: int(v.sum()) for k, v in masks.items()}
        return ArmMasks(kind="categorical", treatment_col=tcol, masks=masks, counts=counts)

    raise ValueError(f"Unknown treatment_spec kind={getattr(ts, 'kind', None)!r}")


# -----------------------------------------------------------------------------
# 2) Univariate overlap / positivity support checks (df-backed)
# -----------------------------------------------------------------------------

def validate_overlap_positivity_univariate(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    arm_masks: ArmMasks,
    # feature selection scope
    use_effect_modifiers: bool = True,
    ignore_cols: Sequence[str] = (),
    max_cols: int = 300,
    # thresholds
    min_arm_n: int = 25,
    min_support_per_arm: int = 25,
    max_levels_checked: int = 30,
    # numeric overlap via quantile intervals
    q_lo: float = 0.10,
    q_hi: float = 0.90,
) -> Tuple[List["ValidationIssue"], Dict[str, Any]]:
    """
    Flags overlap/positivity risks by checking whether covariate/effect-modifier support differs across arms.

    Categorical/string features:
      - find top levels globally, flag levels that appear in one arm (>=min_support) but are absent in another

    Numeric/bool features:
      - compare [q_lo, q_hi] intervals across arms; flag near-disjoint ranges (intersection/union ~ 0)
      - also flag 'nonzero support exclusive' for sparse numeric (nonzero count present in one arm, zero in another)

    Returns:
      (issues, metrics)
    """
    issues: List["ValidationIssue"] = []
    ignore = {c for c in ignore_cols if c.strip()}

    covariates = _dedup_keep_order(list(getattr(protocol, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(protocol, "effect_modifiers", []) or []))

    feat_cols = covariates + (effect_modifiers if use_effect_modifiers else [])
    feat_cols = _dedup_keep_order([c for c in feat_cols if c not in ignore])[: int(max_cols)]

    missing = [c for c in feat_cols if c not in df.columns]
    metrics: Dict[str, Any] = {
        "arm_kind": arm_masks.kind,
        "treatment_col": arm_masks.treatment_col,
        "arm_counts": arm_masks.counts,
        "min_arm_n": int(min_arm_n),
        "min_support_per_arm": int(min_support_per_arm),
        "max_levels_checked": int(max_levels_checked),
        "q_lo": float(q_lo),
        "q_hi": float(q_hi),
        "use_effect_modifiers": bool(use_effect_modifiers),
        "n_checked_cols": int(len(feat_cols)),
        "n_missing_cols": int(len(missing)),
        "missing_cols": missing[:200],
    }

    if missing:
        issues.append(
            _issue(
                severity="FAIL",
                message="Cannot run overlap/positivity diagnostics: some referenced covariates/effect modifiers are missing.",
                evidence=metrics,
                fix_hint="Run presence validation earlier or fix upstream column retention.",
            )
        )
        return issues, metrics

    # Eligible arms (avoid tiny arms creating noisy rates)
    arms = [a for a, n in arm_masks.counts.items() if int(n) >= int(min_arm_n)]
    metrics["eligible_arms"] = arms
    if len(arms) < 2:
        issues.append(
            _issue(
                severity="WARN",
                message="Overlap/positivity diagnostics are inconclusive: too few eligible arms (arms too small).",
                evidence=metrics,
                fix_hint="Increase cohort size or relax filtering so each arm has enough rows.",
            )
        )
        return issues, metrics

    exclusive_flags: List[Dict[str, Any]] = []
    checked = 0

    for c in feat_cols:
        checked += 1
        s = df[c]

        # skip datetimes here (type-risk validator handles them); they need explicit encoding first
        if ptypes.is_datetime64_any_dtype(s.dtype):
            continue

        # Per-arm slices
        per_arm = {a: s.loc[arm_masks.masks[a]] for a in arms}

        # -------------------------
        # Numeric / boolean
        # -------------------------
        if ptypes.is_numeric_dtype(s.dtype) or ptypes.is_bool_dtype(s.dtype):
            xnum = pd.to_numeric(s, errors="coerce")
            per_arm_num = {a: xnum.loc[arm_masks.masks[a]] for a in arms}

            # Nonzero exclusivity (sparse numeric / indicator-ish)
            nz = {a: int((per_arm_num[a].fillna(0.0) != 0.0).sum()) for a in arms} # pyright: ignore[reportUnknownMemberType]
            if max(nz.values()) >= int(min_support_per_arm) and any(v == 0 for v in nz.values()):
                exclusive_flags.append(
                    {
                        "col": c,
                        "kind": "numeric_nonzero_exclusive",
                        "dtype": str(s.dtype),
                        "nonzero_by_arm": nz,
                    }
                )
                continue

            # Quantile interval overlap check (only if enough non-missing)
            intervals: Dict[str, Tuple[float, float, int]] = {}
            for a in arms:
                xa = per_arm_num[a].dropna()
                if int(xa.shape[0]) < int(min_support_per_arm):
                    continue
                lo = float(xa.quantile(float(q_lo)))
                hi = float(xa.quantile(float(q_hi)))
                intervals[a] = (lo, hi, int(xa.shape[0]))

            if len(intervals) >= 2:
                # measure overlap of intervals pairwise: intersection length / union length
                # for multi-arm, take worst-case (min overlap)
                overlaps: List[float] = []
                arm_list = list(intervals.keys())
                for i in range(len(arm_list)):
                    for j in range(i + 1, len(arm_list)):
                        a1, a2 = arm_list[i], arm_list[j]
                        lo1, hi1, _ = intervals[a1]
                        lo2, hi2, _ = intervals[a2]
                        inter = max(0.0, min(hi1, hi2) - max(lo1, lo2))
                        union = max(0.0, max(hi1, hi2) - min(lo1, lo2))
                        ov = float(inter / union) if union > 0 else 0.0
                        overlaps.append(ov)

                min_ov = float(min(overlaps)) if overlaps else 1.0
                if min_ov <= 0.01:  # essentially disjoint
                    exclusive_flags.append(
                        {
                            "col": c,
                            "kind": "numeric_interval_disjoint",
                            "dtype": str(s.dtype),
                            "intervals_by_arm": {k: {"qlo": v[0], "qhi": v[1], "n": v[2]} for k, v in intervals.items()},
                            "min_pairwise_overlap_ratio": min_ov,
                        }
                    )
            continue

        # -------------------------
        # Categorical / string-ish
        # -------------------------
        ss = s.astype("string")
        ss_nn = ss.dropna()
        if ss_nn.empty:
            continue

        vc = ss_nn.value_counts(dropna=True)
        levels = [str(k) for k in vc.head(int(max_levels_checked)).index.tolist()]

        per_arm_norm = {a: per_arm[a].astype("string") for a in arms}

        for lvl in levels:
            counts = {a: int((per_arm_norm[a] == lvl).sum()) for a in arms}
            mx = max(counts.values()) if counts else 0
            if mx >= int(min_support_per_arm) and any(v == 0 for v in counts.values()):
                exclusive_flags.append(
                    {
                        "col": c,
                        "kind": "categorical_level_exclusive",
                        "dtype": str(s.dtype),
                        "level": lvl,
                        "counts_by_arm": counts,
                        "levels_checked": int(len(levels)),
                    }
                )
                break  # one strong exclusive level is enough for this feature

    n_feats = max(1, checked)
    n_ex = int(len(exclusive_flags))
    frac = float(n_ex / n_feats)

    metrics.update(
        {
            "n_cols_checked": int(checked),
            "n_exclusive_flags": n_ex,
            "exclusive_frac": frac,
            "examples": exclusive_flags[:50],
        }
    )

    if n_ex == 0:
        return issues, metrics

    # Severity policy: fail only when the dataset looks fundamentally non-overlapping
    sev: Literal["WARN", "FAIL"] = "WARN"
    if frac >= 0.60 and n_ex >= 10:
        sev = "FAIL"

    issues.append(
        _issue(
            severity=sev,
            message="Overlap/positivity risk: some features or categories appear only in certain treatment arms (support mismatch).",
            evidence=metrics,
            fix_hint="Consider trimming to common support, redefining cohort/treatment, collapsing rare levels, or dropping arm-exclusive variables.",
        )
    )
    return issues, metrics


# -----------------------------------------------------------------------------
# 3) Optional propensity separability proxy (binary treatment only)
# -----------------------------------------------------------------------------

def validate_overlap_propensity_proxy(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    arm_masks: ArmMasks,
    # use covariates only by default; effect modifiers often include post-treatment-ish features by mistake
    use_effect_modifiers: bool = False,
    max_features: int = 200,
    sample_n: int = 10000,
    extreme_lo: float = 0.01,
    extreme_hi: float = 0.99,
    auc_warn: float = 0.90,
    extreme_share_warn: float = 0.20,
) -> Tuple[List["ValidationIssue"], Dict[str, Any]]:
    """
    Binary-treatment-only proxy:
      Fit a simple logistic regression on numeric-coercible covariates (and optionally effect_modifiers).
      Flag if:
        - AUC is very high AND
        - many predicted propensities are extreme

    If sklearn is unavailable, emits WARN and skips.
    """
    issues: List["ValidationIssue"] = []

    metrics: Dict[str, Any] = {
        "enabled": False,
        "reason": None,
        "treatment_col": protocol.treatment_spec.column,
        "arm_kind": arm_masks.kind,
        "auc": None,
        "extreme_prob_share": None,
        "n_rows_used": 0,
        "n_features_used": 0,
        "use_effect_modifiers": bool(use_effect_modifiers),
    }

    if arm_masks.kind != "binary":
        metrics["reason"] = "treatment_not_binary"
        return issues, metrics

    # soft dependency
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except Exception:
        issues.append(
            _issue(
                severity="WARN",
                message="Propensity overlap proxy skipped: scikit-learn not available.",
                evidence={"missing_dependency": "scikit-learn"},
                fix_hint="Install scikit-learn or disable this proxy.",
            )
        )
        metrics["reason"] = "sklearn_missing"
        return issues, metrics

    m_t = arm_masks.masks.get("treated")
    m_c = arm_masks.masks.get("control")
    if m_t is None or m_c is None:
        metrics["reason"] = "missing_binary_masks"
        return issues, metrics

    idx_t = df.index[m_t]
    idx_c = df.index[m_c]
    if len(idx_t) == 0 or len(idx_c) == 0:
        metrics["reason"] = "empty_arm"
        return issues, metrics

    # deterministic stratified sampling
    n_total = min(int(sample_n), int(len(idx_t) + len(idx_c)))
    n_half = max(1, n_total // 2)
    take_t = min(len(idx_t), n_half)
    take_c = min(len(idx_c), n_total - take_t)
    if take_t + take_c < n_total:
        rem = n_total - (take_t + take_c)
        if len(idx_t) - take_t >= len(idx_c) - take_c:
            take_t = min(len(idx_t), take_t + rem)
        else:
            take_c = min(len(idx_c), take_c + rem)

    idx_t_s = pd.Index(idx_t).to_series().sample(n=take_t, random_state=0).to_numpy()
    idx_c_s = pd.Index(idx_c).to_series().sample(n=take_c, random_state=1).to_numpy()
    idx = pd.Index(np.concatenate([idx_t_s, idx_c_s]))

    y = np.concatenate([np.ones(len(idx_t_s), dtype=np.int32), np.zeros(len(idx_c_s), dtype=np.int32)])

    covariates = _dedup_keep_order(list(getattr(protocol, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(protocol, "effect_modifiers", []) or []))
    feat_cols = covariates + (effect_modifiers if use_effect_modifiers else [])
    feat_cols = _dedup_keep_order([c for c in feat_cols if c in df.columns])

    metrics["n_features_candidate"] = int(len(feat_cols))

    X_parts: List[np.ndarray] = []
    used: List[str] = []

    for c in feat_cols:
        if len(used) >= int(max_features):
            break
        s = df.loc[idx, c]
        if ptypes.is_datetime64_any_dtype(s.dtype):
            continue

        if ptypes.is_bool_dtype(s.dtype):
            x = s.astype("boolean").fillna(False).astype("int8").to_numpy() # pyright: ignore[reportUnknownMemberType]
            if np.std(x) <= 0:
                continue
            X_parts.append(x.reshape(-1, 1))
            used.append(c)
            continue

        if ptypes.is_numeric_dtype(s.dtype):
            sn = pd.to_numeric(s, errors="coerce")
            med = float(sn.median()) if sn.notna().any() else 0.0
            x = sn.fillna(med).astype("float64").to_numpy() # pyright: ignore[reportUnknownMemberType]
            if np.std(x) <= 0:
                continue
            X_parts.append(x.reshape(-1, 1))
            used.append(c)
            continue

        # object/string: include only if mostly numeric-coercible
        sn2 = pd.to_numeric(s, errors="coerce")
        ok_rate = float(sn2.notna().mean()) if int(sn2.shape[0]) > 0 else 0.0
        if ok_rate >= 0.80:
            med2 = float(sn2.median()) if sn2.notna().any() else 0.0
            x2 = sn2.fillna(med2).astype("float64").to_numpy() # pyright: ignore[reportUnknownMemberType]
            if np.std(x2) <= 0:
                continue
            X_parts.append(x2.reshape(-1, 1))
            used.append(c)

    if not X_parts:
        metrics["reason"] = "no_numeric_features"
        issues.append(
            _issue(
                severity="WARN",
                message="Propensity overlap proxy skipped: no numeric-coercible covariates available pre-transform.",
                evidence=metrics,
                fix_hint="Run overlap proxy after transform (encoding) or ensure covariates include numeric features.",
            )
        )
        return issues, metrics

    X = np.concatenate(X_parts, axis=1)
    metrics["enabled"] = True
    metrics["n_rows_used"] = int(X.shape[0])
    metrics["n_features_used"] = int(X.shape[1])
    metrics["features_used_sample"] = used[:50]

    try:
        lr = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=0,
        )
        lr.fit(X, y)
        p = lr.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, p))
        extreme = float(((p < float(extreme_lo)) | (p > float(extreme_hi))).mean())

        metrics["auc"] = auc
        metrics["extreme_prob_share"] = extreme
        metrics["extreme_lo"] = float(extreme_lo)
        metrics["extreme_hi"] = float(extreme_hi)

        if auc >= float(auc_warn) and extreme >= float(extreme_share_warn):
            issues.append(
                _issue(
                    severity="WARN",
                    message="Propensity proxy suggests weak overlap (strong separability and many extreme propensities).",
                    evidence=metrics,
                    fix_hint="Consider trimming, redefining cohort/treatment, dropping post-treatment variables, or enforcing common support.",
                )
            )
        return issues, metrics

    except Exception as e:
        issues.append(
            _issue(
                severity="WARN",
                message="Propensity overlap proxy failed to run (non-fatal).",
                evidence={**metrics, "error": _safe_display(e)},
                fix_hint="Inspect covariates for numeric issues; you can still rely on univariate overlap checks.",
            )
        )
        metrics["reason"] = "fit_failed"
        return issues, metrics


# -----------------------------------------------------------------------------
# 4) One entrypoint that runs overlap/positivity suite
# -----------------------------------------------------------------------------

def validate_overlap_and_positivity(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    require_covariates: bool = True,
    # univariate knobs
    use_effect_modifiers_univariate: bool = True,
    # propensity proxy knobs
    enable_propensity_proxy: bool = True,
) -> Tuple[List["ValidationIssue"], Dict[str, Any]]:
    """
    Advanced overlap/positivity validation suite.

    Runs:
      - arm mask construction
      - univariate support/exclusivity checks on covariates (+ optional effect_modifiers)
      - optional propensity proxy (binary treatment only)

    Returns:
      (issues, metrics)
    """
    issues: List["ValidationIssue"] = []

    # Build masks (this can raise if T missing; let caller run treatment presence checks earlier)
    arms = compute_arm_masks_from_protocol(df=df, protocol=protocol)

    # Require covariates for overlap checks (otherwise overlap is not meaningful)
    covariates = _dedup_keep_order(list(getattr(protocol, "covariates", []) or []))
    if require_covariates and not covariates:
        metrics = {"require_covariates": True, "n_covariates": 0, "treatment_col": protocol.treatment_spec.column}
        issues.append(
            _issue(
                severity="FAIL",
                message="Cannot assess overlap/positivity: protocol.covariates is empty (no adjustment set).",
                evidence=metrics,
                fix_hint="Add covariates (confounders) to protocol.covariates before causal estimation.",
            )
        )
        return issues, {"arm_masks": arms.to_dict(), **metrics}

    # Univariate overlap
    iss_u, met_u = validate_overlap_positivity_univariate(
        df=df,
        protocol=protocol,
        arm_masks=arms,
        use_effect_modifiers=use_effect_modifiers_univariate,
    )
    issues.extend(iss_u)

    metrics: Dict[str, Any] = {"arm_masks": arms.to_dict(), "univariate": met_u}

    # Optional propensity proxy
    if enable_propensity_proxy:
        iss_p, met_p = validate_overlap_propensity_proxy(df=df, protocol=protocol, arm_masks=arms)
        issues.extend(iss_p)
        metrics["propensity_proxy"] = met_p

    return issues, metrics





# =============================================================================
# Internal helpers
# =============================================================================

def _issue(
    *,
    severity: ValidationSeverity,
    message: str,
    evidence: Dict[str, Any] | None = None,
    fix_hint: str | None = None,
) -> ValidationIssue:
    return {
        "severity": severity,
        "message": message,
        "evidence": dict(evidence or {}),
        "fix_hint": fix_hint,
    }

def _safe_display(v: Any) -> Any:
    """Return a JSON-safe display representation of a value."""
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def _duplicates(cols: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    dups: List[str] = []
    for c in cols:
        if c in seen and c not in dups:
            dups.append(c)
        seen.add(c)
    return dups