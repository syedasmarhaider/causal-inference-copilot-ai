from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, TypedDict

import pandas as pd
import pandas.api.types as ptypes
from typing import cast
from python.implementation.workflows.nodes.compile_protocol.protocol_specs import (
    BinaryOutcomeSpecModel,
    BinaryTreatmentSpecModel,
    CategoricalOutcomeSpecModel,
    CategoricalTreatmentSpecModel,
    ContinuousOutcomeSpecModel,
    ContinuousTreatmentSpecModel,
)

from python.implementation.workflows.nodes.compile_protocol.protocol_specs import (
    DurationOutcomeSpecModel,
    ProtocolSpec,
)
from python.implementation.workflows.utils.utils import BOOL_FALSE, BOOL_TRUE
# =============================================================================
# 5) Covariates / Effect Modifiers validations (pre-transform, raw df)
# =============================================================================

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, cast

import numpy as np

# TODO: move to tools

ValidationSeverity = Literal["WARN", "FAIL"]


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
    if isinstance(protocol.outcome_spec, DurationOutcomeSpecModel):
        outcome_cols: List[str] = [
            protocol.outcome_spec.duration_column,
            protocol.outcome_spec.event_column,
        ]
    else:
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

    tz_type = getattr(protocol, "time_zero_type", None)
    if tz_type != "COLUMN":
        return issues, {"time_zero_type": tz_type}

    tz = getattr(protocol, "time_zero", None)
    if not isinstance(tz, str) or not tz.strip():
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
def validate_treatment_missingness_protocol(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    allow_missing_rate_fail: float = 0.0,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    """
    ProtocolSpec-native treatment missingness check.

    FAIL if:
      - treatment_spec.column not present in df
      - missing_rate > allow_missing_rate_fail
    """
    tcol: str = protocol.treatment_spec.column
    issues: List[ValidationIssue] = []

    if tcol not in df.columns:
        metrics: Dict[str, Any] = {"treatment_col": tcol, "present": False}
        issues.append(
            _issue(
                severity="FAIL",
                message="treatment_spec.column not found in dataframe.",
                evidence=metrics,
                fix_hint="Ensure the treatment column is retained after filtering and matches the dataset column name exactly.",
            )
        )
        return issues, metrics

    s = df[tcol]
    n_rows = int(s.shape[0])
    miss_rate = float(s.isna().mean()) if n_rows > 0 else 0.0

    metrics = {
        "treatment_col": tcol,
        "present": True,
        "dtype": str(s.dtype),
        "n_rows": n_rows,
        "missing_rate": miss_rate,
        "allow_missing_rate_fail": float(allow_missing_rate_fail),
    }

    if miss_rate > float(allow_missing_rate_fail):
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment column has missing values after filtering.",
                evidence=metrics,
                fix_hint="Fix upstream null-handling for the treatment column (drop/impute) so treatment is fully observed.",
            )
        )

    return issues, metrics


def _treatment_allowed_literals(protocol: ProtocolSpec) -> Optional[List[str]]:
    """
    Returns the allowed literal domain for treatment_spec.
    - binary -> [treated, control]
    - categorical -> levels
    - continuous -> None (no finite domain)
    """
    ts = protocol.treatment_spec
    if isinstance(ts, BinaryTreatmentSpecModel):
        return [ts.treated, ts.control]
    if isinstance(ts, CategoricalTreatmentSpecModel):
        return list(ts.levels)
    if isinstance(ts, ContinuousTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        return None
    # Should not happen if schema is enforced
    return None


def validate_treatment_domain_integrity_protocol(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    """
    ProtocolSpec-native treatment domain check.

    Goal:
      Ensure the observed (non-missing) values in treatment_spec.column
      are a subset of the protocol-specified allowed literals (binary/categorical).

    Continuous treatment: domain check is skipped.
    """
    issues: List[ValidationIssue] = []

    # 0) defensive: duplicated df columns can make df[col] return a DataFrame
    if not df.columns.is_unique:
        dupes = df.columns[df.columns.duplicated()].tolist()
        metrics = {"present": None, "df_has_duplicate_columns": True, "duplicate_columns": dupes[:200]}
        issues.append(
            _issue(
                severity="FAIL",
                message="Dataframe has duplicate column names; cannot validate treatment domain safely.",
                evidence=metrics,
                fix_hint="Deduplicate/rename columns upstream (CSV import/join/concat) so df.columns is unique.",
            )
        )
        return issues, metrics

    # 1) locate treatment column
    tcol: str = protocol.treatment_spec.column

    if tcol not in df.columns:
        metrics = {"treatment_col": tcol, "present": False}
        issues.append(
            _issue(
                severity="FAIL",
                message="treatment_spec.column not found in dataframe.",
                evidence=metrics,
                fix_hint="Ensure the treatment column is retained after filtering and matches the dataset column name exactly.",
            )
        )
        return issues, metrics

    s = df[tcol]
    if isinstance(s, pd.DataFrame):  # ultra-defensive
        metrics = {"treatment_col": tcol, "present": True, "error": "df[tcol] returned DataFrame"}
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment column lookup returned multiple columns (duplicate name).",
                evidence=metrics,
                fix_hint="Ensure dataframe column names are unique.",
            )
        )
        return issues, metrics

    # 2) determine allowed domain from protocol
    allowed_literals = _treatment_allowed_literals(protocol)

    metrics: Dict[str, Any] = {
        "treatment_col": tcol,
        "present": True,
        "dtype": str(s.dtype),
        "treatment_kind": getattr(protocol.treatment_spec, "kind", None),
        "n_rows": int(s.shape[0]),
        "missing_rate": float(s.isna().mean()) if int(s.shape[0]) > 0 else 0.0,
    }

    # Continuous => no finite literal domain to validate
    if allowed_literals is None:
        metrics["domain_check"] = "skipped_continuous"
        return issues, metrics

    metrics["allowed_literals"] = list(allowed_literals)

    # 3) compare in a dtype-aware way
    # Decide comparison mode based on series dtype; fall back to normalized string compare when coercion fails.
    compare_mode: str = "string_norm"

    # Build observed values in comparison space
    s_nonnull = s.dropna()

    if s_nonnull.empty:
        metrics["n_unique_observed"] = 0
        metrics["domain_check"] = "no_nonnull_values"
        # Domain check is vacuously satisfied; missingness validator handles whether this is OK.
        return issues, metrics

    # --- BOOLEAN dtype ---
    if ptypes.is_bool_dtype(s.dtype):
        parsed_allowed: List[Optional[bool]] = [_parse_bool_token(x) for x in allowed_literals]
        if all(v is not None for v in parsed_allowed):
            compare_mode = "bool"
            allowed_set = set(parsed_allowed)  # type: ignore[arg-type]
            obs_set = set(s_nonnull.astype("bool").unique().tolist())
            unexpected = sorted([repr(x) for x in obs_set if x not in allowed_set])
            metrics["compare_mode"] = compare_mode
            metrics["n_unique_observed"] = len(obs_set)

            if unexpected:
                issues.append(
                    _issue(
                        severity="FAIL",
                        message="Treatment column contains values outside the protocol-specified boolean domain.",
                        evidence={**metrics, "unexpected": unexpected[:50]},
                        fix_hint="Ensure upstream filtering/whitelisting maps treatment to the protocol literals only.",
                    )
                )
            return issues, metrics
        # else: fall through to string_norm

    # --- NUMERIC dtype ---
    if ptypes.is_numeric_dtype(s.dtype):
        try:
            allowed_num = [float(x) for x in allowed_literals]
            compare_mode = "numeric_float"
            allowed_set = set(allowed_num)
            obs_num = pd.to_numeric(s_nonnull, errors="coerce")
            obs_set = set(obs_num.dropna().unique().tolist())
            unexpected = sorted([_safe_repr(x) for x in obs_set if x not in allowed_set])
            metrics["compare_mode"] = compare_mode
            metrics["n_unique_observed"] = len(obs_set)

            if unexpected:
                issues.append(
                    _issue(
                        severity="FAIL",
                        message="Treatment column contains numeric values outside the protocol-specified domain.",
                        evidence={**metrics, "unexpected": unexpected[:50]},
                        fix_hint="Ensure upstream filtering/whitelisting maps treatment values to the protocol literals only.",
                    )
                )
            return issues, metrics
        except Exception:
            pass  # fall through to string_norm

    # --- DATETIME dtype ---
    if ptypes.is_datetime64_any_dtype(s.dtype):
        try:
            allowed_dt = [pd.to_datetime(x, errors="raise") for x in allowed_literals]
            compare_mode = "datetime"
            allowed_set = set(allowed_dt)
            obs_dt = pd.to_datetime(s_nonnull, errors="coerce")
            obs_set = set(obs_dt.dropna().unique().tolist())
            unexpected = sorted([_safe_repr(x) for x in obs_set if x not in allowed_set])
            metrics["compare_mode"] = compare_mode
            metrics["n_unique_observed"] = len(obs_set)

            if unexpected:
                issues.append(
                    _issue(
                        severity="FAIL",
                        message="Treatment column contains datetime values outside the protocol-specified domain.",
                        evidence={**metrics, "unexpected": unexpected[:50]},
                        fix_hint="Ensure upstream filtering/whitelisting maps treatment values to the protocol literals only.",
                    )
                )
            return issues, metrics
        except Exception:
            pass  # fall through to string_norm

    # --- STRING/NORMALIZED fallback (object/string/categorical or coercion failed) ---
    compare_mode = "string_norm"
    ss = s_nonnull.astype("string").str.strip().str.casefold()
    obs_set = set(ss.unique().tolist())

    allowed_set = {str(x).strip().casefold() for x in allowed_literals}
    unexpected_vals = sorted([x for x in obs_set if x not in allowed_set])

    metrics["compare_mode"] = compare_mode
    metrics["n_unique_observed"] = len(obs_set)

    if unexpected_vals:
        # include counts for debug (bounded)
        vc = ss.value_counts(dropna=True)
        unexpected_counts = [{ "value": v, "count": int(vc.get(v, 0)) } for v in unexpected_vals[:50]] # pyright: ignore[reportUnknownVariableType]
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment column contains values outside the protocol-specified domain.",
                evidence={**metrics, "unexpected": unexpected_counts},
                fix_hint="Ensure upstream filtering/whitelisting removes or maps unexpected values to the allowed literals.",
            )
        )

    return issues, metrics


def validate_treatment_variation(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    min_count_warn: int = 30,
    imbalance_share_warn: float = 0.05,
) -> Tuple[List["ValidationIssue"], Dict[str, Any]]:
    """
    Validate that the treatment column has *usable variation* after upstream filtering.

    What this checks (deterministic, df-backed):
      - Column presence and non-empty dataset.
      - Binary treatment: both arms present; warn on small arms and strong imbalance.
      - Categorical treatment: at least 2 levels present; warn on rare levels.
      - Continuous treatment: numeric parseability; at least 2 unique numeric values; warn on coercion failures.

    Why it matters:
      - No variation => ATE/CATE is undefined (cannot compare arms).
      - Very small arms / rare levels => high variance, unstable nuisance fits, weak overlap.
      - Non-numeric continuous treatment => breaks most estimators or yields nonsense.

    Returns:
      (issues, metrics) where metrics is JSON-friendly and stable for logging.
    """
    ts = protocol.treatment_spec
    tcol = ts.column

    issues: List["ValidationIssue"] = []

    # -------------------------
    # 0) Structural: column exists + non-empty dataframe
    # -------------------------
    if tcol not in df.columns:
        metrics = {"treatment_col": tcol, "present": False, "n_rows": int(df.shape[0])}
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment column not found in dataframe.",
                evidence=metrics,
                fix_hint="Ensure treatment_spec.column is retained through filtering and matches the dataset column name exactly.",
            )
        )
        return issues, metrics

    s = df[tcol]
    n_rows = int(s.shape[0])

    metrics: Dict[str, Any] = {
        "treatment_col": tcol,
        "present": True,
        "dtype": str(s.dtype),
        "kind": getattr(ts, "kind", None),
        "n_rows": n_rows,
        "min_count_warn": int(min_count_warn),
        "imbalance_share_warn": float(imbalance_share_warn),
    }

    if n_rows == 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="No rows available to validate treatment variation (empty dataframe).",
                evidence=metrics,
                fix_hint="Fix upstream filtering/whitelisting that removed all rows.",
            )
        )
        return issues, metrics

    # -------------------------
    # 1) Binary treatment: both arms must be non-empty
    # -------------------------
    if isinstance(ts, BinaryTreatmentSpecModel):
        allowed = [ts.treated, ts.control]
        counts = _counts_by_allowed_literals(s, allowed)

        n_treated = int(counts.get(ts.treated, 0))
        n_control = int(counts.get(ts.control, 0))
        total = int(n_treated + n_control)

        metrics.update(
            {
                "allowed": allowed,
                "counts": counts,
                "n_treated": n_treated,
                "n_control": n_control,
                "n_total_observed_in_domain": total,
            }
        )

        # Hard gate: must have both arms after filtering
        if n_treated == 0 or n_control == 0:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Binary treatment has no variation: one arm is empty after filtering.",
                    evidence=metrics,
                    fix_hint="Redefine treatment mapping, broaden cohort filters, or fix whitelisting so both arms remain.",
                )
            )
            return issues, metrics

        treated_share = float(n_treated / max(1, total))
        metrics["treated_share"] = treated_share

        # Soft stability warnings
        if min(n_treated, n_control) < int(min_count_warn):
            issues.append(
                _issue(
                    severity="WARN",
                    message="Binary treatment has a small arm count; estimates may be unstable.",
                    evidence={**metrics, "min_arm_count": int(min(n_treated, n_control))},
                    fix_hint="Increase sample size, relax cohort filters, or redefine treatment to increase arm sizes.",
                )
            )

        if treated_share < float(imbalance_share_warn) or treated_share > (1.0 - float(imbalance_share_warn)):
            issues.append(
                _issue(
                    severity="WARN",
                    message="Binary treatment is highly imbalanced; overlap/positivity may be weak.",
                    evidence=metrics,
                    fix_hint="Consider trimming to common support, redefining treatment, or collecting more balanced data.",
                )
            )

        return issues, metrics

    # -------------------------
    # 2) Categorical treatment: need >=2 observed levels; warn on rare levels
    # -------------------------
    if isinstance(ts, CategoricalTreatmentSpecModel):
        allowed = list(ts.levels)
        counts = _counts_by_allowed_literals(s, allowed)

        present_levels = [lvl for lvl, cnt in counts.items() if int(cnt) > 0]
        small_levels = {lvl: int(cnt) for lvl, cnt in counts.items() if 0 < int(cnt) < int(min_count_warn)}

        metrics.update(
            {
                "allowed": allowed,
                "counts": counts,
                "n_levels_present": int(len(present_levels)),
                "present_levels": present_levels[:50],
            }
        )

        # Hard gate: must have at least 2 levels present
        if len(present_levels) < 2:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Categorical treatment has <2 levels present after filtering; no variation.",
                    evidence=metrics,
                    fix_hint="Adjust included levels, broaden cohort filters, or fix whitelisting/mapping.",
                )
            )
            return issues, metrics

        # Soft stability warning: rare levels
        if small_levels:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Some categorical treatment levels have small counts; effects may be unstable.",
                    evidence={**metrics, "small_levels": dict(list(small_levels.items())[:50])},
                    fix_hint="Merge rare levels, drop rare arms, or increase cohort size.",
                )
            )

        return issues, metrics

    # -------------------------
    # 3) Continuous treatment: must be numeric-coercible with >=2 unique numeric values
    # -------------------------
    if isinstance(ts, ContinuousTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        v = pd.to_numeric(s, errors="coerce")

        n_nonmissing = int(s.notna().sum())
        n_numeric = int(v.notna().sum())
        n_bad = int(max(0, n_nonmissing - n_numeric))
        n_unique = int(v.nunique(dropna=True))
        parse_rate = float(n_numeric / max(1, n_nonmissing))

        metrics.update(
            {
                "n_nonmissing": n_nonmissing,
                "n_numeric": n_numeric,
                "n_non_numeric_nonmissing": n_bad,
                "numeric_parse_rate": parse_rate,
                "n_unique_numeric": n_unique,
            }
        )

        # Hard gate: must have at least some numeric content
        if n_numeric == 0:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Continuous treatment has no numeric values after filtering.",
                    evidence=metrics,
                    fix_hint="Ensure treatment column is numeric/coercible, or fix upstream typing/cleaning.",
                )
            )
            return issues, metrics

        # Hard gate: must vary
        if n_unique <= 1:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Continuous treatment has <=1 unique numeric value; no variation.",
                    evidence=metrics,
                    fix_hint="Choose a treatment with variability or broaden cohort filtering.",
                )
            )
            return issues, metrics

        # Soft warning: coercion failures indicate dirty tokens (units, commas, text)
        if n_bad > 0:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Some non-numeric tokens exist in continuous treatment (coercion failures).",
                    evidence=metrics,
                    fix_hint="Normalize treatment values (remove units/suffixes, standardize decimal separators) before modeling.",
                )
            )

        return issues, metrics

    # -------------------------
    # 4) Unknown kind: should be unreachable if protocol schema is enforced
    # -------------------------
    issues.append(
        _issue(
            severity="FAIL",
            message="Unknown treatment_spec kind; cannot validate treatment variation.",
            evidence={"treatment_col": tcol, "kind": getattr(ts, "kind", None)},
            fix_hint="Ensure compiled protocol emits a supported treatment spec model.",
        )
    )
    return issues, metrics


# =============================================================================
# 4) Outcome validations (pre-transform; whitelist already applied upstream)
# =============================================================================

def validate_outcome_missingness(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    allow_missing_rate_fail: float = 0.0,
) -> Tuple[List["ValidationIssue"], Dict[str, Any]]:
    """
    Validate that all outcome columns are present and have acceptable missingness.

    Deterministic checks:
      1) All outcome columns referenced by protocol.outcome_spec exist in df.
      2) For each outcome column, missing_rate <= allow_missing_rate_fail.
         - Default policy is strict: no missing outcome values after filtering (allow_missing_rate_fail=0.0).

    Why this is a hard gate (pre-transform):
      - Missing outcomes typically break downstream estimators or force implicit row dropping,
        which can silently change the target population and invalidate assumptions.
      - This must be checked on the full artifact (not via sampling).

    Returns:
      (issues, metrics) where metrics is JSON-friendly and stable for logging.
    """
    ys = protocol.outcome_spec
    cols = _outcome_cols(ys)

    issues: List["ValidationIssue"] = []

    # -------------------------
    # 0) Structural: outcome columns must exist
    # -------------------------
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        metrics = {
            "present": False,
            "expected_outcome_cols": cols,
            "missing_cols": missing_cols[:50],
            "n_missing": int(len(missing_cols)),
            "n_df_cols": int(df.shape[1]),
            "n_rows": int(df.shape[0]),
            "outcome_kind": getattr(ys, "kind", None),
        }
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome column(s) referenced by protocol are missing from dataframe.",
                evidence=metrics,
                fix_hint="Ensure outcome columns are retained through filtering/column-drop steps and the names match exactly.",
            )
        )
        return issues, metrics

    # -------------------------
    # 1) Missingness by outcome column
    # -------------------------
    n_rows = int(df.shape[0])
    metrics: Dict[str, Any] = {
        "present": True,
        "outcome_cols": cols,
        "outcome_kind": ys.kind,
        "n_rows": n_rows,
        "allow_missing_rate_fail": float(allow_missing_rate_fail),
    }

    offenders: List[Dict[str, Any]] = []
    for c in cols:
        s = df[c]
        miss_rate = float(s.isna().mean()) if n_rows > 0 else 0.0

        metrics[f"{c}.dtype"] = str(s.dtype)
        metrics[f"{c}.missing_rate"] = miss_rate

        if miss_rate > float(allow_missing_rate_fail):
            offenders.append({"col": c, "missing_rate": miss_rate, "dtype": str(s.dtype)})

    if offenders:
        # Emit a single failure issue (cleaner) with bounded evidence payload.
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome column(s) contain missing values above allowed threshold.",
                evidence={**metrics, "offenders": offenders[:50], "n_offenders": int(len(offenders))},
                fix_hint=(
                    "Fix upstream null handling: drop rows with missing outcomes, "
                    "or ensure outcome columns are included in the null-purge subset. "
                    "Avoid implicit dropping during model fit."
                ),
            )
        )

    return issues, metrics

def validate_outcome_domain_integrity(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
) -> Tuple[List["ValidationIssue"], Dict[str, Any]]:
    """
    Validate that observed outcome values are consistent with the protocol-defined outcome domain.

    Deterministic checks (df-backed):
      - Duration outcomes: validate domain on event_column only:
          observed(event_column) ⊆ {event_value, censor_value}
      - Binary/Categorical outcomes: validate observed values are within protocol literal domain.
      - Continuous outcomes: domain validation is skipped (no finite literal set).

    Why this is a hard gate:
      - Rare stray values (e.g., '2' in a binary outcome) can silently invalidate estimator assumptions.
      - Domain constraints must be enforced on the full artifact, not inferred via sampling.

    Returns:
      (issues, metrics) where metrics is JSON-friendly and stable for logging/telemetry.
    """
    ys = protocol.outcome_spec
    issues: List["ValidationIssue"] = []

    # -------------------------
    # 0) Duration outcome: domain check on event indicator only
    # -------------------------
    if isinstance(ys, DurationOutcomeSpecModel):
        ecol = ys.event_column

        if ecol not in df.columns:
            metrics = {"present": False, "kind": "duration", "event_column": ecol, "n_rows": int(df.shape[0])}
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Duration outcome event_column referenced by protocol is missing from dataframe.",
                    evidence=metrics,
                    fix_hint="Ensure outcome_spec.event_column is retained through filtering/column-drop steps.",
                )
            )
            return issues, metrics

        s = df[ecol]
        allowed_literals = [ys.event_value, ys.censor_value]

        obs = _observed_values_set(s)
        allowed_set = _allowed_values_set_for_series(s, allowed_literals)

        unexpected = sorted(_safe_repr(x) for x in obs if x not in allowed_set)

        metrics: Dict[str, Any] = {
            "present": True,
            "kind": "duration",
            "event_column": ecol,
            "dtype": str(s.dtype),
            "allowed": allowed_literals,
            "n_unique_observed": int(len(obs)),
        }

        if unexpected:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Duration event indicator contains values outside the protocol-defined domain.",
                    evidence={**metrics, "unexpected": unexpected[:50], "n_unexpected": int(len(unexpected))},
                    fix_hint="Upstream whitelisting/mapping should restrict the event indicator to the protocol literals.",
                )
            )

        return issues, metrics

    # -------------------------
    # 1) Non-duration outcomes: single outcome column required
    # -------------------------
    ycol = ys.column
    if ycol not in df.columns:
        metrics = {"present": False, "kind": ys.kind, "outcome_col": ycol, "n_rows": int(df.shape[0])}
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome column referenced by protocol is missing from dataframe.",
                evidence=metrics,
                fix_hint="Ensure outcome_spec.column is retained through filtering/column-drop steps.",
            )
        )
        return issues, metrics

    s = df[ycol]
    kind = ys.kind

    # Protocol-defined literal domain (None for continuous)
    allowed_literals = _allowed_outcome_literals(ys)

    metrics2: Dict[str, Any] = {
        "present": True,
        "kind": kind,
        "outcome_col": ycol,
        "dtype": str(s.dtype),
    }

    # -------------------------
    # 2) Continuous outcome: no finite domain to validate
    # -------------------------
    if allowed_literals is None:
        metrics2["domain_check"] = "skipped_continuous"
        return issues, metrics2

    # -------------------------
    # 3) Binary/Categorical outcome: observed ⊆ allowed
    # -------------------------
    obs2 = _observed_values_set(s)
    allowed_set2 = _allowed_values_set_for_series(s, allowed_literals)

    unexpected2 = sorted(_safe_repr(x) for x in obs2 if x not in allowed_set2)

    metrics2.update(
        {
            "allowed": list(allowed_literals),
            "n_unique_observed": int(len(obs2)),
        }
    )

    if unexpected2:
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome values contain unexpected values outside the protocol-defined domain.",
                evidence={**metrics2, "unexpected": unexpected2[:50], "n_unexpected": int(len(unexpected2))},
                fix_hint="Upstream whitelisting/mapping should restrict outcome values to the protocol literals.",
            )
        )

    return issues, metrics2


def validate_outcome_variation(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    min_count_warn: int = 30,
    imbalance_share_warn: float = 0.05,
) -> Tuple[List["ValidationIssue"], Dict[str, Any]]:
    """
    Validate that the outcome has usable variation after upstream filtering.

    Deterministic checks (df-backed):
      - Structural: required outcome columns exist and df is non-empty.
      - Duration outcomes:
          * duration_column is numeric-coercible, non-negative, and not completely constant
          * event_column contains both event and censor (WARN if only one)
          * warn on severe imbalance of event vs censor
          * warn on non-numeric duration tokens (coercion failures)
      - Binary outcomes:
          * both classes present (FAIL if one missing)
          * warn on small class sizes and high imbalance
      - Categorical outcomes:
          * >=2 levels present (FAIL if not)
          * warn on rare levels
      - Continuous outcomes:
          * numeric-coercible (FAIL if none)
          * warn if (near) constant (<=1 unique)
          * warn on coercion failures

    Returns:
      (issues, metrics) where metrics is JSON-friendly and stable for logging.
    """
    ys = protocol.outcome_spec
    issues: List["ValidationIssue"] = []

    # -------------------------
    # 0) Duration outcomes (duration + event indicator)
    # -------------------------
    if isinstance(ys, DurationOutcomeSpecModel):
        dcol = ys.duration_column
        ecol = ys.event_column

        # Structural: required cols must exist
        missing_cols = [c for c in (dcol, ecol) if c not in df.columns]
        if missing_cols:
            metrics = {
                "present": False,
                "kind": "duration",
                "missing_cols": missing_cols,
                "duration_column": dcol,
                "event_column": ecol,
                "n_rows": int(df.shape[0]),
            }
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Duration outcome columns referenced by protocol are missing from dataframe.",
                    evidence=metrics,
                    fix_hint="Ensure outcome_spec.duration_column and outcome_spec.event_column are retained through filtering/column-drop steps.",
                )
            )
            return issues, metrics

        sd = df[dcol]
        se = df[ecol]
        n_rows = int(df.shape[0])

        # Duration numeric diagnostics
        vd = pd.to_numeric(sd, errors="coerce")
        n_nonmissing = int(sd.notna().sum())
        n_numeric = int(vd.notna().sum())
        n_bad = int(max(0, n_nonmissing - n_numeric))

        neg_count = int((vd.dropna() < 0).sum())
        n_unique = int(vd.nunique(dropna=True))

        # Event/censor diagnostics (domain integrity handled elsewhere; here we focus on variation)
        allowed_e = [ys.event_value, ys.censor_value]
        counts_e = _counts_by_allowed_literals(se, allowed_e)
        n_event = int(counts_e.get(ys.event_value, 0))
        n_cens = int(counts_e.get(ys.censor_value, 0))
        denom = int(max(1, n_event + n_cens))
        event_share = float(n_event / denom)

        metrics: Dict[str, Any] = {
            "present": True,
            "kind": "duration",
            "n_rows": n_rows,
            "duration_column": dcol,
            "event_column": ecol,
            "duration_dtype": str(sd.dtype),
            "event_dtype": str(se.dtype),
            "min_count_warn": int(min_count_warn),
            "imbalance_share_warn": float(imbalance_share_warn),
            # duration
            "n_nonmissing_duration": n_nonmissing,
            "n_numeric_duration": n_numeric,
            "n_non_numeric_duration_nonmissing": n_bad,
            "numeric_parse_rate_duration": float(n_numeric / max(1, n_nonmissing)),
            "n_unique_duration": n_unique,
            "n_negative_duration": neg_count,
            # event
            "event_counts": counts_e,
            "n_event": n_event,
            "n_censor": n_cens,
            "event_share": event_share,
        }

        # Hard gates: must have numeric duration values and must be non-negative
        if n_numeric == 0:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Duration column has no numeric values after filtering.",
                    evidence=metrics,
                    fix_hint="Ensure duration is numeric/coercible to float and retained through filtering.",
                )
            )
            return issues, metrics

        if neg_count > 0:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Duration column contains negative values (invalid for durations).",
                    evidence=metrics,
                    fix_hint="Fix parsing/cleaning; durations must be >= 0.",
                )
            )
            return issues, metrics

        # Soft warnings: degenerate duration variability
        if n_unique <= 1:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Duration column has <=1 unique numeric value; survival estimation may be degenerate.",
                    evidence=metrics,
                    fix_hint="Verify duration definition; choose a duration with variability.",
                )
            )

        # Soft warnings: event indicator variation / imbalance
        if n_event == 0 or n_cens == 0:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Duration outcome has only one event class observed (all event or all censored).",
                    evidence=metrics,
                    fix_hint="Verify event coding and cohort definition; many survival methods require both event and censoring.",
                )
            )
        else:
            if event_share < float(imbalance_share_warn) or event_share > (1.0 - float(imbalance_share_warn)):
                issues.append(
                    _issue(
                        severity="WARN",
                        message="Duration event indicator is highly imbalanced; estimates may be unstable.",
                        evidence=metrics,
                        fix_hint="Broaden cohort, verify event definition, or consider methods robust to imbalance.",
                    )
                )

        # Soft warning: non-numeric tokens survived
        if n_bad > 0:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Duration column contains some non-numeric tokens (coercion failures).",
                    evidence=metrics,
                    fix_hint="Normalize duration values (remove units/suffixes) before modeling.",
                )
            )

        return issues, metrics

    # -------------------------
    # 1) Non-duration outcomes (single outcome column)
    # -------------------------
    ycol = ys.column
    if ycol not in df.columns:
        metrics = {"present": False, "kind": ys.kind, "outcome_col": ycol, "n_rows": int(df.shape[0])}
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome column referenced by protocol is missing from dataframe.",
                evidence=metrics,
                fix_hint="Ensure outcome_spec.column is retained through filtering/column-drop steps.",
            )
        )
        return issues, metrics

    s = df[ycol]
    n_rows = int(s.shape[0])

    metrics2: Dict[str, Any] = {
        "present": True,
        "kind": ys.kind,
        "outcome_col": ycol,
        "dtype": str(s.dtype),
        "n_rows": n_rows,
        "min_count_warn": int(min_count_warn),
        "imbalance_share_warn": float(imbalance_share_warn),
    }

    if n_rows == 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="No rows available to validate outcome variation (empty dataframe).",
                evidence=metrics2,
                fix_hint="Fix upstream filtering that removed all rows.",
            )
        )
        return issues, metrics2

    # -------------------------
    # 2) Binary outcome: both classes required
    # -------------------------
    if isinstance(ys, BinaryOutcomeSpecModel):
        allowed = [ys.event, ys.non_event]
        counts = _counts_by_allowed_literals(s, allowed)

        n_event = int(counts.get(ys.event, 0))
        n_nonevent = int(counts.get(ys.non_event, 0))
        total = int(n_event + n_nonevent)

        metrics2.update(
            {
                "allowed": allowed,
                "counts": counts,
                "n_event": n_event,
                "n_non_event": n_nonevent,
                "n_total_observed_in_domain": total,
            }
        )

        # Hard gate: must have both classes
        if n_event == 0 or n_nonevent == 0:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Binary outcome has no variation: one class is empty after filtering.",
                    evidence=metrics2,
                    fix_hint="Redefine outcome mapping or broaden cohort filtering so both classes remain.",
                )
            )
            return issues, metrics2

        event_share = float(n_event / max(1, total))
        metrics2["event_share"] = event_share

        # Soft warnings: small class / strong imbalance
        if min(n_event, n_nonevent) < int(min_count_warn):
            issues.append(
                _issue(
                    severity="WARN",
                    message="Binary outcome has a small class count; estimates may be unstable.",
                    evidence={**metrics2, "min_class_count": int(min(n_event, n_nonevent))},
                    fix_hint="Increase sample size, relax cohort filters, or redefine outcome to increase class sizes.",
                )
            )

        if event_share < float(imbalance_share_warn) or event_share > (1.0 - float(imbalance_share_warn)):
            issues.append(
                _issue(
                    severity="WARN",
                    message="Binary outcome is highly imbalanced; estimates may be unstable.",
                    evidence=metrics2,
                    fix_hint="Broaden cohort, reconsider outcome definition, or use methods robust to imbalance.",
                )
            )

        return issues, metrics2

    # -------------------------
    # 3) Categorical outcome: need >=2 observed levels; warn on rare levels
    # -------------------------
    if isinstance(ys, CategoricalOutcomeSpecModel):
        allowed = list(ys.levels)
        counts = _counts_by_allowed_literals(s, allowed)

        present_levels = [lvl for lvl, cnt in counts.items() if int(cnt) > 0]
        small_levels = {lvl: int(cnt) for lvl, cnt in counts.items() if 0 < int(cnt) < int(min_count_warn)}

        metrics2.update(
            {
                "allowed": allowed,
                "counts": counts,
                "n_levels_present": int(len(present_levels)),
                "present_levels": present_levels[:50],
            }
        )

        # Hard gate: must have at least 2 levels present
        if len(present_levels) < 2:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Categorical outcome has <2 levels present after filtering; no variation.",
                    evidence=metrics2,
                    fix_hint="Adjust included levels or broaden cohort filtering.",
                )
            )
            return issues, metrics2

        # Soft warning: rare levels
        if small_levels:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Some categorical outcome levels have small counts; estimates may be unstable.",
                    evidence={**metrics2, "small_levels": dict(list(small_levels.items())[:50])},
                    fix_hint="Merge rare levels, drop rare classes, or increase cohort size.",
                )
            )

        return issues, metrics2

    # -------------------------
    # 4) Continuous outcome: must be numeric-coercible; warn on degeneracy + coercion failures
    # -------------------------
    if isinstance(ys, ContinuousOutcomeSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        v = pd.to_numeric(s, errors="coerce")

        n_nonmissing = int(s.notna().sum())
        n_numeric = int(v.notna().sum())
        n_bad = int(max(0, n_nonmissing - n_numeric))

        n_unique = int(v.nunique(dropna=True))
        parse_rate = float(n_numeric / max(1, n_nonmissing))

        metrics2.update(
            {
                "n_nonmissing": n_nonmissing,
                "n_numeric": n_numeric,
                "n_non_numeric_nonmissing": n_bad,
                "numeric_parse_rate": parse_rate,
                "n_unique_numeric": n_unique,
            }
        )

        # Hard gate: must have numeric content
        if n_numeric == 0:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Continuous outcome has no numeric values after filtering.",
                    evidence=metrics2,
                    fix_hint="Ensure outcome column is numeric/coercible to float, or fix upstream typing/cleaning.",
                )
            )
            return issues, metrics2

        # Soft warning: near-constant outcome is degenerate for many models
        if n_unique <= 1:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Continuous outcome has <=1 unique numeric value; estimates may be degenerate.",
                    evidence=metrics2,
                    fix_hint="Verify outcome definition or cohort filtering; choose an outcome with variability.",
                )
            )

        # Soft warning: coercion failures
        if n_bad > 0:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Continuous outcome contains some non-numeric tokens (coercion failures).",
                    evidence=metrics2,
                    fix_hint="Normalize outcome values (remove units/suffixes) before modeling.",
                )
            )

        return issues, metrics2

    # -------------------------
    # 5) Unknown kind: should be unreachable if protocol schema is enforced
    # -------------------------
    issues.append(
        _issue(
            severity="FAIL",
            message="Unknown outcome_spec kind; cannot validate outcome variation.",
            evidence={"kind": getattr(ys, "kind", None), "outcome_col": ycol},
            fix_hint="Ensure compiled protocol emits a supported outcome spec model.",
        )
    )
    return issues, metrics2


# =============================================================================
# Shared helpers for 3) and 4)
# =============================================================================

def _allowed_outcome_literals(ys: Any) -> Optional[List[str]]:
    if isinstance(ys, BinaryOutcomeSpecModel):
        return [ys.event, ys.non_event]
    if isinstance(ys, CategoricalOutcomeSpecModel):
        return list(ys.levels)
    if isinstance(ys, ContinuousOutcomeSpecModel):
        return None
    return None


def _outcome_cols(ys: Any) -> List[str]:
    if isinstance(ys, DurationOutcomeSpecModel):
        return [ys.duration_column, ys.event_column]
    return [cast(str, getattr(ys, "column"))]


def _observed_values_set(s: pd.Series) -> set[Any]:
    # Keep raw comparable values where possible; drop missing.
    return set(s.dropna().unique().tolist())


def _allowed_values_set_for_series(s: pd.Series, allowed_literals: Sequence[str]) -> set[Any]:
    """
    Convert protocol string literals into comparable values for this series dtype.

    - bool dtype: strict tokens {true/false/1/0/yes/no}
    - numeric dtype: float(...)
    - datetime dtype: pd.to_datetime(..., errors='raise')
    - object/string/category: normalized string comparison
    """
    dt = s.dtype

    if ptypes.is_bool_dtype(dt):
        outb: set[bool] = set()
        for raw in allowed_literals:
            b = _parse_bool_token(raw)
            if b is None:
                # if protocol literal can't map to bool, fall back to string compare via repr
                return {str(x).strip().casefold() for x in allowed_literals}
            outb.add(b)
        return outb

    if ptypes.is_numeric_dtype(dt):
        outn: set[float] = set()
        for raw in allowed_literals:
            try:
                outn.add(float(raw))
            except Exception:
                # fall back to string compare
                return {str(x).strip().casefold() for x in allowed_literals}
        return outn

    if ptypes.is_datetime64_any_dtype(dt):
        outd: set[pd.Timestamp] = set()
        for raw in allowed_literals:
            try:
                outd.add(pd.to_datetime(raw, errors="raise"))
            except Exception:
                return {str(x).strip().casefold() for x in allowed_literals}
        return outd

    # object/string/category/other => normalized strings
    return {str(x).strip().casefold() for x in allowed_literals}


def _counts_by_allowed_literals(s: pd.Series, allowed_literals: Sequence[str]) -> Dict[str, int]:
    """
    Returns counts keyed by *protocol literal strings* (stable order as in allowed_literals).

    For non-string dtypes we still key by the original literal string, but match using coerced
    comparable values when possible.
    """
    out: Dict[str, int] = {lit: 0 for lit in allowed_literals}
    if s.empty:
        return out

    dt = s.dtype

    # bool/numeric/datetime: compare in native space if coercion works
    if ptypes.is_bool_dtype(dt):
        for lit in allowed_literals:
            b = _parse_bool_token(lit)
            if b is None:
                # fallback to normalized string compare
                out = _counts_by_normalized_string(s, allowed_literals)
                return out
            out[lit] = int((s == b).sum())
        return out

    if ptypes.is_numeric_dtype(dt):
        v = pd.to_numeric(s, errors="coerce")
        for lit in allowed_literals:
            try:
                thr = float(lit)
            except Exception:
                return _counts_by_normalized_string(s, allowed_literals)
            out[lit] = int((v == thr).sum())
        return out

    if ptypes.is_datetime64_any_dtype(dt):
        for lit in allowed_literals:
            try:
                ts = pd.to_datetime(lit, errors="raise")
            except Exception:
                return _counts_by_normalized_string(s, allowed_literals)
            out[lit] = int((s == ts).sum())
        return out

    # object/string/category: normalized string match
    return _counts_by_normalized_string(s, allowed_literals)


def _counts_by_normalized_string(s: pd.Series, allowed_literals: Sequence[str]) -> Dict[str, int]:
    ss = s.astype("string").str.strip().str.casefold()
    out: Dict[str, int] = {lit: 0 for lit in allowed_literals}
    for lit in allowed_literals:
        key = str(lit).strip().casefold()
        out[lit] = int(ss.eq(key).sum())
    return out

def _safe_repr(x: Any) -> str:
    try:
        return repr(x)
    except Exception:
        return "<unrepr>"


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

    elif isinstance(ts, CategoricalTreatmentSpecModel):
        for lvl in list(ts.levels):
            arm_masks[str(lvl)] = _mask_equals_literal(sT, str(lvl))

    elif isinstance(ts, ContinuousTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        issues.append(
            _issue(
                severity="WARN",
                message="Differential missingness by arm is skipped for continuous treatment (not implemented).",
                evidence=metrics,
                fix_hint="If needed, implement quantile binning for continuous treatment and compare missingness across bins.",
            )
        )
        metrics["skipped"] = "continuous_treatment"
        return issues, metrics

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

    if isinstance(ts, CategoricalTreatmentSpecModel):
        masks: Dict[str, pd.Series] = {}
        for lvl in list(ts.levels):
            masks[str(lvl)] = _mask_equals_literal(s, str(lvl))
        counts = {k: int(v.sum()) for k, v in masks.items()}
        return ArmMasks(kind="categorical", treatment_col=tcol, masks=masks, counts=counts)

    if isinstance(ts, ContinuousTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        # Diagnostics-only binning (quantiles). If qcut fails, fall back to single arm.
        sn = pd.to_numeric(s, errors="coerce")
        if sn.dropna().empty:
            masks = {"all": pd.Series([True] * len(df), index=df.index)}
            return ArmMasks(kind="continuous", treatment_col=tcol, masks=masks, counts={"all": int(len(df))}, notes="treatment_all_missing_or_non_numeric")

        q = int(max(2, min(int(max_bins_continuous), 10)))
        try:
            bins = pd.qcut(sn, q=q, duplicates="drop")
            masks = {}
            if hasattr(bins, "cat"):
                for cat in bins.cat.categories:
                    name = f"bin:{cat.left:g}..{cat.right:g}"
                    masks[name] = cast(pd.Series, bins.eq(cat).astype(object).fillna(False).astype(bool)) # type: ignore[call-overload]
            if not masks:
                masks = {"all": pd.Series([True] * len(df), index=df.index)}
                return ArmMasks(kind="continuous", treatment_col=tcol, masks=masks, counts={"all": int(len(df))}, notes="qcut_empty_bins")
            counts = {k: int(v.sum()) for k, v in masks.items()}
            return ArmMasks(kind="continuous", treatment_col=tcol, masks=masks, counts=counts, notes="quantile_bins")
        except Exception:
            masks = {"all": pd.Series([True] * len(df), index=df.index)}
            return ArmMasks(kind="continuous", treatment_col=tcol, masks=masks, counts={"all": int(len(df))}, notes="qcut_exception_fallback_all")

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
                evidence={**metrics, "error": repr(e)},
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

def _duplicates(cols: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    dups: List[str] = []
    for c in cols:
        if c in seen and c not in dups:
            dups.append(c)
        seen.add(c)
    return dups