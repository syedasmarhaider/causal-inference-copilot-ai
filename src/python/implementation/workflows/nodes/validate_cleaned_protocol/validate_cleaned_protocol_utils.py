from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict

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


@dataclass(frozen=True)
class KeyColumns:
    """
    Canonical column roles extracted from a compiled ProtocolSpec.

    Notes:
      - outcome_cols is length 1 for non-duration outcomes
      - outcome_cols is length 2 for duration outcomes: [duration_column, event_column]
      - time_zero_col is present only when time_zero_type == "COLUMN"
      - W_cols / X_cols preserve protocol order (no silent dedupe)
    """

    treatment_col: str
    outcome_cols: List[str]
    time_zero_col: Optional[str]
    W_cols: List[str]
    X_cols: List[str]


# =============================================================================
# 1) Core extractors
# =============================================================================

def extract_key_columns(protocol: ProtocolSpec) -> KeyColumns:
    """
    Extracts the authoritative T/Y/W/X/(optional)time_zero column names from a compiled protocol.

    This is pure structural extraction; it does not touch the DataFrame.
    """
    # Treatment
    tcol = _req_nonempty(getattr(protocol.treatment_spec, "column", None), "treatment_spec.column")

    # Outcome (duration has 2 cols)
    ys = protocol.outcome_spec
    if isinstance(ys, DurationOutcomeSpecModel):
        dcol = _req_nonempty(getattr(ys, "duration_column", None), "outcome_spec.duration_column")
        ecol = _req_nonempty(getattr(ys, "event_column", None), "outcome_spec.event_column")
        outcome_cols = [dcol, ecol]
    else:
        ycol = _req_nonempty(getattr(ys, "column", None), "outcome_spec.column")
        outcome_cols = [ycol]

    # Time zero
    tz_col: Optional[str] = None
    if getattr(protocol, "time_zero_type", None) == "COLUMN":
        tz_col = _req_nonempty(getattr(protocol, "time_zero", None), "time_zero")

    # W/X
    W_cols = _list_str(getattr(protocol, "covariates", []), "covariates")
    X_cols = _list_str(getattr(protocol, "effect_modifiers", []), "effect_modifiers")

    return KeyColumns(
        treatment_col=tcol,
        outcome_cols=outcome_cols,
        time_zero_col=tz_col,
        W_cols=W_cols,
        X_cols=X_cols,
    )


def select_modeling_view(
    df: pd.DataFrame,
    key_cols: KeyColumns,
    *,
    include_time_zero: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Returns a restricted DF view containing only T/Y/W/X (+ optional time_zero).
    Deterministic column order: preserves df.columns order (not protocol order).
    Non-strict: silently drops missing columns (validation should catch missing upstream).
    """
    want: List[str] = [key_cols.treatment_col, *key_cols.outcome_cols, *key_cols.W_cols, *key_cols.X_cols]
    if include_time_zero and key_cols.time_zero_col:
        want.append(key_cols.time_zero_col)

    want_set = {c for c in want if c and c.strip()}
    keep = [c for c in df.columns if str(c) in want_set]
    out = df.loc[:, keep]
    return out.copy() if copy else out


# =============================================================================
# 2) Structural invariants (protocol + df)
# =============================================================================

def validate_min_rows(
    df: pd.DataFrame,
    *,
    min_rows_fail: int = 10,
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


def validate_column_list_invariants(key_cols: KeyColumns) -> List[ValidationIssue]:
    """
    Pure invariants about *lists of names* (no df needed).

    FAIL:
      - duplicates within role lists (W, X, outcome_cols)
      - any overlap of W/X with T or outcome columns
    WARN:
      - overlap between W and X (allowed, but usually undesirable)
      - time_zero overlaps with other roles (often confusing, but not always invalid)
    """
    issues: List[ValidationIssue] = []

    t = key_cols.treatment_col
    ys = list(key_cols.outcome_cols)
    W = list(key_cols.W_cols)
    X = list(key_cols.X_cols)
    tz = key_cols.time_zero_col

    # Duplicates within lists (FAIL)
    for label, cols in (("outcome_cols", ys), ("W_cols", W), ("X_cols", X)):
        dups = _duplicates(cols)
        if dups:
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"Duplicate columns found in {label}.",
                    evidence={"duplicates": dups, "all": cols},
                    fix_hint="Remove duplicates in compiled protocol output; each role list should be unique.",
                )
            )

    # W/X must not include T or outcomes (FAIL)
    forbidden = {t, *ys}
    bad_W = sorted([c for c in W if c in forbidden])
    bad_X = sorted([c for c in X if c in forbidden])

    if bad_W:
        issues.append(
            _issue(
                severity="FAIL",
                message="Covariates (W) overlap with treatment/outcome columns.",
                evidence={"overlap": bad_W, "treatment_col": t, "outcome_cols": ys},
                fix_hint="Remove T/Y columns from covariates; keep them only in their dedicated roles.",
            )
        )
    if bad_X:
        issues.append(
            _issue(
                severity="FAIL",
                message="Effect modifiers (X) overlap with treatment/outcome columns.",
                evidence={"overlap": bad_X, "treatment_col": t, "outcome_cols": ys},
                fix_hint="Remove T/Y columns from effect modifiers; keep them only in their dedicated roles.",
            )
        )

    # Overlap W vs X (WARN)
    inter = sorted(set(W).intersection(set(X)))
    if inter:
        issues.append(
            _issue(
                severity="WARN",
                message="W and X overlap (same column appears in both covariates and effect modifiers).",
                evidence={"overlap": inter},
                fix_hint="Prefer disjoint sets: keep confounders in W and reserve X for heterogeneity drivers.",
            )
        )

    # time_zero overlap (WARN)
    if isinstance(tz, str) and tz:
        tz_overlap = sorted({tz}.intersection({t, *ys, *W, *X}))
        if tz_overlap:
            issues.append(
                _issue(
                    severity="WARN",
                    message="time_zero column overlaps with another model role column.",
                    evidence={"time_zero_col": tz, "overlaps_with": tz_overlap},
                    fix_hint="If intentional, keep it. Otherwise pick a dedicated baseline/time column.",
                )
            )

    # duration outcome must be two distinct cols (FAIL) — already enforced by duplicates check,
    # but keep explicit for readability.
    if len(ys) == 2 and ys[0] == ys[1]:
        issues.append(
            _issue(
                severity="FAIL",
                message="Duration outcome requires distinct duration_column and event_column.",
                evidence={"outcome_cols": ys},
                fix_hint="Set outcome_spec.duration_column and outcome_spec.event_column to different columns.",
            )
        )

    return issues


def validate_time_zero_semantics(
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    key_cols: KeyColumns,
    *,
    sample_n: int = 2000,
    parse_fail_rate_warn: float = 0.10,
    parse_fail_rate_fail: float = 0.50,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    """
    Lightweight time-zero validation for COLUMN time_zero only (pre-transform).

    We do NOT enforce strict datetime dtype, but we do flag clearly unparseable values.
    """
    issues: List[ValidationIssue] = []

    if getattr(protocol, "time_zero_type", None) != "COLUMN":
        return issues, {"time_zero_type": getattr(protocol, "time_zero_type", None)}

    tz = key_cols.time_zero_col
    if not isinstance(tz, str) or not tz.strip():
        # Should not happen if protocol compiled correctly.
        issues.append(
            _issue(
                severity="FAIL",
                message="time_zero_type=='COLUMN' but time_zero column name is missing.",
                evidence={"time_zero": tz},
                fix_hint="Set protocol.time_zero to a non-empty dataset column name.",
            )
        )
        return issues, {"time_zero_col": tz}

    if tz not in df.columns:
        issues.append(
            _issue(
                severity="FAIL",
                message="time_zero column not found in dataframe.",
                evidence={"time_zero_col": tz},
                fix_hint="Ensure time_zero column is retained after filtering and is spelled exactly as in the dataset.",
            )
        )
        return issues, {"time_zero_col": tz}

    s = df[tz]
    n = int(s.shape[0])
    miss_rate = float(s.isna().mean()) if n > 0 else 0.0

    metrics: Dict[str, Any] = {
        "time_zero_col": tz,
        "dtype": str(s.dtype),
        "n_rows": n,
        "missing_rate": miss_rate,
    }

    # If it's already datetime-like => accept.
    if ptypes.is_datetime64_any_dtype(s.dtype):
        return issues, metrics

    # Numeric time representations are acceptable, but warn (user likely expects datetime).
    if ptypes.is_numeric_dtype(s.dtype):
        issues.append(
            _issue(
                severity="WARN",
                message="time_zero column is numeric; ensure downstream logic interprets it correctly (e.g., timestamp vs offset).",
                evidence=metrics,
                fix_hint="If this is a timestamp, consider converting to datetime for clarity.",
            )
        )
        return issues, metrics

    # Try parsing (string/object/category)
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

    # sample for speed
    if int(ss.shape[0]) > int(sample_n):
        ss = ss.sample(n=int(sample_n), random_state=0)  # deterministic

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


def _req_nonempty(v: Any, field: str) -> str:
    if isinstance(v, str):
        s = v.strip()
        if s:
            return s
    raise ValueError(f"ProtocolSpec invalid: {field} must be a non-empty string (got {v!r})")


def _list_str(v: Any, field: str) -> List[str]:
    if v is None:
        return []
    if not isinstance(v, list):
        raise ValueError(f"ProtocolSpec invalid: {field} must be a list[str] (got {type(v).__name__})")
    out: List[str] = []
    for i, x in enumerate(v): # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType] 
        if not isinstance(x, str):
            raise ValueError(f"ProtocolSpec invalid: {field}[{i}] must be str (got {type(x).__name__})") # pyright: ignore[reportUnknownArgumentType]
        s = x.strip()
        if not s:
            raise ValueError(f"ProtocolSpec invalid: {field}[{i}] must be non-empty")
        out.append(s)
    return out


def _duplicates(cols: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    dups: List[str] = []
    for c in cols:
        if c in seen and c not in dups:
            dups.append(c)
        seen.add(c)
    return dups


# =============================================================================
# 3) Treatment validations (pre-transform; whitelist already applied upstream)
# =============================================================================
def validate_treatment_missingness(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    allow_missing_rate_fail: float = 0.0,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    ts = protocol.treatment_spec
    tcol = getattr(ts, "column")

    issues: List[ValidationIssue] = []
    if tcol not in df.columns:
        return (
            [
                _issue(
                    severity="FAIL",
                    message="Treatment column missing in dataframe (unexpected; columns should be verified already).",
                    evidence={"treatment_col": tcol},
                    fix_hint="Ensure you keep treatment_spec.column throughout filtering steps.",
                )
            ],
            {"treatment_col": tcol, "present": False},
        )

    s: pd.Series = df[tcol]
    miss_rate = float(s.isna().mean()) if int(s.shape[0]) > 0 else 0.0
    dtype_str = str(s.dtype)
    metrics = {"treatment_col": tcol, "dtype": dtype_str, "missing_rate": miss_rate, "n_rows": int(s.shape[0])}

    if miss_rate > float(allow_missing_rate_fail):
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment column contains missing values after filtering; pre-transform validation expects none.",
                evidence=metrics,
                fix_hint="Fix upstream null purge or ensure treatment column is included in the null purge subset.",
            )
        )

    return issues, metrics


def validate_treatment_domain_integrity(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    ts = protocol.treatment_spec
    tcol = getattr(ts, "column")

    issues: List[ValidationIssue] = []
    if tcol not in df.columns:
        return (
            [
                _issue(
                    severity="FAIL",
                    message="Treatment column missing in dataframe (unexpected; columns should be verified already).",
                    evidence={"treatment_col": tcol},
                    fix_hint="Ensure you keep treatment_spec.column throughout filtering steps.",
                )
            ],
            {"treatment_col": tcol, "present": False},
        )

    allowed = _allowed_treatment_literals(ts)
    s = df[tcol]
    metrics: Dict[str, Any] = {"treatment_col": tcol, "dtype": str(s.dtype), "kind": getattr(ts, "kind", None)}

    # Continuous: no literal domain to verify
    if allowed is None:
        metrics["domain_check"] = "skipped_continuous"
        return issues, metrics

    obs = _observed_values_set(s)
    allowed_set = _allowed_values_set_for_series(s, allowed)

    metrics["allowed"] = allowed
    metrics["n_unique_observed"] = len(obs)

    # If any observed value is outside allowed => FAIL
    unexpected = sorted([_safe_repr(x) for x in obs if x not in allowed_set])
    if unexpected:
        issues.append(
            _issue(
                severity="FAIL",
                message="Observed treatment values contain unexpected values outside protocol domain.",
                evidence={"treatment_col": tcol, "unexpected": unexpected[:50], **metrics},
                fix_hint="Upstream whitelist should remove these; verify whitelist logic and literal typing.",
            )
        )

    return issues, metrics


def validate_treatment_variation(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    min_count_warn: int = 30,
    imbalance_share_warn: float = 0.05,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    ts = protocol.treatment_spec
    tcol = getattr(ts, "column")

    issues: List[ValidationIssue] = []
    if tcol not in df.columns:
        return (
            [
                _issue(
                    severity="FAIL",
                    message="Treatment column missing in dataframe (unexpected; columns should be verified already).",
                    evidence={"treatment_col": tcol},
                    fix_hint="Ensure you keep treatment_spec.column throughout filtering steps.",
                )
            ],
            {"treatment_col": tcol, "present": False},
        )

    s = df[tcol]
    n = int(s.shape[0])
    metrics: Dict[str, Any] = {"treatment_col": tcol, "dtype": str(s.dtype), "kind": getattr(ts, "kind", None), "n_rows": n}

    if n == 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="No rows available to validate treatment variation.",
                evidence=metrics,
                fix_hint="Fix upstream filtering that removed all rows.",
            )
        )
        return issues, metrics

    if isinstance(ts, BinaryTreatmentSpecModel):
        allowed = [ts.treated, ts.control]
        counts = _counts_by_allowed_literals(s, allowed)
        metrics["counts"] = counts
        metrics["allowed"] = allowed

        n_t = int(counts.get(ts.treated, 0))
        n_c = int(counts.get(ts.control, 0))

        if n_t == 0 or n_c == 0:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Binary treatment has no variation: one arm is empty after filtering.",
                    evidence={"n_treated": n_t, "n_control": n_c, **metrics},
                    fix_hint="Redefine treatment mapping or broaden cohort filtering.",
                )
            )
            return issues, metrics

        share = float(n_t / max(1, (n_t + n_c)))
        metrics["treated_share"] = share

        if min(n_t, n_c) < int(min_count_warn):
            issues.append(
                _issue(
                    severity="WARN",
                    message="Binary treatment arm has small count; estimates may be unstable.",
                    evidence={"min_arm_count": min(n_t, n_c), "min_count_warn": int(min_count_warn), **metrics},
                    fix_hint="Broaden cohort or redefine treatment to increase arm sizes.",
                )
            )

        if share < float(imbalance_share_warn) or share > (1.0 - float(imbalance_share_warn)):
            issues.append(
                _issue(
                    severity="WARN",
                    message="Binary treatment is highly imbalanced; overlap/positivity may be weak.",
                    evidence={"treated_share": share, "imbalance_share_warn": float(imbalance_share_warn), **metrics},
                    fix_hint="Consider redefining treatment, trimming, or collecting more balanced data.",
                )
            )

        return issues, metrics

    if isinstance(ts, CategoricalTreatmentSpecModel):
        allowed = list(ts.levels)
        counts = _counts_by_allowed_literals(s, allowed)
        present_levels = [k for k, v in counts.items() if int(v) > 0]

        metrics["allowed"] = allowed
        metrics["counts"] = counts
        metrics["n_levels_present"] = len(present_levels)

        if len(present_levels) < 2:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Categorical treatment has <2 levels present after filtering; no variation.",
                    evidence=metrics,
                    fix_hint="Adjust included levels or broaden cohort filtering.",
                )
            )
            return issues, metrics

        small = {k: int(v) for k, v in counts.items() if int(v) > 0 and int(v) < int(min_count_warn)}
        if small:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Some categorical treatment levels have small counts; effects may be unstable.",
                    evidence={"small_levels": small, "min_count_warn": int(min_count_warn), **metrics},
                    fix_hint="Merge rare levels or increase cohort size.",
                )
            )

        return issues, metrics

    # Continuous
    if isinstance(ts, ContinuousTreatmentSpecModel):
        v = pd.to_numeric(s, errors="coerce")
        n_nonmissing = int(s.notna().sum())
        n_numeric = int(v.notna().sum())
        n_bad = int(max(0, n_nonmissing - n_numeric))

        metrics.update(
            {
                "n_nonmissing": n_nonmissing,
                "n_numeric": n_numeric,
                "n_non_numeric_nonmissing": n_bad,
                "numeric_parse_rate": float(n_numeric / max(1, n_nonmissing)),
                "n_unique_numeric": int(v.nunique(dropna=True)),
            }
        )

        if n_numeric == 0:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Continuous treatment has no numeric values after filtering.",
                    evidence=metrics,
                    fix_hint="Ensure treatment column is numeric/coercible or fix upstream typing/cleaning.",
                )
            )
            return issues, metrics

        if int(v.nunique(dropna=True)) <= 1:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Continuous treatment has <=1 unique numeric value; no variation.",
                    evidence=metrics,
                    fix_hint="Choose a treatment with variability or broaden cohort filtering.",
                )
            )
            return issues, metrics

        # Warn if many non-numeric tokens survived (should be rare after earlier cleaning)
        if n_bad > 0:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Some non-numeric values exist in continuous treatment (coercion failures).",
                    evidence=metrics,
                    fix_hint="Normalize treatment values (e.g., remove units/suffixes) before modeling.",
                )
            )

        return issues, metrics

    # Unknown kind (should not happen)
    issues.append(
        _issue(
            severity="FAIL",
            message="Unknown treatment_spec kind; cannot validate treatment variation.",
            evidence={"kind": getattr(ts, "kind", None), "treatment_col": tcol},
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
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    ys = protocol.outcome_spec
    issues: List[ValidationIssue] = []

    cols = _outcome_cols(ys)
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        return (
            [
                _issue(
                    severity="FAIL",
                    message="Outcome column(s) missing in dataframe (unexpected; columns should be verified already).",
                    evidence={"missing_cols": missing_cols, "expected_outcome_cols": cols},
                    fix_hint="Ensure you keep outcome columns throughout filtering steps.",
                )
            ],
            {"expected_outcome_cols": cols, "present": False},
        )

    metrics: Dict[str, Any] = {"outcome_cols": cols, "kinds": getattr(ys, "kind", None)}
    for c in cols:
        s = df[c]
        miss_rate = float(s.isna().mean()) if int(s.shape[0]) > 0 else 0.0
        metrics[f"{c}.dtype"] = str(s.dtype)
        metrics[f"{c}.missing_rate"] = miss_rate

        if miss_rate > float(allow_missing_rate_fail):
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Outcome column contains missing values after filtering; pre-transform validation expects none.",
                    evidence={"col": c, "missing_rate": miss_rate, "allow": float(allow_missing_rate_fail), **metrics},
                    fix_hint="Fix upstream null purge or ensure outcome columns are included in the null purge subset.",
                )
            )

    return issues, metrics


def validate_outcome_domain_integrity(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    ys = protocol.outcome_spec
    issues: List[ValidationIssue] = []

    # Duration: domain is on event_column only (event_value/censor_value)
    if isinstance(ys, DurationOutcomeSpecModel):
        ecol = ys.event_column
        if ecol not in df.columns:
            return (
                [
                    _issue(
                        severity="FAIL",
                        message="Duration outcome event_column missing in dataframe.",
                        evidence={"event_column": ecol},
                        fix_hint="Ensure you keep outcome_spec.event_column throughout filtering steps.",
                    )
                ],
                {"event_column": ecol, "present": False},
            )

        s = df[ecol]
        allowed = [ys.event_value, ys.censor_value]
        obs = _observed_values_set(s)
        allowed_set = _allowed_values_set_for_series(s, allowed)

        unexpected = sorted([_safe_repr(x) for x in obs if x not in allowed_set])
        metrics: Dict[str, Any] = {"kind": "duration", "event_column": ecol, "dtype": str(s.dtype), "allowed": allowed, "n_unique_observed": len(obs)}

        if unexpected:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Observed duration event values contain unexpected values outside protocol domain.",
                    evidence={"unexpected": unexpected[:50], **metrics},
                    fix_hint="Upstream whitelist should remove these; verify whitelist logic and literal typing.",
                )
            )

        return issues, metrics

    # Non-duration outcomes
    ycol = cast(str, getattr(ys, "column"))
    if ycol not in df.columns:
        return (
            [
                _issue(
                    severity="FAIL",
                    message="Outcome column missing in dataframe (unexpected; columns should be verified already).",
                    evidence={"outcome_col": ycol},
                    fix_hint="Ensure you keep outcome_spec.column throughout filtering steps.",
                )
            ],
            {"outcome_col": ycol, "present": False},
        )

    allowed = _allowed_outcome_literals(ys)
    s = df[ycol]
    metrics2: Dict[str, Any] = {"kind": getattr(ys, "kind", None), "outcome_col": ycol, "dtype": str(s.dtype)}

    # Continuous: no domain
    if allowed is None:
        metrics2["domain_check"] = "skipped_continuous"
        return issues, metrics2

    obs2 = _observed_values_set(s)
    allowed_set2 = _allowed_values_set_for_series(s, allowed)

    unexpected2 = sorted([_safe_repr(x) for x in obs2 if x not in allowed_set2])
    metrics2["allowed"] = allowed
    metrics2["n_unique_observed"] = len(obs2)

    if unexpected2:
        issues.append(
            _issue(
                severity="FAIL",
                message="Observed outcome values contain unexpected values outside protocol domain.",
                evidence={"unexpected": unexpected2[:50], **metrics2},
                fix_hint="Upstream whitelist should remove these; verify whitelist logic and literal typing.",
            )
        )

    return issues, metrics2


def validate_outcome_variation(
    *,
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    min_count_warn: int = 30,
    imbalance_share_warn: float = 0.05,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    ys = protocol.outcome_spec
    issues: List[ValidationIssue] = []

    # Duration outcome: validate duration numeric/non-negative + event/censor presence
    if isinstance(ys, DurationOutcomeSpecModel):
        dcol = ys.duration_column
        ecol = ys.event_column

        missing_cols = [c for c in (dcol, ecol) if c not in df.columns]
        if missing_cols:
            return (
                [
                    _issue(
                        severity="FAIL",
                        message="Duration outcome column(s) missing in dataframe (unexpected; columns should be verified already).",
                        evidence={"missing_cols": missing_cols, "duration_column": dcol, "event_column": ecol},
                        fix_hint="Ensure you keep outcome_spec.duration_column and outcome_spec.event_column.",
                    )
                ],
                {"kind": "duration", "present": False},
            )

        sd = df[dcol]
        se = df[ecol]
        n = int(df.shape[0])

        vd = pd.to_numeric(sd, errors="coerce")
        n_nonmissing_d = int(sd.notna().sum())
        n_numeric_d = int(vd.notna().sum())
        n_bad_d = int(max(0, n_nonmissing_d - n_numeric_d))

        neg = int((vd.dropna() < 0).sum())
        nunq_d = int(vd.nunique(dropna=True))

        allowed_e = [ys.event_value, ys.censor_value]
        counts_e = _counts_by_allowed_literals(se, allowed_e)
        n_event = int(counts_e.get(ys.event_value, 0))
        n_cens = int(counts_e.get(ys.censor_value, 0))

        metrics: Dict[str, Any] = {
            "kind": "duration",
            "n_rows": n,
            "duration_column": dcol,
            "event_column": ecol,
            "duration_dtype": str(sd.dtype),
            "event_dtype": str(se.dtype),
            "n_numeric_duration": n_numeric_d,
            "n_non_numeric_duration_nonmissing": n_bad_d,
            "n_unique_duration": nunq_d,
            "n_negative_duration": neg,
            "event_counts": counts_e,
        }

        if n_numeric_d == 0:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Duration column has no numeric values after filtering.",
                    evidence=metrics,
                    fix_hint="Ensure duration is numeric/coercible to float.",
                )
            )
            return issues, metrics

        if neg > 0:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Duration column contains negative values.",
                    evidence=metrics,
                    fix_hint="Fix data cleaning; durations must be >= 0.",
                )
            )
            return issues, metrics

        if nunq_d <= 1:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Duration column has <=1 unique value; survival estimation may be degenerate.",
                    evidence=metrics,
                    fix_hint="Verify duration definition; choose a duration with variability.",
                )
            )

        if n_event == 0 or n_cens == 0:
            # Not always fatal (all-events or all-censored), but usually a modeling problem.
            issues.append(
                _issue(
                    severity="WARN",
                    message="Duration outcome has only one event class observed (all event or all censored).",
                    evidence={"n_event": n_event, "n_censor": n_cens, **metrics},
                    fix_hint="Verify event coding and cohort definition; many methods need both event and censoring.",
                )
            )
        else:
            share = float(n_event / max(1, (n_event + n_cens)))
            if share < float(imbalance_share_warn) or share > (1.0 - float(imbalance_share_warn)):
                issues.append(
                    _issue(
                        severity="WARN",
                        message="Duration event indicator is highly imbalanced; estimates may be unstable.",
                        evidence={"event_share": share, "imbalance_share_warn": float(imbalance_share_warn), **metrics},
                        fix_hint="Broaden cohort or verify event definition.",
                    )
                )

        if n_bad_d > 0:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Some non-numeric tokens exist in duration column (coercion failures).",
                    evidence=metrics,
                    fix_hint="Normalize duration values (remove units/suffixes) before modeling.",
                )
            )

        return issues, metrics

    # Non-duration outcomes
    ycol = cast(str, getattr(ys, "column"))
    if ycol not in df.columns:
        return (
            [
                _issue(
                    severity="FAIL",
                    message="Outcome column missing in dataframe (unexpected; columns should be verified already).",
                    evidence={"outcome_col": ycol},
                    fix_hint="Ensure you keep outcome_spec.column throughout filtering steps.",
                )
            ],
            {"outcome_col": ycol, "present": False},
        )

    s = df[ycol]
    assert isinstance(s, pd.Series)
    n = int(s.shape[0])
    metrics2: Dict[str, Any] = {"kind": getattr(ys, "kind", None), "outcome_col": ycol, "dtype": str(s.dtype), "n_rows": n}

    if n == 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="No rows available to validate outcome variation.",
                evidence=metrics2,
                fix_hint="Fix upstream filtering that removed all rows.",
            )
        )
        return issues, metrics2

    if isinstance(ys, BinaryOutcomeSpecModel):
        allowed = [ys.event, ys.non_event]
        counts = _counts_by_allowed_literals(s, allowed)
        metrics2["allowed"] = allowed
        metrics2["counts"] = counts

        n_e = int(counts.get(ys.event, 0))
        n_ne = int(counts.get(ys.non_event, 0))

        if n_e == 0 or n_ne == 0:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Binary outcome has no variation: one class is empty after filtering.",
                    evidence={"n_event": n_e, "n_non_event": n_ne, **metrics2},
                    fix_hint="Redefine outcome mapping or broaden cohort filtering.",
                )
            )
            return issues, metrics2

        share = float(n_e / max(1, (n_e + n_ne)))
        metrics2["event_share"] = share

        if min(n_e, n_ne) < int(min_count_warn):
            issues.append(
                _issue(
                    severity="WARN",
                    message="Binary outcome class has small count; estimates may be unstable.",
                    evidence={"min_class_count": min(n_e, n_ne), "min_count_warn": int(min_count_warn), **metrics2},
                    fix_hint="Broaden cohort or redefine outcome to increase class sizes.",
                )
            )

        if share < float(imbalance_share_warn) or share > (1.0 - float(imbalance_share_warn)):
            issues.append(
                _issue(
                    severity="WARN",
                    message="Binary outcome is highly imbalanced; estimates may be unstable.",
                    evidence={"event_share": share, "imbalance_share_warn": float(imbalance_share_warn), **metrics2},
                    fix_hint="Broaden cohort or reconsider outcome definition.",
                )
            )

        return issues, metrics2

    if isinstance(ys, CategoricalOutcomeSpecModel):
        allowed = list(ys.levels)
        counts = _counts_by_allowed_literals(s, allowed)
        present_levels = [k for k, v in counts.items() if int(v) > 0]

        metrics2["allowed"] = allowed
        metrics2["counts"] = counts
        metrics2["n_levels_present"] = len(present_levels)

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

        small = {k: int(v) for k, v in counts.items() if int(v) > 0 and int(v) < int(min_count_warn)}
        if small:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Some categorical outcome levels have small counts; estimates may be unstable.",
                    evidence={"small_levels": small, "min_count_warn": int(min_count_warn), **metrics2},
                    fix_hint="Merge rare levels or increase cohort size.",
                )
            )

        return issues, metrics2

    # Continuous
    if isinstance(ys, ContinuousOutcomeSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        v = pd.to_numeric(s, errors="coerce")
        n_nonmissing = int(s.notna().sum())
        n_numeric = int(v.notna().sum())
        n_bad = int(max(0, n_nonmissing - n_numeric))

        metrics2.update(
            {
                "n_nonmissing": n_nonmissing,
                "n_numeric": n_numeric,
                "n_non_numeric_nonmissing": n_bad,
                "numeric_parse_rate": float(n_numeric / max(1, n_nonmissing)),
                "n_unique_numeric": int(v.nunique(dropna=True)),
            }
        )

        if n_numeric == 0:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Continuous outcome has no numeric values after filtering.",
                    evidence=metrics2,
                    fix_hint="Ensure outcome column is numeric/coercible to float.",
                )
            )
            return issues, metrics2

        if int(v.nunique(dropna=True)) <= 1:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Continuous outcome has <=1 unique numeric value; estimates may be degenerate.",
                    evidence=metrics2,
                    fix_hint="Verify outcome definition; choose an outcome with variability.",
                )
            )

        if n_bad > 0:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Some non-numeric values exist in continuous outcome (coercion failures).",
                    evidence=metrics2,
                    fix_hint="Normalize outcome values (remove units/suffixes) before modeling.",
                )
            )

        return issues, metrics2

    # Unknown kind (should not happen)
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

def _allowed_treatment_literals(ts: Any) -> Optional[List[str]]:
    if isinstance(ts, BinaryTreatmentSpecModel):
        return [ts.treated, ts.control]
    if isinstance(ts, CategoricalTreatmentSpecModel):
        return list(ts.levels)
    if isinstance(ts, ContinuousTreatmentSpecModel):
        return None
    return None


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


def _parse_bool_token(raw: str) -> Optional[bool]:
    s = str(raw).strip().casefold()
    if s in BOOL_TRUE:
        return True
    if s in BOOL_FALSE:
        return False
    return None


def _safe_repr(x: Any) -> str:
    try:
        return repr(x)
    except Exception:
        return "<unrepr>"


# =============================================================================
# 5) Covariates / Effect Modifiers validations (pre-transform, raw df)
# =============================================================================
FeatureKind = Literal["NUMERIC", "BOOLEAN", "DATETIME", "CATEGORICAL", "STRING", "OTHER"]


class FeatureTopValue(TypedDict):
    value: str
    count: int


class FeatureProfile(TypedDict, total=False):
    name: str
    dtype: str
    kind: FeatureKind
    missing_rate: float
    n_unique: int
    unique_ratio: float
    is_constant: bool
    top_values: List[FeatureTopValue]
    python_types: List[str]
    notes: str


@dataclass(frozen=True)
class FeatureBlockProfile:
    label: Literal["W", "X", "WX"]
    n_rows: int
    cols: List[str]
    profiles: List[FeatureProfile]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "n_rows": self.n_rows,
            "cols": list(self.cols),
            "profiles": list(self.profiles),
        }


def validate_WX_presence(
    df: pd.DataFrame,
    key_cols: KeyColumns,
    *,
    require_W: bool,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    issues: List[ValidationIssue] = []

    W_cols = list(key_cols.W_cols)
    X_cols = list(key_cols.X_cols)

    missing_W = [c for c in W_cols if c not in df.columns]
    missing_X = [c for c in X_cols if c not in df.columns]

    nW = len(W_cols) - len(missing_W)
    nX = len(X_cols) - len(missing_X)

    metrics: Dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_W": int(nW),
        "n_X": int(nX),
        "n_W_requested": int(len(W_cols)),
        "n_X_requested": int(len(X_cols)),
        "missing_W_cols": missing_W,
        "missing_X_cols": missing_X,
    }

    if missing_W or missing_X:
        issues.append(
            _issue(
                severity="FAIL",
                message="Some W/X columns referenced by protocol are missing from the dataframe.",
                evidence=metrics,
                fix_hint="Ensure earlier column-drop step keeps all T/Y/W/X/time_zero columns required by the protocol.",
            )
        )
        # Missing columns is structural; no point emitting other presence warnings here.
        return issues, metrics

    if require_W and nW == 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="No covariates (W) available for adjustment.",
                evidence=metrics,
                fix_hint="Add confounders/covariates in the protocol (W), or relax prior filtering that removed them.",
            )
        )
    elif nW == 0:
        issues.append(
            _issue(
                severity="WARN",
                message="No covariates (W) available; estimates will be unadjusted / likely biased.",
                evidence=metrics,
                fix_hint="Add confounders/covariates in the protocol (W).",
            )
        )

    if nX == 0:
        issues.append(
            _issue(
                severity="WARN",
                message="No effect modifiers (X) available; heterogeneity (CATE) analysis will be limited.",
                evidence=metrics,
                fix_hint="If you want heterogeneous effects, add effect modifiers (X). Otherwise ignore.",
            )
        )

    if (nW + nX) == 0:
        issues.append(
            _issue(
                severity="FAIL" if require_W else "WARN",
                message="No W/X features available after filtering; modeling is likely degenerate.",
                evidence=metrics,
                fix_hint="Verify protocol feature lists and upstream filtering/null purge steps.",
            )
        )

    return issues, metrics


def profile_feature_block(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    label: Literal["W", "X", "WX"],
    top_k: int = 20,
    sample_n: int = 5000,
) -> FeatureBlockProfile:
    cols2 = [c for c in cols if c and c.strip() and c in df.columns]
    n_rows = int(df.shape[0])

    # deterministic sampling indices for expensive string profiling
    if n_rows > int(sample_n) and int(sample_n) > 0:
        sampled_idx = df.sample(n=int(sample_n), random_state=0).index
        df_s = df.loc[sampled_idx, cols2]
    else:
        df_s = df.loc[:, cols2]

    profiles: List[FeatureProfile] = []

    for c in cols2:
        s_full = df[c]
        s = df_s[c] if c in df_s.columns else s_full

        dtype_str = str(s_full.dtype)
        miss_rate = float(s_full.isna().mean()) if n_rows > 0 else 0.0
        nunq = int(s_full.nunique(dropna=True))
        uniq_ratio = float(nunq / max(1, int(s_full.notna().sum())))

        kind: FeatureKind = "OTHER"
        if ptypes.is_bool_dtype(s_full.dtype):
            kind = "BOOLEAN"
        elif ptypes.is_numeric_dtype(s_full.dtype):
            kind = "NUMERIC"
        elif ptypes.is_datetime64_any_dtype(s_full.dtype):
            kind = "DATETIME"
        elif isinstance(s_full.dtype, pd.CategoricalDtype):
            kind = "CATEGORICAL"
        elif ptypes.is_string_dtype(s_full.dtype) or ptypes.is_object_dtype(s_full.dtype):
            # distinguish "STRING" vs "CATEGORICAL-ish"
            kind = "STRING"
        else:
            kind = "OTHER"

        is_constant = nunq <= 1

        # Mixed python type detection (object columns)
        py_types: List[str] = []
        if ptypes.is_object_dtype(s_full.dtype):
            ss = s.dropna()
            if not ss.empty:
                # bounded scan
                if int(ss.shape[0]) > 200:
                    ss = ss.sample(n=200, random_state=0)
                type_set = {type(x).__name__ for x in ss.tolist()}
                py_types = sorted(type_set)

        # top values for non-numeric (bounded)
        top_vals: List[FeatureTopValue] = []
        if kind in ("CATEGORICAL", "STRING", "OTHER", "BOOLEAN", "DATETIME"):
            ss2 = s.dropna()
            if not ss2.empty:
                vc = ss2.astype("string").value_counts(dropna=True)
                head = vc.head(int(top_k))
                top_vals = [{"value": str(k), "count": int(v)} for k, v in head.items()]

        notes: List[str] = []
        if kind in ("STRING", "CATEGORICAL") and uniq_ratio >= 0.50 and nunq >= 50:
            notes.append("high_unique_ratio_stringish")
        if py_types and len(py_types) > 1:
            notes.append("mixed_python_types_object")

        prof: FeatureProfile = {
            "name": c,
            "dtype": dtype_str,
            "kind": kind,
            "missing_rate": float(miss_rate),
            "n_unique": int(nunq),
            "unique_ratio": float(uniq_ratio),
            "is_constant": bool(is_constant),
            "top_values": top_vals,
        }
        if py_types:
            prof["python_types"] = py_types
        if notes:
            prof["notes"] = ",".join(notes)

        profiles.append(prof)

    return FeatureBlockProfile(label=label, n_rows=n_rows, cols=list(cols2), profiles=profiles)


def validate_feature_missingness(
    profile: FeatureBlockProfile,
    *,
    threshold_warn: float = 0.05,
    threshold_fail: float = 0.30,
    ignore_cols: Sequence[str] = (),
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    issues: List[ValidationIssue] = []
    ignore = {c for c in ignore_cols}

    warn_off: List[Dict[str, Any]] = []
    fail_off: List[Dict[str, Any]] = []

    for p in profile.profiles:
        c = p.get("name")
        if not isinstance(c, str) or c in ignore:
            continue
        mr = float(p.get("missing_rate", 0.0))
        if mr >= float(threshold_fail):
            fail_off.append({"col": c, "missing_rate": mr, "dtype": p.get("dtype"), "kind": p.get("kind")})
        elif mr >= float(threshold_warn):
            warn_off.append({"col": c, "missing_rate": mr, "dtype": p.get("dtype"), "kind": p.get("kind")})

    metrics: Dict[str, Any] = {
        "label": profile.label,
        "n_cols": len(profile.cols),
        "threshold_warn": float(threshold_warn),
        "threshold_fail": float(threshold_fail),
        "n_warn": len(warn_off),
        "n_fail": len(fail_off),
    }

    if fail_off:
        issues.append(
            _issue(
                severity="FAIL",
                message=f"{profile.label}: some features have high missingness.",
                evidence={"offenders": fail_off[:50], **metrics},
                fix_hint="Ensure upstream null purge covered these columns, or add targeted imputation in the transform step.",
            )
        )
    if warn_off:
        issues.append(
            _issue(
                severity="WARN",
                message=f"{profile.label}: some features have non-trivial missingness.",
                evidence={"offenders": warn_off[:50], **metrics},
                fix_hint="Consider imputation or dropping these columns during the transform step.",
            )
        )

    return issues, metrics


def validate_feature_constantness(
    profile: FeatureBlockProfile,
    *,
    frac_warn: float = 0.30,
    frac_fail: float = 0.70,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    issues: List[ValidationIssue] = []

    const_cols = [p.get("name") for p in profile.profiles if bool(p.get("is_constant")) and isinstance(p.get("name"), str)]
    n_feat = max(1, len(profile.cols))
    frac = float(len(const_cols) / n_feat)

    metrics: Dict[str, Any] = {
        "label": profile.label,
        "n_cols": int(len(profile.cols)),
        "n_constant": int(len(const_cols)),
        "constant_frac": float(frac),
        "frac_warn": float(frac_warn),
        "frac_fail": float(frac_fail),
    }

    if frac >= float(frac_fail):
        issues.append(
            _issue(
                severity="FAIL",
                message=f"{profile.label}: too many constant columns (likely over-filtering or bad feature selection).",
                evidence={"constant_cols": const_cols[:100], **metrics},
                fix_hint="Drop constant columns; verify protocol W/X selection and upstream filtering did not collapse variation.",
            )
        )
    elif frac >= float(frac_warn) and const_cols:
        issues.append(
            _issue(
                severity="WARN",
                message=f"{profile.label}: many constant columns (adds no signal).",
                evidence={"constant_cols": const_cols[:100], **metrics},
                fix_hint="Drop constant columns during transform; verify feature selection.",
            )
        )

    return issues, metrics


def validate_feature_cardinality(
    profile: FeatureBlockProfile,
    *,
    max_levels_warn: int = 50,
    max_levels_fail: int = 200,
    warn_unique_ratio: float = 0.50,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    issues: List[ValidationIssue] = []

    hi_warn: List[Dict[str, Any]] = []
    hi_fail: List[Dict[str, Any]] = []
    textish: List[Dict[str, Any]] = []

    for p in profile.profiles:
        kind = p.get("kind")
        c = p.get("name")
        if not isinstance(c, str):
            continue

        nunq = int(p.get("n_unique", 0))
        ur = float(p.get("unique_ratio", 0.0))

        # High cardinality categoricals/strings explode later in one-hot.
        if kind in ("CATEGORICAL", "STRING", "OTHER"):
            if nunq >= int(max_levels_fail):
                hi_fail.append({"col": c, "n_unique": nunq, "dtype": p.get("dtype"), "kind": kind})
            elif nunq >= int(max_levels_warn):
                hi_warn.append({"col": c, "n_unique": nunq, "dtype": p.get("dtype"), "kind": kind})

        # “ID/text-ish” heuristic: very high unique ratio in string/object.
        if kind in ("STRING", "OTHER") and ur >= float(warn_unique_ratio) and nunq >= int(max_levels_warn):
            textish.append({"col": c, "unique_ratio": ur, "n_unique": nunq, "dtype": p.get("dtype"), "kind": kind})

    metrics: Dict[str, Any] = {
        "label": profile.label,
        "n_cols": int(len(profile.cols)),
        "max_levels_warn": int(max_levels_warn),
        "max_levels_fail": int(max_levels_fail),
        "warn_unique_ratio": float(warn_unique_ratio),
        "n_hi_warn": int(len(hi_warn)),
        "n_hi_fail": int(len(hi_fail)),
        "n_textish": int(len(textish)),
    }

    if hi_fail:
        issues.append(
            _issue(
                severity="FAIL",
                message=f"{profile.label}: some categorical/string columns have extreme cardinality (one-hot will explode).",
                evidence={"offenders": hi_fail[:50], **metrics},
                fix_hint="Drop/aggregate rare levels, bucketize, or exclude ID-like columns before transform.",
            )
        )
    if hi_warn:
        issues.append(
            _issue(
                severity="WARN",
                message=f"{profile.label}: some columns have high cardinality; transform may create very wide matrices.",
                evidence={"offenders": hi_warn[:50], **metrics},
                fix_hint="Consider bucketing, hashing, target encoding (careful), or dropping these columns.",
            )
        )
    if textish:
        issues.append(
            _issue(
                severity="WARN",
                message=f"{profile.label}: some string columns look like IDs/free-text (high unique ratio).",
                evidence={"offenders": textish[:50], **metrics},
                fix_hint="Drop ID/free-text columns from W/X or implement explicit text feature engineering (not default EconML prep).",
            )
        )

    return issues, metrics


def validate_feature_type_risks(profile: FeatureBlockProfile) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    issues: List[ValidationIssue] = []

    dt_cols = [p.get("name") for p in profile.profiles if p.get("kind") == "DATETIME" and isinstance(p.get("name"), str)]
    mixed_cols: List[str] = []
    for p in profile.profiles:
        if p.get("notes") and "mixed_python_types_object" in str(p.get("notes")) and isinstance(p.get("name"), str):
            mixed_cols.append(p.get("name"))

    metrics: Dict[str, Any] = {
        "label": profile.label,
        "n_cols": int(len(profile.cols)),
        "n_datetime": int(len(dt_cols)),
        "n_mixed_object": int(len(mixed_cols)),
    }

    if dt_cols:
        issues.append(
            _issue(
                severity="WARN",
                message=f"{profile.label}: datetime features detected; you need an explicit transform strategy (e.g., offsets, components).",
                evidence={"datetime_cols": dt_cols[:50], **metrics},
                fix_hint="Convert datetimes to numeric features (e.g., days since time_zero) during the transform step.",
            )
        )

    if mixed_cols:
        issues.append(
            _issue(
                severity="WARN",
                message=f"{profile.label}: object columns with mixed python types detected; transforms may be unstable.",
                evidence={"mixed_type_cols": mixed_cols[:50], **metrics},
                fix_hint="Normalize these columns upstream (cast to string or numeric) before transform.",
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
    masks: Dict[str, pd.Series]          # arm_name -> boolean mask
    counts: Dict[str, int]              # arm_name -> row count
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "treatment_col": self.treatment_col,
            "counts": dict(self.counts),
            "arms": list(self.masks.keys()),
            "notes": self.notes,
        }


def compute_arm_masks(
    df: pd.DataFrame,
    protocol: ProtocolSpec,
    key_cols: KeyColumns,
    *,
    max_bins_continuous: int = 5,
) -> ArmMasks:
    tcol = key_cols.treatment_col
    if tcol not in df.columns:
        # upstream validation should catch this; keep deterministic failure here
        raise KeyError(f"compute_arm_masks: treatment_col not found in df: {tcol!r}")

    ts = protocol.treatment_spec
    s = df[tcol]

    if isinstance(ts, BinaryTreatmentSpecModel):
        treated = ts.treated
        control = ts.control

        m_t = _mask_equals_literal(s, treated)
        m_c = _mask_equals_literal(s, control)

        masks = {"treated": m_t, "control": m_c}
        counts = {k: int(v.sum()) for k, v in masks.items()}
        return ArmMasks(kind="binary", treatment_col=tcol, masks=masks, counts=counts)

    if isinstance(ts, CategoricalTreatmentSpecModel):
        levels = list(ts.levels)
        masks2: Dict[str, pd.Series] = {}
        for lvl in levels:
            masks2[str(lvl)] = _mask_equals_literal(s, str(lvl))
        counts2 = {k: int(v.sum()) for k, v in masks2.items()}
        return ArmMasks(kind="categorical", treatment_col=tcol, masks=masks2, counts=counts2)

    if isinstance(ts, ContinuousTreatmentSpecModel):
        # quantile binning for diagnostics only
        sn = pd.to_numeric(s, errors="coerce")
        sn = sn.dropna()
        if sn.empty:
            # downstream validators should fail on missing/invalid treatment anyway
            masks3 = {"all": pd.Series([True] * len(df), index=df.index)}
            return ArmMasks(kind="continuous", treatment_col=tcol, masks=masks3, counts={"all": int(len(df))}, notes="treatment_non_numeric_or_all_missing")

        q = int(max(2, min(int(max_bins_continuous), 10)))
        try:
            bins = pd.qcut(pd.to_numeric(df[tcol], errors="coerce"), q=q, duplicates="drop")
            # bins is Categorical with interval labels (may have <q levels after drop)
            masks3 = {}
            for cat in bins.cat.categories:
                name = f"bin:{cat.left:g}..{cat.right:g}"
                masks3[name] = bins.eq(cat).fillna(False)
            if not masks3:
                masks3 = {"all": pd.Series([True] * len(df), index=df.index)}
                return ArmMasks(kind="continuous", treatment_col=tcol, masks=masks3, counts={"all": int(len(df))}, notes="qcut_failed_empty_bins")
            counts3 = {k: int(v.sum()) for k, v in masks3.items()}
            return ArmMasks(kind="continuous", treatment_col=tcol, masks=masks3, counts=counts3)
        except Exception:
            masks3 = {"all": pd.Series([True] * len(df), index=df.index)}
            return ArmMasks(kind="continuous", treatment_col=tcol, masks=masks3, counts={"all": int(len(df))}, notes="qcut_exception_fallback_all")

    raise ValueError(f"compute_arm_masks: unknown treatment_spec kind={getattr(ts, 'kind', None)!r}")


def overlap_support_check(
    df: pd.DataFrame,
    *,
    feat_cols: Sequence[str],
    arm_masks: ArmMasks,
    min_support_per_arm: int = 25,
    max_levels_checked: int = 50,
    max_cols: int = 300,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    """
    Flags df-backed exclusivity / support problems that imply overlap/positivity risk.

    - Categorical-like: levels present in one arm but absent in another.
    - Numeric-like: non-zero support present only in one arm, or variation present only in one arm.
    """
    issues: List[ValidationIssue] = []

    cols = [c for c in feat_cols if c in df.columns][: int(max_cols)]
    arms = list(arm_masks.masks.keys())

    exclusive_features: List[Dict[str, Any]] = []
    checked = 0

    for c in cols:
        checked += 1
        s = df[c]

        # Skip datetimes here (you already warn elsewhere); treat them as risky but not a support check target.
        if ptypes.is_datetime64_any_dtype(s.dtype):
            continue

        # Per-arm series slices
        per_arm = {a: s.loc[arm_masks.masks[a]] for a in arms}

        # Numeric / bool
        if ptypes.is_numeric_dtype(s.dtype) or ptypes.is_bool_dtype(s.dtype):
            # non-missing support
            support = {a: int(per_arm[a].notna().sum()) for a in arms}
            # non-zero support (useful for sparse indicators / one-hot-ish numeric)
            xnum = pd.to_numeric(s, errors="coerce").fillna(0.0)
            per_arm_num = {a: xnum.loc[arm_masks.masks[a]] for a in arms}
            nz = {a: int((per_arm_num[a] != 0).sum()) for a in arms}

            # variation per arm
            nunq = {a: int(pd.to_numeric(per_arm[a], errors="coerce").nunique(dropna=True)) for a in arms}

            # Exclusivity rules
            # 1) non-zero present in one arm (>=min_support) and zero in another
            nz_vals = list(nz.values())
            if nz_vals and (max(nz_vals) >= int(min_support_per_arm)) and any(v == 0 for v in nz_vals):
                exclusive_features.append(
                    {
                        "col": c,
                        "kind": "numeric_nonzero_exclusive",
                        "nonzero_by_arm": nz,
                        "support_by_arm": support,
                    }
                )
                continue

            # 2) variation present in one arm but constant in another (with enough support)
            nunq_vals = list(nunq.values())
            if max(nunq_vals) >= 2 and any(v <= 1 for v in nunq_vals):
                exclusive_features.append(
                    {
                        "col": c,
                        "kind": "numeric_variation_imbalanced",
                        "nunique_by_arm": nunq,
                        "support_by_arm": support,
                    }
                )
                continue

            continue

        # Categorical / string / object: check level support overlap
        ss = s.astype("string")
        ss_nn = ss.dropna()
        if ss_nn.empty:
            continue

        # Determine which levels to check (bounded)
        vc = ss_nn.value_counts(dropna=True)
        levels = [str(k) for k in vc.head(int(max_levels_checked)).index.tolist()]

        # counts per arm per level
        per_arm_norm = {a: per_arm[a].astype("string") for a in arms}
        for lvl in levels:
            counts = {a: int((per_arm_norm[a] == lvl).sum()) for a in arms}
            mx = max(counts.values()) if counts else 0
            if mx >= int(min_support_per_arm) and any(v == 0 for v in counts.values()):
                exclusive_features.append(
                    {
                        "col": c,
                        "kind": "categorical_level_exclusive",
                        "level": lvl,
                        "counts_by_arm": counts,
                        "levels_checked": int(len(levels)),
                    }
                )
                break  # one strong exclusive level is enough to flag the feature

    n_feats = max(1, checked)
    n_ex = len(exclusive_features)
    frac = float(n_ex / n_feats)

    metrics: Dict[str, Any] = {
        "arm_kind": arm_masks.kind,
        "arms": arms,
        "n_cols_checked": int(checked),
        "min_support_per_arm": int(min_support_per_arm),
        "max_levels_checked": int(max_levels_checked),
        "n_exclusive_flags": int(n_ex),
        "exclusive_frac": float(frac),
    }

    if n_ex == 0:
        return issues, metrics

    # Severity thresholds (match your earlier philosophy)
    severity: ValidationSeverity = "WARN"
    if frac >= 0.60 and n_ex >= 10:
        severity = "FAIL"
    elif frac >= 0.30 or n_ex >= 10:
        severity = "WARN"

    issues.append(
        _issue(
            severity=severity,
            message="Overlap/positivity risk: some features or categories appear only in certain treatment arms.",
            evidence={"examples": exclusive_features[:50], **metrics},
            fix_hint="Consider redefining cohort/treatment, trimming, collapsing rare levels, or dropping post-treatment / arm-exclusive variables.",
        )
    )
    return issues, metrics


def overlap_summary_univariate(
    df: pd.DataFrame,
    *,
    feat_cols: Sequence[str],
    arm_masks: ArmMasks,
    max_cols: int = 200,
    top_k_cats: int = 10,
) -> Dict[str, Any]:
    """
    Interpretable univariate overlap summary per feature, per arm.

    Returns a JSON-friendly dict report (no issues; consumers may convert into WARN messages).
    """
    cols = [c for c in feat_cols if isinstance(c, str) and c in df.columns][: int(max_cols)]
    arms = list(arm_masks.masks.keys())

    features: List[Dict[str, Any]] = []

    for c in cols:
        s = df[c]
        kind: str = "other"
        if ptypes.is_bool_dtype(s.dtype):
            kind = "boolean"
        elif ptypes.is_numeric_dtype(s.dtype):
            kind = "numeric"
        elif ptypes.is_datetime64_any_dtype(s.dtype):
            kind = "datetime"
        else:
            kind = "categorical"

        per_arm_stats: Dict[str, Any] = {}

        if kind in ("numeric", "boolean"):
            x = pd.to_numeric(s, errors="coerce")
            for a in arms:
                xa = x.loc[arm_masks.masks[a]]
                xa2 = xa.dropna()
                if xa2.empty:
                    per_arm_stats[a] = {"n": int(xa.shape[0]), "n_nonnull": 0}
                    continue
                arr = xa2.to_numpy(dtype="float64", copy=False)
                per_arm_stats[a] = {
                    "n": int(xa.shape[0]),
                    "n_nonnull": int(xa2.shape[0]),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                    "p10": float(np.quantile(arr, 0.10)),
                    "p50": float(np.quantile(arr, 0.50)),
                    "p90": float(np.quantile(arr, 0.90)),
                    "missing_rate": float(xa.isna().mean()) if int(xa.shape[0]) > 0 else 0.0,
                }

            # If exactly two arms, compute SMD (standardized mean diff)
            if len(arms) == 2 and arms[0] in per_arm_stats and arms[1] in per_arm_stats:
                a0, a1 = arms[0], arms[1]
                m0 = float(per_arm_stats[a0].get("mean", 0.0))
                m1 = float(per_arm_stats[a1].get("mean", 0.0))
                s0 = float(per_arm_stats[a0].get("std", 0.0))
                s1 = float(per_arm_stats[a1].get("std", 0.0))
                pooled = np.sqrt((s0 * s0 + s1 * s1) / 2.0)
                smd = float((m1 - m0) / pooled) if pooled > 0 else 0.0
                per_arm_stats["two_arm"] = {"smd": smd}

        else:
            # categorical-ish: compare top categories globally
            ss = s.astype("string")
            ss_nn = ss.dropna()
            if ss_nn.empty:
                per_arm_stats = {a: {"n": int(df.loc[arm_masks.masks[a]].shape[0]), "n_nonnull": 0} for a in arms}
            else:
                vc = ss_nn.value_counts(dropna=True)
                levels = [str(k) for k in vc.head(int(top_k_cats)).index.tolist()]

                support_sets: Dict[str, set[str]] = {}

                for a in arms:
                    sa = ss.loc[arm_masks.masks[a]]
                    sa_nn = sa.dropna()
                    support_sets[a] = set(map(str, sa_nn.unique().tolist()))
                    # frequencies for chosen levels
                    denom = max(1, int(sa_nn.shape[0]))
                    freqs = {}
                    for lvl in levels:
                        cnt = int((sa_nn == lvl).sum())
                        freqs[lvl] = {"count": cnt, "share": float(cnt / denom)}
                    per_arm_stats[a] = {
                        "n": int(sa.shape[0]),
                        "n_nonnull": int(sa_nn.shape[0]),
                        "missing_rate": float(sa.isna().mean()) if int(sa.shape[0]) > 0 else 0.0,
                        "top_levels": freqs,
                    }

                if len(arms) == 2:
                    a0, a1 = arms[0], arms[1]
                    s0, s1 = support_sets.get(a0, set()), support_sets.get(a1, set())
                    inter = len(s0.intersection(s1))
                    union = len(s0.union(s1))
                    jac = float(inter / union) if union > 0 else 0.0

                    # max abs diff among top levels
                    diffs: List[float] = []
                    for lvl in levels:
                        p0 = float(per_arm_stats[a0]["top_levels"][lvl]["share"])
                        p1 = float(per_arm_stats[a1]["top_levels"][lvl]["share"])
                        diffs.append(abs(p1 - p0))
                    per_arm_stats["two_arm"] = {"jaccard_support": jac, "max_abs_share_diff_topk": float(max(diffs) if diffs else 0.0)}

        features.append({"col": c, "dtype": str(s.dtype), "kind": kind, "by_arm": per_arm_stats})

    return {
        "arm_kind": arm_masks.kind,
        "treatment_col": arm_masks.treatment_col,
        "arms": arms,
        "n_features_reported": int(len(features)),
        "features": features,
    }


def validate_WX_overlap(key_cols: KeyColumns) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    W = list(key_cols.W_cols)
    X = list(key_cols.X_cols)

    overlap = sorted(set(W).intersection(set(X)))

    m: Dict[str, Any] = {
        "n_W": len(W),
        "n_X": len(X),
        "n_overlap": len(overlap),
        "overlap_cols": overlap,
    }

    if not overlap:
        return [], m

    issues: List[ValidationIssue] = [
        _issue(
            severity="WARN",
            message="W/X overlap detected: some columns appear in both covariates (W) and effect modifiers (X).",
            evidence=m,
            fix_hint=(
                "This is allowed but often redundant. Prefer disjoint sets: keep heterogeneity drivers in X and "
                "keep pure confounders/controls in W. If intentional, ensure downstream code dedupes W+X lists."
            ),
        )
    ]
    return issues, m

def overlap_propensity_proxy(
    df: pd.DataFrame,
    *,
    W_cols: Sequence[str],
    treatment_col: str,
    arm_masks: ArmMasks,
    max_features: int = 200,
    sample_n: int = 10000,
) -> Tuple[List[ValidationIssue], Dict[str, Any]]:
    """
    Optional pre-transform overlap proxy for *binary* treatment only.
    Uses a fast logistic regression on numeric-coercible W columns (no encoding).
    """
    issues: List[ValidationIssue] = []
    metrics: Dict[str, Any] = {
        "enabled": False,
        "reason": None,
        "auc": None,
        "extreme_prob_share": None,
        "n_rows_used": 0,
        "n_features_used": 0,
        "n_features_candidate": 0,
    }

    if arm_masks.kind != "binary":
        metrics["reason"] = "treatment_not_binary"
        return issues, metrics

    if treatment_col not in df.columns:
        metrics["reason"] = "treatment_col_missing"
        return issues, metrics

    # Soft dependency on sklearn (prod-safe: if missing, just warn and skip)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except Exception:
        issues.append(
            _issue(
                severity="WARN",
                message="Propensity proxy skipped: scikit-learn not available.",
                evidence={"missing_dependency": "scikit-learn"},
                fix_hint="Install scikit-learn or disable propensity proxy diagnostics.",
            )
        )
        metrics["reason"] = "sklearn_missing"
        return issues, metrics

    # Build binary target y using arm masks (robust to raw literal values)
    m_t = arm_masks.masks.get("treated")
    m_c = arm_masks.masks.get("control")
    if m_t is None or m_c is None:
        metrics["reason"] = "binary_arm_masks_missing"
        return issues, metrics

    idx_t = df.index[m_t]
    idx_c = df.index[m_c]
    if len(idx_t) == 0 or len(idx_c) == 0:
        metrics["reason"] = "empty_arm"
        return issues, metrics

    # Deterministic stratified sampling
    n_total = min(int(sample_n), int(len(idx_t) + len(idx_c)))
    n_half = max(1, n_total // 2)
    take_t = min(len(idx_t), n_half)
    take_c = min(len(idx_c), n_total - take_t)

    # If one arm is tiny, take all from it and fill from the other
    if take_c < 1:
        take_c = min(len(idx_c), n_half)
    if take_t + take_c < n_total:
        # top-up from the larger arm
        rem = n_total - (take_t + take_c)
        if len(idx_t) - take_t >= len(idx_c) - take_c:
            take_t = min(len(idx_t), take_t + rem)
        else:
            take_c = min(len(idx_c), take_c + rem)

    idx_t_s = pd.Index(idx_t).to_series().sample(n=take_t, random_state=0).to_numpy()
    idx_c_s = pd.Index(idx_c).to_series().sample(n=take_c, random_state=1).to_numpy()
    idx = pd.Index(np.concatenate([idx_t_s, idx_c_s]))

    y = np.concatenate([np.ones(len(idx_t_s), dtype=np.int32), np.zeros(len(idx_c_s), dtype=np.int32)])

    # Build numeric feature matrix from W (pre-transform)
    candidates = [c for c in W_cols if isinstance(c, str) and c in df.columns]
    metrics["n_features_candidate"] = int(len(candidates))

    X_cols_used: List[str] = []
    X_parts: List[np.ndarray] = []

    for c in candidates:
        if len(X_cols_used) >= int(max_features):
            break
        s = df.loc[idx, c]
        if ptypes.is_datetime64_any_dtype(s.dtype):
            continue

        if ptypes.is_bool_dtype(s.dtype):
            x = s.astype("boolean").fillna(False).astype("int8").to_numpy()
            X_parts.append(x.reshape(-1, 1))
            X_cols_used.append(c)
            continue

        if ptypes.is_numeric_dtype(s.dtype):
            sn = pd.to_numeric(s, errors="coerce")
            med = float(sn.median()) if sn.notna().any() else 0.0
            x = sn.fillna(med).astype("float64").to_numpy()
            if np.nanstd(x) <= 0:
                continue
            X_parts.append(x.reshape(-1, 1))
            X_cols_used.append(c)
            continue

        # object/string: try numeric coercion if mostly parseable
        sn2 = pd.to_numeric(s, errors="coerce")
        ok_rate = float(sn2.notna().mean()) if int(sn2.shape[0]) > 0 else 0.0
        if ok_rate >= 0.80:
            med2 = float(sn2.median()) if sn2.notna().any() else 0.0
            x2 = sn2.fillna(med2).astype("float64").to_numpy()
            if np.nanstd(x2) <= 0:
                continue
            X_parts.append(x2.reshape(-1, 1))
            X_cols_used.append(c)

    if not X_parts:
        metrics["reason"] = "no_numeric_features"
        issues.append(
            _issue(
                severity="WARN",
                message="Propensity proxy skipped: no numeric-coercible W features available pre-transform.",
                evidence={"n_candidates": len(candidates), "sample_n": int(len(idx))},
                fix_hint="Run this proxy after transform (encoding), or ensure W contains numeric features.",
            )
        )
        return issues, metrics

    X = np.concatenate(X_parts, axis=1)
    metrics["enabled"] = True
    metrics["n_rows_used"] = int(X.shape[0])
    metrics["n_features_used"] = int(X.shape[1])

    # Fit logistic regression proxy (fast, stable)
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
        extreme = float(((p < 0.01) | (p > 0.99)).mean())

        metrics["auc"] = auc
        metrics["extreme_prob_share"] = extreme

        # WARN-level heuristic gate
        if auc >= 0.90 and extreme >= 0.20:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Propensity proxy suggests poor overlap (strong separability and many extreme propensities).",
                    evidence={"auc": auc, "extreme_prob_share": extreme, "n_rows_used": int(X.shape[0]), "n_features_used": int(X.shape[1])},
                    fix_hint="Consider trimming, redefining cohort/treatment, adding overlap-driving covariates, or verifying post-treatment leakage.",
                )
            )

        return issues, metrics

    except Exception as e:
        issues.append(
            _issue(
                severity="WARN",
                message="Propensity proxy failed to run (non-fatal).",
                evidence={"error": repr(e), "n_rows_used": int(X.shape[0]), "n_features_used": int(X.shape[1])},
                fix_hint="Inspect W columns for numeric issues; consider running overlap diagnostics without propensity proxy.",
            )
        )
        metrics["reason"] = "fit_failed"
        return issues, metrics


# =============================================================================
# Internal helper: literal equality mask on raw series
# =============================================================================
def _mask_equals_literal(s: pd.Series, literal: str) -> pd.Series:
    """
    Compare series values to a protocol literal in a dtype-aware way.
    - numeric: float(literal)
    - bool: strict tokens only (true/false/1/0/yes/no)
    - datetime: pd.to_datetime(literal)
    - other: normalized string compare (strip+casefold)
    """
    lit = str(literal).strip()
    if not lit:
        return pd.Series([False] * len(s), index=s.index)

    if ptypes.is_bool_dtype(s.dtype):
        b = _parse_bool_token_strict(lit)
        if b is None:
            # strict: no match
            return pd.Series([False] * len(s), index=s.index)
        return s.astype("boolean").fillna(False).eq(bool(b))

    if ptypes.is_numeric_dtype(s.dtype):
        try:
            v = float(lit)
        except Exception:
            return pd.Series([False] * len(s), index=s.index)
        sn = pd.to_numeric(s, errors="coerce")
        return sn.eq(v)

    if ptypes.is_datetime64_any_dtype(s.dtype):
        ts = pd.to_datetime(lit, errors="coerce")
        if pd.isna(ts):
            return pd.Series([False] * len(s), index=s.index)
        return pd.to_datetime(s, errors="coerce").eq(ts)

    # default: string normalization
    ss = s.astype("string").str.strip().str.casefold()
    return ss.eq(lit.casefold())


def _parse_bool_token_strict(v: str) -> Optional[bool]:
    s = v.strip().casefold()
    if s in BOOL_TRUE:
        return True
    if s in BOOL_FALSE:
        return False
    return None
