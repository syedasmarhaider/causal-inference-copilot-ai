from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from python.implementation.workflows.tools.causal.causal_spec import (
    BinaryOutcomeSpecModel,
    BinaryTreatmentSpecModel,
    CausalSpec,
    ContinuousOutcomeSpecModel,
)
from python.implementation.workflows.utils.utils import BOOL_FALSE, BOOL_TRUE
from python.implementation.workflows.utils.validation import ValidationSeverity


# TODO: move to tools
class ValidationIssue(TypedDict):
    severity: ValidationSeverity
    message: str
    evidence: dict[str, Any]
    fix_hint: str | None

# =============================================================================
# 2) Structural invariants (protocol + df)
# =============================================================================

def validate_min_rows(
    df: pd.DataFrame,
    *,
    min_rows_fail: int = 20,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
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


def validate_protocol_role_columns_invariants(causal_spec: CausalSpec) -> list[ValidationIssue]:
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
    issues: list[ValidationIssue] = []

    # -------------------------
    # Treatment column
    # -------------------------
    treatment_col: str = causal_spec.treatment_spec.column

    # -------------------------
    # Outcome columns (duration has two)
    # -------------------------

    outcome_cols = [causal_spec.outcome_spec.column]

    # -------------------------
    # Covariates / effect modifiers
    # -------------------------
    covariates: list[str] = list(causal_spec.covariates or [])
    effect_modifiers: list[str] = list(causal_spec.effect_modifiers or [])


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
    forbidden: set[str] = {treatment_col, *outcome_cols}

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
                severity="FAIL",
                message="covariates and effect_modifiers overlap: some columns appear in both lists.",
                evidence={"overlap_cols": overlap, "n_overlap": len(overlap)},
                fix_hint="This is allowed is some estimation frameworks, but it's clearer to separate covariates (for adjustment) from effect modifiers (for heterogeneity). Consider assigning each column to one role. So we dont allow it",
            )
        )
        
    return issues

# =============================================================================
# 3) Treatment validations (pre-transform; whitelist already applied upstream)
# =============================================================================
def _treatment_allowed_literals(causal_spec: CausalSpec) -> list[Any]:
    """
    Returns the allowed literal domain for treatment_spec.
    - binary -> [treated, control]

    Note:
    Protocol values may come from DB as strings; downstream validation
    normalizes them with best-effort typed parsing.
    """
    ts = causal_spec.treatment_spec
    if isinstance(ts, BinaryTreatmentSpecModel):  # pyright: ignore[reportUnnecessaryIsInstance]
        return [ts.treated, ts.control]
    raise ValueError(f"Unknown treatment_spec type: {type(ts)}")


def validate_treatment(
    *,
    df: pd.DataFrame,
    causal_spec: CausalSpec,
    min_count_per_literal_fail: int = 15,
    min_share_fail: float = 0.05,
    max_ratio_fail: float = 20.0,
    min_neff_fail: float = 100.0,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """
    STRICT (FAIL-only) treatment validation.

    Discrete comparison semantics:
      - "0", 0, 0.0 -> same numeric literal
      - "true", True -> same boolean literal
      - booleans are NOT collapsed into 0/1
      - non-numeric / non-boolean strings remain strings
    """
    issues: list[ValidationIssue] = []

    # -------------------------
    # Step 0: Column presence
    # -------------------------
    tcol = causal_spec.treatment_spec.column
    if tcol not in df.columns:
        logging.warning(f"Treatment column '{tcol}' not found in dataframe columns: {df.columns.tolist()}")
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

    metrics: dict[str, Any] = {
        "treatment_col": tcol,
        "present": True,
        "treatment_kind": getattr(causal_spec.treatment_spec, "kind", None),
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
    # Step 2: Allowed literals (protocol truth, normalized best-effort)
    # -------------------------
    allowed_literals = _treatment_allowed_literals(causal_spec)
    allowed_unique, allowed_norm_keys, allowed_collisions = _build_allowed_literal_meta(allowed_literals)
    allowed_key_set = set(allowed_norm_keys)

    metrics["allowed_literals"] = list(allowed_literals)
    metrics["allowed_unique"] = list(allowed_unique)
    metrics["allowed_normalized"] = [
        {"literal": _safe_display(raw), "normalized_key": _discrete_key_text(key)}
        for raw, key in zip(allowed_unique, allowed_norm_keys, strict=False)
    ]

    if allowed_collisions:
        metrics["allowed_normalized_collisions"] = [
            {
                "normalized_key": _discrete_key_text(k),
                "raw_literals": [_safe_display(v) for v in vals],
            }
            for k, vals in allowed_collisions.items()
        ]
        issues.append(
            _issue(
                severity="FAIL",
                message='Protocol treatment literals collapse to the same semantic value (e.g. 0 and "0").',
                evidence=metrics,
                fix_hint="Keep only one semantic literal per treatment arm in the protocol.",
            )
        )
        return issues, metrics

    # -------------------------
    # Step 3: Observed tokens
    # -------------------------
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

    obs_counts, obs_examples = _normalized_value_counts(s_nonnull)
    obs_key_set = set(obs_counts.keys())

    counts_by_allowed: dict[str, int] = {
        str(_safe_display(raw)): int(obs_counts.get(key, 0))
        for raw, key in zip(allowed_unique, allowed_norm_keys, strict=False)
    }

    metrics.update(
        {
            "n_unique_observed": int(len(obs_key_set)),
            "counts_by_allowed": counts_by_allowed,
        }
    )

    # -------------------------
    # Step 4: Strict domain equality gates
    # -------------------------
    unexpected_keys = sorted(list(obs_key_set - allowed_key_set), key=_discrete_key_text)
    missing_allowed = [
        raw for raw, key in zip(allowed_unique, allowed_norm_keys, strict=False) if key not in obs_key_set
    ]

    metrics.update(
        {
            "n_unexpected": int(len(unexpected_keys)),
            "unexpected": [
                {
                    "normalized_key": _discrete_key_text(k),
                    "count": int(obs_counts.get(k, 0)),
                    "raw_examples": [_safe_display(v) for v in obs_examples.get(k, [])[:5]],
                }
                for k in unexpected_keys[:50]
            ],
            "n_missing_allowed": int(len(missing_allowed)),
            "missing_allowed": [_safe_display(v) for v in missing_allowed[:50]],
        }
    )

    if unexpected_keys:
        issues.append(
            _issue(
                severity="FAIL",
                message="Strict protocol violation: treatment contains values outside causal specs literals.",
                evidence=metrics,
                fix_hint="Map/filter upstream so treatment values match the causal specs literals semantically.",
            )
        )
        return issues, metrics

    if missing_allowed:
        issues.append(
            _issue(
                severity="FAIL",
                message="Strict protocol violation: not all causal specs treatment literals are present after filtering.",
                evidence=metrics,
                fix_hint="Relax filtering or fix mapping so every allowed literal appears at least once.",
            )
        )
        return issues, metrics

    # -------------------------
    # Step 5: Minimum count per literal gate
    # -------------------------
    low_counts: list[dict[str, Any]] = [
        {"literal": _safe_display(raw), "count": int(obs_counts.get(key, 0))}
        for raw, key in zip(allowed_unique, allowed_norm_keys, strict=False)
        if int(obs_counts.get(key, 0)) < int(min_count_per_literal_fail)
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
    # Step 6: Imbalance gates
    # -------------------------
    ordered_counts = [int(obs_counts.get(key, 0)) for key in allowed_norm_keys]
    total = int(sum(ordered_counts))

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

    shares_by_allowed = {
        str(_safe_display(raw)): float(obs_counts.get(key, 0) / total)
        for raw, key in zip(allowed_unique, allowed_norm_keys, strict=False)
    }
    min_share = float(min(shares_by_allowed.values()))
    min_count = int(min(ordered_counts))
    max_count = int(max(ordered_counts))
    ratio = float((max_count / min_count) if min_count > 0 else float("inf"))

    hhi = float(sum(p * p for p in shares_by_allowed.values()))
    entropy = float(-sum(p * math.log(p) for p in shares_by_allowed.values() if p > 0.0))

    metrics.update(
        {
            "total_in_allowed": total,
            "shares_by_allowed": shares_by_allowed,
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
                message="Treatment arm imbalance: minimum arm share is below threshold (positivity risk).",
                evidence=metrics,
                fix_hint="Relax filtering / broaden cohort / redefine treatment so each arm has sufficient support.",
            )
        )
        return issues, metrics

    if ratio > float(max_ratio_fail):
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment arm imbalance: max/min count ratio exceeds threshold.",
                evidence=metrics,
                fix_hint="Relax filtering or collapse levels; extreme imbalance makes estimates unstable.",
            )
        )
        return issues, metrics

    if len(allowed_norm_keys) == 2 and float(min_neff_fail) > 0.0:
        n0, n1 = ordered_counts[0], ordered_counts[1]
        neff = float((2.0 * n0 * n1) / (n0 + n1) if (n0 + n1) > 0 else 0.0)
        metrics["n_eff_harmonic_mean"] = neff

        if neff < float(min_neff_fail):
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
DiscreteKey = tuple[str, Any]


def _discrete_key_text(k: DiscreteKey) -> str:
    return f"{k[0]}:{k[1]!r}"


def _normalize_discrete_literal(v: Any) -> DiscreteKey:
    try:
        if pd.isna(v):
            return ("na", None)
    except Exception:
        pass

    if isinstance(v, (bool, np.bool_)):
        return ("bool", bool(v)) # pyright: ignore[reportUnknownArgumentType]

    if isinstance(v, (numbers.Integral, np.integer)) and not isinstance(v, (bool, np.bool_)):
        return ("num", float(v)) # pyright: ignore[reportUnknownArgumentType]

    if isinstance(v, (numbers.Real, np.floating)) and not isinstance(v, (bool, np.bool_)):
        fv = float(v) # pyright: ignore[reportUnknownArgumentType]
        if math.isfinite(fv):
            return ("num", fv)
        return ("str", str(v).strip().lower()) # pyright: ignore[reportUnknownArgumentType]

    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return ("str", "")

        s_lower = s.lower()

        if s_lower == "true":
            return ("bool", True)
        if s_lower == "false":
            return ("bool", False)

        try:
            fv = float(s_lower)
            if math.isfinite(fv):
                return ("num", fv)
        except Exception:
            pass

        return ("str", s_lower)

    return ("str", str(v).strip().lower())


def _build_allowed_literal_meta(
    allowed_literals: list[Any],
) -> tuple[list[Any], list[DiscreteKey], dict[DiscreteKey, list[Any]]]:
    allowed_unique: list[Any] = []
    allowed_norm_keys: list[DiscreteKey] = []
    collisions: dict[DiscreteKey, list[Any]] = {}

    seen: dict[DiscreteKey, Any] = {}
    for raw in allowed_literals:
        k = _normalize_discrete_literal(raw)
        if k not in seen:
            seen[k] = raw
            allowed_unique.append(raw)
            allowed_norm_keys.append(k)
        else:
            collisions.setdefault(k, [seen[k]])
            collisions[k].append(raw)

    return allowed_unique, allowed_norm_keys, collisions


def _normalized_value_counts(s: pd.Series) -> tuple[dict[DiscreteKey, int], dict[DiscreteKey, list[Any]]]:
    counts: dict[DiscreteKey, int] = {}
    examples: dict[DiscreteKey, list[Any]] = {}

    for v in s.tolist():
        k = _normalize_discrete_literal(v)
        counts[k] = counts.get(k, 0) + 1
        ex = examples.setdefault(k, [])
        if len(ex) < 5:
            ex.append(v)

    return counts, examples


def _strict_binary_outcome_literal_meta(
    outcome_spec: BinaryOutcomeSpecModel,
) -> tuple[list[Any], list[DiscreteKey], dict[DiscreteKey, list[Any]], DiscreteKey, DiscreteKey]:
    """
    STRICT:
      - requires explicit event / non_event
      - no heuristic inference
      - fails if they collapse semantically
    """
    allowed_literals = [outcome_spec.non_event, outcome_spec.event]
    allowed_unique, allowed_norm_keys, collisions = _build_allowed_literal_meta(allowed_literals)

    if len(allowed_unique) != 2:
        raise ValueError("Binary outcome must define exactly two distinct literals: event and non_event.")

    non_event_key = _normalize_discrete_literal(outcome_spec.non_event)
    event_key = _normalize_discrete_literal(outcome_spec.event)

    if non_event_key == event_key:
        raise ValueError("Binary outcome event and non_event collapse to the same semantic value.")

    return allowed_unique, allowed_norm_keys, collisions, non_event_key, event_key


def _strict_binary_treatment_literal_meta(
    treatment_spec: BinaryTreatmentSpecModel,
) -> tuple[list[Any], list[DiscreteKey], dict[DiscreteKey, list[Any]], DiscreteKey, DiscreteKey]:
    """
    STRICT:
      - requires explicit treated / control
      - no heuristic inference
      - fails if they collapse semantically
    """
    allowed_literals = [treatment_spec.control, treatment_spec.treated]
    allowed_unique, allowed_norm_keys, collisions = _build_allowed_literal_meta(allowed_literals)

    if len(allowed_unique) != 2:
        raise ValueError("Binary treatment must define exactly two distinct literals: treated and control.")

    control_key = _normalize_discrete_literal(treatment_spec.control)
    treated_key = _normalize_discrete_literal(treatment_spec.treated)

    if control_key == treated_key:
        raise ValueError("Binary treatment treated and control collapse to the same semantic value.")

    return allowed_unique, allowed_norm_keys, collisions, control_key, treated_key


def _missingness_by_expected_treatment_arm(
    t: pd.Series,
    y: pd.Series,
    control_key: DiscreteKey,
    treated_key: DiscreteKey,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns:
      - per-expected-arm stats for control/treated
      - unexpected treatment value stats
    """
    expected_rows: dict[DiscreteKey, dict[str, Any]] = {
        control_key: {
            "treatment_key": _discrete_key_text(control_key),
            "arm_role": "control",
            "n": 0,
            "n_y_missing": 0,
            "n_y_nonmissing": 0,
            "treatment_examples": [],
        },
        treated_key: {
            "treatment_key": _discrete_key_text(treated_key),
            "arm_role": "treated",
            "n": 0,
            "n_y_missing": 0,
            "n_y_nonmissing": 0,
            "treatment_examples": [],
        },
    }
    unexpected_rows: dict[DiscreteKey, dict[str, Any]] = {}

    for tv, yv in zip(t.tolist(), y.tolist(), strict=False):
        tkey = _normalize_discrete_literal(tv)

        if tkey in expected_rows:
            bucket = expected_rows[tkey]
        else:
            bucket = unexpected_rows.setdefault(
                tkey,
                {
                    "treatment_key": _discrete_key_text(tkey),
                    "n": 0,
                    "n_y_missing": 0,
                    "n_y_nonmissing": 0,
                    "treatment_examples": [],
                },
            )

        bucket["n"] += 1
        if len(bucket["treatment_examples"]) < 5:
            bucket["treatment_examples"].append(_safe_display(tv))

        try:
            if pd.isna(yv):
                bucket["n_y_missing"] += 1
            else:
                bucket["n_y_nonmissing"] += 1
        except Exception:
            bucket["n_y_nonmissing"] += 1

    expected_out: list[dict[str, Any]] = []
    for key in [control_key, treated_key]:
        d = expected_rows[key]
        n = int(d["n"])
        expected_out.append(
            {
                **d,
                "missing_rate_y": float(d["n_y_missing"] / max(1, n)),
            }
        )

    unexpected_out: list[dict[str, Any]] = []
    for _, d in sorted(unexpected_rows.items(), key=lambda kv: kv[1]["treatment_key"]):
        n = int(d["n"])
        unexpected_out.append(
            {
                **d,
                "missing_rate_y": float(d["n_y_missing"] / max(1, n)),
            }
        )

    return expected_out, unexpected_out


def _binary_event_stats_by_expected_treatment_arm(
    t: pd.Series,
    y: pd.Series,
    control_key: DiscreteKey,
    treated_key: DiscreteKey,
    event_key: DiscreteKey,
    non_event_key: DiscreteKey,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_rows: dict[DiscreteKey, dict[str, Any]] = {
        control_key: {
            "treatment_key": _discrete_key_text(control_key),
            "arm_role": "control",
            "n": 0,
            "n_y_missing": 0,
            "n_y_nonmissing": 0,
            "event_count": 0,
            "non_event_count": 0,
        },
        treated_key: {
            "treatment_key": _discrete_key_text(treated_key),
            "arm_role": "treated",
            "n": 0,
            "n_y_missing": 0,
            "n_y_nonmissing": 0,
            "event_count": 0,
            "non_event_count": 0,
        },
    }
    unexpected_rows: dict[DiscreteKey, dict[str, Any]] = {}

    for tv, yv in zip(t.tolist(), y.tolist(), strict=False):
        tkey = _normalize_discrete_literal(tv)

        if tkey in expected_rows:
            bucket = expected_rows[tkey]
        else:
            bucket = unexpected_rows.setdefault(
                tkey,
                {
                    "treatment_key": _discrete_key_text(tkey),
                    "n": 0,
                    "n_y_missing": 0,
                    "n_y_nonmissing": 0,
                    "event_count": 0,
                    "non_event_count": 0,
                },
            )

        bucket["n"] += 1

        try:
            if pd.isna(yv):
                bucket["n_y_missing"] += 1
                continue
        except Exception:
            pass

        bucket["n_y_nonmissing"] += 1
        ykey = _normalize_discrete_literal(yv)

        if ykey == event_key:
            bucket["event_count"] += 1
        elif ykey == non_event_key:
            bucket["non_event_count"] += 1

    expected_out: list[dict[str, Any]] = []
    for key in [control_key, treated_key]:
        d = expected_rows[key]
        n_nonmissing = int(d["n_y_nonmissing"])
        expected_out.append(
            {
                **d,
                "event_rate": float(d["event_count"] / max(1, n_nonmissing)),
            }
        )

    unexpected_out: list[dict[str, Any]] = []
    for _, d in sorted(unexpected_rows.items(), key=lambda kv: kv[1]["treatment_key"]):
        n_nonmissing = int(d["n_y_nonmissing"])
        unexpected_out.append(
            {
                **d,
                "event_rate": float(d["event_count"] / max(1, n_nonmissing)),
            }
        )

    return expected_out, unexpected_out


def _modifier_binary_support_one_at_a_time(
    *,
    df: pd.DataFrame,
    modifier_col: str,
    treatment_col: str,
    outcome_col: str,
    control_key: DiscreteKey,
    treated_key: DiscreteKey,
    event_key: DiscreteKey,
    min_rows_per_arm_per_level_warn: int,
    min_events_per_level_warn: int,
    max_levels_report: int = 50,
) -> dict[str, Any]:
    """
    Strict heterogeneity-support diagnostics for ONE modifier at a time.

    - Numeric modifiers: not assessed here (no auto-binning).
    - Non-numeric modifiers: levels are treated exactly as observed.
    """
    s_mod = df[modifier_col]
    s_t = df[treatment_col]
    s_y = df[outcome_col]

    n_missing_modifier = int(s_mod.isna().sum())

    if ptypes.is_numeric_dtype(s_mod):
        return {
            "modifier_col": modifier_col,
            "modifier_kind": "numeric",
            "n_missing_modifier": n_missing_modifier,
            "can_assess_support": False,
            "message": "Numeric effect modifier support not assessed in outcome validation without explicit discretization spec.",
        }

    level_rows: dict[DiscreteKey, dict[str, Any]] = {}

    for mv, tv, yv in zip(s_mod.tolist(), s_t.tolist(), s_y.tolist(), strict=False):
        try:
            if pd.isna(mv):
                continue
        except Exception:
            pass

        mkey = _normalize_discrete_literal(mv)
        tkey = _normalize_discrete_literal(tv)

        if tkey not in {control_key, treated_key}:
            continue

        bucket = level_rows.setdefault(
            mkey,
            {
                "modifier_level_key": _discrete_key_text(mkey),
                "modifier_level_examples": [],
                "n": 0,
                "n_y_nonmissing": 0,
                "event_count_total": 0,
                "arm_rows_nonmissing_y": {
                    _discrete_key_text(control_key): 0,
                    _discrete_key_text(treated_key): 0,
                },
                "arm_event_counts": {
                    _discrete_key_text(control_key): 0,
                    _discrete_key_text(treated_key): 0,
                },
            },
        )

        bucket["n"] += 1
        if len(bucket["modifier_level_examples"]) < 5:
            bucket["modifier_level_examples"].append(_safe_display(mv))

        try:
            if pd.isna(yv):
                continue
        except Exception:
            pass

        bucket["n_y_nonmissing"] += 1
        tkey_text = _discrete_key_text(tkey)
        bucket["arm_rows_nonmissing_y"][tkey_text] += 1

        ykey = _normalize_discrete_literal(yv)
        if ykey == event_key:
            bucket["event_count_total"] += 1
            bucket["arm_event_counts"][tkey_text] += 1

    levels_report: list[dict[str, Any]] = []
    unsupported_levels: list[dict[str, Any]] = []
    n_levels_supported = 0

    control_key_text = _discrete_key_text(control_key)
    treated_key_text = _discrete_key_text(treated_key)

    for _, level in sorted(level_rows.items(), key=lambda kv: kv[1]["modifier_level_key"])[:max_levels_report]:
        arm_rows = dict(level["arm_rows_nonmissing_y"])
        min_rows_any_arm = min(
            int(arm_rows.get(control_key_text, 0)),
            int(arm_rows.get(treated_key_text, 0)),
        )
        event_count_total = int(level["event_count_total"])
        has_both_expected_arms = (
            int(arm_rows.get(control_key_text, 0)) > 0
            and int(arm_rows.get(treated_key_text, 0)) > 0
        )

        supported = (
            has_both_expected_arms
            and min_rows_any_arm >= int(min_rows_per_arm_per_level_warn)
            and event_count_total >= int(min_events_per_level_warn)
        )

        level_out: dict[str, Any] = {
            **level,
            "has_both_expected_arms": has_both_expected_arms,
            "min_rows_any_expected_arm_nonmissing_y": int(min_rows_any_arm),
            "supported_for_simple_modifier_level_comparison": bool(supported),
        }
        levels_report.append(level_out)

        if supported:
            n_levels_supported += 1
        else:
            unsupported_levels.append(level_out)

    return {
        "modifier_col": modifier_col,
        "modifier_kind": "categorical",
        "n_missing_modifier": n_missing_modifier,
        "can_assess_support": True,
        "n_levels_observed": int(len(level_rows)),
        "n_levels_supported": int(n_levels_supported),
        "levels_report": levels_report,
        "unsupported_levels_sample": unsupported_levels[:20],
    }


def validate_outcome(
    *,
    df: pd.DataFrame,
    causal_spec: CausalSpec,
    allow_missing_outcome: bool = False,
    missing_rate_warn: float = 0.01,
    missing_rate_fail: float = 0.20,
    arm_missing_rate_diff_warn: float = 0.05,
    arm_missing_rate_diff_fail: float = 0.15,
    min_arm_n_for_missingness_gates: int = 50,
    min_unique_numeric_fail: int = 2,
    require_strict_numeric: bool = True,
    low_event_count_warn: int = 30,
    low_event_count_fail: int = 1,
    low_arm_event_count_warn: int = 10,
    require_both_expected_treatment_arms_with_nonmissing_y: bool = True,
    assess_effect_modifier_support: bool = True,
    min_rows_per_arm_per_modifier_level_warn: int = 10,
    min_events_per_modifier_level_warn: int = 5,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """
    Outcome validation ONLY.

    Responsibilities:
      - outcome column presence / parseability / literal-domain integrity
      - outcome missingness overall and by expected treatment arm
      - observed support for continuous or binary outcomes
      - binary event support overall and by expected treatment arm
      - optional one-modifier-at-a-time heterogeneity-support diagnostics

    Explicitly NOT responsible for:
      - overlap / positivity
      - confounding / no-adjustment warnings
      - propensity logic
      - broad treatment-design validation beyond minimum arm visibility needed here
    """
    issues: list[ValidationIssue] = []

    ys = causal_spec.outcome_spec
    ts = causal_spec.treatment_spec
    ycol = ys.column
    tcol = ts.column

    # ------------------------------------------------------------------
    # Step 1: presence + empty df
    # ------------------------------------------------------------------
    n_rows = int(df.shape[0])
    missing_cols = [c for c in [ycol, tcol] if c not in df.columns]
    if missing_cols:
        metrics = {
            "present": False,
            "required_cols": [ycol, tcol],
            "missing_cols": missing_cols[:50],
            "n_missing": int(len(missing_cols)),
            "n_rows": n_rows,
            "n_df_cols": int(df.shape[1]),
            "outcome_kind": getattr(ys, "kind", None),
        }
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome/treatment column referenced by protocol is missing from dataframe.",
                evidence=metrics,
                fix_hint="Ensure treatment/outcome columns are retained and names match the protocol exactly.",
            )
        )
        return issues, metrics

    if n_rows == 0:
        metrics = {
            "present": True,
            "n_rows": 0,
            "outcome_kind": getattr(ys, "kind", None),
            "outcome_col": ycol,
            "treatment_col": tcol,
        }
        issues.append(
            _issue(
                severity="FAIL",
                message="No rows available to validate outcome.",
                evidence=metrics,
                fix_hint="Fix upstream filtering that removed all rows.",
            )
        )
        return issues, metrics

    y = df[ycol]
    t = df[tcol]

    # ------------------------------------------------------------------
    # Step 2: strict treatment literal meta
    # ------------------------------------------------------------------
    if not isinstance(ts, BinaryTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        metrics = {
            "present": True,
            "treatment_col": tcol,
            "treatment_kind": getattr(ts, "kind", None),
        }
        issues.append(
            _issue(
                severity="FAIL",
                message="validate_outcome currently requires BinaryTreatmentSpecModel.",
                evidence=metrics,
                fix_hint="Use a binary treatment spec for this validator path.",
            )
        )
        return issues, metrics

    try:
        _, _, treatment_collisions, control_key, treated_key = _strict_binary_treatment_literal_meta(ts)
    except ValueError as e:
        metrics = {
            "present": True,
            "treatment_col": tcol,
            "treatment_kind": getattr(ts, "kind", None),
            "treated": _safe_display(getattr(ts, "treated", None)),
            "control": _safe_display(getattr(ts, "control", None)),
        }
        issues.append(
            _issue(
                severity="FAIL",
                message=str(e),
                evidence=metrics,
                fix_hint="Ensure treated/control are explicitly defined and semantically distinct.",
            )
        )
        return issues, metrics

    if treatment_collisions:
        metrics = {
            "present": True,
            "treatment_col": tcol,
            "treated": _safe_display(ts.treated),
            "control": _safe_display(ts.control),
            "treatment_normalized_collisions": [
                {
                    "normalized_key": _discrete_key_text(k),
                    "raw_literals": [_safe_display(v) for v in vals],
                }
                for k, vals in treatment_collisions.items()
            ],
        }
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment literals collapse to the same semantic value.",
                evidence=metrics,
                fix_hint="Use exactly two semantically distinct treatment literals.",
            )
        )
        return issues, metrics

    # ------------------------------------------------------------------
    # Step 3: outcome missingness overall and by expected treatment arm
    # ------------------------------------------------------------------
    n_y_missing = int(y.isna().sum())
    miss_rate = float(n_y_missing / max(1, n_rows))
    n_y_nonmissing = int(n_rows - n_y_missing)

    arm_stats, unexpected_treatment_stats = _missingness_by_expected_treatment_arm(
        t=t,
        y=y,
        control_key=control_key,
        treated_key=treated_key,
    )
    eligible_arm_rates = [
        a["missing_rate_y"]
        for a in arm_stats
        if int(a["n"]) >= int(min_arm_n_for_missingness_gates)
    ]
    arm_diff = float(max(eligible_arm_rates) - min(eligible_arm_rates)) if len(eligible_arm_rates) >= 2 else 0.0

    metrics: dict[str, Any] = {
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
        "expected_treatment_arms": [
            {
                "role": "control",
                "literal": _safe_display(ts.control),
                "normalized_key": _discrete_key_text(control_key),
            },
            {
                "role": "treated",
                "literal": _safe_display(ts.treated),
                "normalized_key": _discrete_key_text(treated_key),
            },
        ],
        "missingness_by_expected_treatment_arm": arm_stats,
        "unexpected_treatment_value_stats": unexpected_treatment_stats[:50],
        "n_unexpected_treatment_values": int(len(unexpected_treatment_stats)),
        "arm_missing_rate_diff": arm_diff,
    }

    if unexpected_treatment_stats:
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment column contains values outside the explicit treated/control protocol literals.",
                evidence=metrics,
                fix_hint="Map or filter treatment values so they match the binary treatment protocol exactly.",
            )
        )
        return issues, metrics

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

    if n_y_missing > 0:
        if miss_rate >= float(missing_rate_fail):
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Outcome missingness is too high; observed-only analysis is unreliable.",
                    evidence=metrics,
                    fix_hint="Improve outcome capture, redefine the outcome window, or handle missing outcomes explicitly.",
                )
            )
            return issues, metrics

        if miss_rate >= float(missing_rate_warn):
            issues.append(
                _issue(
                    severity="WARN",
                    message="Outcome has non-trivial missingness.",
                    evidence=metrics,
                    fix_hint="Report missingness clearly and inspect differences by expected treatment arm.",
                )
            )

        if arm_diff >= float(arm_missing_rate_diff_fail) and len(eligible_arm_rates) >= 2:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Outcome missingness differs substantially by expected treatment arm.",
                    evidence=metrics,
                    fix_hint="Do not rely on naive observed-only analysis; investigate differential outcome capture by arm.",
                )
            )
            return issues, metrics

        if arm_diff >= float(arm_missing_rate_diff_warn) and len(eligible_arm_rates) >= 2:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Outcome missingness differs by expected treatment arm.",
                    evidence=metrics,
                    fix_hint="At minimum, report arm-specific missingness and assess whether observed-only analysis is acceptable.",
                )
            )

    if n_y_nonmissing == 0:
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome is missing for all rows; cannot validate or estimate effects.",
                evidence=metrics,
                fix_hint="Fix outcome extraction or mapping.",
            )
        )
        return issues, metrics

    y_nm = y.dropna()

    # ------------------------------------------------------------------
    # Step 4A: continuous outcome
    # ------------------------------------------------------------------
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
                    message="Continuous outcome contains non-numeric tokens among non-missing entries.",
                    evidence={**metrics, "bad_value_sample": [_safe_display(x) for x in bad_sample]},
                    fix_hint="Clean the outcome column so all observed values are numeric.",
                )
            )
            return issues, metrics

        n_unique = int(v.nunique(dropna=True))
        metrics["n_unique_numeric_used"] = n_unique
        metrics["continuous_outcome_support"] = {
            "has_variation": bool(n_unique >= int(min_unique_numeric_fail)),
            "min_unique_numeric_fail": int(min_unique_numeric_fail),
        }

        if n_unique < int(min_unique_numeric_fail):
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Continuous outcome is degenerate (too few unique numeric values among observed outcomes).",
                    evidence=metrics,
                    fix_hint="Use an outcome with variability or adjust filters that collapsed variation.",
                )
            )
            return issues, metrics

        return issues, metrics

    # ------------------------------------------------------------------
    # Step 4B: binary outcome
    # ------------------------------------------------------------------
    if isinstance(ys, BinaryOutcomeSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        try:
            allowed_unique, allowed_norm_keys, allowed_collisions, non_event_key, event_key = _strict_binary_outcome_literal_meta(ys)
        except ValueError as e:
            issues.append(
                _issue(
                    severity="FAIL",
                    message=str(e),
                    evidence={
                        **metrics,
                        "dtype": str(y.dtype),
                        "event": _safe_display(getattr(ys, "event", None)),
                        "non_event": _safe_display(getattr(ys, "non_event", None)),
                    },
                    fix_hint="Define explicit event and non_event literals in the protocol and ensure they are semantically distinct.",
                )
            )
            return issues, metrics

        if allowed_collisions:
            metrics["outcome_normalized_collisions"] = [
                {
                    "normalized_key": _discrete_key_text(k),
                    "raw_literals": [_safe_display(v) for v in vals],
                }
                for k, vals in allowed_collisions.items()
            ]
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Binary outcome literals collapse to the same semantic value.",
                    evidence=metrics,
                    fix_hint="Use exactly two semantically distinct outcome literals: event and non_event.",
                )
            )
            return issues, metrics

        metrics.update(
            {
                "dtype": str(y.dtype),
                "allowed_outcome_literals": [_safe_display(x) for x in allowed_unique],
                "allowed_outcome_normalized": [
                    {"literal": _safe_display(raw), "normalized_key": _discrete_key_text(key)}
                    for raw, key in zip(allowed_unique, allowed_norm_keys, strict=False)
                ],
                "event_literal": _safe_display(ys.event),
                "non_event_literal": _safe_display(ys.non_event),
                "event_key": _discrete_key_text(event_key),
                "non_event_key": _discrete_key_text(non_event_key),
            }
        )

        obs_counts, obs_examples = _normalized_value_counts(y_nm)
        obs_key_set = set(obs_counts.keys())
        allowed_key_set = set(allowed_norm_keys)
        unexpected_keys = sorted(list(obs_key_set - allowed_key_set), key=_discrete_key_text)

        metrics.update(
            {
                "n_unique_observed_nonmissing": int(len(obs_key_set)),
                "unexpected_outcome_values": [
                    {
                        "normalized_key": _discrete_key_text(k),
                        "count": int(obs_counts.get(k, 0)),
                        "raw_examples": [_safe_display(v) for v in obs_examples.get(k, [])[:5]],
                    }
                    for k in unexpected_keys[:50]
                ],
                "n_unexpected_outcome_values": int(len(unexpected_keys)),
                "counts_by_observed_outcome": {
                    _discrete_key_text(k): int(obs_counts.get(k, 0))
                    for k in sorted(obs_key_set, key=_discrete_key_text)
                },
            }
        )

        if unexpected_keys:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Binary outcome contains values outside the explicit event/non_event protocol literals among non-missing entries.",
                    evidence=metrics,
                    fix_hint="Map or filter outcome values so observed non-missing entries match event/non_event exactly.",
                )
            )
            return issues, metrics

        if len(obs_key_set) < 2:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Binary outcome has fewer than 2 observed levels among non-missing entries.",
                    evidence=metrics,
                    fix_hint="Relax filters, increase cohort size, or redefine outcome so both event states appear.",
                )
            )
            return issues, metrics

        total_nonmissing = int(sum(obs_counts.values()))
        event_count_total = int(obs_counts.get(event_key, 0))
        non_event_count_total = int(obs_counts.get(non_event_key, 0))
        event_rate_total = float(event_count_total / max(1, total_nonmissing))

        arm_event_stats, unexpected_arm_event_stats = _binary_event_stats_by_expected_treatment_arm(
            t=t,
            y=y,
            control_key=control_key,
            treated_key=treated_key,
            event_key=event_key,
            non_event_key=non_event_key,
        )

        if unexpected_arm_event_stats:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Unexpected treatment values were encountered while computing binary outcome support.",
                    evidence={**metrics, "unexpected_arm_event_stats": unexpected_arm_event_stats[:50]},
                    fix_hint="Normalize or filter treatment values before outcome validation.",
                )
            )
            return issues, metrics

        arms_with_nonmissing_y = [a for a in arm_event_stats if int(a["n_y_nonmissing"]) > 0]

        metrics.update(
            {
                "total_nonmissing_in_domain": total_nonmissing,
                "event_count_total": event_count_total,
                "non_event_count_total": non_event_count_total,
                "event_rate_total": event_rate_total,
                "event_stats_by_expected_treatment_arm": arm_event_stats,
                "n_expected_treatment_arms_with_nonmissing_y": int(len(arms_with_nonmissing_y)),
                "low_event_count_warn": int(low_event_count_warn),
                "low_event_count_fail": int(low_event_count_fail),
                "low_arm_event_count_warn": int(low_arm_event_count_warn),
                "require_both_expected_treatment_arms_with_nonmissing_y": bool(require_both_expected_treatment_arms_with_nonmissing_y),
            }
        )

        if event_count_total < int(low_event_count_fail):
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Binary outcome has too few observed events overall.",
                    evidence=metrics,
                    fix_hint="Use a more common outcome, expand the cohort, or revise filtering.",
                )
            )
            return issues, metrics

        if non_event_count_total == 0:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Binary outcome has no observed non-events.",
                    evidence=metrics,
                    fix_hint="Use a cohort/outcome definition where both event states are observed.",
                )
            )
            return issues, metrics

        if require_both_expected_treatment_arms_with_nonmissing_y and len(arms_with_nonmissing_y) < 2:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Observed binary outcome data remain in fewer than both expected treatment arms.",
                    evidence=metrics,
                    fix_hint="Check filtering and missingness; outcome support must remain visible in both treated and control arms.",
                )
            )
            return issues, metrics

        if event_count_total < int(low_event_count_warn):
            issues.append(
                _issue(
                    severity="WARN",
                    message="Binary outcome has low event count overall; effect estimates may be imprecise.",
                    evidence=metrics,
                    fix_hint="Interpret overall effect estimates cautiously and avoid over-claiming precision.",
                )
            )

        low_event_arms: list[dict[str, Any]] = [
            {
                "arm_role": a["arm_role"],
                "treatment_key": a["treatment_key"],
                "event_count": int(a["event_count"]),
                "n_y_nonmissing": int(a["n_y_nonmissing"]),
            }
            for a in arms_with_nonmissing_y
            if int(a["event_count"]) < int(low_arm_event_count_warn)
        ]
        if low_event_arms:
            issues.append(
                _issue(
                    severity="WARN",
                    message="Some expected treatment arms have low event support for the binary outcome.",
                    evidence={**metrics, "low_event_arms": low_event_arms[:20]},
                    fix_hint="Arm-specific estimates may be unstable; interpret contrasts carefully.",
                )
            )

        # --------------------------------------------------------------
        # Step 5: optional effect-modifier support diagnostics
        # --------------------------------------------------------------
        if assess_effect_modifier_support and getattr(causal_spec, "effect_modifiers", None):
            modifier_support_reports: list[dict[str, Any]] = []
            missing_modifier_cols = [c for c in causal_spec.effect_modifiers if c not in df.columns]

            if missing_modifier_cols:
                issues.append(
                    _issue(
                        severity="FAIL",
                        message="Some effect modifier columns referenced by protocol are missing from dataframe.",
                        evidence={**metrics, "missing_effect_modifier_cols": missing_modifier_cols[:50]},
                        fix_hint="Retain these columns or remove them from the protocol.",
                    )
                )
                return issues, metrics

            for mod_col in causal_spec.effect_modifiers:
                report = _modifier_binary_support_one_at_a_time(
                    df=df,
                    modifier_col=mod_col,
                    treatment_col=tcol,
                    outcome_col=ycol,
                    control_key=control_key,
                    treated_key=treated_key,
                    event_key=event_key,
                    min_rows_per_arm_per_level_warn=min_rows_per_arm_per_modifier_level_warn,
                    min_events_per_level_warn=min_events_per_modifier_level_warn,
                )
                modifier_support_reports.append(report)

                if not report["can_assess_support"]:
                    issues.append(
                        _issue(
                            severity="WARN",
                            message=f'Outcome support for numeric effect modifier "{mod_col}" was not assessed in outcome validation.',
                            evidence={**metrics, "modifier_support_report": report},
                            fix_hint="Provide an explicit discretization rule elsewhere if heterogeneity support must be checked numerically.",
                        )
                    )
                    continue

                unsupported_levels = report.get("unsupported_levels_sample", [])
                if unsupported_levels:
                    issues.append(
                        _issue(
                            severity="WARN",
                            message=f'Binary outcome support is sparse in some levels of effect modifier "{mod_col}".',
                            evidence={**metrics, "modifier_support_report": report},
                            fix_hint="Treat heterogeneity claims for this modifier cautiously or simplify the modifier structure.",
                        )
                    )

            metrics["effect_modifier_support_reports"] = modifier_support_reports

        return issues, metrics

    # Defensive fallback
    issues.append(
        _issue(
            severity="FAIL",
            message="Unsupported outcome spec type encountered in validate_outcome.",
            evidence={**metrics, "outcome_spec_type": type(ys).__name__},
            fix_hint="Use BinaryOutcomeSpecModel or ContinuousOutcomeSpecModel.",
        )
    )
    return issues, metrics


# =============================================================================
# 5) Covariates / Effect Modifiers validations (pre-transform, raw df)
# =============================================================================
def validate_covariate_and_effect_modifier_presence(
    *,
    df: pd.DataFrame,
    causal_spec: CausalSpec,
    require_covariates: bool,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
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
    issues: list[ValidationIssue] = []

    # -------------------------
    # 0) Global schema sanity: duplicate df column labels are ambiguous in pandas
    # -------------------------
    if not df.columns.is_unique:
        dupes = df.columns[df.columns.duplicated()].tolist()
        counts: dict[str, int] = {}
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
    def _dedup_keep_order(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if  x.strip() and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    covariates = _dedup_keep_order(list(getattr(causal_spec, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(causal_spec, "effect_modifiers", []) or []))

    # -------------------------
    # 2) Existence checks
    # -------------------------
    missing_covariates = [c for c in covariates if c not in df.columns]
    missing_effect_modifiers = [c for c in effect_modifiers if c not in df.columns]

    n_covariates_present = int(len(covariates) - len(missing_covariates))
    n_effect_modifiers_present = int(len(effect_modifiers) - len(missing_effect_modifiers))

    overlap_cols = sorted(set(covariates).intersection(set(effect_modifiers)))

    metrics: dict[str, Any] = {
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
    causal_spec: CausalSpec,
    missing_rate_warn: float = 0.05,
    missing_rate_fail: float = 0.30,
    ignore_cols: Sequence[str] = (),
    max_cols: int = 500,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
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
    issues: list[ValidationIssue] = []
    ignore = {c for c in ignore_cols if c.strip()}

    def _dedup_keep_order(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x.strip() and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    covariates = _dedup_keep_order(list(getattr(causal_spec, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(causal_spec, "effect_modifiers", []) or []))

    # Combined list (stable): covariates first
    cols_all = _dedup_keep_order([c for c in (covariates + effect_modifiers) if c not in ignore])[: int(max_cols)]

    missing_cols = [c for c in cols_all if c not in df.columns]
    n_rows = int(df.shape[0])

    metrics: dict[str, Any] = {
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

    warn_off: list[dict[str, Any]] = []
    fail_off: list[dict[str, Any]] = []

    for c in cols_all:
        s = df[c]
        mr = float(s.isna().mean()) if n_rows > 0 else 0.0
        row: dict[str, Any] = {"col": c, "missing_rate": mr, "dtype": str(s.dtype)}
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
                fix_hint="Drop these columns or fix your data",
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
    causal_spec: CausalSpec,
    delta_warn: float = 0.05,
    delta_fail: float = 0.20,
    ignore_cols: Sequence[str] = (),
    max_cols: int = 300,
    min_arm_n: int = 25,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
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
    issues: list[ValidationIssue] = []
    ignore = {c for c in ignore_cols if c.strip()}

    ts = causal_spec.treatment_spec
    tcol = ts.column

    metrics: dict[str, Any] = {
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

    arm_masks: dict[str, pd.Series] = {}

    if isinstance(ts, BinaryTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        arm_masks["treated"] = _mask_equals_literal(sT, ts.treated)
        arm_masks["control"] = _mask_equals_literal(sT, ts.control)
            
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

    def _dedup_keep_order(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if  x not in seen:
                seen.add(x)
                out.append(x)
        return out

    covariates = _dedup_keep_order(list(getattr(causal_spec, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(causal_spec, "effect_modifiers", []) or []))
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

    offenders_warn: list[dict[str, Any]] = []
    offenders_fail: list[dict[str, Any]] = []

    for c in cols_all:
        s = df[c]
        per_arm: dict[str, float] = {}
        for a in eligible_arms:
            m = arm_masks[a]
            sa = s.loc[m]
            per_arm[a] = float(sa.isna().mean()) if int(sa.shape[0]) > 0 else 0.0

        gap = float(max(per_arm.values()) - min(per_arm.values())) if per_arm else 0.0

        row: dict[str, Any] = {
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
    causal_spec:  CausalSpec,
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
) -> tuple[list[ValidationIssue], dict[str, Any]]:
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
    issues: list[ValidationIssue] = []
    ignore = {c for c in ignore_cols if c.strip()}

    def _dedup_keep_order(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x.strip() and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    covariates = _dedup_keep_order(list(getattr(causal_spec, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(causal_spec, "effect_modifiers", []) or []))
    cols_all = _dedup_keep_order([c for c in (covariates + effect_modifiers) if c not in ignore])[: int(max_cols)]

    missing_cols = [c for c in cols_all if c not in df.columns]
    metrics: dict[str, Any] = {
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

    constant_like: list[dict[str, Any]] = []
    numeric_near_constant: list[dict[str, Any]] = []

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
    causal_spec: CausalSpec,
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
) -> tuple[list[ValidationIssue], dict[str, Any]]:
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
    issues: list[ValidationIssue] = []
    ignore = {c for c in ignore_cols if c.strip()}

    def _dedup_keep_order(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x.strip() and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    covariates = _dedup_keep_order(list(getattr(causal_spec, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(causal_spec, "effect_modifiers", []) or []))
    cols_all = _dedup_keep_order([c for c in (covariates + effect_modifiers) if c not in ignore])[: int(max_cols)]

    missing_cols = [c for c in cols_all if c not in df.columns]
    n_rows = int(df.shape[0])

    metrics: dict[str, Any] = {
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

    hi_warn: list[dict[str, Any]] = []
    hi_fail: list[dict[str, Any]] = []
    id_warn: list[dict[str, Any]] = []
    id_fail: list[dict[str, Any]] = []

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

        row: dict[str, Any] = {
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
    causal_spec: CausalSpec,
    ignore_cols: Sequence[str] = (),
    max_cols: int = 500,
    # object-type scan is bounded for speed/determinism
    obj_type_scan_n: int = 200,
    # if True, datetime columns are WARN; if False they are INFO-level (but you only support WARN/FAIL)
    warn_on_datetime: bool = True,
    # If you have a strict policy that "object dtype is not allowed" pre-transform, set to True
    fail_on_object_mixed_types: bool = False,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
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
    issues: list[ValidationIssue] = []
    ignore = {c for c in ignore_cols if c and c.strip()}

    def _dedup_keep_order(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x.strip() and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    covariates = _dedup_keep_order(list(getattr(causal_spec, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(causal_spec, "effect_modifiers", []) or []))
    cols_all = _dedup_keep_order([c for c in (covariates + effect_modifiers) if c not in ignore])[: int(max_cols)]

    missing_cols = [c for c in cols_all if c not in df.columns]
    n_rows = int(df.shape[0])

    metrics: dict[str, Any] = {
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

    datetime_cols: list[dict[str, Any]] = []
    mixed_object_cols: list[dict[str, Any]] = []
    long_text_cols: list[dict[str, Any]] = []
    object_high_card_cols: list[dict[str, Any]] = []

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

ArmKind = Literal["binary", "continuous"]


@dataclass(frozen=True)
class ArmMasks:
    kind: ArmKind
    treatment_col: str
    masks: dict[str, pd.Series]  # arm_name -> bool mask
    counts: dict[str, int]       # arm_name -> count
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
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

def _parse_bool_token(raw: str) -> bool | None:
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


def _dedup_keep_order(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
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
    causal_spec: CausalSpec,
    max_bins_continuous: int = 5,
) -> ArmMasks:
    tcol = causal_spec.treatment_spec.column
    if tcol not in df.columns:
        raise KeyError(f"treatment_col not found in df: {tcol!r}")

    ts = causal_spec.treatment_spec
    s = df[tcol]

    if isinstance(ts, BinaryTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        m_t = _mask_equals_literal(s, ts.treated)
        m_c = _mask_equals_literal(s, ts.control)
        masks = {"treated": m_t, "control": m_c}
        counts = {k: int(v.sum()) for k, v in masks.items()}
        return ArmMasks(kind="binary", treatment_col=tcol, masks=masks, counts=counts)

    raise ValueError(f"Unknown treatment_spec kind={getattr(ts, 'kind', None)!r}")


# -----------------------------------------------------------------------------
# 2) Univariate overlap / positivity support checks (df-backed)
# -----------------------------------------------------------------------------

def validate_overlap_positivity_univariate(
    *,
    df: pd.DataFrame,
    causal_spec: CausalSpec,
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
) -> tuple[list[ValidationIssue], dict[str, Any]]:
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
    issues: list[ValidationIssue] = []
    ignore = {c for c in ignore_cols if c.strip()}

    covariates = _dedup_keep_order(list(getattr(causal_spec, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(causal_spec, "effect_modifiers", []) or []))

    feat_cols = covariates + (effect_modifiers if use_effect_modifiers else [])
    feat_cols = _dedup_keep_order([c for c in feat_cols if c not in ignore])[: int(max_cols)]

    missing = [c for c in feat_cols if c not in df.columns]
    metrics: dict[str, Any] = {
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

    exclusive_flags: list[dict[str, Any]] = []
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
            intervals: dict[str, tuple[float, float, int]] = {}
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
                overlaps: list[float] = []
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
    causal_spec: CausalSpec,
    arm_masks: ArmMasks,
    # use covariates only by default; effect modifiers often include post-treatment-ish features by mistake
    use_effect_modifiers: bool = False,
    max_features: int = 200,
    sample_n: int = 10000,
    extreme_lo: float = 0.01,
    extreme_hi: float = 0.99,
    auc_warn: float = 0.90,
    extreme_share_warn: float = 0.20,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """
    Binary-treatment-only proxy:
      Fit a simple logistic regression on numeric-coercible covariates (and optionally effect_modifiers).
      Flag if:
        - AUC is very high AND
        - many predicted propensities are extreme

    If sklearn is unavailable, emits WARN and skips.
    """
    issues: list[ValidationIssue] = []

    metrics: dict[str, Any] = {
        "enabled": False,
        "reason": None,
        "treatment_col": causal_spec.treatment_spec.column,
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

    covariates = _dedup_keep_order(list(getattr(causal_spec, "covariates", []) or []))
    effect_modifiers = _dedup_keep_order(list(getattr(causal_spec, "effect_modifiers", []) or []))
    feat_cols = covariates + (effect_modifiers if use_effect_modifiers else [])
    feat_cols = _dedup_keep_order([c for c in feat_cols if c in df.columns])

    metrics["n_features_candidate"] = int(len(feat_cols))

    X_parts: list[np.ndarray] = []
    used: list[str] = []

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
    causal_spec: CausalSpec,
    require_covariates: bool = True,
    # univariate knobs
    use_effect_modifiers_univariate: bool = True,
    # propensity proxy knobs
    enable_propensity_proxy: bool = True,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """
    Advanced overlap/positivity validation suite.

    Runs:
      - arm mask construction
      - univariate support/exclusivity checks on covariates (+ optional effect_modifiers)
      - optional propensity proxy (binary treatment only)

    Returns:
      (issues, metrics)
    """
    issues: list[ValidationIssue] = []

    # Build masks (this can raise if T missing; let caller run treatment presence checks earlier)
    arms = compute_arm_masks_from_protocol(df=df, causal_spec=causal_spec)

    # Require covariates for overlap checks (otherwise overlap is not meaningful)
    covariates = _dedup_keep_order(list(getattr(causal_spec, "covariates", []) or []))
    if require_covariates and not covariates:
        metrics = {"require_covariates": True, "n_covariates": 0, "treatment_col": causal_spec.treatment_spec.column}
        issues.append(
            _issue(
                severity="FAIL",
                message="Cannot assess overlap/positivity: causal_spec.covariates is empty (no adjustment set).",
                evidence=metrics,
                fix_hint="Add covariates (confounders) to causal_spec.covariates before causal estimation.",
            )
        )
        return issues, {"arm_masks": arms.to_dict(), **metrics}

    # Univariate overlap
    iss_u, met_u = validate_overlap_positivity_univariate(
        df=df,
        causal_spec=causal_spec,
        arm_masks=arms,
        use_effect_modifiers=use_effect_modifiers_univariate,
    )
    issues.extend(iss_u)

    metrics: dict[str, Any] = {"arm_masks": arms.to_dict(), "univariate": met_u}

    # Optional propensity proxy
    if enable_propensity_proxy:
        iss_p, met_p = validate_overlap_propensity_proxy(df=df, causal_spec=causal_spec, arm_masks=arms)
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
    evidence: dict[str, Any] | None = None,
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


def _duplicates(cols: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    dups: list[str] = []
    for c in cols:
        if c in seen and c not in dups:
            dups.append(c)
        seen.add(c)
    return dups