from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from python.implementation.workflows.utils.validation import ValidationIssueModel

Severity = Literal["WARN", "FAIL"]
SemanticKind = Literal["binary", "continuous", "duration"]


# =============================================================================
# Inputs the validator needs (these come from existing states, not humans)
# =============================================================================

@dataclass(frozen=True)
class RoleSpec:
    raw_columns: Sequence[str]
    kind: SemanticKind


@dataclass(frozen=True)
class RoleSet:
    # From protocol / inference-ready state
    treatment: RoleSpec
    outcome: RoleSpec
    covariates_w: Sequence[str]
    effect_modifiers_x: Sequence[str]


@dataclass(frozen=True)
class FeatureMap:
    # From transform application (raw -> produced cols, plus dropped)
    produced_columns: Mapping[str, Sequence[str]]
    dropped: Sequence[str]


# =============================================================================
# Config: NOT user-filled. Defaults here; any “dynamic” parts are inferred.
# =============================================================================

@dataclass(frozen=True)
class TransformPostValidationConfig:
    # --- Artifact integrity ---
    require_same_row_count: bool = True
    require_unique_columns: bool = True

    # If we can infer a stable key, we validate it; otherwise we skip.
    validate_stable_key_if_available: bool = True
    stable_key_uniqueness_ratio: float = 0.999  # candidate id-like key threshold

    # --- Domain constraints ---
    binary_allowed_values: Tuple[int, int] = (0, 1)

    # Continuous sanity (constant columns are problematic)
    min_variance: float = 1e-12

    # Duration sanity
    duration_min_value: float = 0.0


# =============================================================================
# Issue helper
# =============================================================================

def _issue(
    *,
    severity: Severity,
    message: str,
    evidence: Optional[Dict[str, Any]] = None,
    fix_hint: Optional[str] = None,
) -> ValidationIssueModel:
    return ValidationIssueModel(
        severity=severity,
        message=message,
        evidence=evidence or {},
        fix_hint=fix_hint,
    )


# =============================================================================
# Small deterministic utilities
# =============================================================================

def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s.dtype)


def _unique_ratio(s: pd.Series) -> float:
    n = int(len(s))
    if n <= 0:
        return 0.0
    return float(s.nunique(dropna=False)) / float(n)


def _infer_stable_row_key(
    *,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    profiling_summary: Optional[Mapping[str, Any]],
    cfg: TransformPostValidationConfig,
) -> Optional[str]:
    """
    Infer a stable row key automatically.

    Priority:
      1) profiling_summary["stable_row_key"] if present
      2) profiling_summary["candidate_row_keys"] (first that exists in both dfs)
      3) heuristic: pick a column present in both, with high uniqueness ratio in both

    This avoids “manual config filling”.
    """
    if profiling_summary:
        key = profiling_summary.get("stable_row_key")
        if isinstance(key, str) and key in df_before.columns and key in df_after.columns:
            return key

        cands = profiling_summary.get("candidate_row_keys")
        if isinstance(cands, list):
            for c in cands:
                if isinstance(c, str) and c in df_before.columns and c in df_after.columns:
                    return c

    # Heuristic fallback: scan shared columns for high uniqueness
    shared = [c for c in df_before.columns if c in df_after.columns]
    best: Optional[Tuple[str, float]] = None

    for c in shared:
        # avoid expensive scanning of huge object columns by skipping very wide text-like columns
        # (still deterministic; just a pragmatic guard)
        s0 = df_before[c]
        s1 = df_after[c]
        r0 = _unique_ratio(s0)
        r1 = _unique_ratio(s1)
        r = min(r0, r1)
        if r >= cfg.stable_key_uniqueness_ratio:
            if best is None or r > best[1]:
                best = (c, r)

    return best[0] if best else None


def _resolve_raw_to_produced(
    raw: str,
    *,
    df_after: pd.DataFrame,
    feature_map: Optional[FeatureMap],
) -> List[str]:
    """
    Resolve raw protocol col -> transformed cols.
    Deterministic rules:
      - If feature_map says dropped -> []
      - If feature_map provides produced list -> that list
      - Else identity fallback if raw exists in df_after
    """
    if feature_map is not None:
        if raw in set(feature_map.dropped):
            return []
        produced = feature_map.produced_columns.get(raw)
        if produced is not None:
            return list(produced)

    return [raw] if raw in df_after.columns else []


def _resolve_many(
    raws: Sequence[str],
    *,
    df_after: pd.DataFrame,
    feature_map: Optional[FeatureMap],
) -> List[str]:
    out: List[str] = []
    for r in raws:
        out.extend(_resolve_raw_to_produced(r, df_after=df_after, feature_map=feature_map))
    # de-dupe, preserve order
    seen: Set[str] = set()
    dedup: List[str] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


# =============================================================================
# Validation 1: Artifact integrity invariants (math / exactness)
# =============================================================================

def validate_artifact_integrity(
    *,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    profiling_summary: Optional[Mapping[str, Any]] = None,
    cfg: Optional[TransformPostValidationConfig] = None,
) -> List[ValidationIssueModel]:
    """
    Post-transform artifact checks:
      - row count unchanged
      - no duplicate column names
      - if a stable row key can be inferred, ensure key set unchanged (and key is unique)

    Why static:
      - exact equality/count checks over full data
      - referential integrity (unit alignment) must be deterministic
    """
    c = cfg or TransformPostValidationConfig()
    issues: List[ValidationIssueModel] = []

    n0 = int(len(df_before))
    n1 = int(len(df_after))

    if c.require_same_row_count and n0 != n1:
        issues.append(
            _issue(
                severity="FAIL",
                message="Row count changed after transform.",
                evidence={"n_rows_before": n0, "n_rows_after": n1},
                fix_hint="Transforms must not filter rows; ensure encoding does not drop rows implicitly.",
            )
        )

    if c.require_unique_columns:
        dup = df_after.columns[df_after.columns.duplicated()].tolist()
        if dup:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Duplicate column names detected after transform.",
                    evidence={"duplicate_columns": dup[:50], "n_duplicates": len(dup)},
                    fix_hint="Ensure deterministic naming for derived features and prevent collisions.",
                )
            )

    if c.validate_stable_key_if_available:
        key = _infer_stable_row_key(
            df_before=df_before,
            df_after=df_after,
            profiling_summary=profiling_summary,
            cfg=c,
        )
        if key is not None:
            # uniqueness
            r0 = _unique_ratio(df_before[key])
            r1 = _unique_ratio(df_after[key])
            if r0 < 1.0 or r1 < 1.0:
                issues.append(
                    _issue(
                        severity="FAIL",
                        message="Stable row key is not unique; cannot guarantee row identity preservation.",
                        evidence={"stable_row_key": key, "unique_ratio_before": r0, "unique_ratio_after": r1},
                        fix_hint="Use a truly unique row key (row_id/patient_id+time) or store a row fingerprint.",
                    )
                )
            else:
                set0 = set(df_before[key].astype(str).tolist())
                set1 = set(df_after[key].astype(str).tolist())
                if set0 != set1:
                    issues.append(
                        _issue(
                            severity="FAIL",
                            message="Stable row key set changed after transform (row identity not preserved).",
                            evidence={
                                "stable_row_key": key,
                                "missing_in_after_sample": sorted(list(set0 - set1))[:10],
                                "new_in_after_sample": sorted(list(set1 - set0))[:10],
                            },
                            fix_hint="Transform must not drop/duplicate/rewrite row identifiers.",
                        )
                    )

    return issues


# =============================================================================
# Validation 2: Role resolution correctness (referential integrity)
# =============================================================================

def validate_role_resolution(
    *,
    roles: RoleSet,
    df_after: pd.DataFrame,
    feature_map: Optional[FeatureMap] = None,
) -> List[ValidationIssueModel]:
    """
    Ensures protocol roles are satisfiable on the transformed dataset:
      - each raw role column resolves to >=1 produced column
      - all produced columns exist in df_after

    Why static:
      - exact key existence / lookup correctness
    """
    issues: List[ValidationIssueModel] = []

    required: List[Tuple[str, str]] = []
    required.extend((c, "outcome") for c in roles.outcome.raw_columns)
    required.extend((c, "treatment") for c in roles.treatment.raw_columns)
    required.extend((c, "covariates_w") for c in roles.covariates_w)
    required.extend((c, "effect_modifiers_x") for c in roles.effect_modifiers_x)

    dropped_set = set(feature_map.dropped) if feature_map is not None else set()

    for raw, role_name in required:
        if raw in dropped_set:
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"Required column was dropped by transform: '{raw}'.",
                    evidence={"raw_column": raw, "role": role_name},
                    fix_hint="Do not drop protocol-required columns; change encoding for this column.",
                )
            )
            continue

        produced = _resolve_raw_to_produced(raw, df_after=df_after, feature_map=feature_map)
        if not produced:
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"Required column not resolvable after transform: '{raw}'.",
                    evidence={"raw_column": raw, "role": role_name},
                    fix_hint="Ensure column exists or feature_map correctly maps raw->produced.",
                )
            )
            continue

        missing = [p for p in produced if p not in df_after.columns]
        if missing:
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"Feature map references missing produced columns for '{raw}'.",
                    evidence={"raw_column": raw, "role": role_name, "missing_produced": missing},
                    fix_hint="Encoding implementation or naming convention mismatch; fix producer or feature_map.",
                )
            )

    return issues


# =============================================================================
# Validation 3: Treatment + Outcome domain checks (math)
# =============================================================================

def validate_treatment_outcome_domains(
    *,
    roles: RoleSet,
    df_after: pd.DataFrame,
    feature_map: Optional[FeatureMap] = None,
    cfg: Optional[TransformPostValidationConfig] = None,
) -> List[ValidationIssueModel]:
    """
    Validates that T/Y match declared semantic kinds after transform.

    Why static:
      - requires computing exact unique sets, variance, min bounds on full arrays
      - these are estimator preconditions
    """
    c = cfg or TransformPostValidationConfig()
    issues: List[ValidationIssueModel] = []

    t_cols = _resolve_many(roles.treatment.raw_columns, df_after=df_after, feature_map=feature_map)
    y_cols = _resolve_many(roles.outcome.raw_columns, df_after=df_after, feature_map=feature_map)

    if not t_cols:
        issues.append(
            _issue(
                severity="FAIL",
                message="Treatment columns not resolvable after transform.",
                evidence={"raw_treatment": list(roles.treatment.raw_columns)},
            )
        )
        return issues

    if not y_cols:
        issues.append(
            _issue(
                severity="FAIL",
                message="Outcome columns not resolvable after transform.",
                evidence={"raw_outcome": list(roles.outcome.raw_columns)},
            )
        )
        return issues

    # --- helpers ---
    def _check_binary(col: str, role: str) -> None:
        s = df_after[col]
        if not _is_numeric_series(s):
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' is not numeric after transform (binary required).",
                    evidence={"column": col, "dtype": str(s.dtype)},
                    fix_hint="Ensure encoding outputs numeric 0/1 values.",
                )
            )
            return

        allowed = set(c.binary_allowed_values)
        observed = set(pd.unique(s.to_numpy()))
        # cast numpy scalars to python ints/floats for evidence readability
        observed_norm = {int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v for v in observed}

        if not observed.issubset(allowed):
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' is not binary {sorted(list(allowed))}.",
                    evidence={"column": col, "observed_values": sorted(list(observed_norm))},
                    fix_hint="Fix mapping/encoding so only 0/1 remain.",
                )
            )

        if len(observed) < 2:
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' is constant after transform (no variation).",
                    evidence={"column": col, "observed_values": sorted(list(observed_norm))},
                    fix_hint="Check encoding/mapping; you likely collapsed categories or filtered implicitly.",
                )
            )

    def _check_continuous(col: str, role: str) -> None:
        s = df_after[col]
        if not _is_numeric_series(s):
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' is not numeric after transform (continuous required).",
                    evidence={"column": col, "dtype": str(s.dtype)},
                    fix_hint="Use to_numeric or a numeric mapping encoding.",
                )
            )
            return

        x = s.to_numpy(dtype=float)
        if not np.all(np.isfinite(x)):
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' contains non-finite values after transform.",
                    evidence={"column": col},
                    fix_hint="Fix transform to avoid inf/-inf (e.g., log on negatives, divide by ~0).",
                )
            )
            return

        var = float(np.var(x))
        if var <= c.min_variance:
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' is (near) constant after transform (variance too low).",
                    evidence={"column": col, "variance": var, "min_variance": c.min_variance},
                    fix_hint="Constant outcomes/features break identification or model fitting; check encoding.",
                )
            )

    def _check_duration(col: str, role: str) -> None:
        s = df_after[col]
        if not _is_numeric_series(s):
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' is not numeric after transform (duration required).",
                    evidence={"column": col, "dtype": str(s.dtype)},
                )
            )
            return

        x = s.to_numpy(dtype=float)
        if not np.all(np.isfinite(x)):
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' contains non-finite values after transform.",
                    evidence={"column": col},
                )
            )
            return

        mn = float(np.min(x))
        if mn < c.duration_min_value:
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' has values below allowed minimum.",
                    evidence={"column": col, "min_value": mn, "duration_min_value": c.duration_min_value},
                    fix_hint="Duration must be non-negative (or positive) as declared; fix encoding/parsing.",
                )
            )

        var = float(np.var(x))
        if var <= c.min_variance:
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' is (near) constant after transform (variance too low).",
                    evidence={"column": col, "variance": var, "min_variance": c.min_variance},
                )
            )

    # --- Treatment checks ---
    for col in t_cols:
        if roles.treatment.kind == "binary":
            _check_binary(col, "Treatment")
        elif roles.treatment.kind == "continuous":
            _check_continuous(col, "Treatment")
        else:  # duration
            _check_duration(col, "Treatment")

    # --- Outcome checks ---
    for col in y_cols:
        if roles.outcome.kind == "binary":
            _check_binary(col, "Outcome")
        elif roles.outcome.kind == "continuous":
            _check_continuous(col, "Outcome")
        else:
            _check_duration(col, "Outcome")

    return issues