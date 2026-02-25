from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from python.implementation.workflows.utils.validation import ValidationIssueModel, ValidationSeverity

from python.implementation.workflows.nodes.transform_protocol.transform_protocol_specs import TransformedProtocolSpec


def _issue(
    *,
     severity: ValidationSeverity,
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


def validate_input_columns_exist_and_are_unambiguous(
    *,
    df_after: pd.DataFrame,
    spec: TransformedProtocolSpec,
) -> List[ValidationIssueModel]:
    """
    Validation #1 (HARD GATE): Schema referential integrity.

    Checks (deterministic):
      1) df_after must not contain duplicate column labels (Pandas permits duplicates).
      2) Every column referenced by TransformedProtocolSpec (Y/T/W/X) must exist in df_after.
      3) Any referenced column name must be unambiguous (not duplicated).

    Why this must be static:
      - This is exact key membership / ambiguity detection on the real artifact.
      - An LLM cannot "validate" exact schema correctness; it can only guess from samples.
    """
    issues: List[ValidationIssueModel] = []

    # ---- 1) Detect duplicate dataframe column labels (global schema ambiguity) ----
    dup_mask = df_after.columns.duplicated(keep=False)
    if bool(dup_mask.any()):
        dup_names = df_after.columns[dup_mask].tolist()
        # Count occurrences for evidence (limit to keep payload small)
        counts: Dict[str, int] = {}
        for n in dup_names:
            counts[n] = counts.get(n, 0) + 1

        # Provide a stable, small sample for debugging
        sample = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:25]
        issues.append(
            _issue(
                severity="FAIL",
                message="Transformed dataframe contains duplicate column labels (ambiguous schema).",
                evidence={
                    "n_total_columns": int(len(df_after.columns)),
                    "n_duplicated_labels": int(len(counts)),
                    "duplicated_label_sample": [{"name": k, "count": v} for k, v in sample],
                },
                fix_hint=(
                    "Fix transform naming collisions (e.g., join/concat suffix rules, one-hot prefixing, "
                    "derived-feature naming). Pandas duplicate labels can silently corrupt column selection."
                ),
            )
        )

    # ---- 2) Collect required columns from the spec ----
    # Prefer spec.all_input_cols if you added it; else build from properties.
    if hasattr(spec, "all_input_cols"):
        required_cols = list(getattr(spec, "all_input_cols"))
    else:
        required_cols = []
        required_cols.extend(list(getattr(spec, "y_cols", [])))
        required_cols.extend(list(getattr(spec, "t_cols", [])))
        required_cols.extend(list(getattr(spec, "w_cols", [])))
        required_cols.extend(list(getattr(spec, "x_cols", [])))

    # De-dupe while preserving order
    seen = set()
    dedup_required: List[str] = []
    for c in required_cols:
        if c not in seen:
            seen.add(c)
            dedup_required.append(c)

    # ---- 3) Check existence and ambiguity of each required column ----
    cols_index = df_after.columns
    missing: List[str] = []
    duplicated_required: List[str] = []

    for c in dedup_required:
        if c not in cols_index:
            missing.append(c)
            continue

        # If df has duplicates, a required column might appear multiple times -> ambiguous selection
        # (even if we already flagged global duplicates, this pinpoints which required names are affected).
        if int((cols_index == c).sum()) > 1:
            duplicated_required.append(c)

    if missing:
        issues.append(
            _issue(
                severity="FAIL",
                message="TransformedProtocolSpec references columns that do not exist in the transformed dataframe.",
                evidence={
                    "missing_columns": missing[:50],
                    "n_missing": int(len(missing)),
                },
                fix_hint=(
                    "Resolver/spec builder is out of sync with the transform output naming. "
                    "Ensure the transform emits deterministic column names and the spec uses those exact names."
                ),
            )
        )

    if duplicated_required:
        issues.append(
            _issue(
                severity="FAIL",
                message="TransformedProtocolSpec references column names that are duplicated in the transformed dataframe (ambiguous).",
                evidence={
                    "duplicated_required_columns": duplicated_required[:50],
                    "n_duplicated_required": int(len(duplicated_required)),
                },
                fix_hint=(
                    "Even if the column name exists, duplicates make selection ambiguous. "
                    "Fix the transform to produce globally unique column labels."
                ),
            )
        )

    return issues


# =============================================================================
# Validation #2: Numeric dtype enforcement (HARD GATE)
# =============================================================================
def validate_model_inputs_are_numeric_dtypes(
    *,
    df_after: pd.DataFrame,
    spec: TransformedProtocolSpec,
    allow_bool: bool = True,
) -> List[ValidationIssueModel]:
    """
    Validation #2 (HARD GATE): All model inputs must be numeric dtypes.

    Why static (non-replaceable by LLM):
      - Pandas dtype drives runtime behavior. "Looks numeric" strings still crash or coerce inconsistently.
      - This must be checked on the actual artifact, not inferred from samples.

    Checks:
      - For every col in spec.y/t/w/x: dtype is numeric (or bool if allow_bool).
    """
    issues: List[ValidationIssueModel] = []

    if hasattr(spec, "all_input_cols"):
        cols = list(getattr(spec, "all_input_cols"))
    else:
        cols = []
        cols.extend(list(getattr(spec, "y_cols", [])))
        cols.extend(list(getattr(spec, "t_cols", [])))
        cols.extend(list(getattr(spec, "w_cols", [])))
        cols.extend(list(getattr(spec, "x_cols", [])))

    # de-dupe preserve order
    seen = set()
    dedup_cols: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            dedup_cols.append(c)

    non_numeric: List[Dict[str, str]] = []
    for c in dedup_cols:
        if c not in df_after.columns:
            # Existence is validated in Validation #1; skip here.
            continue

        s = df_after[c]
        is_num = pd.api.types.is_numeric_dtype(s.dtype)
        is_bool = pd.api.types.is_bool_dtype(s.dtype)

        if not is_num and not (allow_bool and is_bool):
            non_numeric.append({"column": c, "dtype": str(s.dtype)})

    if non_numeric:
        issues.append(
            _issue(
                severity="FAIL",
                message="Non-numeric dtypes present in model inputs (Y/T/W/X).",
                evidence={
                    "non_numeric_sample": non_numeric[:50],
                    "n_non_numeric": int(len(non_numeric)),
                },
                fix_hint=(
                    "Ensure transforms output numeric dtypes. Common fixes: apply to_numeric, "
                    "use *_map_idx encoders, one_hot, or parse datetimes to epoch."
                ),
            )
        )

    return issues


# =============================================================================
# Validation #3: Treatment/Outcome domain checks by kind (HARD GATE)
# =============================================================================
def validate_treatment_outcome_domains_by_kind(
    *,
    df_after: pd.DataFrame,
    spec: TransformedProtocolSpec,
    binary_allowed_values: Tuple[int, int] = (0, 1),
    min_variance: float = 1e-12,
    duration_min_value: float = 0.0,
) -> List[ValidationIssueModel]:
    """
    Validation #3 (HARD GATE): Validate Y/T value domains based on spec.y_kind and spec.t_kind.

    Why static (non-replaceable by LLM):
      - Requires global aggregation (exact unique sets, min/max, variance, finiteness) over full columns.
      - A single rare bad value (e.g., '2' in binary) can silently break estimator assumptions.

    Notes:
      - This validates *data content*, not just schema.
      - It does NOT repeat constraints already enforced by Pydantic (like len(y)==1 for non-duration).
    """
    issues: List[ValidationIssueModel] = []

    def _ensure_finite_numeric(col: str, role: str) -> Optional[np.ndarray]:
        if col not in df_after.columns:
            return None

        s = df_after[col]

        # dtype check is handled in Validation #2, but keep a tight guard:
        if not (pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype)):
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' is not numeric; cannot validate domain.",
                    evidence={"column": col, "dtype": str(s.dtype)},
                    fix_hint="Fix encoding to output numeric dtype.",
                )
            )
            return None

        x = s.to_numpy()

        # bool is okay; treat as 0/1 later for binary checks
        if x.dtype == np.bool_:
            x = x.astype(np.int64)

        # Convert to float for finite checks/variance
        xf = x.astype(float, copy=False)

        if not np.all(np.isfinite(xf)):
            # You said missingness handled upstream; this catches inf/-inf too.
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' contains non-finite values (NaN/inf/-inf) after transform.",
                    evidence={"column": col},
                    fix_hint="Fix transform step that produced non-finite values (e.g., log on zeros/negatives).",
                )
            )
            return None

        return xf

    def _check_binary(col: str, role: str) -> None:
        xf = _ensure_finite_numeric(col, role)
        if xf is None:
            return

        # Exact unique set (global)
        uniq = set(np.unique(xf))
        # Accept 0.0/1.0 as well
        uniq_norm = set(float(u) for u in uniq)

        bad = sorted([u for u in uniq_norm if u not in {0.0, 1.0}])
        if bad:
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' is declared binary but contains values outside {{0,1}}.",
                    evidence={"column": col, "unique_values": sorted(list(uniq_norm))},
                    fix_hint="Fix mapping/encoding so the output is strictly 0/1.",
                )
            )
            return

        # Must contain both classes
        if uniq_norm != {0.0, 1.0}:
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' is binary but constant / missing one class.",
                    evidence={"column": col, "unique_values": sorted(list(uniq_norm))},
                    fix_hint="You need both classes present; check filtering, mapping, or cohort construction.",
                )
            )

    def _check_continuous(col: str, role: str) -> None:
        xf = _ensure_finite_numeric(col, role)
        if xf is None:
            return

        var = float(np.var(xf))
        if var <= float(min_variance):
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' is (near) constant (variance too low).",
                    evidence={"column": col, "variance": var, "min_variance": float(min_variance)},
                    fix_hint="Constant outcomes/features break model fitting; check transform or cohort.",
                )
            )

    def _check_duration(col: str, role: str) -> None:
        xf = _ensure_finite_numeric(col, role)
        if xf is None:
            return

        mn = float(np.min(xf))
        if mn < float(duration_min_value):
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' has values below allowed minimum for duration.",
                    evidence={"column": col, "min_value": mn, "duration_min_value": float(duration_min_value)},
                    fix_hint="Duration should be non-negative (or per your declared minimum); fix parsing/encoding.",
                )
            )

        var = float(np.var(xf))
        if var <= float(min_variance):
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role} '{col}' is (near) constant (variance too low).",
                    evidence={"column": col, "variance": var, "min_variance": float(min_variance)},
                )
            )

    # -------------------------
    # Outcome checks (Y)
    # -------------------------
    y_cols = list(getattr(spec, "y_cols", []))
    y_kind = str(getattr(spec, "y_kind", ""))

    if y_kind == "binary":
        # Pydantic already enforces 1 col for non-duration; still loop defensively.
        for c in y_cols:
            _check_binary(c, "Outcome")
    elif y_kind == "continuous":
        for c in y_cols:
            _check_continuous(c, "Outcome")
    elif y_kind == "duration":
        # For now: treat spec.y as the duration columns being passed to the model adapter.
        # If you later represent duration outcome as (duration,event) pair, extend ModelInputSpecs.
        for c in y_cols:
            _check_duration(c, "Outcome")
    elif y_kind == "categorical":
        # If you don't support categorical outcomes in your runner, fail here.
        issues.append(
            _issue(
                severity="FAIL",
                message="Categorical outcomes are not supported by the current runner contract.",
                evidence={"y_cols": y_cols},
                fix_hint="Encode outcome to binary/continuous (or extend runner + estimators to support multi-class).",
            )
        )
    else:
        issues.append(
            _issue(
                severity="FAIL",
                message="Unknown y_kind in ModelInputSpecs.",
                evidence={"y_kind": y_kind},
            )
        )

    # -------------------------
    # Treatment checks (T)
    # -------------------------
    t_cols = list(getattr(spec, "t_cols", []))
    t_kind = str(getattr(spec, "t_kind", ""))

    if t_kind == "binary":
        for c in t_cols:
            _check_binary(c, "Treatment")
    elif t_kind == "continuous":
        for c in t_cols:
            _check_continuous(c, "Treatment")
    elif t_kind == "categorical":
        # Two supported patterns:
        #  (A) single integer-coded column (ordinal_map_idx style)
        #  (B) multi-column one-hot treatment (multi-arm) -> each col in {0,1} and row-sum == 1
        if len(t_cols) == 1:
            col = t_cols[0]
            xf = _ensure_finite_numeric(col, "Treatment")
            if xf is not None:
                # integer-like check
                if not np.all(np.isclose(xf, np.round(xf))):
                    issues.append(
                        _issue(
                            severity="FAIL",
                            message="Categorical treatment is single-column but not integer-coded.",
                            evidence={"column": col},
                            fix_hint="Use ordinal_map_idx / binary_map_idx (integer codes) or one-hot multi-arm.",
                        )
                    )
                if len(np.unique(xf)) < 2:
                    issues.append(
                        _issue(
                            severity="FAIL",
                            message="Categorical treatment has <2 unique levels after transform.",
                            evidence={"column": col},
                        )
                    )
        else:
            # multi-arm one-hot style
            mats: List[np.ndarray] = []
            for c in t_cols:
                _check_binary(c, "Treatment")  # ensures 0/1 + both? (both isn't required per dummy, but ok)
                xf = _ensure_finite_numeric(c, "Treatment")
                if xf is not None:
                    mats.append(xf)

            if mats:
                M = np.column_stack(mats)
                row_sum = M.sum(axis=1)
                # strict: every row assigned to exactly one arm
                if not np.all(np.isclose(row_sum, 1.0)):
                    # summarize violations
                    bad_n = int(np.sum(~np.isclose(row_sum, 1.0)))
                    issues.append(
                        _issue(
                            severity="FAIL",
                            message="Multi-arm one-hot treatment violates row-sum==1 invariant.",
                            evidence={
                                "t_cols": t_cols[:50],
                                "n_bad_rows": bad_n,
                                "row_sum_min": float(np.min(row_sum)),
                                "row_sum_max": float(np.max(row_sum)),
                            },
                            fix_hint="Ensure exactly one treatment arm is active per row (add explicit 'Unknown' arm if needed).",
                        )
                    )
    else:
        issues.append(
            _issue(
                severity="FAIL",
                message="Unknown t_kind in ModelInputSpecs.",
                evidence={"t_kind": t_kind},
            )
        )

    return issues


# =============================================================================
# Validation #4: One-hot / binary invariants (HARD GATE)
# =============================================================================
def validate_binary_and_one_hot_invariants(
    *,
    df_after: pd.DataFrame,
    spec: TransformedProtocolSpec,
    value_tol: float = 1e-9,
    # If you provide source_raw in your ColumnRefModel, we can do grouped row-sum checks.
    check_one_hot_group_row_sums: bool = True,
    allow_zero_sum_rows: bool = True,  # allow (0) when category missing/unknown not explicitly modeled
) -> List[ValidationIssueModel]:
    """
    Validation #4 (HARD GATE): Enforce invariants for columns declared as feature_kind in {"binary","one_hot"}.

    Checks (deterministic):
      - For every W/X column with feature_kind binary/one_hot: values are in {0,1} (within tol)
      - Optional group-wise check (requires ColumnRefModel.source_raw):
          For each (role, source_raw) group of one_hot columns:
            * no row has sum > 1  (impossible for single-valued categorical one-hot)
            * (optional) warn if some rows have sum == 0 when allow_zero_sum_rows=True

    Why static (non-replaceable by LLM):
      - This is a global algebraic constraint across *all rows* (and sometimes across groups of columns).
      - A rare illegal value or rare multi-hot row is enough to silently corrupt meaning.
    """
    issues: List[ValidationIssueModel] = []

    # Helper to iterate role columns with metadata
    def _iter_role(role_name: str) -> List[Any]:
        role_obj = getattr(spec, role_name, None)
        if role_obj is None:
            return []
        cols = getattr(role_obj, "columns", None)
        return list(cols) if cols else []

    # Only enforce invariants on W/X (because Y/T domain is handled in Validation #3)
    candidates: List[Tuple[str, Any]] = []
    for role_name in ("w", "x"):
        for cref in _iter_role(role_name):
            fk = str(getattr(cref, "feature_kind", "unknown"))
            if fk in ("binary", "one_hot"):
                candidates.append((role_name, cref))

    if not candidates:
        return issues

    # --- 4a) Per-column {0,1} check ---
    bad_cols: List[Dict[str, Any]] = []
    constant_cols: List[Dict[str, Any]] = []

    for role_name, cref in candidates:
        col = str(getattr(cref, "name"))
        if col not in df_after.columns:
            # Existence already checked in Validation #1; skip to avoid noise.
            continue

        s = df_after[col]
        # dtype numeric check is in Validation #2; keep a tight guard
        if not (pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype)):
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role_name.upper()} column '{col}' declared {getattr(cref, 'feature_kind', 'unknown')} but is non-numeric dtype.",
                    evidence={"column": col, "dtype": str(s.dtype), "role": role_name},
                    fix_hint="Ensure encoder outputs numeric 0/1 (not strings or categories).",
                )
            )
            continue

        x = s.to_numpy()
        if x.dtype == np.bool_:
            x = x.astype(np.int8)

        xf = x.astype(float, copy=False)

        if not np.all(np.isfinite(xf)):
            issues.append(
                _issue(
                    severity="FAIL",
                    message=f"{role_name.upper()} column '{col}' contains non-finite values (NaN/inf/-inf).",
                    evidence={"column": col, "role": role_name},
                    fix_hint="Fix the transform step producing NaN/inf (e.g., bad cast, invalid arithmetic).",
                )
            )
            continue

        # Accept values close to 0 or 1 within tolerance
        is0 = np.isclose(xf, 0.0, atol=value_tol, rtol=0.0)
        is1 = np.isclose(xf, 1.0, atol=value_tol, rtol=0.0)
        ok = is0 | is1
        if not bool(np.all(ok)):
            bad_idx = np.where(~ok)[0]
            sample_idx = bad_idx[:10].tolist()
            sample_vals = [float(xf[i]) for i in sample_idx]
            bad_cols.append(
                {
                    "column": col,
                    "role": role_name,
                    "feature_kind": str(getattr(cref, "feature_kind", "unknown")),
                    "n_bad": int(len(bad_idx)),
                    "bad_value_sample": sample_vals,
                }
            )
            continue

        # Constant dummy detection is useful but not always fatal -> WARN
        uniq = np.unique(xf)
        if uniq.size == 1:
            constant_cols.append(
                {
                    "column": col,
                    "role": role_name,
                    "feature_kind": str(getattr(cref, "feature_kind", "unknown")),
                    "constant_value": float(uniq[0]),
                }
            )

    if bad_cols:
        issues.append(
            _issue(
                severity="FAIL",
                message="Binary/one-hot columns contain values outside {0,1}.",
                evidence={"bad_columns_sample": bad_cols[:25], "n_bad_columns": int(len(bad_cols))},
                fix_hint="Fix encoding/casting. One-hot and binary features must be strict 0/1.",
            )
        )

    if constant_cols:
        issues.append(
            _issue(
                severity="WARN",
                message="Some binary/one-hot columns are constant (all 0 or all 1).",
                evidence={"constant_columns_sample": constant_cols[:25], "n_constant": int(len(constant_cols))},
                fix_hint="Consider dropping constant dummies or check if your cohort/encoding collapsed variability.",
            )
        )

    # --- 4b) Optional grouped row-sum checks for one_hot columns (needs source_raw) ---
    if not check_one_hot_group_row_sums:
        return issues

    # Group one_hot by (role, source_raw)
    groups: DefaultDict[Tuple[str, str], List[str]] = defaultdict(list)
    for role_name, cref in candidates:
        fk = str(getattr(cref, "feature_kind", "unknown"))
        if fk != "one_hot":
            continue
        src = getattr(cref, "source_raw", None)
        if not src:
            continue  # cannot group without provenance
        col = str(getattr(cref, "name"))
        if col in df_after.columns:
            groups[(role_name, str(src))].append(col)

    # Only check groups where it makes sense (>=2 dummies)
    for (role_name, src), cols in groups.items():
        if len(cols) < 2:
            continue
        M = df_after[cols].to_numpy(dtype=float)
        if not np.all(np.isfinite(M)):
            issues.append(
                _issue(
                    severity="FAIL",
                    message="One-hot group contains non-finite values.",
                    evidence={"role": role_name, "source_raw": src, "cols_sample": cols[:20]},
                )
            )
            continue

        row_sum = M.sum(axis=1)

        # Hard fail: any row has more than 1 active category (single-valued categorical violated)
        too_many = ~np.isclose(row_sum, 0.0, atol=value_tol, rtol=0.0) & (row_sum > 1.0 + value_tol)
        if bool(np.any(too_many)):
            n_bad = int(np.sum(too_many))
            issues.append(
                _issue(
                    severity="FAIL",
                    message="One-hot group violates single-valued categorical constraint (row sum > 1).",
                    evidence={
                        "role": role_name,
                        "source_raw": src,
                        "n_bad_rows": n_bad,
                        "row_sum_min": float(np.min(row_sum)),
                        "row_sum_max": float(np.max(row_sum)),
                        "cols_sample": cols[:20],
                    },
                    fix_hint="Your one-hot encoding produced multi-hot rows. Fix preprocessing/joins or use a multi-label encoding policy.",
                )
            )

        # If zero-sum is not allowed: fail. If allowed: warn (often indicates missing/unknown not handled).
        zeros = np.isclose(row_sum, 0.0, atol=value_tol, rtol=0.0)
        if bool(np.any(zeros)):
            if allow_zero_sum_rows:
                issues.append(
                    _issue(
                        severity="WARN",
                        message="One-hot group has rows with no active category (row sum == 0).",
                        evidence={
                            "role": role_name,
                            "source_raw": src,
                            "n_zero_rows": int(np.sum(zeros)),
                            "cols_sample": cols[:20],
                        },
                        fix_hint="If missing/unknown should be represented, add an explicit 'Unknown' level so row sums become 1.",
                    )
                )
            else:
                issues.append(
                    _issue(
                        severity="FAIL",
                        message="One-hot group violates constraint (row sum == 0 not allowed).",
                        evidence={
                            "role": role_name,
                            "source_raw": src,
                            "n_zero_rows": int(np.sum(zeros)),
                            "cols_sample": cols[:20],
                        },
                        fix_hint="Add explicit unknown level or ensure every row maps to exactly one category.",
                    )
                )

    return issues


# =============================================================================
# Validation #5: ID-like / near-unique feature detection in W/X (HARD GATE by default)
# =============================================================================
def validate_id_like_features_in_controls(
    *,
    df_after: pd.DataFrame,
    spec: TransformedProtocolSpec,
    uniqueness_ratio_threshold: float = 0.98,
    max_allowed_id_like: int = 0,  # 0 => FAIL if any; >0 => WARN unless count exceeds
) -> List[ValidationIssueModel]:
    """
    Validation #5: Identify ID-like features in W/X using uniqueness ratio.

    Check (deterministic):
      - For each W/X column: nunique(dropna=False)/n_rows >= threshold => ID-like.

    Why static (non-replaceable by LLM):
      - This is numeric aggregation over full columns. Name semantics are unreliable.
      - A single near-unique key can let nuisance models memorize rows (bad for causal estimation).

    Policy:
      - Default is hard gate (max_allowed_id_like=0). Adjust if you want warnings only.
    """
    issues: List[ValidationIssueModel] = []

    n = int(len(df_after))
    if n <= 0:
        return issues

    def _iter_role_cols(role_name: str) -> List[Any]:
        role_obj = getattr(spec, role_name, None)
        if role_obj is None:
            return []
        cols = getattr(role_obj, "columns", None)
        return list(cols) if cols else []

    flagged: List[Dict[str, Any]] = []

    for role_name in ("w", "x"):
        for cref in _iter_role_cols(role_name):
            col = str(getattr(cref, "name"))
            if col not in df_after.columns:
                continue

            s = df_after[col]
            # One-hot/binary typically not near-unique; skip to reduce noise
            fk = str(getattr(cref, "feature_kind", "unknown"))
            if fk in ("one_hot", "binary"):
                continue

            nunique = int(s.nunique(dropna=False))
            ratio = float(nunique) / float(n)

            if ratio >= float(uniqueness_ratio_threshold):
                flagged.append(
                    {
                        "column": col,
                        "role": role_name,
                        "feature_kind": fk,
                        "n_rows": n,
                        "nunique": nunique,
                        "uniqueness_ratio": ratio,
                    }
                )

    if flagged:
        sev: ValidationSeverity = "FAIL"
        if max_allowed_id_like > 0 and len(flagged) <= int(max_allowed_id_like):
            sev = "WARN"

        issues.append(
            _issue(
                severity=sev,
                message="ID-like / near-unique features detected in controls (W/X).",
                evidence={
                    "uniqueness_ratio_threshold": float(uniqueness_ratio_threshold),
                    "n_flagged": int(len(flagged)),
                    "flagged_sample": sorted(flagged, key=lambda d: -float(d["uniqueness_ratio"]))[:25],
                },
                fix_hint=(
                    "Drop or mask near-unique identifiers (patient_id, encounter_id, timestamps-as-keys). "
                    "They enable memorization and destabilize nuisance fits, harming causal estimates."
                ),
            )
        )

    return issues


# =============================================================================
# Validation #6: Constant / near-constant features in W/X (WARN or FAIL by policy)
# =============================================================================
def validate_constant_or_near_constant_controls(
    *,
    df_after: pd.DataFrame,
    spec: TransformedProtocolSpec,
    min_variance: float = 1e-12,
    # if set, fail when constant-like features exceed this count; otherwise WARN only
    max_constant_allowed: Optional[int] = None,
    # skip one_hot/binary because #4 already handles those (avoids duplicate warnings)
    skip_binary_one_hot: bool = True,
) -> List[ValidationIssueModel]:
    """
    Validation #6: detect constant / near-constant controls in W/X.

    Deterministic checks:
      - For each W/X column (optionally excluding one_hot/binary):
          * nunique(dropna=False) <= 1  OR  variance <= min_variance => constant-like

    Why static (non-replaceable by LLM):
      - Requires full-column aggregation (unique counts, variance). Sampling misses rare variation patterns.
    """
    issues: List[ValidationIssueModel] = []

    def _iter_role_cols(role_name: str) -> List[Any]:
        role_obj = getattr(spec, role_name, None)
        if role_obj is None:
            return []
        cols = getattr(role_obj, "columns", None)
        return list(cols) if cols else []

    constant_like: List[Dict[str, Any]] = []

    for role_name in ("w", "x"):
        for cref in _iter_role_cols(role_name):
            col = str(getattr(cref, "name"))
            if col not in df_after.columns:
                continue

            fk = str(getattr(cref, "feature_kind", "unknown"))
            if skip_binary_one_hot and fk in ("binary", "one_hot"):
                continue

            s = df_after[col]

            # dtype numeric is validated in #2; keep guard to avoid noisy exceptions
            if not (pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype)):
                continue

            nunique = int(s.nunique(dropna=False))
            if nunique <= 1:
                constant_like.append(
                    {
                        "column": col,
                        "role": role_name,
                        "feature_kind": fk,
                        "dtype": str(s.dtype),
                        "nunique": nunique,
                        "variance": 0.0,
                    }
                )
                continue

            x = s.to_numpy()
            if x.dtype == np.bool_:
                x = x.astype(np.int8)
            xf = x.astype(float, copy=False)

            if not np.all(np.isfinite(xf)):
                # non-finite is handled by #3 for Y/T; for W/X you can optionally hard-fail elsewhere
                continue

            var = float(np.var(xf))
            if var <= float(min_variance):
                constant_like.append(
                    {
                        "column": col,
                        "role": role_name,
                        "feature_kind": fk,
                        "dtype": str(s.dtype),
                        "nunique": nunique,
                        "variance": var,
                    }
                )

    if not constant_like:
        return issues

    sev: ValidationSeverity = "WARN"
    if max_constant_allowed is not None and len(constant_like) > int(max_constant_allowed):
        sev = "FAIL"

    issues.append(
        _issue(
            severity=sev,
            message="Constant / near-constant control features detected in W/X.",
            evidence={
                "min_variance": float(min_variance),
                "n_constant_like": int(len(constant_like)),
                "constant_like_sample": constant_like[:25],
            },
            fix_hint=(
                "Constant or near-constant controls add no information and can destabilize some estimators. "
                "Consider dropping them or investigate why the cohort/encoding collapsed variability."
            ),
        )
    )

    return issues


# =============================================================================
# Validation #7: Dimensionality caps (HARD FAIL)
# =============================================================================
def validate_dimensionality_caps(
    *,
    df_after: pd.DataFrame,
    spec: TransformedProtocolSpec,
    max_total_features: Optional[int] = 5000,
    max_w_features: Optional[int] = None,
    max_x_features: Optional[int] = None,
    # requires ColumnRefModel.source_raw to be effective; otherwise skipped
    max_features_per_source_raw: Optional[int] = None,
) -> List[ValidationIssueModel]:
    """
    Validation #7 (HARD GATE): dimensionality explosion prevention.

    Deterministic checks:
      - total features = |W| + |X| <= max_total_features (if set)
      - |W| <= max_w_features, |X| <= max_x_features (if set)
      - per raw source expansion cap (if source_raw present + max_features_per_source_raw set)

    Why static (non-replaceable by LLM):
      - This is exact counting against hard resource/stability limits. Qualitative reasoning is not sufficient.
    """
    issues: List[ValidationIssueModel] = []

    n_w = len(list(getattr(spec, "w_cols", [])))
    n_x = len(list(getattr(spec, "x_cols", [])))
    total = int(n_w + n_x)

    if max_total_features is not None and total > int(max_total_features):
        issues.append(
            _issue(
                severity="FAIL",
                message="Total feature count exceeds cap (|W|+|X|).",
                evidence={"n_w": int(n_w), "n_x": int(n_x), "total": total, "cap": int(max_total_features)},
                fix_hint="Reduce dimensionality: cap one-hot levels, use hashing/frequency encoding, or drop weak features.",
            )
        )

    if max_w_features is not None and n_w > int(max_w_features):
        issues.append(
            _issue(
                severity="FAIL",
                message="W feature count exceeds cap.",
                evidence={"n_w": int(n_w), "cap": int(max_w_features)},
                fix_hint="Reduce W dimensionality: cap one-hot levels, collapse categories, or drop low-value covariates.",
            )
        )

    if max_x_features is not None and n_x > int(max_x_features):
        issues.append(
            _issue(
                severity="FAIL",
                message="X feature count exceeds cap.",
                evidence={"n_x": int(n_x), "cap": int(max_x_features)},
                fix_hint="Reduce X dimensionality or move some effect modifiers to W-only.",
            )
        )

    if max_features_per_source_raw is not None:
        # Group by source_raw across W/X if available on ColumnRefModel
        def _iter_role_cols(role_name: str) -> List[Any]:
            role_obj = getattr(spec, role_name, None)
            if role_obj is None:
                return []
            cols = getattr(role_obj, "columns", None)
            return list(cols) if cols else []

        counts: DefaultDict[str, int] = defaultdict(int)
        sample_cols: DefaultDict[str, List[str]] = defaultdict(list)

        for role_name in ("w", "x"):
            for cref in _iter_role_cols(role_name):
                src = getattr(cref, "source_raw", None)
                if not src:
                    continue
                col = str(getattr(cref, "name"))
                counts[str(src)] += 1
                if len(sample_cols[str(src)]) < 10:
                    sample_cols[str(src)].append(col)

        too_big: List[Dict[str, Any]] = [
            {"source_raw": k, "n_features": int(v), "cols_sample": sample_cols[k]}
            for k, v in counts.items()
            if int(v) > int(max_features_per_source_raw)
        ]
        if too_big:
            issues.append(
                _issue(
                    severity="FAIL",
                    message="Per-source_raw feature expansion exceeds cap.",
                    evidence={"cap": int(max_features_per_source_raw), "violations_sample": too_big[:25]},
                    fix_hint="Cap one-hot levels per raw column, use hashing/frequency encoding, or group rare categories.",
                )
            )

    return issues


# =============================================================================
# Validation #8: Encoding post-condition sanity (minmax/standardize/epoch/log1p) (HARD FAIL for bounds)
# =============================================================================
def validate_encoding_postconditions(
    *,
    df_after: pd.DataFrame,
    spec: TransformedProtocolSpec,
    minmax_tol: float = 1e-6,
    zscore_warn_abs: float = 10.0,
    zscore_fail_abs: Optional[float] = None,
    # epoch plausibility bounds (seconds since Unix epoch) — conservative defaults
    epoch_min: int = int(pd.Timestamp("1900-01-01").value // 10**9),
    epoch_max: int = int(pd.Timestamp("2100-01-01").value // 10**9),
) -> List[ValidationIssueModel]:
    """
    Validation #8: enforce deterministic post-conditions for known encodings.

    This validator only triggers if ColumnRefModel has an `encoding` attribute.
    If not present in your spec model, it becomes a no-op (safe).

    Deterministic checks:
      - encoding == "minmax": values within [0,1] ± tol   (FAIL)
      - encoding == "standardize": finite; warn/fail on extreme |z|
      - encoding == "datetime_to_epoch_seconds": values within plausible range (WARN/FAIL)
      - encoding == "log1p": finite (already covered elsewhere); optional WARN if output has strong negatives

    Why static (non-replaceable by LLM):
      - Post-conditions are mathematical. Must be verified via global min/max/extremes over the real artifact.
    """
    issues: List[ValidationIssueModel] = []

    def _iter_all_role_cols() -> List[Any]:
        out: List[Any] = []
        for role_name in ("y", "t", "w", "x"):
            role_obj = getattr(spec, role_name, None)
            if role_obj is None:
                continue
            cols = getattr(role_obj, "columns", None)
            if cols:
                out.extend(list(cols))
        return out

    cols_with_encoding: List[Any] = []
    for cref in _iter_all_role_cols():
        enc = getattr(cref, "encoding", None)
        if enc:
            cols_with_encoding.append(cref)

    if not cols_with_encoding:
        return issues  # spec doesn't carry encoding; nothing to validate

    minmax_bad: List[Dict[str, Any]] = []
    z_warn: List[Dict[str, Any]] = []
    z_fail: List[Dict[str, Any]] = []
    epoch_out: List[Dict[str, Any]] = []
    log1p_neg: List[Dict[str, Any]] = []

    for cref in cols_with_encoding:
        col = str(getattr(cref, "name"))
        enc = str(getattr(cref, "encoding"))
        if col not in df_after.columns:
            continue

        s = df_after[col]
        if not (pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype)):
            # dtype issues handled by #2; skip
            continue

        x = s.to_numpy()
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        xf = x.astype(float, copy=False)

        if not np.all(np.isfinite(xf)):
            # Non-finite can be hard-failed elsewhere if you want; keep focused here.
            continue

        if enc == "minmax":
            mn = float(np.min(xf))
            mx = float(np.max(xf))
            if mn < -float(minmax_tol) or mx > 1.0 + float(minmax_tol):
                minmax_bad.append(
                    {
                        "column": col,
                        "min": mn,
                        "max": mx,
                        "tol": float(minmax_tol),
                        "feature_kind": str(getattr(cref, "feature_kind", "unknown")),
                    }
                )

        elif enc == "standardize":
            max_abs = float(np.max(np.abs(xf)))
            if zscore_fail_abs is not None and max_abs > float(zscore_fail_abs):
                z_fail.append({"column": col, "max_abs": max_abs, "fail_abs": float(zscore_fail_abs)})
            elif max_abs > float(zscore_warn_abs):
                z_warn.append({"column": col, "max_abs": max_abs, "warn_abs": float(zscore_warn_abs)})

        elif enc == "datetime_to_epoch_seconds":
            mn = int(np.min(xf))
            mx = int(np.max(xf))
            if mn < int(epoch_min) or mx > int(epoch_max):
                epoch_out.append(
                    {
                        "column": col,
                        "min": mn,
                        "max": mx,
                        "epoch_min": int(epoch_min),
                        "epoch_max": int(epoch_max),
                    }
                )

        elif enc == "log1p":
            # log1p output can be negative if original x in (-1,0).
            # Not always wrong, but often indicates bad parsing (e.g., negative lab values).
            mn = float(np.min(xf))
            if mn < -1e-6:
                log1p_neg.append({"column": col, "min": mn})

    if minmax_bad:
        issues.append(
            _issue(
                severity="FAIL",
                message="minmax-encoded columns violate [0,1] bounds.",
                evidence={"violations_sample": minmax_bad[:25]},
                fix_hint="Bug in min/max computation or wrong column stats used. Enforce correct minmax parameters.",
            )
        )

    if z_fail:
        issues.append(
            _issue(
                severity="FAIL",
                message="standardize-encoded columns have absurd |z| values (hard threshold exceeded).",
                evidence={"violations_sample": z_fail[:25]},
                fix_hint="Likely near-zero variance, wrong units, or incorrect standardization parameters.",
            )
        )

    if z_warn:
        issues.append(
            _issue(
                severity="WARN",
                message="standardize-encoded columns have very large |z| values (distribution sanity warning).",
                evidence={"warnings_sample": z_warn[:25]},
                fix_hint="Check for outliers, unit errors, or incorrect scaling; consider robust scaling or clipping.",
            )
        )

    if epoch_out:
        # Range plausibility is domain-dependent; default WARN.
        issues.append(
            _issue(
                severity="WARN",
                message="datetime_to_epoch_seconds columns fall outside plausible time range.",
                evidence={"violations_sample": epoch_out[:25]},
                fix_hint="Check datetime parsing/timezone/unit errors (seconds vs ms) or unrealistic dates in source.",
            )
        )

    if log1p_neg:
        issues.append(
            _issue(
                severity="WARN",
                message="log1p-encoded columns have negative outputs (possible negative inputs).",
                evidence={"warnings_sample": log1p_neg[:25]},
                fix_hint="If negatives are not expected, inspect raw values and parsing. Otherwise, this can be OK.",
            )
        )

    return issues