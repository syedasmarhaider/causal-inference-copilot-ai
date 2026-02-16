from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Final, List,  Mapping, Optional, Sequence, cast
from uuid import UUID

import numpy as np
import pandas as pd

from python.domain.repo.data_repo import DataRepo
from python.workflows.state.control_state import ACTION
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState, ConversationStateHelpers
from python.workflows.state.inference_ready_state import InferenceReadyState
from python.workflows.state.validate_inference_ready_state import InferenceReadyValidationIssue, InferenceReadyValidationReport, ValidationSeverity, ValidationStatus

log = logging.getLogger(__name__)

# =============================================================================
# Thresholds / constants
# =============================================================================

MIN_N_TOTAL_FAIL: Final[int] = 200
MIN_N_ARM_FAIL: Final[int] = 50
MIN_ARM_SHARE_WARN: Final[float] = 0.05

# After *inference-ready* prep, NaNs should be gone in modeling columns.
ALLOW_MISSING_RATE_MODEL_COLS_FAIL: Final[float] = 0.0

# If too many features are constant or treatment-exclusive, it indicates encoding problems / positivity risks.
MAX_CONST_FEATURE_FRAC_WARN: Final[float] = 0.30
MAX_CONST_FEATURE_FRAC_FAIL: Final[float] = 0.60

MAX_EXCLUSIVE_FEATURE_FRAC_WARN: Final[float] = 0.30
MAX_EXCLUSIVE_FEATURE_FRAC_FAIL: Final[float] = 0.60

# Safety: avoid expensive scans on ultra-wide one-hot matrices
MAX_FEATURES_FOR_EXCLUSIVE_SCAN: Final[int] = 5000


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class ValidateInferenceReadyConfig:
    # If True, treat WARN as DONE (no user action required) like your protocol validator.
    warn_is_done: bool = True


# =============================================================================
# Public node factory
# =============================================================================

def make_validate_inference_ready_static_node(*, data_repo: DataRepo) -> CallableNodeFunc:
    """
    Validate *InferenceReadyState* + the prepared dataset it references.

    - No LLM
    - Deterministic
    - Hard-fails on schema mismatches, NaNs/infs in modeling columns, tiny cohorts, missing arms, etc.
    """

    def _run(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        cfg = ValidateInferenceReadyConfig()

        ir = state.get("inference_ready")
        if not isinstance(ir, dict):
            msg = "InferenceReadyState missing; run prepare_inference_ready first."
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NEEDS_INPUT"), msg)

        issues: List[InferenceReadyValidationIssue] = []
        issues.extend(_validate_ir_invariants(cast(InferenceReadyState, ir)))

        prepared_ds_id = _require_prepared_dataset_id(cast(InferenceReadyState, ir))
        if prepared_ds_id is None:
            issues.append(
                _mk_issue(
                    rule_id="PREPARED_DATASET_ID_MISSING",
                    severity="FAIL",
                    message="InferenceReadyState.prepared_dataset.id missing; cannot validate prepared data.",
                    evidence={"prepared_dataset": ir.get("prepared_dataset")},
                    fix_hint="Ensure prepare_inference_ready saves a prepared DatasetState with a valid id.",
                )
            )
            report = _build_report(issues=issues, metrics={})
            _attach_report(state, report)
            msg = _render_report_text(report)
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NEEDS_INPUT"), msg)

        df = _load_prepared_df(data_repo=data_repo, user_id=user_id, conversation_id=conversation_id, dataset_id=prepared_ds_id)
        if df is None or df.empty:
            issues.append(
                _mk_issue(
                    rule_id="PREPARED_DF_LOAD_FAIL",
                    severity="FAIL",
                    message="Failed to load prepared dataset (or it is empty).",
                    evidence={"prepared_dataset_id": str(prepared_ds_id)},
                    fix_hint="Verify the prepared dataset was saved and is readable by DataRepo.get_csv_data.",
                )
            )
            report = _build_report(issues=issues, metrics={})
            _attach_report(state, report)
            msg = _render_report_text(report)
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NEEDS_INPUT"), msg)

        # Dataset-backed checks
        issues.extend(_validate_columns_exist(df=df, ir=cast(InferenceReadyState, ir)))
        if _has_fail(issues):
            report = _build_report(issues=issues, metrics=_basic_metrics(df, cast(InferenceReadyState, ir)))
            _attach_report(state, report)
            msg = _render_report_text(report)
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NEEDS_INPUT"), msg)

        issues.extend(_validate_cohort_size(df=df))
        issues.extend(_validate_treatment(df=df, ir=cast(InferenceReadyState, ir)))
        issues.extend(_validate_outcome(df=df, ir=cast(InferenceReadyState, ir)))
        issues.extend(_validate_features(df=df, ir=cast(InferenceReadyState, ir)))

        report = _build_report(issues=issues, metrics=_basic_metrics(df, cast(InferenceReadyState, ir)))
        _attach_report(state, report)

        msg = _render_report_text(report)
        ConversationStateHelpers.append_ai_message(state, msg)

        if report["status"] == "FAIL":
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NEEDS_INPUT"), msg)

        # WARN/PASS
        if report["status"] == "WARN" and not cfg.warn_is_done:
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NEEDS_INPUT"), msg)

        return ConversationStateHelpers.set_done(state, cast(ACTION, "NONE"), msg)

    return _run


# =============================================================================
# Report helpers
# =============================================================================

def _mk_issue(
    *,
    rule_id: str,
    severity: ValidationSeverity,
    message: str,
    evidence: Mapping[str, Any] | None = None,
    fix_hint: str | None = None,
) -> InferenceReadyValidationIssue:
    return {
        "rule_id": rule_id,
        "severity": severity,
        "message": message,
        "evidence": dict(evidence or {}),
        "fix_hint": fix_hint,
    }


def _derive_status(issues: Sequence[InferenceReadyValidationIssue]) -> ValidationStatus:
    if any(i["severity"] == "FAIL" for i in issues):
        return "FAIL"
    if any(i["severity"] == "WARN" for i in issues):
        return "WARN"
    return "PASS"


def _build_report(*, issues: List[InferenceReadyValidationIssue], metrics: Dict[str, Any]) -> InferenceReadyValidationReport:
    return {"status": _derive_status(issues), "issues": issues, "metrics": metrics}


def _has_fail(issues: Sequence[InferenceReadyValidationIssue]) -> bool:
    return any(i["severity"] == "FAIL" for i in issues)


def _render_report_text(report: InferenceReadyValidationReport) -> str:
    status = report["status"]
    issues = report["issues"]

    lines: List[str] = [f"Inference-ready validation: {status}"]

    if not issues:
        lines.append("No issues found.")
        return "\n".join(lines)

    # group by severity, stable order
    fails = [i for i in issues if i["severity"] == "FAIL"]
    warns = [i for i in issues if i["severity"] == "WARN"]

    if fails:
        lines.append("")
        lines.append("FAILURES:")
        for i in fails:
            hint = f" | fix: {i['fix_hint']}" if i.get("fix_hint") else ""
            lines.append(f"- [{i['rule_id']}] {i['message']}{hint}")

    if warns:
        lines.append("")
        lines.append("WARNINGS:")
        for i in warns:
            hint = f" | fix: {i['fix_hint']}" if i.get("fix_hint") else ""
            lines.append(f"- [{i['rule_id']}] {i['message']}{hint}")

    return "\n".join(lines)


def _attach_report(state: ConversationState, report: InferenceReadyValidationReport) -> None:
    """
    You should add this to ConversationState TypedDict:
        inference_ready_validation: InferenceReadyValidationState | None
    """
    state["inference_ready_validation"] = cast(InferenceReadyValidationState, {"report": report})  # type: ignore[typeddict-unknown-key]


# =============================================================================
# Data loading
# =============================================================================

def _require_prepared_dataset_id(ir: InferenceReadyState) -> UUID | None:
    ds = ir.get("prepared_dataset")
    if not isinstance(ds, dict):
        return None
    ds_id = ds.get("id")
    return ds_id if isinstance(ds_id, UUID) else None


def _load_prepared_df(*, data_repo: DataRepo, user_id: UUID, conversation_id: UUID, dataset_id: UUID) -> Optional[pd.DataFrame]:
    try:
        return data_repo.get_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
            limit=None,
        )
    except Exception:
        log.exception("Inference-ready validation: failed to load prepared dataset")
        return None


# =============================================================================
# InferenceReadyState-only invariants (no df)
# =============================================================================

def _validate_ir_invariants(ir: InferenceReadyState) -> List[InferenceReadyValidationIssue]:
    issues: List[InferenceReadyValidationIssue] = []

    err = ir.get("error")
    if isinstance(err, str) and err.strip():
        issues.append(
            _mk_issue(
                rule_id="IR_ERROR_SET",
                severity="FAIL",
                message="InferenceReadyState.error is set.",
                evidence={"error": err},
                fix_hint="Fix preparation step; inference-ready state must be produced without errors.",
            )
        )

    # Feature set consistency
    fs = ir.get("feature_sets")
    if not isinstance(fs, dict):
        issues.append(
            _mk_issue(
                rule_id="FEATURE_SETS_INVALID",
                severity="FAIL",
                message="feature_sets must be a dict.",
                evidence={"feature_sets": fs},
                fix_hint="Set feature_sets={'W':W_cols,'X':X_cols,'XW':X+W stable union}.",
            )
        )
        return issues

    W_cols = ir.get("W_cols", [])
    X_cols = ir.get("X_cols", [])
    if not isinstance(W_cols, list) or not isinstance(X_cols, list):
        issues.append(
            _mk_issue(
                rule_id="WX_COLS_INVALID",
                severity="FAIL",
                message="W_cols and X_cols must be lists.",
                evidence={"W_cols": W_cols, "X_cols": X_cols},
                fix_hint="Ensure preparation emits list[str] for W_cols and X_cols.",
            )
        )
        return issues

    if fs.get("W") != W_cols:
        issues.append(
            _mk_issue(
                rule_id="FEATURE_SETS_W_MISMATCH",
                severity="FAIL",
                message="feature_sets['W'] must equal W_cols.",
                evidence={"feature_sets.W": fs.get("W"), "W_cols": W_cols},
                fix_hint="Recompute feature_sets from W_cols/X_cols after encoding.",
            )
        )

    if fs.get("X") != X_cols:
        issues.append(
            _mk_issue(
                rule_id="FEATURE_SETS_X_MISMATCH",
                severity="FAIL",
                message="feature_sets['X'] must equal X_cols.",
                evidence={"feature_sets.X": fs.get("X"), "X_cols": X_cols},
                fix_hint="Recompute feature_sets from W_cols/X_cols after encoding.",
            )
        )

    expected_xw = _stable_union(X_cols, W_cols)
    if fs.get("XW") != expected_xw:
        issues.append(
            _mk_issue(
                rule_id="FEATURE_SETS_XW_MISMATCH",
                severity="FAIL",
                message="feature_sets['XW'] must be stable union of X_cols then W_cols.",
                evidence={"feature_sets.XW": fs.get("XW"), "expected": expected_xw},
                fix_hint="Set feature_sets['XW'] = X_cols + [w for w in W_cols if w not in X_cols].",
            )
        )

    # T/Y consistency
    tcol = ir.get("T_col")
    ycols = ir.get("Y_cols")

    if not isinstance(tcol, str) or not tcol.strip():
        issues.append(
            _mk_issue(
                rule_id="T_COL_INVALID",
                severity="FAIL",
                message="T_col must be a non-empty string.",
                evidence={"T_col": tcol},
                fix_hint="Set T_col to the prepared treatment column name.",
            )
        )

    if not isinstance(ycols, list) or len(ycols) != 1 or not isinstance(ycols[0], str) or not ycols[0].strip():
        issues.append(
            _mk_issue(
                rule_id="Y_COLS_INVALID",
                severity="FAIL",
                message="Y_cols must be a list[str] of length 1 for v1.",
                evidence={"Y_cols": ycols},
                fix_hint="Set Y_cols=[prepared_outcome_column].",
            )
        )

    return issues


def _stable_union(a: Sequence[str], b: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in list(a) + list(b):
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =============================================================================
# DF-backed validation
# =============================================================================

def _validate_columns_exist(*, df: pd.DataFrame, ir: InferenceReadyState) -> List[InferenceReadyValidationIssue]:
    issues: List[InferenceReadyValidationIssue] = []

    cols = set(map(str, df.columns))

    tcol = ir.get("T_col")
    ycols = ir.get("Y_cols", [])
    W_cols = ir.get("W_cols", [])
    X_cols = ir.get("X_cols", [])

    def _missing(name: str, c: str) -> None:
        issues.append(
            _mk_issue(
                rule_id="COLUMN_MISSING",
                severity="FAIL",
                message=f"Required column missing in prepared dataset: {name}='{c}'.",
                evidence={"column": c},
                fix_hint="Re-run preparation; ensure you keep T/Y/W/X columns and save the prepared dataset.",
            )
        )

    if isinstance(tcol, str) and tcol not in cols:
        _missing("T_col", tcol)

    for c in ycols:
        if isinstance(c, str) and c not in cols:
            _missing("Y_col", c)

    for c in W_cols:
        if isinstance(c, str) and c not in cols:
            _missing("W_col", c)

    for c in X_cols:
        if isinstance(c, str) and c not in cols:
            _missing("X_col", c)

    return issues


def _validate_cohort_size(*, df: pd.DataFrame) -> List[InferenceReadyValidationIssue]:
    n = int(df.shape[0])
    if n < MIN_N_TOTAL_FAIL:
        return [
            _mk_issue(
                rule_id="N_TOO_SMALL",
                severity="FAIL",
                message="Too few rows after preparation for causal modeling.",
                evidence={"n_rows": n, "min_required": MIN_N_TOTAL_FAIL},
                fix_hint="Broaden cohort (relax exclusions) or use a larger dataset.",
            )
        ]
    return []


def _validate_treatment(*, df: pd.DataFrame, ir: InferenceReadyState) -> List[InferenceReadyValidationIssue]:
    issues: List[InferenceReadyValidationIssue] = []

    t = ir.get("treatment")
    tcol = ir.get("T_col")

    if not isinstance(t, dict) or not isinstance(tcol, str) or tcol not in df.columns:
        return issues

    kind = t.get("kind")
    s = df[tcol]

    # No NaNs / infs allowed for model columns in inference-ready data
    issues.extend(_validate_no_nan_inf(df=df, cols=[tcol], label="treatment"))

    if kind == "binary":
        v = pd.to_numeric(s, errors="coerce")
        uniq = sorted(set(v.dropna().astype(int).unique().tolist()))
        if uniq and any(x not in (0, 1) for x in uniq):
            issues.append(
                _mk_issue(
                    rule_id="T_BINARY_NOT_01",
                    severity="FAIL",
                    message="Binary treatment must be encoded as 0/1 in prepared data.",
                    evidence={"T_col": tcol, "unique_values": uniq[:20]},
                    fix_hint="Normalize treatment to 0/1 during preparation.",
                )
            )
            return issues

        n_t = int((v == 1).sum())
        n_c = int((v == 0).sum())
        if n_t == 0 or n_c == 0:
            issues.append(
                _mk_issue(
                    rule_id="T_ARM_EMPTY",
                    severity="FAIL",
                    message="One treatment arm has zero rows after preparation.",
                    evidence={"n_treated": n_t, "n_control": n_c},
                    fix_hint="Fix treatment inclusion mapping or relax exclusions.",
                )
            )
            return issues

        if n_t < MIN_N_ARM_FAIL or n_c < MIN_N_ARM_FAIL:
            issues.append(
                _mk_issue(
                    rule_id="T_ARM_TOO_SMALL",
                    severity="FAIL",
                    message="One treatment arm is too small after preparation.",
                    evidence={"n_treated": n_t, "n_control": n_c, "min_arm": MIN_N_ARM_FAIL},
                    fix_hint="Broaden cohort or redefine treatment to increase arm sizes.",
                )
            )

        share = float(n_t / (n_t + n_c))
        if share < MIN_ARM_SHARE_WARN or share > (1.0 - MIN_ARM_SHARE_WARN):
            issues.append(
                _mk_issue(
                    rule_id="T_ARM_IMBALANCE",
                    severity="WARN",
                    message="Treatment is highly imbalanced; overlap and estimates may be unstable.",
                    evidence={"treated_share": share, "n_treated": n_t, "n_control": n_c},
                    fix_hint="Consider trimming, redefining treatment, or improving overlap features.",
                )
            )

        return issues

    if kind == "continuous":
        v = pd.to_numeric(s, errors="coerce")
        if int(v.notna().sum()) == 0:
            issues.append(
                _mk_issue(
                    rule_id="T_CONT_NONNUM",
                    severity="FAIL",
                    message="Continuous treatment has no numeric values after preparation.",
                    evidence={"T_col": tcol},
                    fix_hint="Ensure treatment is numeric and coercible to float during preparation.",
                )
            )
            return issues
        if int(v.nunique(dropna=True)) <= 1:
            issues.append(
                _mk_issue(
                    rule_id="T_CONT_CONSTANT",
                    severity="FAIL",
                    message="Continuous treatment has <=1 unique value after preparation.",
                    evidence={"T_col": tcol},
                    fix_hint="Choose a treatment with variability or broaden cohort.",
                )
            )
        return issues

    if kind == "categorical":
        v = pd.to_numeric(s, errors="coerce")
        if int(v.notna().sum()) == 0:
            issues.append(
                _mk_issue(
                    rule_id="T_CAT_NONNUM",
                    severity="FAIL",
                    message="Categorical treatment must be encoded as integer codes after preparation.",
                    evidence={"T_col": tcol, "dtype": str(s.dtype)},
                    fix_hint="Encode categorical treatment to ordinal integer codes during preparation.",
                )
            )
            return issues

        nunq = int(v.nunique(dropna=True))
        if nunq < 2:
            issues.append(
                _mk_issue(
                    rule_id="T_CAT_TOO_FEW_LEVELS",
                    severity="FAIL",
                    message="Categorical treatment has <2 levels present after preparation.",
                    evidence={"n_unique_levels": nunq},
                    fix_hint="Adjust included_levels or relax exclusions.",
                )
            )
            return issues

        # Warn if some levels tiny (cannot enforce per-level min like binary; but still warn)
        counts = v.value_counts(dropna=True)
        small = {int(k): int(val) for k, val in counts.items() if int(val) < MIN_N_ARM_FAIL}
        if small:
            issues.append(
                _mk_issue(
                    rule_id="T_CAT_SMALL_LEVELS",
                    severity="WARN",
                    message="Some categorical treatment levels have small counts; estimates may be unstable.",
                    evidence={"small_levels": small},
                    fix_hint="Merge rare treatment levels or increase cohort size.",
                )
            )
        return issues

    # Unknown kind
    issues.append(
        _mk_issue(
            rule_id="T_KIND_UNKNOWN",
            severity="FAIL",
            message="Unknown treatment.kind in InferenceReadyState.",
            evidence={"treatment.kind": kind},
            fix_hint="Set treatment.kind to 'binary'|'continuous'|'categorical'.",
        )
    )
    return issues


def _validate_outcome(*, df: pd.DataFrame, ir: InferenceReadyState) -> List[InferenceReadyValidationIssue]:
    issues: List[InferenceReadyValidationIssue] = []

    o = ir.get("outcome")
    ycols = ir.get("Y_cols", [])

    if not isinstance(o, dict) or not isinstance(ycols, list) or len(ycols) != 1:
        return issues

    ycol = ycols[0]
    if not isinstance(ycol, str) or ycol not in df.columns:
        return issues

    kind = o.get("kind")
    s = df[ycol]

    issues.extend(_validate_no_nan_inf(df=df, cols=[ycol], label="outcome"))

    if kind == "binary":
        v = pd.to_numeric(s, errors="coerce")
        uniq = sorted(set(v.dropna().astype(int).unique().tolist()))
        if uniq and any(x not in (0, 1) for x in uniq):
            issues.append(
                _mk_issue(
                    rule_id="Y_BINARY_NOT_01",
                    severity="FAIL",
                    message="Binary outcome must be encoded as 0/1 in prepared data.",
                    evidence={"Y_col": ycol, "unique_values": uniq[:20]},
                    fix_hint="Normalize outcome to 0/1 during preparation.",
                )
            )
            return issues

        n_e = int((v == 1).sum())
        n_ne = int((v == 0).sum())
        if n_e == 0 or n_ne == 0:
            issues.append(
                _mk_issue(
                    rule_id="Y_CLASS_EMPTY",
                    severity="FAIL",
                    message="One outcome class has zero rows after preparation.",
                    evidence={"n_event": n_e, "n_non_event": n_ne},
                    fix_hint="Fix outcome inclusion mapping or relax exclusions.",
                )
            )
        return issues

    if kind == "continuous":
        v = pd.to_numeric(s, errors="coerce")
        if int(v.notna().sum()) == 0:
            issues.append(
                _mk_issue(
                    rule_id="Y_CONT_NONNUM",
                    severity="FAIL",
                    message="Continuous outcome has no numeric values after preparation.",
                    evidence={"Y_col": ycol},
                    fix_hint="Ensure outcome is numeric and coercible to float during preparation.",
                )
            )
            return issues
        if int(v.nunique(dropna=True)) <= 1:
            issues.append(
                _mk_issue(
                    rule_id="Y_CONT_CONSTANT",
                    severity="WARN",
                    message="Continuous outcome has <=1 unique value; effect estimation may be degenerate.",
                    evidence={"Y_col": ycol},
                    fix_hint="Verify outcome definition; choose an outcome with variability.",
                )
            )
        return issues

    if kind == "categorical":
        v = pd.to_numeric(s, errors="coerce")
        nunq = int(v.nunique(dropna=True))
        if nunq < 2:
            issues.append(
                _mk_issue(
                    rule_id="Y_CAT_TOO_FEW_LEVELS",
                    severity="FAIL",
                    message="Categorical outcome has <2 levels present after preparation.",
                    evidence={"n_unique_levels": nunq},
                    fix_hint="Adjust included_levels or relax exclusions.",
                )
            )
        return issues

    issues.append(
        _mk_issue(
            rule_id="Y_KIND_UNKNOWN",
            severity="FAIL",
            message="Unknown outcome.kind in InferenceReadyState.",
            evidence={"outcome.kind": kind},
            fix_hint="Set outcome.kind to 'binary'|'continuous'|'categorical'.",
        )
    )
    return issues


def _validate_features(*, df: pd.DataFrame, ir: InferenceReadyState) -> List[InferenceReadyValidationIssue]:
    issues: List[InferenceReadyValidationIssue] = []

    W_cols = ir.get("W_cols", [])
    X_cols = ir.get("X_cols", [])
    if not isinstance(W_cols, list) or not isinstance(X_cols, list):
        return issues

    feats = _stable_union([c for c in X_cols if isinstance(c, str)], [c for c in W_cols if isinstance(c, str)])
    if not feats:
        # observational w/ no W is handled earlier in protocol stage; here we just warn
        issues.append(
            _mk_issue(
                rule_id="NO_FEATURES",
                severity="WARN",
                message="No W/X features available after preparation. Models may be unstable or unadjusted.",
                evidence={},
                fix_hint="Add covariates/effect modifiers or verify preparation did not drop them.",
            )
        )
        return issues

    # Must be numeric after prep (one-hot -> int8, numeric -> float64)
    non_numeric: List[str] = []
    for c in feats:
        if c not in df.columns:
            continue
        dt = df[c].dtype
        if not (np.issubdtype(dt, np.number) or str(dt).lower().startswith("bool")):
            non_numeric.append(c)

    if non_numeric:
        issues.append(
            _mk_issue(
                rule_id="FEATURE_NON_NUMERIC",
                severity="FAIL",
                message="All W/X features must be numeric after preparation (EconML expects numeric arrays).",
                evidence={"non_numeric_cols": non_numeric[:50], "count": len(non_numeric)},
                fix_hint="One-hot encode categorical features and coerce numeric features to float during preparation.",
            )
        )
        return issues

    # No NaN/Inf in feature columns
    issues.extend(_validate_no_nan_inf(df=df, cols=feats, label="features"))

    # Constant feature diagnostics
    const_cols = _constant_columns(df=df, cols=feats, max_scan=MAX_FEATURES_FOR_EXCLUSIVE_SCAN)
    if const_cols:
        frac = float(len(const_cols) / max(1, len(feats)))
        sev: ValidationSeverity = "WARN"
        if frac >= MAX_CONST_FEATURE_FRAC_FAIL:
            sev = "FAIL"
        elif frac >= MAX_CONST_FEATURE_FRAC_WARN:
            sev = "WARN"
        else:
            sev = "WARN"

        issues.append(
            _mk_issue(
                rule_id="FEATURES_CONSTANT",
                severity=sev,
                message="Many feature columns are constant after preparation (adds no signal; may indicate encoding issues).",
                evidence={"n_constant": len(const_cols), "n_features": len(feats), "fraction": frac},
                fix_hint="Drop constant columns; verify one-hot encoding and cohort filtering.",
            )
        )

    # Treatment-exclusive feature diagnostics (binary treatment only)
    t = ir.get("treatment")
    tcol = ir.get("T_col")
    if isinstance(t, dict) and t.get("kind") == "binary" and isinstance(tcol, str) and tcol in df.columns:
        ex_cols = _treatment_exclusive_features(df=df, tcol=tcol, feat_cols=feats, max_scan=MAX_FEATURES_FOR_EXCLUSIVE_SCAN)
        if ex_cols:
            frac2 = float(len(ex_cols) / max(1, len(feats)))
            sev2: ValidationSeverity = "WARN"
            if frac2 >= MAX_EXCLUSIVE_FEATURE_FRAC_FAIL:
                sev2 = "FAIL"
            elif frac2 >= MAX_EXCLUSIVE_FEATURE_FRAC_WARN:
                sev2 = "WARN"

            issues.append(
                _mk_issue(
                    rule_id="FEATURES_TREATMENT_EXCLUSIVE",
                    severity=sev2,
                    message="Many features are effectively treatment-exclusive (positivity / overlap risk).",
                    evidence={"n_exclusive": len(ex_cols), "n_features": len(feats), "fraction": frac2},
                    fix_hint="Consider trimming/overlap checks, redefining cohort/treatment, or removing post-treatment variables.",
                )
            )

    return issues


def _validate_no_nan_inf(*, df: pd.DataFrame, cols: Sequence[str], label: str) -> List[InferenceReadyValidationIssue]:
    issues: List[InferenceReadyValidationIssue] = []

    present = [c for c in cols if isinstance(c, str) and c in df.columns]
    if not present:
        return issues

    miss = {c: float(df[c].isna().mean()) for c in present if float(df[c].isna().mean()) > 0.0}
    if miss:
        worst = max(miss.values())
        if worst > ALLOW_MISSING_RATE_MODEL_COLS_FAIL:
            issues.append(
                _mk_issue(
                    rule_id=f"{label.upper()}_MISSING_NOT_ALLOWED",
                    severity="FAIL",
                    message=f"Prepared {label} columns contain missing values; inference-ready data should have none.",
                    evidence={"missing_rates": dict(sorted(miss.items(), key=lambda kv: -kv[1]) )},
                    fix_hint="Impute W/X and filter/encode T/Y so required columns have no missing values.",
                )
            )

    # Inf/-Inf checks on numeric cols only
    bad_inf: List[str] = []
    for c in present:
        dt = df[c].dtype
        if np.issubdtype(dt, np.number):
            x = df[c].to_numpy(dtype="float64", copy=False)
            if np.isfinite(x).all() is False:
                bad_inf.append(c)

    if bad_inf:
        issues.append(
            _mk_issue(
                rule_id=f"{label.upper()}_NONFINITE",
                severity="FAIL",
                message=f"Prepared {label} columns contain inf/-inf; models will crash.",
                evidence={"nonfinite_cols": bad_inf[:50], "count": len(bad_inf)},
                fix_hint="Fix transforms (e.g., log on non-positive) and clip values during preparation.",
            )
        )

    return issues


def _constant_columns(*, df: pd.DataFrame, cols: Sequence[str], max_scan: int) -> List[str]:
    # cheap: limit scan to avoid huge cost on ultra-wide matrices
    scan = list(cols[:max_scan])
    out: List[str] = []
    for c in scan:
        if c not in df.columns:
            continue
        nunq = int(df[c].nunique(dropna=True))
        if nunq <= 1:
            out.append(c)
    return out


def _treatment_exclusive_features(*, df: pd.DataFrame, tcol: str, feat_cols: Sequence[str], max_scan: int) -> List[str]:
    """
    For binary treatment (0/1), flag feature columns where:
      - feature==1 occurs only in one arm (or feature has non-zero only in one arm),
    which can create overlap/positivity issues for strata defined by that feature.
    """
    t = pd.to_numeric(df[tcol], errors="coerce")
    mask_t = t == 1
    mask_c = t == 0

    if int(mask_t.sum()) == 0 or int(mask_c.sum()) == 0:
        return []

    scan = list(feat_cols[:max_scan])
    ex: List[str] = []

    for c in scan:
        if c not in df.columns:
            continue
        dt = df[c].dtype
        if not np.issubdtype(dt, np.number):
            continue

        x = pd.to_numeric(df[c], errors="coerce").fillna(0.0) # pyright: ignore[reportUnknownMemberType]
        # for one-hot, this is counts of ones; for numeric, counts of non-zero
        nz_t = int((x[mask_t] != 0).sum())
        nz_c = int((x[mask_c] != 0).sum())
        if (nz_t == 0 and nz_c > 0) or (nz_c == 0 and nz_t > 0):
            ex.append(c)

    return ex


def _basic_metrics(df: pd.DataFrame, ir: InferenceReadyState) -> Dict[str, Any]:
    tcol = ir.get("T_col")
    ycols = ir.get("Y_cols", [])
    W_cols = ir.get("W_cols", [])
    X_cols = ir.get("X_cols", [])
    feats = _stable_union([c for c in X_cols], [c for c in W_cols])

    m: Dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "T_col": tcol,
        "Y_cols": list(ycols) if isinstance(ycols, list) else ycols, # pyright: ignore[reportUnknownArgumentType]
        "n_W": len(W_cols) ,
        "n_X": len(X_cols),
        "n_features_XW": len(feats),
    }

    # counts for binary treatment if applicable
    t = ir.get("treatment")
    if t.get("kind") == "binary" and tcol in df.columns:
        tv = pd.to_numeric(df[tcol], errors="coerce")
        m["n_treated"] = int((tv == 1).sum())
        m["n_control"] = int((tv == 0).sum())
        denom = m["n_treated"] + m["n_control"]
        m["treated_share"] = float(m["n_treated"] / denom) if denom > 0 else None

    return m
