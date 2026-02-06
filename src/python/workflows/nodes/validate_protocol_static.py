from __future__ import annotations

import json
import logging
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence, Tuple, cast
from uuid import UUID

import pandas as pd

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMConfig, LLMService
from python.workflows.nodes.prompts.validate_protocol_static import static_validation_message_prompt
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState, ConversationStateHelpers
from python.workflows.state.control_state import ACTION
from python.workflows.state.protocol_state import (
    ALLOWED_OPS,
    ALLOWED_OUT_KINDS,
    ALLOWED_TIME_ZERO,
    ALLOWED_TREAT_KINDS,
    ALLOWED_UNITS,
    ExclusionRule,
    ProtocolState,
)
from python.workflows.state.validate_protocol_state import (
    ProtocolStaticValidationState,
    ProtocolValidationIssue,
    ProtocolValidationReport,
    ValidationSeverity,
    ValidationStatus,
)

log = logging.getLogger(__name__)

# =============================================================================
# Thresholds / constants
# =============================================================================

MAX_COVARIATE_MISSING_FAIL: Final[float] = 0.50
MAX_COVARIATE_MISSING_WARN: Final[float] = 0.20

MAX_OUTCOME_MISSING_FAIL: Final[float] = 0.05
MAX_TREATMENT_MISSING_FAIL: Final[float] = 0.05

MIN_N_TOTAL_FAIL: Final[int] = 200
MIN_N_ARM_FAIL: Final[int] = 50
MIN_ARM_SHARE_WARN: Final[float] = 0.05  # warn if treated share <5% or >95%

# =============================================================================
# Exclusion normalization (defensive against compiler inversion)
# =============================================================================

SENTINEL_STRINGS: Final[set[str]] = {
    "unknown",
    "unk",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "missing",
    "",
}


def _is_sentinel_value(v: str) -> bool:
    return v.strip().lower() in SENTINEL_STRINGS


def _normalize_exclusion_rule(ex: ExclusionRule) -> Tuple[ExclusionRule, Optional[ProtocolValidationIssue]]:
    """
    Semantics in this node:
        - ExclusionRule builds a mask of rows to REMOVE, then we drop ~mask.

    Common compiler/LLM inversion:
        - User intent: "exclude Unknown"
        - Compiler emits: op='!=' values=['Unknown']  (this actually means "remove NOT Unknown", i.e., keep only Unknown)
      We normalize that to:
        - op='in' values=['Unknown'] (remove Unknown rows), and emit a WARN issue.
    """
    col = str(ex.get("column", ""))
    op_raw = str(ex.get("op", ""))
    values_raw = ex.get("values", [])
    reason = str(ex.get("reason", ""))

    values: List[str] = []
    if isinstance(values_raw, list):
        values = [str(v) for v in values_raw]

    op_applied = op_raw
    issue: Optional[ProtocolValidationIssue] = None

    # Normalize "!= Unknown" -> "in Unknown"  (exclude sentinel rows)
    if op_raw == "!=" and any(_is_sentinel_value(v) for v in values):
        op_applied = "in"
        issue = _mk_issue(
            rule_id="EXCL_OP_NORMALIZED",
            severity="WARN",
            message="Exclusion operator normalized: '!= <sentinel>' interpreted as 'exclude <sentinel>'.",
            evidence={"column": col, "op_raw": op_raw, "op_applied": op_applied, "values": values},
            fix_hint="Fix compiler: for 'exclude Unknown', emit op='in' values=['Unknown'] (or op='==').",
        )

    # Normalize "not_in Unknown" -> "in Unknown" (same failure mode: would keep only Unknown)
    if op_raw == "not_in" and values and all(_is_sentinel_value(v) for v in values):
        op_applied = "in"
        issue = issue or _mk_issue(
            rule_id="EXCL_OP_NORMALIZED",
            severity="WARN",
            message="Exclusion operator normalized: 'not_in <sentinel>' interpreted as 'exclude <sentinel>'.",
            evidence={"column": col, "op_raw": op_raw, "op_applied": op_applied, "values": values},
            fix_hint="Fix compiler: for 'exclude Unknown', emit op='in' values=['Unknown'] (or op='==').",
        )

    ex2: ExclusionRule = {
        "column": col,
        "op": cast(Any, op_applied),
        "values": values,
        "reason": reason,
    }
    return ex2, issue


# =============================================================================
# Issue helpers (single job: create report issues)
# =============================================================================


def _mk_issue(
    *,
    rule_id: str,
    severity: ValidationSeverity,
    message: str,
    evidence: Mapping[str, Any] | None = None,
    fix_hint: str | None = None,
) -> ProtocolValidationIssue:
    return {
        "rule_id": rule_id,
        "severity": severity,
        "message": message,
        "evidence": dict(evidence or {}),
        "fix_hint": fix_hint,
    }


def _derive_status(issues: Sequence[ProtocolValidationIssue]) -> ValidationStatus:
    if any(i["severity"] == "FAIL" for i in issues):
        return "FAIL"
    if any(i["severity"] == "WARN" for i in issues):
        return "WARN"
    return "PASS"


def _build_report(*, issues: List[ProtocolValidationIssue], metrics: Dict[str, Any]) -> ProtocolValidationReport:
    return {
        "status": cast(Any, _derive_status(issues)),
        "issues": issues,
        "metrics": metrics,
    }


# =============================================================================
# Public factory
# =============================================================================


def make_validate_protocol_static_node(*, data_repo: DataRepo, llm: LLMService, model_name: str) -> CallableNodeFunc:
    def _run(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        protocol = state.get("protocol")
        if protocol is None:
            msg = "ProtocolState missing; cannot run static validation."
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

        dataset_id = _require_dataset_id(state)
        if dataset_id is None:
            msg = "Dataset missing; load dataset first."
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

        df0 = _load_df(
            data_repo=data_repo,
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
        )
        if df0 is None:
            msg = "Failed to load dataset for validation."
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

        # Apply exclusions (filtering)
        exclusions = _safe_exclusions(protocol)
        df1, excl_issues, excl_metrics = _apply_exclusions(df0, exclusions)

        issues: List[ProtocolValidationIssue] = []
        issues.extend(_validate_protocol_enums(protocol))
        issues.extend(excl_issues)
        issues.extend(_validate_nonempty_after_exclusions(df_before=df0, df_after=df1, excl_metrics=excl_metrics))
        issues.extend(_validate_required_columns_exist(df=df1, protocol=protocol))
        issues.extend(_validate_missingness(df=df1, protocol=protocol))
        issues.extend(_validate_treatment_distribution(df=df1, protocol=protocol))
        issues.extend(_validate_outcome_distribution(df=df1, protocol=protocol))
        issues.extend(_validate_covariates(df=df1, protocol=protocol))
        issues.extend(_validate_effect_modifiers(df=df1, protocol=protocol))

        metrics = _build_metrics(df_before=df0, df_after=df1, protocol=protocol, excl_metrics=excl_metrics)
        report = _build_report(issues=issues, metrics=metrics)

        state["protocol_static_validation"] = cast(ProtocolStaticValidationState, {"report": report})

        if report["status"] == "PASS":
            msg = "Static validation passed. Proceeding."
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_done(state, cast(ACTION, "NONE"), msg)

        user_msg = _llm_render_validation_message(llm=llm, model_name=model_name, report=report)
        ConversationStateHelpers.append_ai_message(state, user_msg)
        return ConversationStateHelpers.set_abort(state, cast(ACTION, "NEEDS_INPUT"), user_msg)

    return _run


# =============================================================================
# Dataset helpers
# =============================================================================


def _require_dataset_id(state: ConversationState) -> UUID | None:
    ds = state.get("dataset") or {}
    ds_id = ds.get("id")
    return ds_id if isinstance(ds_id, UUID) else None


def _load_df(*, data_repo: DataRepo, user_id: UUID, conversation_id: UUID, dataset_id: UUID) -> Optional[pd.DataFrame]:
    try:
        return data_repo.get_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
            limit=1_000_000,
        )
    except Exception:
        log.exception("Static validation: failed to load dataset")
        return None


# =============================================================================
# Validation: protocol enums (single job)
# =============================================================================


def _validate_protocol_enums(protocol: ProtocolState) -> List[ProtocolValidationIssue]:
    out: List[ProtocolValidationIssue] = []

    tz = protocol.get("time_zero_type")
    if tz not in ALLOWED_TIME_ZERO:
        out.append(
            _mk_issue(
                rule_id="ENUM_TIMEZERO",
                severity="FAIL",
                message="Invalid time_zero_type enum.",
                evidence={"value": tz, "allowed": sorted(ALLOWED_TIME_ZERO)},
                fix_hint="Set time_zero_type to 'COLUMN' or 'CONCEPTUAL'.",
            )
        )

    twu = protocol.get("treatment_window_unit")
    if twu not in ALLOWED_UNITS:
        out.append(
            _mk_issue(
                rule_id="ENUM_TREAT_UNIT",
                severity="FAIL",
                message="Invalid treatment_window_unit enum.",
                evidence={"value": twu, "allowed": sorted(ALLOWED_UNITS)},
                fix_hint="Set treatment_window_unit to a valid WindowUnit.",
            )
        )

    owu = protocol.get("outcome_window_unit")
    if owu not in ALLOWED_UNITS:
        out.append(
            _mk_issue(
                rule_id="ENUM_OUTCOME_UNIT",
                severity="FAIL",
                message="Invalid outcome_window_unit enum.",
                evidence={"value": owu, "allowed": sorted(ALLOWED_UNITS)},
                fix_hint="Set outcome_window_unit to a valid WindowUnit.",
            )
        )

    # NOTE: conceptual time zero -> ignore time logic.
    # So: do NOT fail on duration outcomes here.

    return out


# =============================================================================
# Exclusions: apply filters (single job)
# =============================================================================


def _safe_exclusions(protocol: ProtocolState) -> List[ExclusionRule]:
    ex = protocol.get("exclusions") or []
    if not isinstance(ex, list):
        return []
    return [cast(ExclusionRule, e) for e in ex if isinstance(e, dict)]


def _apply_exclusions(
    df: pd.DataFrame, exclusions: Sequence[ExclusionRule]
) -> Tuple[pd.DataFrame, List[ProtocolValidationIssue], Dict[str, Any]]:
    out_df = df
    issues: List[ProtocolValidationIssue] = []
    rules_metrics: List[Dict[str, Any]] = []

    for i, ex in enumerate(exclusions):
        ex_norm, norm_issue = _normalize_exclusion_rule(cast(ExclusionRule, ex))
        if norm_issue is not None:
            issues.append(norm_issue)

        col = str(ex_norm.get("column", ""))
        op = str(ex_norm.get("op", ""))
        values = ex_norm.get("values", [])
        reason = str(ex_norm.get("reason", ""))

        if col not in out_df.columns:
            issues.append(
                _mk_issue(
                    rule_id="EXCL_COL_MISSING",
                    severity="FAIL",
                    message="Exclusion column missing; cannot apply exclusion rule.",
                    evidence={"index": i, "column": col},
                    fix_hint="Fix exclusions[].column to match an existing dataset column.",
                )
            )
            continue

        if op not in ALLOWED_OPS:
            issues.append(
                _mk_issue(
                    rule_id="EXCL_OP_INVALID",
                    severity="FAIL",
                    message="Exclusion operator invalid; cannot apply exclusion rule.",
                    evidence={"index": i, "op": op, "allowed": sorted(ALLOWED_OPS)},
                    fix_hint="Fix exclusions[].op to a supported operator.",
                )
            )
            continue

        before = int(out_df.shape[0])
        mask = _build_exclusion_mask(out_df, col=col, op=op, values=values)
        out_df = out_df.loc[~mask].copy()
        after = int(out_df.shape[0])

        values_list = values if isinstance(values, list) else []
        rules_metrics.append(
            {
                "index": i,
                "column": col,
                "op_raw": str(ex.get("op", "")),
                "op_applied": op,
                "values": [str(v) for v in values_list],
                "reason": reason,
                "removed": before - after,
                "n_after": after,
            }
        )

    metrics = {"n_before": int(df.shape[0]), "n_after": int(out_df.shape[0]), "rules": rules_metrics}
    return out_df, issues, metrics  # pyright: ignore[reportUnknownVariableType]


def _build_exclusion_mask(df: pd.DataFrame, *, col: str, op: str, values: Any) -> pd.Series:
    s = df[col]

    if op == "is_null":
        return s.isna()
    if op == "not_null":
        return ~s.isna()

    vals: List[str] = [str(v) for v in values] if isinstance(values, list) else []

    if op in (">=", "<=", ">", "<"):
        x = pd.to_numeric(s, errors="coerce")
        v = float(vals[0]) if vals else float("nan")
        if op == ">=":
            return x >= v
        if op == "<=":
            return x <= v
        if op == ">":
            return x > v
        return x < v

    ss = s.astype(str)

    if op in ("==", "in"):
        return ss.isin(vals)
    if op in ("!=", "not_in"):
        return ~ss.isin(vals)

    return pd.Series([False] * df.shape[0], index=df.index)


# =============================================================================
# Validation: cohort size after exclusions (single job)
# =============================================================================


def _validate_nonempty_after_exclusions(
    *, df_before: pd.DataFrame, df_after: pd.DataFrame, excl_metrics: Dict[str, Any]
) -> List[ProtocolValidationIssue]:
    n0 = int(df_before.shape[0])
    n1 = int(df_after.shape[0])

    if n1 <= 0:
        return [
            _mk_issue(
                rule_id="EMPTY_AFTER_EXCL",
                severity="FAIL",
                message="All rows removed after exclusions; cohort is empty.",
                evidence={"n_before": n0, "n_after": n1, "exclusions": excl_metrics},
                fix_hint="Relax exclusions or confirm the target population definition.",
            )
        ]

    if n1 < MIN_N_TOTAL_FAIL:
        return [
            _mk_issue(
                rule_id="N_TOO_SMALL",
                severity="FAIL",
                message="Too few rows after exclusions for causal modeling.",
                evidence={"n_after": n1, "min_required": MIN_N_TOTAL_FAIL, "exclusions": excl_metrics},
                fix_hint="Relax exclusions, broaden the population, or use a larger dataset.",
            )
        ]

    return []


# =============================================================================
# Validation: required columns exist (single job)
# =============================================================================


def _validate_required_columns_exist(*, df: pd.DataFrame, protocol: ProtocolState) -> List[ProtocolValidationIssue]:
    cols = set(map(str, df.columns))
    out: List[ProtocolValidationIssue] = []

    # treatment_spec
    ts = protocol.get("treatment_spec")
    if not isinstance(ts, dict):
        out.append(
            _mk_issue(
                rule_id="TREAT_SPEC_MISSING",
                severity="FAIL",
                message="treatment_spec missing or invalid.",
                evidence={},
                fix_hint="Compile protocol again to produce treatment_spec.",
            )
        )
    else:
        tkind = ts.get("kind")
        tcol = ts.get("column")
        if not isinstance(tkind, str) or tkind not in ALLOWED_TREAT_KINDS:
            out.append(
                _mk_issue(
                    rule_id="TREAT_KIND_INVALID",
                    severity="FAIL",
                    message="Invalid treatment_spec.kind.",
                    evidence={"value": tkind, "allowed": sorted(ALLOWED_TREAT_KINDS)},
                    fix_hint="Set treatment_spec.kind to 'binary'|'continuous'|'categorical'.",
                )
            )
        if not isinstance(tcol, str) or not tcol:
            out.append(
                _mk_issue(
                    rule_id="TREAT_COL_INVALID",
                    severity="FAIL",
                    message="treatment_spec.column must be a non-empty string.",
                    evidence={"value": tcol},
                    fix_hint="Set treatment_spec.column to an existing dataset column name.",
                )
            )
        elif tcol not in cols:
            out.append(
                _mk_issue(
                    rule_id="TREAT_COL_MISSING",
                    severity="FAIL",
                    message="Treatment column not found in dataset.",
                    evidence={"column": tcol},
                    fix_hint="Choose an existing column for treatment_spec.column.",
                )
            )

    # outcome_spec
    ys = protocol.get("outcome_spec")
    if not isinstance(ys, dict):
        out.append(
            _mk_issue(
                rule_id="OUT_SPEC_MISSING",
                severity="FAIL",
                message="outcome_spec missing or invalid.",
                evidence={},
                fix_hint="Compile protocol again to produce outcome_spec.",
            )
        )
        return out

    ykind = ys.get("kind")
    if not isinstance(ykind, str) or ykind not in ALLOWED_OUT_KINDS:
        out.append(
            _mk_issue(
                rule_id="OUT_KIND_INVALID",
                severity="FAIL",
                message="Invalid outcome_spec.kind.",
                evidence={"value": ykind, "allowed": sorted(ALLOWED_OUT_KINDS)},
                fix_hint="Set outcome_spec.kind to 'binary'|'continuous'|'categorical'|'duration'.",
            )
        )
        return out

    if ykind == "duration":
        dcol = ys.get("duration_column")
        ecol = ys.get("event_column")
        if not isinstance(dcol, str) or not dcol:
            out.append(
                _mk_issue(
                    rule_id="OUT_DUR_COL_INVALID",
                    severity="FAIL",
                    message="Duration outcome_spec requires duration_column.",
                    evidence={"duration_column": dcol},
                    fix_hint="Set outcome_spec.duration_column to an existing dataset column.",
                )
            )
        elif dcol not in cols:
            out.append(
                _mk_issue(
                    rule_id="OUT_DUR_COL_MISSING",
                    severity="FAIL",
                    message="Duration column not found in dataset.",
                    evidence={"duration_column": dcol},
                    fix_hint="Choose an existing column for outcome_spec.duration_column.",
                )
            )

        if not isinstance(ecol, str) or not ecol:
            out.append(
                _mk_issue(
                    rule_id="OUT_EVENT_COL_INVALID",
                    severity="FAIL",
                    message="Duration outcome_spec requires event_column.",
                    evidence={"event_column": ecol},
                    fix_hint="Set outcome_spec.event_column to an existing dataset column.",
                )
            )
        elif ecol not in cols:
            out.append(
                _mk_issue(
                    rule_id="OUT_EVENT_COL_MISSING",
                    severity="FAIL",
                    message="Event indicator column not found in dataset.",
                    evidence={"event_column": ecol},
                    fix_hint="Choose an existing column for outcome_spec.event_column.",
                )
            )
    else:
        ycol = ys.get("column")
        if not isinstance(ycol, str) or not ycol:
            out.append(
                _mk_issue(
                    rule_id="OUT_COL_INVALID",
                    severity="FAIL",
                    message="outcome_spec.column must be a non-empty string.",
                    evidence={"value": ycol},
                    fix_hint="Set outcome_spec.column to an existing dataset column name.",
                )
            )
        elif ycol not in cols:
            out.append(
                _mk_issue(
                    rule_id="OUT_COL_MISSING",
                    severity="FAIL",
                    message="Outcome column not found in dataset.",
                    evidence={"column": ycol},
                    fix_hint="Choose an existing column for outcome_spec.column.",
                )
            )

    # covariates + effect modifiers
    for c in protocol.get("covariates", []):
        if c not in cols:
            out.append(
                _mk_issue(
                    rule_id="COV_COL_MISSING",
                    severity="FAIL",
                    message="Covariate column not found in dataset.",
                    evidence={"column": c},
                    fix_hint="Remove the covariate or map it to an existing dataset column.",
                )
            )

    for c in protocol.get("effect_modifiers", []):
        if c not in cols:
            out.append(
                _mk_issue(
                    rule_id="EM_COL_MISSING",
                    severity="FAIL",
                    message="Effect modifier column not found in dataset.",
                    evidence={"column": c},
                    fix_hint="Remove the effect modifier or map it to an existing dataset column.",
                )
            )

    return out


# =============================================================================
# Validation: missingness thresholds (single job)
# =============================================================================


def _validate_missingness(*, df: pd.DataFrame, protocol: ProtocolState) -> List[ProtocolValidationIssue]:
    out: List[ProtocolValidationIssue] = []

    # treatment missingness
    ts = protocol.get("treatment_spec")
    if isinstance(ts, dict):
        tcol = ts.get("column")
        if isinstance(tcol, str) and tcol in df.columns:
            miss = float(df[tcol].isna().mean())
            if miss > MAX_TREATMENT_MISSING_FAIL:
                out.append(
                    _mk_issue(
                        rule_id="TREAT_MISSING_HIGH",
                        severity="FAIL",
                        message="Treatment missingness too high.",
                        evidence={"column": tcol, "missing_rate": miss, "threshold": MAX_TREATMENT_MISSING_FAIL},
                        fix_hint="Impute treatment, change treatment definition, or drop rows with missing treatment.",
                    )
                )

    # outcome missingness
    ys = protocol.get("outcome_spec")
    if isinstance(ys, dict):
        ykind = ys.get("kind")
        if ykind == "duration":
            dcol = ys.get("duration_column")
            ecol = ys.get("event_column")
            if isinstance(dcol, str) and dcol in df.columns:
                miss_d = float(df[dcol].isna().mean())
                if miss_d > MAX_OUTCOME_MISSING_FAIL:
                    out.append(
                        _mk_issue(
                            rule_id="OUT_DURATION_MISSING_HIGH",
                            severity="FAIL",
                            message="Duration missingness too high.",
                            evidence={"column": dcol, "missing_rate": miss_d, "threshold": MAX_OUTCOME_MISSING_FAIL},
                            fix_hint="Impute/drop missing duration or redefine outcome.",
                        )
                    )
            if isinstance(ecol, str) and ecol in df.columns:
                miss_e = float(df[ecol].isna().mean())
                if miss_e > MAX_OUTCOME_MISSING_FAIL:
                    out.append(
                        _mk_issue(
                            rule_id="OUT_EVENT_MISSING_HIGH",
                            severity="FAIL",
                            message="Event indicator missingness too high.",
                            evidence={"column": ecol, "missing_rate": miss_e, "threshold": MAX_OUTCOME_MISSING_FAIL},
                            fix_hint="Impute/drop missing event indicator or redefine outcome.",
                        )
                    )
        else:
            ycol = ys.get("column")
            if isinstance(ycol, str) and ycol in df.columns:
                miss = float(df[ycol].isna().mean())
                if miss > MAX_OUTCOME_MISSING_FAIL:
                    out.append(
                        _mk_issue(
                            rule_id="OUT_MISSING_HIGH",
                            severity="FAIL",
                            message="Outcome missingness too high.",
                            evidence={"column": ycol, "missing_rate": miss, "threshold": MAX_OUTCOME_MISSING_FAIL},
                            fix_hint="Impute outcome, change outcome definition, or drop rows with missing outcome.",
                        )
                    )

    # covariates missingness
    for c in protocol.get("covariates", []):
        if c in df.columns:
            miss = float(df[c].isna().mean())
            if miss > MAX_COVARIATE_MISSING_FAIL:
                out.append(
                    _mk_issue(
                        rule_id="COV_MISSING_HIGH",
                        severity="FAIL",
                        message="Covariate missingness too high.",
                        evidence={"column": c, "missing_rate": miss, "threshold": MAX_COVARIATE_MISSING_FAIL},
                        fix_hint="Impute covariate, drop covariate, or restrict to rows with observed covariate.",
                    )
                )
            elif miss > MAX_COVARIATE_MISSING_WARN:
                out.append(
                    _mk_issue(
                        rule_id="COV_MISSING_WARN",
                        severity="WARN",
                        message="Covariate missingness is substantial.",
                        evidence={"column": c, "missing_rate": miss, "threshold": MAX_COVARIATE_MISSING_WARN},
                        fix_hint="Consider imputation or dropping this covariate.",
                    )
                )

    return out


# =============================================================================
# Validation: treatment distribution (single job)
# =============================================================================


def _validate_treatment_distribution(*, df: pd.DataFrame, protocol: ProtocolState) -> List[ProtocolValidationIssue]:
    ts = protocol.get("treatment_spec")
    if not isinstance(ts, dict):
        return []

    kind = ts.get("kind")
    col = ts.get("column")
    if not isinstance(kind, str) or not isinstance(col, str) or col not in df.columns:
        return []

    if kind == "binary":
        return _validate_binary_treatment(df=df, col=col, ts=ts)
    if kind == "categorical":
        return _validate_categorical_treatment(df=df, col=col, ts=ts)
    if kind == "continuous":
        return _validate_continuous_treatment(df=df, col=col)
    return []


def _validate_binary_treatment(*, df: pd.DataFrame, col: str, ts: Mapping[str, Any]) -> List[ProtocolValidationIssue]:
    treated = ts.get("treated")
    control = ts.get("control")
    if not isinstance(treated, str) or not isinstance(control, str):
        return [
            _mk_issue(
                rule_id="TREAT_BINARY_INCOMPLETE",
                severity="FAIL",
                message="Binary treatment_spec requires treated/control strings.",
                evidence={"treated": treated, "control": control},
                fix_hint="Set treatment_spec.treated and treatment_spec.control.",
            )
        ]

    vals = df[col].astype(str)
    n = int(df.shape[0])
    n_t = int((vals == treated).sum())
    n_c = int((vals == control).sum())

    if n_t == 0 or n_c == 0:
        return [
            _mk_issue(
                rule_id="TREAT_LEVEL_MISSING",
                severity="FAIL",
                message="One treatment arm has zero rows after exclusions.",
                evidence={"n_total": n, "treated": treated, "n_treated": n_t, "control": control, "n_control": n_c},
                fix_hint="Check treated/control labels or relax exclusions.",
            )
        ]

    out: List[ProtocolValidationIssue] = []
    if n_t < MIN_N_ARM_FAIL or n_c < MIN_N_ARM_FAIL:
        out.append(
            _mk_issue(
                rule_id="ARM_TOO_SMALL",
                severity="FAIL",
                message="One arm too small for causal modeling.",
                evidence={"n_total": n, "n_treated": n_t, "n_control": n_c, "min_arm": MIN_N_ARM_FAIL},
                fix_hint="Broaden the cohort or use a larger dataset.",
            )
        )

    share_t = (n_t / n) if n > 0 else 0.0
    if share_t < MIN_ARM_SHARE_WARN or share_t > (1.0 - MIN_ARM_SHARE_WARN):
        out.append(
            _mk_issue(
                rule_id="ARM_IMBALANCE",
                severity="WARN",
                message="Treatment assignment is highly imbalanced; estimates may be unstable.",
                evidence={"n_total": n, "n_treated": n_t, "n_control": n_c, "treated_share": share_t},
                fix_hint="Consider re-defining treatment or using trimming/overlap checks later.",
            )
        )

    return out


def _validate_categorical_treatment(*, df: pd.DataFrame, col: str, ts: Mapping[str, Any]) -> List[ProtocolValidationIssue]:
    levels = ts.get("levels")
    if not isinstance(levels, list) or len(levels) < 2 or any(not isinstance(x, str) for x in levels):  # type: ignore
        return [
            _mk_issue(
                rule_id="TREAT_CAT_LEVELS",
                severity="FAIL",
                message="Categorical treatment_spec requires levels: list[str] len>=2.",
                evidence={"levels": levels},
                fix_hint="Provide >=2 valid levels in treatment_spec.levels.",
            )
        ]

    vals = df[col].astype(str)
    counts = {lvl: int((vals == lvl).sum()) for lvl in levels}
    present = {k: v for k, v in counts.items() if v > 0}

    if len(present) < 2:
        return [
            _mk_issue(
                rule_id="TREAT_CAT_PRESENT",
                severity="FAIL",
                message="Fewer than 2 treatment levels present after exclusions.",
                evidence={"counts": counts},
                fix_hint="Adjust levels or relax exclusions.",
            )
        ]

    small = {k: v for k, v in present.items() if v < MIN_N_ARM_FAIL}
    if small:
        return [
            _mk_issue(
                rule_id="TREAT_CAT_SMALL_ARMS",
                severity="WARN",
                message="Some treatment levels have small counts; estimates may be unstable.",
                evidence={"small_counts": small},
                fix_hint="Consider merging levels or increasing cohort size.",
            )
        ]

    return []


def _validate_continuous_treatment(*, df: pd.DataFrame, col: str) -> List[ProtocolValidationIssue]:
    x = pd.to_numeric(df[col], errors="coerce")
    if int(x.notna().sum()) == 0:
        return [
            _mk_issue(
                rule_id="TREAT_CONT_NONNUM",
                severity="FAIL",
                message="Continuous treatment has no numeric values after coercion.",
                evidence={"column": col},
                fix_hint="Use a numeric treatment column or change treatment_spec.kind.",
            )
        ]

    if int(x.nunique(dropna=True)) <= 1:
        return [
            _mk_issue(
                rule_id="TREAT_CONT_CONSTANT",
                severity="FAIL",
                message="Continuous treatment has <=1 unique numeric value after exclusions.",
                evidence={"column": col},
                fix_hint="Choose a treatment column with variability.",
            )
        ]

    return []


# =============================================================================
# Validation: outcome distribution (single job)
# =============================================================================


def _validate_outcome_distribution(*, df: pd.DataFrame, protocol: ProtocolState) -> List[ProtocolValidationIssue]:
    ys = protocol.get("outcome_spec")
    if not isinstance(ys, dict):
        return []

    kind = ys.get("kind")
    if not isinstance(kind, str) or kind not in ALLOWED_OUT_KINDS:
        return []

    if kind == "duration":
        return _validate_duration_outcome(df=df, ys=ys)

    col = ys.get("column")
    if not isinstance(col, str) or col not in df.columns:
        return []

    if kind == "binary":
        return _validate_binary_outcome(df=df, col=col, ys=ys)
    if kind == "categorical":
        return _validate_categorical_outcome(df=df, col=col, ys=ys)
    if kind == "continuous":
        return _validate_continuous_outcome(df=df, col=col)

    return []


def _validate_binary_outcome(*, df: pd.DataFrame, col: str, ys: Mapping[str, Any]) -> List[ProtocolValidationIssue]:
    event = ys.get("event")
    non_event = ys.get("non_event")
    if not isinstance(event, str) or not isinstance(non_event, str):
        return [
            _mk_issue(
                rule_id="OUT_BINARY_INCOMPLETE",
                severity="FAIL",
                message="Binary outcome_spec requires event/non_event strings.",
                evidence={"event": event, "non_event": non_event},
                fix_hint="Set outcome_spec.event and outcome_spec.non_event.",
            )
        ]

    vals = df[col].astype(str)
    n_e = int((vals == event).sum())
    n_ne = int((vals == non_event).sum())

    if n_e == 0 or n_ne == 0:
        return [
            _mk_issue(
                rule_id="OUT_LEVEL_MISSING",
                severity="FAIL",
                message="One outcome class has zero rows after exclusions.",
                evidence={"event": event, "n_event": n_e, "non_event": non_event, "n_non_event": n_ne},
                fix_hint="Check event/non_event labels or relax exclusions.",
            )
        ]

    return []


def _validate_categorical_outcome(*, df: pd.DataFrame, col: str, ys: Mapping[str, Any]) -> List[ProtocolValidationIssue]:
    levels = ys.get("levels")
    if not isinstance(levels, list) or len(levels) < 2 or any(not isinstance(x, str) for x in levels):
        return [
            _mk_issue(
                rule_id="OUT_CAT_LEVELS",
                severity="FAIL",
                message="Categorical outcome_spec requires levels: list[str] len>=2.",
                evidence={"levels": levels},
                fix_hint="Provide >=2 valid levels in outcome_spec.levels.",
            )
        ]

    vals = df[col].astype(str)
    counts = {lvl: int((vals == lvl).sum()) for lvl in levels}
    present = {k: v for k, v in counts.items() if v > 0}

    if len(present) < 2:
        return [
            _mk_issue(
                rule_id="OUT_CAT_PRESENT",
                severity="FAIL",
                message="Fewer than 2 outcome levels present after exclusions.",
                evidence={"counts": counts},
                fix_hint="Adjust levels or relax exclusions.",
            )
        ]

    return []


def _validate_continuous_outcome(*, df: pd.DataFrame, col: str) -> List[ProtocolValidationIssue]:
    y = pd.to_numeric(df[col], errors="coerce")
    if int(y.notna().sum()) == 0:
        return [
            _mk_issue(
                rule_id="OUT_CONT_NONNUM",
                severity="FAIL",
                message="Continuous outcome has no numeric values after coercion.",
                evidence={"column": col},
                fix_hint="Use a numeric outcome column or change outcome_spec.kind.",
            )
        ]

    if int(y.nunique(dropna=True)) <= 1:
        return [
            _mk_issue(
                rule_id="OUT_CONT_CONSTANT",
                severity="WARN",
                message="Continuous outcome has <=1 unique numeric value after exclusions; effect estimation may be degenerate.",
                evidence={"column": col},
                fix_hint="Verify outcome definition; choose an outcome with variability.",
            )
        ]

    return []


def _validate_duration_outcome(*, df: pd.DataFrame, ys: Mapping[str, Any]) -> List[ProtocolValidationIssue]:
    dcol = ys.get("duration_column")
    ecol = ys.get("event_column")
    ev = ys.get("event_value")
    cv = ys.get("censor_value")

    if not isinstance(dcol, str) or not isinstance(ecol, str) or dcol not in df.columns or ecol not in df.columns:
        return []

    out: List[ProtocolValidationIssue] = []

    d = pd.to_numeric(df[dcol], errors="coerce")
    if int(d.notna().sum()) == 0:
        out.append(
            _mk_issue(
                rule_id="OUT_DUR_NONNUM",
                severity="FAIL",
                message="Duration column has no numeric values after coercion.",
                evidence={"duration_column": dcol},
                fix_hint="Use a numeric duration column.",
            )
        )
    else:
        nonpos = int((d <= 0).sum())
        if nonpos > 0:
            out.append(
                _mk_issue(
                    rule_id="OUT_DUR_NONPOS",
                    severity="WARN",
                    message="Duration column contains non-positive values.",
                    evidence={"duration_column": dcol, "non_positive_count": nonpos},
                    fix_hint="Check units/encoding; consider filtering non-positive durations.",
                )
            )

    if isinstance(ev, str) and isinstance(cv, str):
        s = df[ecol].astype(str)
        n_ev = int((s == ev).sum())
        n_cv = int((s == cv).sum())
        if n_ev == 0 or n_cv == 0:
            out.append(
                _mk_issue(
                    rule_id="OUT_EVENT_LABELS_MISSING",
                    severity="FAIL",
                    message="Event/censor labels not both present after exclusions.",
                    evidence={"event_value": ev, "n_event": n_ev, "censor_value": cv, "n_censor": n_cv},
                    fix_hint="Fix outcome_spec.event_value/censor_value to match dataset encoding.",
                )
            )

    return out


# =============================================================================
# Validation: covariate sanity (single job)
# =============================================================================


def _validate_covariates(*, df: pd.DataFrame, protocol: ProtocolState) -> List[ProtocolValidationIssue]:
    covs = protocol.get("covariates", [])
    exp_type = str(protocol.get("experiment_type", "Observational"))

    if not covs:
        # Only warn if observational (for RCT, covariates can be optional)
        if exp_type.lower().startswith("obs"):
            return [
                _mk_issue(
                    rule_id="NO_COVARIATES",
                    severity="WARN",
                    message="No covariates specified. Observational estimates will be vulnerable to confounding.",
                    evidence={"covariates_count": 0},
                    fix_hint="Add baseline confounders as covariates (W) or justify RCT/randomization.",
                )
            ]
        return []

    out: List[ProtocolValidationIssue] = []
    for c in covs:
        if c in df.columns:
            nunq = int(df[c].nunique(dropna=True))
            if nunq <= 1:
                out.append(
                    _mk_issue(
                        rule_id="COV_CONSTANT",
                        severity="WARN",
                        message="Covariate has <=1 unique value after exclusions; it adds no adjustment power.",
                        evidence={"column": c, "n_unique": nunq},
                        fix_hint="Drop constant covariates.",
                    )
                )

    return out


# =============================================================================
# Validation: effect modifiers sanity (single job)
# =============================================================================


def _validate_effect_modifiers(*, df: pd.DataFrame, protocol: ProtocolState) -> List[ProtocolValidationIssue]:
    em = protocol.get("effect_modifiers", [])
    if not em:
        return []

    out: List[ProtocolValidationIssue] = []
    for c in em:
        if c in df.columns:
            nunq = int(df[c].nunique(dropna=True))
            if nunq <= 1:
                out.append(
                    _mk_issue(
                        rule_id="EM_CONSTANT",
                        severity="WARN",
                        message="Effect modifier has <=1 unique value after exclusions; it adds no heterogeneity signal.",
                        evidence={"column": c, "n_unique": nunq},
                        fix_hint="Drop constant effect modifiers or choose better heterogeneity features.",
                    )
                )
    return out


# =============================================================================
# Metrics builder (single job)
# =============================================================================


def _build_metrics(
    *, df_before: pd.DataFrame, df_after: pd.DataFrame, protocol: ProtocolState, excl_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "n_before": int(df_before.shape[0]),
        "n_after": int(df_after.shape[0]),
        "exclusions": excl_metrics,
        "experiment_type": protocol.get("experiment_type"),
        "time_zero_type": protocol.get("time_zero_type"),
        "covariates_count": len(protocol.get("covariates", [])),
        "effect_modifiers_count": len(protocol.get("effect_modifiers", [])),
    }

    ts = protocol.get("treatment_spec")
    if isinstance(ts, dict):
        m["treatment_kind"] = ts.get("kind")
        tcol = ts.get("column")
        if isinstance(tcol, str) and tcol in df_after.columns:
            m["treatment_missing_rate"] = float(df_after[tcol].isna().mean())

    ys = protocol.get("outcome_spec")
    if isinstance(ys, dict):
        m["outcome_kind"] = ys.get("kind")

    return m


# =============================================================================
# LLM: user-facing message rendering (single job)
# =============================================================================


def _llm_render_validation_message(*, llm: LLMService, model_name: str, report: ProtocolValidationReport) -> str:
    prompt = static_validation_message_prompt().replace("{{REPORT_JSON}}", json.dumps(report, ensure_ascii=False))
    cfg = LLMConfig(model=model_name, temperature=0.5)

    try:
        resp = llm.generate(
            config=cfg,
            system_prompt="Be detail to the report. No JSON.",
            user_prompt=prompt,
            history=None,
        )
        text = str(cast(Any, resp).content or "").strip()
        return text or "Static validation produced warnings/failures. Please review the report."
    except Exception:
        log.exception("Static validation message LLM failed")
        return "Static validation produced warnings/failures. Please review the report."
