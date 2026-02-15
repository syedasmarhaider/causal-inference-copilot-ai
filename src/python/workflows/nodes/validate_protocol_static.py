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

# --- Overlap / positivity (static, heuristic; not identification) ---
OVERLAP_NUM_BINS: Final[int] = 10
OVERLAP_MAX_CATEG_LEVELS: Final[int] = 20
OVERLAP_MIN_BIN_COUNT_WARN: Final[int] = 5

# fraction of bins/levels where BOTH arms appear (binary)
OVERLAP_COVERAGE_WARN: Final[float] = 0.70
OVERLAP_COVERAGE_FAIL: Final[float] = 0.50


# =============================================================================
# Public factory
# =============================================================================


def make_validate_protocol_static_node(*, data_repo: DataRepo, llm: LLMService, model_name: str) -> CallableNodeFunc:
    def _run(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        protocol = state.get("protocol")
        if not isinstance(protocol, dict):
            msg = "ProtocolState missing; cannot run static validation."
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

        dataset_id = _require_dataset_id(state)
        if dataset_id is None:
            msg = "Dataset missing; load dataset first."
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

        df_raw = _load_df(data_repo=data_repo, user_id=user_id, conversation_id=conversation_id, dataset_id=dataset_id)
        if df_raw is None:
            msg = "Failed to load dataset for validation."
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

        issues: List[ProtocolValidationIssue] = []
        issues.extend(_validate_protocol_enums(protocol))
        issues.extend(_validate_supported_identification(protocol))

        # Gate: required protocol specs + required columns must exist BEFORE we can drop NA / apply exclusions.
        issues.extend(_validate_required_columns_exist(df=df_raw, protocol=protocol))
        if _has_required_column_failure(issues):
            report = _build_report(
                issues=issues,
                metrics=_build_metrics(
                    df_raw=df_raw,
                    df_after_dropna=df_raw,
                    df_after_exclusions=df_raw,
                    protocol=protocol,
                    dropna_metrics={"status": "SKIP", "reason": "required_columns_missing"},
                    excl_metrics={"status": "SKIP", "reason": "required_columns_missing"},
                    overlap_metrics=None,
                ),
            )
            state["protocol_static_validation"] = cast(ProtocolStaticValidationState, {"report": report})
            user_msg = _llm_render_validation_message(llm=llm, model_name=model_name, report=report)
            ConversationStateHelpers.append_ai_message(state, user_msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NEEDS_INPUT"), user_msg)

        # Step 1: drop TRUE missing values (NaN/None/pd.NA) for required T/Y columns.
        # IMPORTANT: we do NOT treat "Unknown" (or any string sentinel) as missing.
        df_dropna, dropna_issues, dropna_metrics = _drop_required_ty_missing(df_raw, protocol=protocol)
        issues.extend(dropna_issues)

        # Step 2: apply user exclusions exactly as provided (no heuristics, no auto-normalization).
        exclusions = _safe_exclusions(protocol)
        df_excl, excl_issues, excl_metrics = _apply_exclusions(df_dropna, exclusions)
        issues.extend(excl_issues)

        # Cohort viability after exclusions
        issues.extend(
            _validate_nonempty_after_exclusions(
                df_before=df_dropna,
                df_after=df_excl,
                excl_metrics=excl_metrics,
            )
        )

        if _has_hard_cohort_failure(issues):
            report = _build_report(
                issues=issues,
                metrics=_build_metrics(
                    df_raw=df_raw,
                    df_after_dropna=df_dropna,
                    df_after_exclusions=df_excl,
                    protocol=protocol,
                    dropna_metrics=dropna_metrics,
                    excl_metrics=excl_metrics,
                    overlap_metrics=None,
                ),
            )
            state["protocol_static_validation"] = cast(ProtocolStaticValidationState, {"report": report})
            user_msg = _llm_render_validation_message(llm=llm, model_name=model_name, report=report)
            ConversationStateHelpers.append_ai_message(state, user_msg)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NEEDS_INPUT"), user_msg)

        # Distributions / missingness
        issues.extend(_validate_missingness(df=df_excl, protocol=protocol))
        issues.extend(_validate_treatment_distribution(df=df_excl, protocol=protocol))
        issues.extend(_validate_outcome_distribution(df=df_excl, protocol=protocol))
        issues.extend(_validate_covariates(df=df_excl, protocol=protocol))
        issues.extend(_validate_effect_modifiers(df=df_excl, protocol=protocol))

        # Overlap / positivity heuristics (binary/categorical only)
        overlap_issues, overlap_metrics = _validate_overlap(df=df_excl, protocol=protocol)
        issues.extend(overlap_issues)

        report = _build_report(
            issues=issues,
            metrics=_build_metrics(
                df_raw=df_raw,
                df_after_dropna=df_dropna,
                df_after_exclusions=df_excl,
                protocol=protocol,
                dropna_metrics=dropna_metrics,
                excl_metrics=excl_metrics,
                overlap_metrics=overlap_metrics,
            ),
        )
        state["protocol_static_validation"] = cast(ProtocolStaticValidationState, {"report": report})

        if report["status"] in ("FAIL", "WARN"):
            user_msg = _llm_render_validation_message(llm=llm, model_name=model_name, report=report)
            ConversationStateHelpers.append_ai_message(state, user_msg)
            if report["status"] == "FAIL":
                return ConversationStateHelpers.set_abort(state, cast(ACTION, "NEEDS_INPUT"), user_msg)
            return ConversationStateHelpers.set_done(state, cast(ACTION, "NONE"), user_msg)

        msg = "Static validation passed. Proceeding."
        ConversationStateHelpers.append_ai_message(state, msg)
        return ConversationStateHelpers.set_done(state, cast(ACTION, "NONE"), msg)

    return _run


# =============================================================================
# Issue helpers
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
    return {"status": cast(Any, _derive_status(issues)), "issues": issues, "metrics": metrics}


def _has_hard_cohort_failure(issues: Sequence[ProtocolValidationIssue]) -> bool:
    hard = {"EMPTY_AFTER_EXCL", "N_TOO_SMALL"}
    return any(i["severity"] == "FAIL" and i["rule_id"] in hard for i in issues)


def _has_required_column_failure(issues: Sequence[ProtocolValidationIssue]) -> bool:
    blocking = {
        "TREAT_SPEC_MISSING",
        "TREAT_KIND_INVALID",
        "TREAT_COL_INVALID",
        "TREAT_COL_MISSING",
        "OUT_SPEC_MISSING",
        "OUT_KIND_INVALID",
        "OUT_COL_INVALID",
        "OUT_COL_MISSING",
        "OUT_DUR_COL_INVALID",
        "OUT_DUR_COL_MISSING",
        "OUT_EVENT_COL_INVALID",
        "OUT_EVENT_COL_MISSING",
        "COV_COL_MISSING",
        "EM_COL_MISSING",
    }
    return any(i["severity"] == "FAIL" and i["rule_id"] in blocking for i in issues)


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
# Drop NA in required T/Y (and duration/event if duration outcome)
# =============================================================================


def _required_ty_columns(protocol: ProtocolState) -> Tuple[List[str], List[ProtocolValidationIssue]]:
    """
    Returns columns that must be NON-NA for modeling to make sense.

    IMPORTANT:
      - Only true missing values are handled here (NaN/None/pd.NA).
      - Strings like "Unknown" are treated as ordinary values.
    """
    issues: List[ProtocolValidationIssue] = []

    ts = protocol.get("treatment_spec")
    if not isinstance(ts, dict):
        issues.append(
            _mk_issue(
                rule_id="TREAT_SPEC_MISSING",
                severity="FAIL",
                message="treatment_spec missing or invalid.",
                evidence={"treatment_spec": ts},
                fix_hint="Provide treatment_spec with kind and column.",
            )
        )
        return [], issues

    tcol = ts.get("column")
    if not isinstance(tcol, str) or not tcol.strip():
        issues.append(
            _mk_issue(
                rule_id="TREAT_COL_INVALID",
                severity="FAIL",
                message="treatment_spec.column must be a non-empty string.",
                evidence={"value": tcol},
                fix_hint="Set treatment_spec.column to an existing dataset column name.",
            )
        )
        return [], issues

    ys = protocol.get("outcome_spec")
    if not isinstance(ys, dict):
        issues.append(
            _mk_issue(
                rule_id="OUT_SPEC_MISSING",
                severity="FAIL",
                message="outcome_spec missing or invalid.",
                evidence={"outcome_spec": ys},
                fix_hint="Provide outcome_spec with kind and required columns.",
            )
        )
        return [], issues

    ykind = ys.get("kind")
    if ykind not in ALLOWED_OUT_KINDS:
        issues.append(
            _mk_issue(
                rule_id="OUT_KIND_INVALID",
                severity="FAIL",
                message="Invalid outcome_spec.kind.",
                evidence={"value": ykind, "allowed": sorted(ALLOWED_OUT_KINDS)},
                fix_hint="Set outcome_spec.kind to 'binary'|'continuous'|'categorical'|'duration'.",
            )
        )
        return [], issues

    cols: List[str] = [tcol]

    if ykind == "duration":
        dcol = ys.get("duration_column")
        ecol = ys.get("event_column")
        if not isinstance(dcol, str) or not dcol.strip():
            issues.append(
                _mk_issue(
                    rule_id="OUT_DUR_COL_INVALID",
                    severity="FAIL",
                    message="Duration outcome_spec requires duration_column.",
                    evidence={"duration_column": dcol},
                    fix_hint="Set outcome_spec.duration_column to an existing dataset column.",
                )
            )
            return [], issues
        if not isinstance(ecol, str) or not ecol.strip():
            issues.append(
                _mk_issue(
                    rule_id="OUT_EVENT_COL_INVALID",
                    severity="FAIL",
                    message="Duration outcome_spec requires event_column.",
                    evidence={"event_column": ecol},
                    fix_hint="Set outcome_spec.event_column to an existing dataset column.",
                )
            )
            return [], issues
        cols.extend([dcol, ecol])
    else:
        ycol = ys.get("column")
        if not isinstance(ycol, str) or not ycol.strip():
            issues.append(
                _mk_issue(
                    rule_id="OUT_COL_INVALID",
                    severity="FAIL",
                    message="outcome_spec.column must be a non-empty string.",
                    evidence={"value": ycol},
                    fix_hint="Set outcome_spec.column to an existing dataset column name.",
                )
            )
            return [], issues
        cols.append(ycol)

    # de-dup while preserving order
    seen: set[str] = set()
    cols2: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            cols2.append(c)

    return cols2, issues


def _drop_required_ty_missing(
    df: pd.DataFrame,
    *,
    protocol: ProtocolState,
) -> Tuple[pd.DataFrame, List[ProtocolValidationIssue], Dict[str, Any]]:
    required_cols, req_issues = _required_ty_columns(protocol)
    if req_issues:
        # Caller should already have gated required columns, but stay defensive.
        metrics = {"status": "FAIL", "reason": "required_ty_columns_invalid"}
        return df, req_issues, metrics

    # Column existence should be guaranteed by _validate_required_columns_exist, but stay defensive.
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues = [
            _mk_issue(
                rule_id="DROPNA_COL_MISSING",
                severity="FAIL",
                message="Cannot drop missing T/Y because required columns are missing.",
                evidence={"missing_columns": missing},
                fix_hint="Fix treatment/outcome column names to match the dataset.",
            )
        ]
        metrics = {"status": "FAIL", "missing_columns": missing}
        return df, issues, metrics

    n0 = int(df.shape[0])
    out_df = df.dropna(subset=required_cols).copy()
    n1 = int(out_df.shape[0])
    dropped = n0 - n1

    issues: List[ProtocolValidationIssue] = []
    if dropped > 0:
        issues.append(
            _mk_issue(
                rule_id="DROPNA_TY_APPLIED",
                severity="WARN",
                message="Dropped rows with true missing values in required treatment/outcome columns.",
                evidence={"required_cols": required_cols, "n_before": n0, "n_after": n1, "dropped": dropped},
                fix_hint="If you need to keep these rows, impute missingness upstream (not via string sentinels).",
            )
        )

    metrics = {"status": "OK", "required_cols": required_cols, "n_before": n0, "n_after": n1, "dropped": dropped}
    return out_df, issues, metrics


# =============================================================================
# Exclusions (apply exactly as provided)
# =============================================================================


def _safe_exclusions(protocol: ProtocolState) -> List[ExclusionRule]:
    ex = protocol.get("exclusions") or []
    if not isinstance(ex, list):
        return []
    out: List[ExclusionRule] = []
    for e in ex:
        if isinstance(e, dict):
            out.append(cast(ExclusionRule, e))
    return out


def _apply_exclusions(
    df: pd.DataFrame, exclusions: Sequence[ExclusionRule]
) -> Tuple[pd.DataFrame, List[ProtocolValidationIssue], Dict[str, Any]]:
    out_df = df
    issues: List[ProtocolValidationIssue] = []
    rules_metrics: List[Dict[str, Any]] = []

    for i, ex in enumerate(exclusions):
        col = str(ex.get("column", ""))
        op = str(ex.get("op", ""))
        values = ex.get("values", [])
        reason = str(ex.get("reason", ""))

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

        rules_metrics.append(
            {
                "index": i,
                "column": col,
                "op": op,
                "values": _coerce_str_list(values),
                "reason": reason,
                "removed": before - after,
                "n_after": after,
            }
        )

    metrics: Dict[str, Any] = {"n_before": int(df.shape[0]), "n_after": int(out_df.shape[0]), "rules": rules_metrics}
    return out_df, issues, metrics


def _coerce_str_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(v) for v in values]


def _build_exclusion_mask(df: pd.DataFrame, *, col: str, op: str, values: Any) -> pd.Series:
    s = df[col]

    if op == "is_null":
        return s.isna()
    if op == "not_null":
        return ~s.isna()

    vals = _coerce_str_list(values)

    if op in (">=", "<=", ">", "<"):
        if not vals:
            return pd.Series([False] * int(df.shape[0]), index=df.index)
        x = pd.to_numeric(s, errors="coerce")
        try:
            v = float(vals[0])
        except Exception:
            return pd.Series([False] * int(df.shape[0]), index=df.index)

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

    return pd.Series([False] * int(df.shape[0]), index=df.index)


# =============================================================================
# Validation: supported identification (best-effort)
# =============================================================================


def _validate_supported_identification(protocol: ProtocolState) -> List[ProtocolValidationIssue]:
    val = None
    for k in ("identification_strategy", "id_strategy", "identification", "design"):
        if k in protocol:
            val = protocol.get(k)
            break

    if val is None:
        return []

    s = str(val).strip().lower()
    ok = {"backdoor", "unconfoundedness", "ignorability", "rct", "randomized"}
    if s in ok:
        return []

    return [
        _mk_issue(
            rule_id="UNSUPPORTED_IDENTIFICATION",
            severity="FAIL",
            message="Identification strategy is not supported by this copilot (only unconfoundedness/backdoor supported).",
            evidence={"value": val},
            fix_hint="Change to backdoor/unconfoundedness assumptions or use a different pipeline for IV/frontdoor.",
        )
    ]


# =============================================================================
# Validation: protocol enums
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

    return out


# =============================================================================
# Validation: cohort size after exclusions
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
# Validation: required columns exist (defensive + non-crashing)
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
                evidence={"treatment_spec": ts},
                fix_hint="Provide treatment_spec with kind and column.",
            )
        )
        return out

    tkind = ts.get("kind")
    tcol = ts.get("column")

    if tkind not in ALLOWED_TREAT_KINDS:
        out.append(
            _mk_issue(
                rule_id="TREAT_KIND_INVALID",
                severity="FAIL",
                message="Invalid treatment_spec.kind.",
                evidence={"value": tkind, "allowed": sorted(ALLOWED_TREAT_KINDS)},
                fix_hint="Set treatment_spec.kind to 'binary'|'continuous'|'categorical'.",
            )
        )

    if not isinstance(tcol, str) or not tcol.strip():
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
                evidence={"outcome_spec": ys},
                fix_hint="Provide outcome_spec with kind and required columns.",
            )
        )
        return out

    ykind = ys.get("kind")
    if ykind not in ALLOWED_OUT_KINDS:
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

        if not isinstance(dcol, str) or not dcol.strip():
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

        if not isinstance(ecol, str) or not ecol.strip():
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
        if not isinstance(ycol, str) or not ycol.strip():
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

    for c in protocol.get("covariates", []) or []:
        if isinstance(c, str) and c not in cols:
            out.append(
                _mk_issue(
                    rule_id="COV_COL_MISSING",
                    severity="FAIL",
                    message="Covariate column not found in dataset.",
                    evidence={"column": c},
                    fix_hint="Remove the covariate or map it to an existing dataset column.",
                )
            )

    for c in protocol.get("effect_modifiers", []) or []:
        if isinstance(c, str) and c not in cols:
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
# Validation: missingness thresholds
# =============================================================================


def _validate_missingness(*, df: pd.DataFrame, protocol: ProtocolState) -> List[ProtocolValidationIssue]:
    out: List[ProtocolValidationIssue] = []

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

    for c in protocol.get("covariates", []) or []:
        if isinstance(c, str) and c in df.columns:
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
# Validation: treatment distribution
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
                fix_hint="Consider redefining treatment or using trimming/overlap checks later.",
            )
        )

    return out


def _validate_categorical_treatment(*, df: pd.DataFrame, col: str, ts: Mapping[str, Any]) -> List[ProtocolValidationIssue]:
    levels = ts.get("levels")
    if not isinstance(levels, list) or len(levels) < 2 or any(not isinstance(x, str) for x in levels):
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
# Validation: outcome distribution
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
                message="Continuous outcome has <=1 unique value after exclusions; effect estimation may be degenerate.",
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
# Validation: covariates / effect modifiers sanity
# =============================================================================


def _validate_covariates(*, df: pd.DataFrame, protocol: ProtocolState) -> List[ProtocolValidationIssue]:
    covs = protocol.get("covariates", []) or []
    exp_type = str(protocol.get("experiment_type", "Observational"))

    if not covs:
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
        if isinstance(c, str) and c in df.columns:
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


def _validate_effect_modifiers(*, df: pd.DataFrame, protocol: ProtocolState) -> List[ProtocolValidationIssue]:
    em = protocol.get("effect_modifiers", []) or []
    if not em:
        return []

    out: List[ProtocolValidationIssue] = []
    for c in em:
        if isinstance(c, str) and c in df.columns:
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
# Validation: overlap / positivity heuristics (binary + categorical only)
# =============================================================================


def _validate_overlap(*, df: pd.DataFrame, protocol: ProtocolState) -> Tuple[List[ProtocolValidationIssue], Dict[str, Any]]:
    ts = protocol.get("treatment_spec")
    if not isinstance(ts, dict):
        return [], {"status": "SKIP", "reason": "treatment_spec missing"}

    kind = ts.get("kind")
    tcol = ts.get("column")
    if not isinstance(kind, str) or not isinstance(tcol, str) or tcol not in df.columns:
        return [], {"status": "SKIP", "reason": "treatment_spec invalid or column missing"}

    if kind not in ("binary", "categorical"):
        return [], {"status": "SKIP", "reason": f"not_applicable kind={kind}"}

    covs = [c for c in (protocol.get("covariates", []) or []) if isinstance(c, str)]
    ems = [c for c in (protocol.get("effect_modifiers", []) or []) if isinstance(c, str)]
    strata_cols = [c for c in (covs + ems) if c in df.columns]

    if not strata_cols:
        issues = [
            _mk_issue(
                rule_id="OVERLAP_NO_STRATA",
                severity="WARN",
                message="No covariates/effect modifiers available to run overlap heuristics.",
                evidence={},
                fix_hint="Add baseline covariates/effect modifiers to enable overlap diagnostics.",
            )
        ]
        return issues, {"status": cast(Any, _derive_status(issues)), "reason": "no_strata_cols"}

    t = df[tcol].astype(str)

    issues: List[ProtocolValidationIssue] = []
    per_feature: List[Dict[str, Any]] = []

    if kind == "binary":
        treated = ts.get("treated")
        control = ts.get("control")
        if not isinstance(treated, str) or not isinstance(control, str):
            return [], {"status": "SKIP", "reason": "treated/control missing"}

        mask_t = t == treated
        mask_c = t == control
        if int(mask_t.sum()) == 0 or int(mask_c.sum()) == 0:
            return [], {"status": "SKIP", "reason": "one_arm_empty"}

        for col in strata_cols:
            feat_issues, feat_metrics = _overlap_feature_binary(df=df, col=col, mask_t=mask_t, mask_c=mask_c)
            issues.extend(feat_issues)
            per_feature.append(feat_metrics)

    else:
        levels = ts.get("levels")
        if not isinstance(levels, list) or len(levels) < 2 or any(not isinstance(x, str) for x in levels):
            return [], {"status": "SKIP", "reason": "levels invalid"}

        present_levels = [lvl for lvl in levels if int((t == lvl).sum()) > 0]
        if len(present_levels) < 2:
            return [], {"status": "SKIP", "reason": "fewer_than_2_levels_present"}

        for col in strata_cols:
            feat_issues, feat_metrics = _overlap_feature_multitreat(df=df, col=col, t=t, levels=present_levels)
            issues.extend(feat_issues)
            per_feature.append(feat_metrics)

    status = cast(Any, _derive_status(issues))
    metrics: Dict[str, Any] = {
        "status": status,
        "treatment_column": tcol,
        "treatment_kind": kind,
        "checked_cols": strata_cols,
        "per_feature": per_feature,
    }
    return issues, metrics


def _overlap_feature_binary(
    *,
    df: pd.DataFrame,
    col: str,
    mask_t: pd.Series,
    mask_c: pd.Series,
) -> Tuple[List[ProtocolValidationIssue], Dict[str, Any]]:
    s = df[col]
    nunq = int(s.nunique(dropna=True))
    is_low_card = nunq <= OVERLAP_MAX_CATEG_LEVELS

    x = pd.to_numeric(s, errors="coerce")
    numeric_usable = int(x.notna().sum()) > 0 and (int(x.nunique(dropna=True)) > 1)

    if numeric_usable and not (s.dtype == object and is_low_card):
        valid = x.notna()
        x2 = x[valid]
        mt = mask_t[valid]
        mc = mask_c[valid]

        if int(x2.shape[0]) < 10:
            return [], {"col": col, "type": "numeric", "status": "SKIP", "reason": "too_few_valid_rows"}

        try:
            bins = pd.qcut(x2, q=OVERLAP_NUM_BINS, duplicates="drop")
        except Exception:
            return [], {"col": col, "type": "numeric", "status": "SKIP", "reason": "qcut_failed"}

        dfb = pd.DataFrame({"bin": bins.astype(str), "t": mt.astype(int), "c": mc.astype(int)})
        g = dfb.groupby("bin", dropna=False).agg(n_t=("t", "sum"), n_c=("c", "sum"))
        g["n"] = g["n_t"] + g["n_c"]

        good = (g["n_t"] > 0) & (g["n_c"] > 0)
        coverage = float(good.mean()) if int(g.shape[0]) > 0 else 0.0
        tiny = int((g["n"] < OVERLAP_MIN_BIN_COUNT_WARN).sum())

        issues: List[ProtocolValidationIssue] = []
        sev = _overlap_severity_from_coverage(coverage)
        if sev is not None:
            issues.append(
                _mk_issue(
                    rule_id="OVERLAP_LOW_COVERAGE",
                    severity=sev,
                    message="Low overlap coverage across numeric strata bins (binary treatment).",
                    evidence={"feature": col, "coverage": coverage, "n_bins": int(g.shape[0])},
                    fix_hint="Consider trimming, redefining cohort/treatment, or reducing heterogeneity feature set.",
                )
            )

        if tiny > 0:
            issues.append(
                _mk_issue(
                    rule_id="OVERLAP_TINY_BINS",
                    severity="WARN",
                    message="Some numeric strata bins have very small sample size; overlap diagnostics may be noisy.",
                    evidence={"feature": col, "tiny_bins": tiny, "min_bin_count_warn": OVERLAP_MIN_BIN_COUNT_WARN},
                    fix_hint="Consider fewer bins or stronger cohort definition.",
                )
            )

        return issues, {"col": col, "type": "numeric", "n_bins": int(g.shape[0]), "coverage": coverage, "tiny_bins": tiny}

    ss = s.astype(str)
    if nunq > OVERLAP_MAX_CATEG_LEVELS:
        issues = [
            _mk_issue(
                rule_id="OVERLAP_HIGH_CARDINALITY",
                severity="WARN",
                message="Stratifier has high cardinality; categorical overlap check skipped.",
                evidence={"feature": col, "n_unique": nunq, "max_levels": OVERLAP_MAX_CATEG_LEVELS},
                fix_hint="Consider binning/featurizing this variable before using it for overlap diagnostics.",
            )
        ]
        return issues, {"col": col, "type": "categorical", "status": "SKIP", "reason": "high_cardinality"}

    dfc = pd.DataFrame({"lvl": ss, "t": mask_t.astype(int), "c": mask_c.astype(int)})
    g = dfc.groupby("lvl", dropna=False).agg(n_t=("t", "sum"), n_c=("c", "sum"))
    good = (g["n_t"] > 0) & (g["n_c"] > 0)
    coverage = float(good.mean()) if int(g.shape[0]) > 0 else 0.0

    issues2: List[ProtocolValidationIssue] = []
    sev2 = _overlap_severity_from_coverage(coverage)
    if sev2 is not None:
        issues2.append(
            _mk_issue(
                rule_id="OVERLAP_LOW_COVERAGE",
                severity=sev2,
                message="Low overlap coverage across categorical strata levels (binary treatment).",
                evidence={"feature": col, "coverage": coverage, "n_levels": int(g.shape[0])},
                fix_hint="Consider merging rare levels, trimming, or redefining cohort/treatment.",
            )
        )

    return issues2, {"col": col, "type": "categorical", "n_levels": int(g.shape[0]), "coverage": coverage}


def _overlap_feature_multitreat(
    *, df: pd.DataFrame, col: str, t: pd.Series, levels: Sequence[str]
) -> Tuple[List[ProtocolValidationIssue], Dict[str, Any]]:
    s = df[col]
    nunq = int(s.nunique(dropna=True))

    x = pd.to_numeric(s, errors="coerce")
    numeric_usable = int(x.notna().sum()) > 0 and (int(x.nunique(dropna=True)) > 1)

    if numeric_usable and not (s.dtype == object and nunq <= OVERLAP_MAX_CATEG_LEVELS):
        valid = x.notna()
        x2 = x[valid]
        t2 = t[valid]

        try:
            bins = pd.qcut(x2, q=OVERLAP_NUM_BINS, duplicates="drop").astype(str)
        except Exception:
            return [], {"col": col, "type": "numeric", "status": "SKIP", "reason": "qcut_failed"}

        dfb = pd.DataFrame({"bin": bins, "t": t2})
        counts = dfb.groupby(["bin", "t"]).size().unstack(fill_value=0)

        for lvl in levels:
            if lvl not in counts.columns:
                counts[lvl] = 0
        counts = counts[list(levels)]

        present_k = (counts > 0).sum(axis=1)
        coverage = float((present_k >= 2).mean()) if int(counts.shape[0]) > 0 else 0.0

        issues: List[ProtocolValidationIssue] = []
        sev = _overlap_severity_from_coverage(coverage)
        if sev is not None:
            issues.append(
                _mk_issue(
                    rule_id="OVERLAP_LOW_COVERAGE",
                    severity=sev,
                    message="Low overlap coverage across strata bins (categorical multi-treatment).",
                    evidence={"feature": col, "coverage_ge2_levels": coverage, "n_bins": int(counts.shape[0])},
                    fix_hint="Consider merging treatment levels, trimming, or redefining cohort/treatment.",
                )
            )

        return issues, {"col": col, "type": "numeric", "n_bins": int(counts.shape[0]), "coverage_ge2_levels": coverage}

    ss = s.astype(str)
    if nunq > OVERLAP_MAX_CATEG_LEVELS:
        issues = [
            _mk_issue(
                rule_id="OVERLAP_HIGH_CARDINALITY",
                severity="WARN",
                message="Stratifier has high cardinality; categorical overlap check skipped.",
                evidence={"feature": col, "n_unique": nunq, "max_levels": OVERLAP_MAX_CATEG_LEVELS},
                fix_hint="Consider binning/featurizing this variable before overlap diagnostics.",
            )
        ]
        return issues, {"col": col, "type": "categorical", "status": "SKIP", "reason": "high_cardinality"}

    dfc = pd.DataFrame({"lvl": ss, "t": t})
    counts = dfc.groupby(["lvl", "t"]).size().unstack(fill_value=0)
    for lvl in levels:
        if lvl not in counts.columns:
            counts[lvl] = 0
    counts = counts[list(levels)]

    present_k = (counts > 0).sum(axis=1)
    coverage = float((present_k >= 2).mean()) if int(counts.shape[0]) > 0 else 0.0

    issues2: List[ProtocolValidationIssue] = []
    sev2 = _overlap_severity_from_coverage(coverage)
    if sev2 is not None:
        issues2.append(
            _mk_issue(
                rule_id="OVERLAP_LOW_COVERAGE",
                severity=sev2,
                message="Low overlap coverage across strata levels (categorical multi-treatment).",
                evidence={"feature": col, "coverage_ge2_levels": coverage, "n_levels": int(counts.shape[0])},
                fix_hint="Consider merging rare levels or redefining cohort/treatment.",
            )
        )

    return issues2, {"col": col, "type": "categorical", "n_levels": int(counts.shape[0]), "coverage_ge2_levels": coverage}


def _overlap_severity_from_coverage(coverage: float) -> ValidationSeverity | None:
    if coverage < OVERLAP_COVERAGE_FAIL:
        return "FAIL"
    if coverage < OVERLAP_COVERAGE_WARN:
        return "WARN"
    return None


# =============================================================================
# Metrics builder
# =============================================================================


def _build_metrics(
    *,
    df_raw: pd.DataFrame,
    df_after_dropna: pd.DataFrame,
    df_after_exclusions: pd.DataFrame,
    protocol: ProtocolState,
    dropna_metrics: Dict[str, Any],
    excl_metrics: Dict[str, Any],
    overlap_metrics: Dict[str, Any] | None,
) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "n_raw": int(df_raw.shape[0]),
        "n_after_dropna_required_ty": int(df_after_dropna.shape[0]),
        "n_after_exclusions": int(df_after_exclusions.shape[0]),
        "dropna_required_ty": dropna_metrics,
        "exclusions": excl_metrics,
        "experiment_type": protocol.get("experiment_type"),
        "time_zero_type": protocol.get("time_zero_type"),
        "covariates_count": len(protocol.get("covariates", []) or []),
        "effect_modifiers_count": len(protocol.get("effect_modifiers", []) or []),
    }

    ts = protocol.get("treatment_spec")
    if isinstance(ts, dict):
        m["treatment_kind"] = ts.get("kind")
        tcol = ts.get("column")
        if isinstance(tcol, str) and tcol in df_after_exclusions.columns:
            m["treatment_missing_rate"] = float(df_after_exclusions[tcol].isna().mean())

    ys = protocol.get("outcome_spec")
    if isinstance(ys, dict):
        m["outcome_kind"] = ys.get("kind")

    if overlap_metrics is not None:
        m["overlap"] = overlap_metrics

    return m


# =============================================================================
# LLM: user-facing message rendering
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
