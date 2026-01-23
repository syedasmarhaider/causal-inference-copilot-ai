from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
from uuid import UUID

import numpy as np
import pandas as pd

from python.domain.repo.data_repo import DataRepo
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState
from python.workflows.state.dataset_state import DatasetState
from python.workflows.state.protocol_state import ProtocolState
from python.workflows.state.validate_protocol_state import (
    ProtocolValidationIssue,
    ProtocolValidationReport,
    ValidationSeverity,
    ValidationStatus,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationPolicy:
    """
    Deterministic thresholds for the static validation gate.
    Tune per cohort size and expected missingness.
    """

    # Cohort size / feasibility
    min_total_rows: int = 200
    min_arm_size: int = 50
    max_arm_imbalance_ratio: float = 20.0

    # Missingness thresholds (rates in [0, 1])
    max_missing_outcome: float = 0.35
    max_missing_treatment: float = 0.20
    min_complete_case_rate: float = 0.30

    # Outcome sanity
    min_outcome_variance: float = 1e-12

    # Behavioral strictness knobs
    warn_if_observational_and_no_adjusters: bool = True  # heuristic via protocol.experiment_type only


def make_validate_protocol_static_node(*, data_repo: DataRepo) -> CallableNodeFunc:
    """
    Static protocol validation node (no LLM).

    Reads:
      - state.dataset (must be loaded, no load_error)
      - state.protocol (ProtocolState)

    Writes:
      - state.protocol_static_validation (ProtocolStaticValidationState)
          - report (ProtocolValidationReport)
    """
    policy = ValidationPolicy()

    def _run(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        dataset = state.get("dataset")
        protocol = state.get("protocol")

        if dataset is None:
            _write_report(state, _fatal_report("Dataset is missing from state."))
            return state

        load_error = dataset.get("load_error")
        if load_error:
            _write_report(state, _fatal_report(f"Dataset load_error is set: {load_error}"))
            return state

        if protocol is None:
            _write_report(state, _fatal_report("Protocol is missing from state."))
            return state

        try:
            df = _load_dataframe(data_repo, user_id, conversation_id, dataset)
        except Exception as e:
            log.exception("Failed to load dataframe for static validation.")
            _write_report(state, _fatal_report(f"Failed to load dataset for validation: {e}"))
            return state

        report = _validate(protocol=protocol, df=df, policy=policy)
        _write_report(state, report)
        return state

    return _run


# -----------------------------------------------------------------------------
# Validation core
# -----------------------------------------------------------------------------

def _validate(*, protocol: ProtocolState, df: pd.DataFrame, policy: ValidationPolicy) -> ProtocolValidationReport:
    issues: List[ProtocolValidationIssue] = []
    metrics: Dict[str, Any] = {}

    # 1) Protocol structural / numeric sanity
    issues.extend(_check_protocol_integrity(protocol))
    if _has_fail(issues):
        return _finalize_report(issues, metrics)

    # 2) Schema alignment
    issues.extend(_check_schema_alignment(protocol, df))
    if _has_fail(issues):
        return _finalize_report(issues, metrics)

    # 3) Dataset size
    n_total = int(df.shape[0])
    metrics["n_rows_total"] = n_total
    if n_total < policy.min_total_rows:
        issues.append(
            _issue(
                "D001_DATASET_TOO_SMALL",
                "FAIL",
                "Dataset has too few rows for reliable estimation.",
                evidence={"n_rows_total": n_total, "min_total_rows": policy.min_total_rows},
                fix_hint="Use a larger dataset, broaden the population definition, or revise the question.",
            )
        )
        return _finalize_report(issues, metrics)

    # 4) Arm feasibility (binary / two-class treatment)
    eligible_mask = pd.Series(True, index=df.index)
    treated_mask, control_mask, arm_issue, arm_metrics = _infer_treatment_arms(df, eligible_mask, protocol)
    metrics.update(arm_metrics)
    if arm_issue is not None:
        issues.append(arm_issue)
        return _finalize_report(issues, metrics)

    n_treated = int(treated_mask.sum())
    n_control = int(control_mask.sum())
    metrics["n_treated"] = n_treated
    metrics["n_control"] = n_control

    if min(n_treated, n_control) < policy.min_arm_size:
        issues.append(
            _issue(
                "F020_MIN_ARM_SIZE",
                "FAIL",
                "One treatment arm is too small for stable estimation.",
                evidence={"n_treated": n_treated, "n_control": n_control, "min_arm_size": policy.min_arm_size},
                fix_hint="Broaden population criteria, revise comparator, or use a dataset with more treated/control cases.",
            )
        )
        return _finalize_report(issues, metrics)

    imbalance_ratio = float(max(n_treated, n_control) / max(1, min(n_treated, n_control)))
    metrics["arm_imbalance_ratio"] = imbalance_ratio
    if imbalance_ratio > policy.max_arm_imbalance_ratio:
        issues.append(
            _issue(
                "F021_ARM_IMBALANCE",
                "WARN",
                "Arm imbalance is severe; downstream estimators may be unstable.",
                evidence={
                    "arm_imbalance_ratio": imbalance_ratio,
                    "max_arm_imbalance_ratio": policy.max_arm_imbalance_ratio,
                },
                fix_hint="Proceed with caution; consider weighting/stratification later.",
            )
        )

    # 5) Missingness gates
    required_cols = _required_columns(protocol)
    missingness = _compute_missingness(df, eligible_mask, required_cols)
    metrics["missingness_by_col"] = missingness

    outcome = protocol["outcome"]
    treatment = protocol["treatment"]

    if missingness.get(outcome, 0.0) > policy.max_missing_outcome:
        issues.append(
            _issue(
                "M010_OUTCOME_MISSINGNESS",
                "FAIL",
                "Outcome missingness is too high.",
                evidence={
                    "outcome": outcome,
                    "missing_rate": missingness.get(outcome),
                    "max_allowed": policy.max_missing_outcome,
                },
                fix_hint="Choose a better-recorded outcome or improve data completeness.",
            )
        )
        return _finalize_report(issues, metrics)

    if missingness.get(treatment, 0.0) > policy.max_missing_treatment:
        issues.append(
            _issue(
                "M011_TREATMENT_MISSINGNESS",
                "FAIL",
                "Treatment missingness is too high.",
                evidence={
                    "treatment": treatment,
                    "missing_rate": missingness.get(treatment),
                    "max_allowed": policy.max_missing_treatment,
                },
                fix_hint="Fix treatment encoding or choose a reliably observed treatment column.",
            )
        )
        return _finalize_report(issues, metrics)

    # Complete-case rate
    complete_case_mask = eligible_mask.copy()
    for c in required_cols:
        complete_case_mask &= df[c].notna()

    n_complete = int(complete_case_mask.sum())
    denom = max(1, int(eligible_mask.sum()))
    complete_rate = float(n_complete / denom)
    metrics["n_complete_case"] = n_complete
    metrics["complete_case_rate"] = complete_rate

    if complete_rate < policy.min_complete_case_rate:
        issues.append(
            _issue(
                "M020_COMPLETE_CASE_TOO_LOW",
                "FAIL",
                "Too few rows have all required variables present (complete-case rate too low).",
                evidence={
                    "complete_case_rate": complete_rate,
                    "min_required": policy.min_complete_case_rate,
                    "n_complete_case": n_complete,
                    "n_rows_total": n_total,
                },
                fix_hint="Reduce covariates/effect modifiers or plan imputation later.",
            )
        )
        return _finalize_report(issues, metrics)

    # 6) Outcome sanity
    issues.extend(_check_outcome_sanity(df, eligible_mask, protocol, policy, metrics))
    if _has_fail(issues):
        return _finalize_report(issues, metrics)

    # 7) Covariate hygiene (WARN only)
    issues.extend(_check_covariate_hygiene(df, eligible_mask, protocol, metrics))

    # 8) Optional heuristic warning
    issues.extend(_heuristic_warn_no_adjusters(protocol, policy))

    return _finalize_report(issues, metrics)


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def _check_protocol_integrity(protocol: ProtocolState) -> List[ProtocolValidationIssue]:
    issues: List[ProtocolValidationIssue] = []

    if protocol.get("time_zero_type") not in ("COLUMN", "CONCEPTUAL"):
        issues.append(
            _issue(
                "P001_BAD_TIME_ZERO_TYPE",
                "FAIL",
                "Invalid time_zero_type; must be COLUMN or CONCEPTUAL.",
                evidence={"time_zero_type": protocol.get("time_zero_type")},
                fix_hint="Set time_zero_type to COLUMN or CONCEPTUAL.",
            )
        )

    tws = _parse_float(protocol.get("treatment_window_start", ""))
    twe = _parse_float(protocol.get("treatment_window_end", ""))
    if tws is None or twe is None:
        issues.append(
            _issue(
                "P002_BAD_TREATMENT_WINDOW",
                "FAIL",
                "Treatment window start/end must be parseable numbers (stored as strings).",
                evidence={
                    "treatment_window_start": protocol.get("treatment_window_start"),
                    "treatment_window_end": protocol.get("treatment_window_end"),
                },
                fix_hint="Use numeric strings like '0', '7', etc., consistent with treatment_window_unit.",
            )
        )
    elif tws > twe:
        issues.append(
            _issue(
                "P003_TREATMENT_WINDOW_ORDER",
                "FAIL",
                "Treatment window start must be <= treatment window end.",
                evidence={"start": tws, "end": twe},
                fix_hint="Swap/correct the treatment window bounds.",
            )
        )

    ow = _parse_float(protocol.get("outcome_window", ""))
    if ow is None or ow <= 0:
        issues.append(
            _issue(
                "P004_BAD_OUTCOME_WINDOW",
                "FAIL",
                "Outcome window must be a parseable positive number (stored as string).",
                evidence={"outcome_window": protocol.get("outcome_window")},
                fix_hint="Use numeric strings like '30' with outcome_window_unit.",
            )
        )

    covs = protocol.get("covariates", []) or []
    ems = protocol.get("effect_modifiers", []) or []
    if len(set(covs)) != len(covs) or len(set(ems)) != len(ems):
        issues.append(
            _issue(
                "P005_DUPLICATE_VARS",
                "WARN",
                "Duplicate entries detected in covariates/effect_modifiers.",
                evidence={"covariates": covs, "effect_modifiers": ems},
                fix_hint="Deduplicate these lists.",
            )
        )

    return issues


def _check_schema_alignment(protocol: ProtocolState, df: pd.DataFrame) -> List[ProtocolValidationIssue]:
    issues: List[ProtocolValidationIssue] = []
    cols = set(df.columns)

    required = _required_columns(protocol)
    missing = [c for c in required if c not in cols]
    if missing:
        issues.append(
            _issue(
                "S001_MISSING_COLUMNS",
                "FAIL",
                "Protocol references missing columns in the dataset.",
                evidence={"missing": missing},
                fix_hint="Fix protocol column names to exactly match dataset schema.",
            )
        )
        return issues

    if protocol.get("outcome_is_duration") is True:
        y = protocol["outcome"]
        if not pd.api.types.is_numeric_dtype(df[y]):
            issues.append(
                _issue(
                    "S010_OUTCOME_NOT_NUMERIC_FOR_DURATION",
                    "FAIL",
                    "Outcome is marked as duration, but dtype is not numeric.",
                    evidence={"outcome": y, "dtype": str(df[y].dtype)},
                    fix_hint="Choose a numeric duration outcome or set outcome_is_duration=False.",
                )
            )

    return issues


def _check_outcome_sanity(
    df: pd.DataFrame,
    eligible_mask: pd.Series,
    protocol: ProtocolState,
    policy: ValidationPolicy,
    metrics: Dict[str, Any],
) -> List[ProtocolValidationIssue]:
    issues: List[ProtocolValidationIssue] = []
    y = protocol["outcome"]

    s = df.loc[eligible_mask, y].dropna()
    if s.empty:
        issues.append(
            _issue(
                "Y001_OUTCOME_ALL_MISSING",
                "FAIL",
                "Outcome is entirely missing in the dataset.",
                evidence={"outcome": y},
                fix_hint="Choose a different outcome or fix missingness upstream.",
            )
        )
        return issues

    y_stats: Dict[str, Any] = {
        "count": int(s.shape[0]),
        "dtype": str(df[y].dtype),
        "n_unique": int(s.nunique(dropna=True)),
    }

    if pd.api.types.is_numeric_dtype(s):
        arr = s.to_numpy(dtype=float, copy=False)
        y_stats["min"] = float(np.nanmin(arr))
        y_stats["max"] = float(np.nanmax(arr))
        y_stats["var"] = float(np.nanvar(arr))

    metrics["outcome_stats"] = y_stats

    if protocol.get("outcome_is_duration") is True:
        if not pd.api.types.is_numeric_dtype(s):
            issues.append(
                _issue(
                    "Y010_DURATION_NOT_NUMERIC",
                    "FAIL",
                    "Duration outcome must be numeric.",
                    evidence=y_stats,
                    fix_hint="Use a numeric duration outcome column.",
                )
            )
            return issues

        if y_stats.get("min", 0.0) < 0.0:
            issues.append(
                _issue(
                    "Y011_NEGATIVE_DURATION",
                    "FAIL",
                    "Duration outcome contains negative values.",
                    evidence=y_stats,
                    fix_hint="Clean negative durations or select correct duration field.",
                )
            )
            return issues

        if y_stats.get("var", 0.0) < policy.min_outcome_variance:
            issues.append(
                _issue(
                    "Y020_OUTCOME_NEAR_CONSTANT",
                    "FAIL",
                    "Outcome variance is near zero; effect estimation is not meaningful.",
                    evidence={"var": y_stats.get("var"), "min_outcome_variance": policy.min_outcome_variance},
                    fix_hint="Choose an outcome with sufficient variability.",
                )
            )
            return issues
    else:
        if y_stats["n_unique"] <= 1:
            issues.append(
                _issue(
                    "Y021_OUTCOME_CONSTANT",
                    "FAIL",
                    "Outcome has <= 1 unique value; cannot estimate an effect.",
                    evidence=y_stats,
                    fix_hint="Choose an outcome with variability or revise population definition.",
                )
            )

    return issues


def _check_covariate_hygiene(
    df: pd.DataFrame,
    eligible_mask: pd.Series,
    protocol: ProtocolState,
    metrics: Dict[str, Any],
) -> List[ProtocolValidationIssue]:
    issues: List[ProtocolValidationIssue] = []
    cols = list(dict.fromkeys((protocol.get("covariates", []) or []) + (protocol.get("effect_modifiers", []) or [])))
    if not cols:
        return issues

    stats: Dict[str, Any] = {}
    denom = max(1, int(eligible_mask.sum()))
    for c in cols:
        s = df.loc[eligible_mask, c]
        s_nonnull = s.dropna()
        n_unique = int(s_nonnull.nunique(dropna=True)) if not s_nonnull.empty else 0
        missing_rate = float(1.0 - (s_nonnull.shape[0] / denom))
        stats[c] = {"dtype": str(df[c].dtype), "n_unique": n_unique, "missing_rate": missing_rate}

        if n_unique <= 1:
            issues.append(
                _issue(
                    "X010_COVARIATE_CONSTANT",
                    "WARN",
                    "Covariate has no variability; it will not help adjustment.",
                    evidence={"column": c, "n_unique": n_unique},
                    fix_hint="Remove it from covariates/effect modifiers or accept dropping later.",
                )
            )

    metrics["covariate_stats"] = stats
    return issues


def _heuristic_warn_no_adjusters(protocol: ProtocolState, policy: ValidationPolicy) -> List[ProtocolValidationIssue]:
    if not policy.warn_if_observational_and_no_adjusters:
        return []
    covs = protocol.get("covariates", []) or []
    ems = protocol.get("effect_modifiers", []) or []
    if (len(covs) + len(ems)) > 0:
        return []

    exp = str(protocol.get("experiment_type", "") or "").strip().lower()
    observational_like = any(k in exp for k in ("observ", "tte", "target", "trial", "ate", "att"))
    if not observational_like:
        return []

    return [
        _issue(
            "H001_NO_ADJUSTERS_HEURISTIC",
            "WARN",
            "No covariates/effect modifiers provided; if this is observational, unadjusted estimates may be biased.",
            evidence={"experiment_type": protocol.get("experiment_type")},
            fix_hint="Add baseline covariates for adjustment (or explicitly accept an unadjusted estimate).",
        )
    ]


# -----------------------------------------------------------------------------
# Arm inference + missingness
# -----------------------------------------------------------------------------

def _infer_treatment_arms(
    df: pd.DataFrame,
    eligible_mask: pd.Series,
    protocol: ProtocolState,
) -> Tuple[pd.Series, pd.Series, Optional[ProtocolValidationIssue], Dict[str, Any]]:
    metrics: Dict[str, Any] = {}
    tcol = protocol["treatment"]
    s = df.loc[eligible_mask, tcol]

    if pd.api.types.is_bool_dtype(s):
        metrics["treatment_arm_mode"] = "boolean"
        treated = eligible_mask & (df[tcol] == True)   # noqa: E712
        control = eligible_mask & (df[tcol] == False)  # noqa: E712
        return treated, control, None, metrics

    uniques = pd.unique(s.dropna())
    metrics["treatment_unique_values_sample"] = [str(x) for x in uniques[:10]]
    n_unique = int(uniques.shape[0])
    metrics["treatment_unique_values_count"] = n_unique

    if n_unique != 2:
        return (
            pd.Series(False, index=df.index),
            pd.Series(False, index=df.index),
            _issue(
                "T011_TREATMENT_NOT_BINARY",
                "FAIL",
                "Treatment is not binary/two-class.",
                evidence={"treatment_col": tcol, "n_unique": n_unique, "sample_values": [str(x) for x in uniques[:10]]},
                fix_hint="Transform treatment into a 2-class column (or add multi-arm support).",
            ),
            metrics,
        )

    comparator = (protocol.get("comparator") or "").strip()
    u0, u1 = uniques[0], uniques[1]

    chosen_control = None
    if comparator:
        if str(u0) == comparator:
            chosen_control = u0
        elif str(u1) == comparator:
            chosen_control = u1

    if chosen_control is None:
        chosen_control = u0 if str(u0) <= str(u1) else u1
        metrics["comparator_mapping_warning"] = {
            "comparator": comparator,
            "values": [str(u0), str(u1)],
            "control_value": str(chosen_control),
        }

    chosen_treated = u1 if chosen_control is u0 else u0
    treated = eligible_mask & (df[tcol] == chosen_treated)
    control = eligible_mask & (df[tcol] == chosen_control)

    metrics["treatment_arm_mode"] = "two_class"
    metrics["treated_value"] = str(chosen_treated)
    metrics["control_value"] = str(chosen_control)

    if int(treated.sum()) == 0 or int(control.sum()) == 0:
        return (
            pd.Series(False, index=df.index),
            pd.Series(False, index=df.index),
            _issue(
                "T012_TREATMENT_ARM_EMPTY",
                "FAIL",
                "Resolved treatment/control mapping produced an empty arm.",
                evidence={"treatment_col": tcol, "treated_value": str(chosen_treated), "control_value": str(chosen_control)},
                fix_hint="Confirm treatment encoding and comparator; ensure both classes exist.",
            ),
            metrics,
        )

    return treated, control, None, metrics


def _compute_missingness(df: pd.DataFrame, mask: pd.Series, cols: Sequence[str]) -> Dict[str, float]:
    denom = float(max(1, int(mask.sum())))
    out: Dict[str, float] = {}
    for c in cols:
        non_missing = float(df.loc[mask, c].notna().sum())
        out[c] = float(1.0 - (non_missing / denom))
    return out


# -----------------------------------------------------------------------------
# Report helpers
# -----------------------------------------------------------------------------

def _write_report(state: ConversationState, report: ProtocolValidationReport) -> None:
    state["protocol_static_validation"] = {"report": report}  # type: ignore[typeddict-item]


def _fatal_report(message: str) -> ProtocolValidationReport:
    return {
        "status": "FAIL",
        "issues": [_issue("NODE_FATAL", "FAIL", message, evidence={}, fix_hint=None)],
        "metrics": {},
    }


def _finalize_report(issues: List[ProtocolValidationIssue], metrics: Dict[str, Any]) -> ProtocolValidationReport:
    return {"status": _resolve_status(issues), "issues": issues, "metrics": metrics}


def _issue(
    rule_id: str,
    severity: ValidationSeverity,
    message: str,
    *,
    evidence: Dict[str, Any],
    fix_hint: Optional[str],
) -> ProtocolValidationIssue:
    return {"rule_id": rule_id, "severity": severity, "message": message, "evidence": evidence, "fix_hint": fix_hint}


def _has_fail(issues: Sequence[ProtocolValidationIssue]) -> bool:
    return any(i.get("severity") == "FAIL" for i in issues)


def _resolve_status(issues: Sequence[ProtocolValidationIssue]) -> ValidationStatus:
    if any(i.get("severity") == "FAIL" for i in issues):
        return "FAIL"
    if any(i.get("severity") == "WARN" for i in issues):
        return "WARN"
    return "PASS"


def _required_columns(protocol: ProtocolState) -> List[str]:
    cols: List[str] = [protocol["treatment"], protocol["outcome"]]
    if protocol.get("time_zero_type") == "COLUMN":
        cols.append(protocol.get("time_zero", ""))
    cols.extend(protocol.get("covariates", []) or [])
    cols.extend(protocol.get("effect_modifiers", []) or [])
    return list(dict.fromkeys([c for c in cols if c]))


def _parse_float(x: Any) -> Optional[float]:
    try:
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _load_dataframe(data_repo: DataRepo, user_id: UUID, conversation_id: UUID, dataset: DatasetState) -> pd.DataFrame:
    """
    Adapter to avoid tight coupling to a single DataRepo method name.
    Update once your DataRepo interface is fixed.
    """
    candidates = ("get_dataframe", "get_df", "load_dataframe", "load_df", "get_dataset_df")
    for name in candidates:
        fn = getattr(data_repo, name, None)
        if not callable(fn):
            continue
        try:
            if name in {"load_dataframe", "load_df"}:
                path = dataset.get("path")
                if not path:
                    continue
                return cast(pd.DataFrame, fn(path))  # type: ignore[misc]
            if name == "get_dataset_df":
                return cast(pd.DataFrame, fn(user_id, conversation_id, dataset.get("id")))  # type: ignore[misc]
            return cast(pd.DataFrame, fn(user_id, conversation_id))  # type: ignore[misc]
        except TypeError:
            continue
    raise RuntimeError("DataRepo has no compatible dataframe loader method.")
