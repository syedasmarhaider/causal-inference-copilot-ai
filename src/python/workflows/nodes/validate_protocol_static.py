# src/python/workflows/nodes/validate_protocol_static.py
from __future__ import annotations

import logging
from difflib import get_close_matches
from typing import Any, Iterable, List, Optional, Tuple, cast
from uuid import UUID

import pandas as pd
from langchain_core.messages import AIMessage

from python.domain.repo.data_repo import DataRepo
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState
from python.workflows.state.dataset_state import DatasetState
from python.workflows.state.metadata_state import MetadataState
from python.workflows.state.protocol_state import ProtocolState

log = logging.getLogger(__name__)


def make_validate_protocol_static_node(
    data_repo: DataRepo,
) -> CallableNodeFunc:
    def _node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        dataset, _, protocol, err = _require_dataset_meta_protocol(state)
        if err:
            return _fatal(state, err)

        # Protocol must be accepted before we validate it against data.
        if protocol.get("accepted") is not True:
            msg = "Protocol is not accepted yet. stage COMPILE_PROTOCOL must complete successfully first."
            _append_ai(state, msg)
            return _need_input(state, msg)

        # Dataset must be loadable.
        load_error = dataset.get("load_error")
        if isinstance(load_error, str) and load_error.strip():
            return _fatal(state, f"Dataset could not be loaded ({load_error}). Please reload and retry.")

        dataset_id = dataset.get("id")
        if dataset_id is None:
            return _fatal(state, "Dataset ID is missing. Please reload the dataset.")

        # Load ALL rows (as requested). We still validate using only the needed columns after load.
        try:
            df = _load_all_rows(data_repo, user_id, conversation_id, dataset_id)
        except Exception as e:
            return _fatal(state, f"Failed to load dataset: {type(e).__name__}: {e}")

        if df.empty:
            msg = "Dataset is empty. Cannot validate protocol against an empty dataset."
            _append_ai(state, msg)
            return _fatal(state, msg)

        issues = _validate_protocol_against_df(protocol=protocol, df=df)

        if issues:
            # IMPORTANT: mark protocol as not accepted to allow upstream “compile_protocol_state”
            # to re-enter non-LOCKED mode and let the user fix/confirm.
            protocol["accepted"] = False
            protocol["open_questions"] = issues
            state["protocol"] = protocol

            msg = "Static protocol validation failed:\n- " + "\n- ".join(issues)
            _append_ai(state, msg)
            return _need_input(state, msg)

        msg = "Static protocol validation passed."
        _append_ai(state, msg)
        return _succeed(state, msg)

    return _node


# ----------------------------
# Core validation
# ----------------------------

def _validate_protocol_against_df(
    *,
    protocol: ProtocolState,
    df: pd.DataFrame,
) -> List[str]:
    issues: List[str] = []

    # 1) Window sanity: prevents immortal-time / nonsense window specs downstream.
    tws = _parse_int(protocol.get("treatment_window_start"), "treatment_window_start", issues)
    twe = _parse_int(protocol.get("treatment_window_end"), "treatment_window_end", issues)
    if tws is not None and twe is not None and tws > twe:
        issues.append("treatment_window_start must be <= treatment_window_end.")

    ow = _parse_int(protocol.get("outcome_window"), "outcome_window", issues)
    if ow is not None and ow <= 0:
        issues.append("outcome_window must be a positive integer string.")

    # 2) Required timestamp availability when time_zero is a dataset column.
    #    This is a hard operational requirement for cohort anchoring.
    if protocol.get("time_zero_type") == "COLUMN":
        tz = (protocol.get("time_zero") or "").strip()
        if not tz:
            issues.append("time_zero_type=COLUMN but time_zero is empty.")
        else:
            if tz not in df.columns:
                issues.append(_missing_col_msg(tz, df.columns, field="time_zero"))
            else:
                # Scientific requirement: time_zero must be a valid time axis.
                # We accept datetime dtype OR object/string that is highly parseable to datetime.
                if not _is_datetime_like_or_parseable(df[tz], min_parseable_ratio=0.98):
                    issues.append(
                        f"time_zero column '{tz}' must be datetime-like or reliably parseable (>=98% parseable)."
                    )

    # 3) T and Y existence: cannot proceed without them.
    t = (protocol.get("treatment") or "").strip()
    y = (protocol.get("outcome") or "").strip()

    if not t:
        issues.append("treatment is empty.")
    elif t not in df.columns:
        issues.append(_missing_col_msg(t, df.columns, field="treatment"))

    if not y:
        issues.append("outcome is empty.")
    elif y not in df.columns:
        issues.append(_missing_col_msg(y, df.columns, field="outcome"))

    # If columns missing, type/viability checks below would be misleading.
    if any("not found" in x for x in issues):
        return _dedupe(issues)

    # 4) Type compatibility checks (estimator-agnostic but scientifically necessary).
    #    - T cannot be datetime/timedelta.
    #    - If outcome_is_duration=True, Y must be numeric.
    t_series = df[t]
    y_series = df[y]

    if _is_datetime_dtype(t_series) or _is_timedelta_dtype(t_series):
        issues.append(f"treatment '{t}' has invalid type (datetime/duration). Use binary/categorical/numeric treatment.")

    if protocol.get("outcome_is_duration") is True:
        # duration must be numeric (or coercible with low failure)
        if not _is_numeric_or_numeric_coercible(y_series, min_coercible_ratio=0.98):
            issues.append(
                f"outcome_is_duration=True but outcome '{y}' is not numeric or not reliably coercible to numeric (>=98%)."
            )
    else:
        # direct outcomes rarely should be datetime/timedelta
        if _is_datetime_dtype(y_series) or _is_timedelta_dtype(y_series):
            issues.append(f"outcome '{y}' has invalid type (datetime/duration) for a direct outcome variable.")

    # 5) Minimal data viability checks (prevents silent degeneracy).
    #    These are not causal identification checks; they’re “can we even estimate something?” checks.
    #    - T must vary (at least 2 unique non-null values)
    #    - Y must have non-missing signal
    t_unique = _nunique_nonnull(t_series)
    if t_unique < 2:
        issues.append(f"treatment '{t}' is degenerate (unique non-null values < 2).")

    y_nonnull = int(y_series.notna().sum())
    if y_nonnull == 0:
        issues.append(f"outcome '{y}' is entirely missing (0 non-null rows).")

    # For duration outcomes, negative values are typically invalid unless explicitly allowed.
    # This is a common scientific sanity check.
    if protocol.get("outcome_is_duration") is True and _is_numeric_or_numeric_coercible(y_series, 0.98):
        yn = pd.to_numeric(y_series, errors="coerce")
        if int((yn.notna() & (yn < 0)).sum()) > 0:
            issues.append(f"outcome '{y}' contains negative durations; verify encoding or cleaning rules.")

    # 6) Protocol hygiene: never adjust for T/Y in covariates/modifiers (prevents trivial leakage).
    covs = protocol.get("covariates", [])
    mods = protocol.get("effect_modifiers", [])
    if  (t in covs or y in covs):
        issues.append("covariates contains treatment/outcome; remove T/Y from covariates.")
    if  (t in mods or y in mods):
        issues.append("effect_modifiers contains treatment/outcome; remove T/Y from effect_modifiers.")

    return _dedupe(issues)


# ----------------------------
# Data loading (ALL rows)
# ----------------------------

def _load_all_rows(
    data_repo: DataRepo,
    user_id: UUID,
    conversation_id: UUID,
    dataset_id: Any,
) -> pd.DataFrame:
    df = data_repo.get_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
        )
    return  df

# ----------------------------
# Column/type helpers
# ----------------------------

def _missing_col_msg(col: str, cols: Iterable[str], *, field: str) -> str:
    cols_list = [str(c) for c in cols]  # works for pandas Index + any iterable
    sugg = get_close_matches(col, cols_list, n=3, cutoff=0.6)
    if sugg:
        return f"{field} column '{col}' not found in dataset. Closest matches: {sugg}"
    return f"{field} column '{col}' not found in dataset."


def _is_datetime_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def _is_timedelta_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_timedelta64_dtype(s)


def _is_datetime_like_or_parseable(s: pd.Series, *, min_parseable_ratio: float) -> bool:
    if _is_datetime_dtype(s):
        return True
    # Try parse; require high parseability among non-null entries.
    nonnull = s.dropna()
    if nonnull.empty:
        return False
    parsed = pd.to_datetime(nonnull, errors="coerce", utc=False)
    ratio = float(parsed.notna().mean())
    return ratio >= min_parseable_ratio


def _is_numeric_or_numeric_coercible(s: pd.Series, min_coercible_ratio: float) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    nonnull = s.dropna()
    if nonnull.empty:
        return False
    coerced = pd.to_numeric(nonnull, errors="coerce")
    ratio = float(coerced.notna().mean())
    return ratio >= min_coercible_ratio


def _nunique_nonnull(s: pd.Series) -> int:
    nonnull = s.dropna()
    if nonnull.empty:
        return 0
    # nunique on object can be expensive; this is still acceptable for validation
    return int(nonnull.nunique(dropna=True))


# ----------------------------
# Window parsing
# ----------------------------

def _parse_int(v: Any, field: str, issues: List[str]) -> Optional[int]:
    """
    Protocol windows are stored as strings.
    For a deterministic validator, we enforce a strict grammar:
      - integer strings like "-30", "0", "7"
    This keeps downstream temporal logic unambiguous.
    """
    if not isinstance(v, str) or not v.strip():
        issues.append(f"{field} is empty.")
        return None
    try:
        return int(v.strip())
    except Exception:
        issues.append(f"{field} must be an integer string (e.g., '-30', '0', '7'); got '{v}'.")
        return None


def _dedupe(xs: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# ----------------------------
# State plumbing (matches your node style)
# ----------------------------

def _require_dataset_meta_protocol(
    state: ConversationState,
) -> Tuple[DatasetState, MetadataState, ProtocolState, Optional[str]]:
    dataset = state.get("dataset")
    meta = state.get("metadata")
    protocol = state.get("protocol")
    if not isinstance(dataset, dict) or not isinstance(meta, dict) or not isinstance(protocol, dict):
        return cast(DatasetState, {}), cast(MetadataState, {}), cast(ProtocolState, {}), "dataset/metadata/protocol missing."
    return dataset, meta, protocol, None


def _append_ai(state: ConversationState, content: str) -> None:
    msgs = state.get("messages") or []
    msgs = list(msgs)
    msgs.append(AIMessage(content=content))
    state["messages"] = msgs


def _succeed(state: ConversationState, msg: Optional[str]) -> ConversationState:
    c = state.get("control")
    c["current_stage_status"] = "DONE"
    c["action_required"] = "NONE"
    c["node_message"] = msg
    state["control"] = c
    return state


def _need_input(state: ConversationState, msg: str) -> ConversationState:
    c = state.get("control")
    c["current_stage_status"] = "PENDING"
    c["action_required"] = "NEEDS_INPUT"
    c["node_message"] = msg
    state["control"] = c
    return state


def _fatal(state: ConversationState, msg: str) -> ConversationState:
    c = state.get("control")
    c["current_stage_status"] = "ABORTED"
    c["action_required"] = "NEEDS_INPUT"
    c["node_message"] = msg
    state["control"] = c
    return state
