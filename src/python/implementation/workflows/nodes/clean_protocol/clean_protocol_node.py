from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple
from uuid import UUID, uuid4

import pandas as pd
import pandas.api.types as ptypes

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.clean_protocol.clean_protocol_deps import CleanProtocolDeps

from python.implementation.workflows.nodes.clean_protocol.clean_protocol_prompts import get_clean_protocol_node_info
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolPayloadModel, CleanProtocolState
from python.implementation.workflows.nodes.compile_protocol.protocol_specs import (
    BinaryOutcomeSpecModel,
    BinaryTreatmentSpecModel,
    CategoricalOutcomeSpecModel,
    CategoricalTreatmentSpecModel,
    ContinuousOutcomeSpecModel,
    ContinuousTreatmentSpecModel,
    DurationOutcomeSpecModel,
    ExclusionRuleModel,
    ProtocolSpec,
)
from python.implementation.workflows.utils.utils import BOOL_FALSE, BOOL_TRUE


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanProtocolNode(Node):
    """
    Clean dataset into an "inference-ready" dataset artifact:

      1) Drop cols to required-by-protocol
      2) Normalize missing sentinels + global null purge + exclusions
      3) Keep treatment/outcome domains (whitelist)
      4) Basic feasibility checks
      5) Save cleaned dataset -> new dataset_id
      6) Return InferenceReadyState(clean_dataset_id=...)

    No LLM, deterministic.
    """

    NAME: ClassVar[str] = CleanProtocolState.NAME

    data_repo: DataRepo

    # behavior knobs
    strict_required_cols: bool = True
    missing_sentinels: Tuple[str, ...] = ("na", "nan", "null")

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_clean_protocol_node_info()

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        tool_factory: ToolFactory,
        previous_state_dependencies: Mapping[str, Any],
        messages_history: Optional[Sequence[ChatMessage]],
    ) -> State:
        try:
            deps = CleanProtocolDeps.from_loaded(previous_state_dependencies)
            dataset_id = deps.load_dataset.payload.id
            if dataset_id is None:
                return CleanProtocolState(
                    payload=CleanProtocolPayloadModel(
                        clean_dataset_id=None,
                        cleaning_error="LOAD_DATASET.id is missing; cannot load data.",
                        user_message="Dataset id missing. Re-run LOAD_DATASET.",
                    )
                )

            compiled_protocol = _require_compiled_protocol(deps.compile_protocol)
            if compiled_protocol is None:
                return CleanProtocolState(
                    payload=CleanProtocolPayloadModel(
                        clean_dataset_id=None,
                        cleaning_error="COMPILE_PROTOCOL produced no protocol.",
                        user_message="Protocol missing. Re-run COMPILE_PROTOCOL.",
                    )
                )

            df = self.data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=dataset_id,
                limit=None,
            )
            if df.empty:
                return CleanProtocolState(
                    payload=CleanProtocolPayloadModel(
                        clean_dataset_id=None,
                        cleaning_error=f"Dataset is empty (dataset_id={dataset_id}).",
                        user_message="Dataset is empty; cannot prepare inference-ready data.",
                    )
                )

            n_rows_0 = int(df.shape[0])
            n_cols_0 = int(df.shape[1])

            # 1) Drop columns to required-by-protocol
            df1, drop_summary = edit_df_drop_cols_expect_required(
                df,
                compiled_protocol,
                keep_all_original=False,
                strict=self.strict_required_cols,
            )

            # 2) Null purge + exclusions
            df2, excl_summary = apply_null_purge_then_exclusions(
                df1,
                list(compiled_protocol.exclusions),
                missing_sentinels=self.missing_sentinels,
            )

            # 3) Treatment/outcome domain keep (whitelist)
            df3, domain_summary = apply_treatment_outcome_domain_keep(
                df2,
                compiled_protocol,
                keep_treatment_domain=True,
                keep_outcome_domain=True,
                dropna_on_domain_cols=False,
            )

            # 4) Basic feasibility checks (fast + actionable)
            feas_err = _feasibility_error(df3, compiled_protocol)
            if feas_err is not None:
                msg = _render_failure_message(
                    cleaning_error=feas_err,
                    n_rows_0=n_rows_0,
                    n_cols_0=n_cols_0,
                    drop_summary=drop_summary,
                    excl_summary=excl_summary,
                    domain_summary=domain_summary,
                )
                return CleanProtocolState(
                    payload=CleanProtocolPayloadModel(
                        clean_dataset_id=None,
                        cleaning_error=feas_err,
                        user_message=msg,
                    )
                )

            # 5) Save cleaned dataset
            clean_id = uuid4()
            self.data_repo.save_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=clean_id,
                df=df3,
                overwrite=True,
                include_index=False,
            )

            # 6) Success state
            msg_ok = _render_success_message(
                clean_dataset_id=clean_id,
                n_rows_0=n_rows_0,
                n_cols_0=n_cols_0,
                df_clean=df3,
                drop_summary=drop_summary,
                excl_summary=excl_summary,
                domain_summary=domain_summary,
            )
            return CleanProtocolState(
                payload=CleanProtocolPayloadModel(
                    clean_dataset_id=clean_id,
                    cleaning_error=None,
                    user_message=msg_ok,
                )
            )

        except Exception as e:
            log.exception("CleanProtocolNode failed")
            return CleanProtocolState(
                payload=CleanProtocolPayloadModel(
                    clean_dataset_id=None,
                    cleaning_error=f"Compile inference failed: {e!r}",
                    user_message=f"Compile inference failed: {e!r}",
                )
            )


# =============================================================================
# Protocol extraction (no guessing of dependency keys; only internal field)
# =============================================================================

def _require_compiled_protocol(compile_protocol_state: Any) -> Optional[ProtocolSpec]:
    """
    CompileProtocolState is expected to carry the compiled ProtocolSpec in `.protocol`.

    If your CompileProtocolState uses a different attribute name, change it here (single source of truth).
    """
    proto = getattr(compile_protocol_state, "protocol", None)
    return proto if isinstance(proto, ProtocolSpec) else None


# =============================================================================
# Feasibility checks (minimal + deterministic)
# =============================================================================

def _feasibility_error(df: pd.DataFrame, protocol: ProtocolSpec) -> Optional[str]:
    if df.empty:
        return "Cleaned dataset has zero rows after preprocessing (null purge/exclusions/domain filters)."

    if int(df.shape[1]) == 0:
        return "Cleaned dataset has zero columns after dropping to required columns."

    # Required modeling columns
    tcol = protocol.treatment_spec.column
    ys = protocol.outcome_spec

    needed = {tcol}
    if isinstance(ys, DurationOutcomeSpecModel):
        needed.add(ys.duration_column)
        needed.add(ys.event_column)
    else:
        needed.add(ys.column)

    missing = [c for c in needed if c not in df.columns]
    if missing:
        return f"Cleaned dataset is missing required modeling columns: {missing}"

    # Treatment variability
    ts = protocol.treatment_spec
    if isinstance(ts, BinaryTreatmentSpecModel):
        nunq = int(df[tcol].nunique(dropna=True))
        if nunq < 2:
            return f"Binary treatment column '{tcol}' has <2 unique values after filtering."
    elif isinstance(ts, CategoricalTreatmentSpecModel):
        nunq = int(df[tcol].nunique(dropna=True))
        if nunq < 2:
            return f"Categorical treatment column '{tcol}' has <2 levels present after filtering."
    elif isinstance(ts, ContinuousTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        nunq = int(df[tcol].nunique(dropna=True))
        if nunq < 2:
            return f"Continuous treatment column '{tcol}' has <=1 unique value after filtering."
    else:
        return f"Unsupported treatment spec kind: {getattr(ts, 'kind', None)!r}"    

    # Outcome variability
    if isinstance(ys, DurationOutcomeSpecModel):
        ecol = ys.event_column
        nunq_e = int(df[ecol].nunique(dropna=True))
        if nunq_e < 2:
            return f"Duration outcome event column '{ecol}' has <2 unique values after filtering."
    elif isinstance(ys, BinaryOutcomeSpecModel):
        ycol = ys.column
        nunq = int(df[ycol].nunique(dropna=True))
        if nunq < 2:
            return f"Binary outcome column '{ycol}' has <2 unique values after filtering."
    elif isinstance(ys, CategoricalOutcomeSpecModel):
        ycol = ys.column
        nunq = int(df[ycol].nunique(dropna=True))
        if nunq < 2:
            return f"Categorical outcome column '{ycol}' has <2 levels present after filtering."
    elif isinstance(ys, ContinuousOutcomeSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        ycol = ys.column
        nunq = int(df[ycol].nunique(dropna=True))
        if nunq < 2:
            return f"Continuous outcome column '{ycol}' has <=1 unique value after filtering."
    else:
        return f"Unsupported outcome spec kind: {getattr(ys, 'kind', None)!r}"    

    return None


# =============================================================================
# Message rendering
# =============================================================================

def _render_success_message(
    *,
    clean_dataset_id: UUID,
    n_rows_0: int,
    n_cols_0: int,
    df_clean: pd.DataFrame,
    drop_summary: DropColsSummary,
    excl_summary: Dict[str, Any],
    domain_summary: TreatmentOutcomeDomainSummary,
) -> str:
    n_rows_1 = int(df_clean.shape[0])
    n_cols_1 = int(df_clean.shape[1])

    parts: list[str] = []
    parts.append("Inference-ready dataset compiled successfully.")
    parts.append(f"- clean_dataset_id: {clean_dataset_id}")
    parts.append(f"- rows: {n_rows_0} -> {n_rows_1}")
    parts.append(f"- cols: {n_cols_0} -> {n_cols_1}")
    parts.append(f"- kept_cols: {len(drop_summary.kept_cols)}, dropped_cols: {len(drop_summary.dropped_cols)}")

    parts.append(
        f"- null_purge_removed: {int(excl_summary.get('n_removed_null_purge', 0))}, "
        f"exclusions_removed_total: {sum(int(r.get('n_removed', 0)) for r in excl_summary.get('applied', []))}"
    )

    parts.append(f"- domain_removed_total: {domain_summary.total_removed}")

    return "\n".join(parts)


def _render_failure_message(
    *,
    cleaning_error: str,
    n_rows_0: int,
    n_cols_0: int,
    drop_summary: DropColsSummary,
    excl_summary: Dict[str, Any],
    domain_summary: TreatmentOutcomeDomainSummary,
) -> str:
    parts: list[str] = []
    parts.append("Failed to compile inference-ready dataset.")
    parts.append(f"Error: {cleaning_error}")
    parts.append("")
    parts.append("Diagnostics:")
    parts.append(f"- rows_before: {n_rows_0}, cols_before: {n_cols_0}")
    parts.append(f"- kept_cols: {len(drop_summary.kept_cols)}, dropped_cols: {len(drop_summary.dropped_cols)}")
    if drop_summary.missing_required:
        parts.append(f"- missing_required_cols: {drop_summary.missing_required}")

    parts.append(f"- rows_after_null_purge: {int(excl_summary.get('n_rows_after_null_purge', 0))}")
    parts.append(f"- rows_after_exclusions: {int(excl_summary.get('n_rows_after', 0))}")
    parts.append(f"- rows_after_domain_keep: {domain_summary.n_rows_after}")

    # Optional: include first few exclusion rules audit rows
    applied = excl_summary.get("applied", [])
    if isinstance(applied, list) and applied:
        preview = applied[:5] # pyright: ignore[reportUnknownVariableType]
        parts.append(f"- exclusions_applied_preview: {preview}")

    return "\n".join(parts)


@dataclass(frozen=True)
class DropColsSummary:
    kept_cols: List[str]
    dropped_cols: List[str]
    missing_required: List[str]
    required_cols: List[str]


def edit_df_drop_cols_expect_required(
    df: pd.DataFrame,
    compiled_protocol: ProtocolSpec,
    *,
    keep_all_original: bool = False,
    strict: bool = True,
) -> Tuple[pd.DataFrame, DropColsSummary]:
    """
    Keep only columns required by the compiled protocol.

    Required columns:
      - treatment_spec.column
      - outcome_spec.column (or duration_column + event_column for duration)
      - covariates
      - effect_modifiers
      - optionally time_zero if time_zero_type == "COLUMN"

    Args:
      keep_all_original: if True, returns df unchanged but still computes summary.
      strict: if True, raise ValueError when any required column is missing.

    Returns:
      (df_filtered, summary)
    """
    required: Set[str] = set()

    # time zero
    if getattr(compiled_protocol, "time_zero_type", None) == "COLUMN":
        required.add(compiled_protocol.time_zero)

    # exclusions
    for ex in compiled_protocol.exclusions:
        required.add(ex.column)

    # treatment column
    required.add(compiled_protocol.treatment_spec.column)

    # outcome column(s)
    ys = compiled_protocol.outcome_spec
    # duration has two columns; others have .column
    if getattr(ys, "kind", None) == "duration":
        required.add(getattr(ys, "duration_column"))
        required.add(getattr(ys, "event_column"))
    else:
        required.add(getattr(ys, "column"))

    # covariates / effect modifiers
    required.update(list(compiled_protocol.covariates))
    required.update(list(compiled_protocol.effect_modifiers))

    # normalize: strip + remove empties (shouldn’t happen due to NonEmptyStr, but defensive)
    required = {c.strip() for c in required if c.strip()}

    df_cols = [str(c) for c in df.columns]
    df_col_set = set(df_cols)

    missing = sorted([c for c in required if c not in df_col_set])
    required_sorted = sorted(required)

    if strict and missing:
        raise ValueError(f"edit_df_drop_cols_expect_required: missing required columns: {missing}")

    if keep_all_original:
        kept = df_cols
        dropped: List[str] = []
        out = df.copy()
    else:
        kept = [c for c in df_cols if c in required]
        dropped = [c for c in df_cols if c not in required]
        out = df.loc[:, kept].copy()

    summary = DropColsSummary(
        kept_cols=kept,
        dropped_cols=dropped,
        missing_required=missing,
        required_cols=required_sorted,
    )
    return out, summary

# exclusions

def _normalize_missing_sentinels(
    df: pd.DataFrame,
    *,
    missing_sentinels: Sequence[str],
) -> pd.DataFrame:
    """
    Convert common *string* missing sentinels (e.g. 'NE', 'nan', 'null') into pd.NA
    so dropna() will purge them.
    Only touches object/string/category columns.
    """
    sent = {s.strip().casefold() for s in missing_sentinels if s.strip()}
    if not sent:
        return df.copy()

    out = df.copy()
    for c in out.columns:
        s = out[c]
        dt = s.dtype

        is_obj_or_str = ptypes.is_object_dtype(dt) or ptypes.is_string_dtype(dt)
        is_cat = isinstance(dt, pd.CategoricalDtype)

        if is_obj_or_str or is_cat:
            out[c] = s.map(lambda x: pd.NA if isinstance(x, str) and x.strip().casefold() in sent else x)

    return out


def _parse_bool_token_strict(v: str) -> Optional[bool]:
    s = v.strip().casefold()
    if s in BOOL_TRUE:
        return True
    if s in BOOL_FALSE:
        return False
    return None


def _parse_iso_datetime(v: str) -> Optional[pd.Timestamp]:
    try:
        return pd.Timestamp(datetime.fromisoformat(v.strip()))
    except Exception:
        return None


def _coerce_literals_for_series(s: pd.Series, values: Sequence[str]) -> List[Any]:
    """
    Coerce exclusion literal strings into a comparable type for this series *without modifying the df*.

    - bool dtype  -> strict tokens only: {true,1,yes,false,0,no}
    - numeric     -> float(...)
    - datetime    -> ISO datetime -> Timestamp
    - other       -> keep as strings
    """
    if ptypes.is_bool_dtype(s.dtype):
        out: List[bool] = []
        for raw in values:
            b = _parse_bool_token_strict(raw)
            if b is None:
                raise ValueError(
                    f"Invalid boolean literal {raw!r}. Allowed: {sorted(BOOL_TRUE | BOOL_FALSE)} (no y/n)."
                )
            out.append(b)
        return out

    if ptypes.is_numeric_dtype(s.dtype):
        outn: List[float] = []
        for raw in values:
            try:
                outn.append(float(raw))
            except Exception as e:
                raise ValueError(f"Numeric literal not parseable as float: {raw!r} ({e!r})") from e
        return outn

    if ptypes.is_datetime64_any_dtype(s.dtype):
        outd: List[pd.Timestamp] = []
        for raw in values:
            ts = _parse_iso_datetime(raw)
            if ts is None:
                raise ValueError(f"Datetime literal not parseable as ISO-8601: {raw!r}")
            outd.append(ts)
        return outd

    return [str(v) for v in values]


def apply_null_purge_then_exclusions(
    df: pd.DataFrame,
    exclusions: List[ExclusionRuleModel],
    *,
    missing_sentinels: Sequence[str] = ("na", "nan", "null"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    1) Normalize missing sentinels (string columns only) -> pd.NA
    2) Drop EVERY row with ANY missing in ANY column (global null purge)
    3) Apply exclusions sequentially (remove rows matching each rule)

    Supported ops: '==', 'in', 'not_in', '>', '>=', '<', '<='
    (No '!=' by design.)

    Returns: (new_df, summary_dict)
    """
    n0 = int(df.shape[0])

    cur = _normalize_missing_sentinels(df, missing_sentinels=missing_sentinels)
    cur = cur.dropna(axis=0, how="any").copy() # pyright: ignore[reportUnknownMemberType]

    n_after_nulls = int(cur.shape[0])
    applied: List[Dict[str, Any]] = []

    for i, rule in enumerate(exclusions):
        col = rule.column
        op = rule.op
        vals = list(rule.values)

        if col not in cur.columns:
            raise KeyError(f"Exclusion column not found in df: {col!r}")

        # Arity rules (this is not schema validation; it's required semantics)
        if op in ("==", ">", ">=", "<", "<=") and len(vals) != 1:
            raise ValueError(f"Exclusion {i}: op {op!r} requires exactly 1 value, got {len(vals)}")
        if op in ("in", "not_in") and len(vals) < 1:
            raise ValueError(f"Exclusion {i}: op {op!r} requires >= 1 value, got {len(vals)}")

        s = cur[col]
        n_before = int(cur.shape[0])

        coerced = _coerce_literals_for_series(s, vals)

        if op == "==":
            v0 = coerced[0]
            mask = s.eq(v0)

        elif op == "in":
            mask = s.isin(coerced)

        elif op == "not_in":
            mask = ~s.isin(coerced)

        elif op in (">", ">=", "<", "<="):
            # Only allow inequality on numeric/datetime dtypes (no silent coercion of the series)
            if not (ptypes.is_numeric_dtype(s.dtype) or ptypes.is_datetime64_any_dtype(s.dtype)):
                raise ValueError(f"Exclusion {i}: op {op!r} requires numeric/datetime column, got {s.dtype!r} for {col!r}")

            v0 = coerced[0]
            if op == ">":
                mask = s.gt(v0)
            elif op == ">=":
                mask = s.ge(v0)
            elif op == "<":
                mask = s.lt(v0)
            else:  # "<="
                mask = s.le(v0)

        else:
            # Should be unreachable because FilterOp restricts it, but keep explicit.
            raise ValueError(f"Unsupported exclusion operator: {op!r}")

        # Exclusion semantics: remove rows that match
        cur = cur.loc[~mask].copy()

        n_after = int(cur.shape[0])
        applied.append(
            {
                "index": i,
                "column": col,
                "op": op,
                "values": vals,
                "n_rows_before": n_before,
                "n_rows_after": n_after,
                "n_removed": n_before - n_after,
            }
        )

    n_final = int(cur.shape[0])
    summary: Dict[str, Any] = {
        "n_rows_before": n0,
        "n_rows_after_null_purge": n_after_nulls,
        "n_removed_null_purge": n0 - n_after_nulls,
        "applied": applied,
        "n_rows_after": n_final,
        "total_removed": n0 - n_final,
    }
    return cur, summary



@dataclass(frozen=True)
class TreatmentOutcomeDomainSummary:
    n_rows_before: int
    n_rows_after: int
    total_removed: int
    treatment: Optional[Dict[str, Any]]
    outcome: Optional[Dict[str, Any]]


def _norm_str_series(s: pd.Series) -> pd.Series:
    # preserves NA as <NA>; comparisons treat NA as not-in-domain
    return s.astype("string").str.strip().str.casefold()


def _mask_keep_in_domain(s: pd.Series, allowed_literals: Sequence[str]) -> pd.Series:
    """
    Returns a boolean mask KEEPING rows whose s value is in allowed_literals.

    - bool/numeric/datetime: type-coerce literals, then isin on the series
    - object/string/category: compare via normalized string representations (casefold+strip)
    """
    if not allowed_literals:
        # Nothing allowed => keep nothing (but this should not happen for your specs)
        return pd.Series([False] * len(s), index=s.index)

    if ptypes.is_bool_dtype(s.dtype) or ptypes.is_numeric_dtype(s.dtype) or ptypes.is_datetime64_any_dtype(s.dtype):
        coerced = _coerce_literals_for_series(s, allowed_literals)
        return s.isin(coerced)

    # object/string/category/other: normalize string representations
    s_norm = _norm_str_series(s)
    allowed_norm = [str(x).strip().casefold() for x in allowed_literals]
    return s_norm.isin(allowed_norm)


def apply_treatment_outcome_domain_keep(
    df: pd.DataFrame,
    compiled_protocol: ProtocolSpec,
    *,
    keep_treatment_domain: bool = True,
    keep_outcome_domain: bool = True,
    dropna_on_domain_cols: bool = False,
) -> Tuple[pd.DataFrame, TreatmentOutcomeDomainSummary]:
    """
    Keep only rows whose treatment/outcome values are within protocol-defined domains.

    This is a *separate* whitelist step (safe), intended to run AFTER:
      - edit_df_drop_cols_expect_required(...)
      - apply_null_purge_then_exclusions(...)

    Args:
      keep_treatment_domain: apply treatment domain whitelist if treatment is binary/categorical
      keep_outcome_domain: apply outcome domain whitelist if outcome is binary/categorical/duration
      dropna_on_domain_cols: if True, drop NA on the specific domain columns before whitelisting
                            (even if you already did global dropna earlier)

    Returns:
      (df_filtered, summary)
    """
    cur = df.copy()
    n0 = int(cur.shape[0])

    t_summary: Optional[Dict[str, Any]] = None
    y_summary: Optional[Dict[str, Any]] = None

    # ----------------------------
    # Treatment domain keep
    # ----------------------------
    if keep_treatment_domain:
        ts = compiled_protocol.treatment_spec
        tcol = ts.column

        if tcol not in cur.columns:
            raise KeyError(f"Treatment column not found in df: {tcol!r}")

        allowed_t: Optional[List[str]] = None
        if isinstance(ts, BinaryTreatmentSpecModel):
            allowed_t = [ts.treated, ts.control]
        elif isinstance(ts, CategoricalTreatmentSpecModel):
            allowed_t = list(ts.levels)
        elif isinstance(ts, ContinuousTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
            allowed_t = None  # no whitelist for continuous
        else:
            raise ValueError(f"Unknown treatment_spec kind: {getattr(ts, 'kind', None)!r}")   

        if allowed_t is not None:
            n_before = int(cur.shape[0])
            if dropna_on_domain_cols:
                cur = cur.dropna(axis=0, how="any", subset=[tcol]).copy() # pyright: ignore[reportUnknownMemberType]
            mask_keep = _mask_keep_in_domain(cur[tcol], allowed_t)
            cur = cur.loc[mask_keep].copy()
            n_after = int(cur.shape[0])

            t_summary = {
                "column": tcol,
                "allowed": allowed_t,
                "n_rows_before": n_before,
                "n_rows_after": n_after,
                "n_removed": n_before - n_after,
            }

    # ----------------------------
    # Outcome domain keep
    # ----------------------------
    if keep_outcome_domain:
        ys = compiled_protocol.outcome_spec

        # Duration: whitelist event_column values (event_value/censor_value)
        if isinstance(ys, DurationOutcomeSpecModel):
            ecol = ys.event_column
            if ecol not in cur.columns:
                raise KeyError(f"Outcome event_column not found in df: {ecol!r}")

            allowed_y = [ys.event_value, ys.censor_value]

            n_before = int(cur.shape[0])
            if dropna_on_domain_cols:
                # event_column plus duration_column are logically required for survival analysis
                dcol = ys.duration_column
                subset = [ecol] + ([dcol] if dcol in cur.columns else [])
                cur = cur.dropna(axis=0, how="any", subset=subset).copy() # pyright: ignore[reportUnknownMemberType]

            mask_keep = _mask_keep_in_domain(cur[ecol], allowed_y)
            cur = cur.loc[mask_keep].copy()
            n_after = int(cur.shape[0])

            y_summary = {
                "kind": "duration",
                "event_column": ecol,
                "allowed": allowed_y,
                "n_rows_before": n_before,
                "n_rows_after": n_after,
                "n_removed": n_before - n_after,
            }

        # Non-duration: column-based outcomes
        else:
            ycol = ys.column
            if ycol not in cur.columns:
                raise KeyError(f"Outcome column not found in df: {ycol!r}")

            allowed_y2: Optional[List[str]] = None
            if isinstance(ys, BinaryOutcomeSpecModel):
                allowed_y2 = [ys.event, ys.non_event]
            elif isinstance(ys, CategoricalOutcomeSpecModel):
                allowed_y2 = list(ys.levels)
            elif isinstance(ys, ContinuousOutcomeSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
                allowed_y2 = None  # no whitelist for continuous
            else:
                raise ValueError(f"Unknown outcome_spec kind: {getattr(ys, 'kind', None)!r}")    

            if allowed_y2 is not None:
                n_before = int(cur.shape[0])
                if dropna_on_domain_cols:
                    cur = cur.dropna(axis=0, how="any", subset=[ycol]).copy() # pyright: ignore[reportUnknownMemberType]

                mask_keep = _mask_keep_in_domain(cur[ycol], allowed_y2)
                cur = cur.loc[mask_keep].copy()
                n_after = int(cur.shape[0])

                y_summary = {
                    "kind": getattr(ys, "kind", "unknown"),
                    "column": ycol,
                    "allowed": allowed_y2,
                    "n_rows_before": n_before,
                    "n_rows_after": n_after,
                    "n_removed": n_before - n_after,
                }

    n_final = int(cur.shape[0])
    summary = TreatmentOutcomeDomainSummary(
        n_rows_before=n0,
        n_rows_after=n_final,
        total_removed=n0 - n_final,
        treatment=t_summary,
        outcome=y_summary,
    )
    return cur, summary