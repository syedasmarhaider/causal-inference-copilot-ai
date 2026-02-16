from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any,  Dict, List, Literal, Optional, Sequence, Tuple, cast
from uuid import UUID, uuid4

import pandas as pd

from python.domain.repo.data_repo import DataRepo
from python.workflows.state.control_state import ACTION
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState, ConversationStateHelpers
from python.workflows.state.dataset_state import  DatasetState, DatasetStateHelpers
from python.workflows.state.inference_ready_state import InferenceReadyState
from python.workflows.state.protocol_state import (
    ExclusionRule,
    FilterOp,
    ProtocolState,
    TreatmentSpec,
    OutcomeSpec,
)

log = logging.getLogger(__name__)



# =============================================================================
# Node config
# =============================================================================

@dataclass(frozen=True)
class PrepareInferenceReadyConfig:
    max_categories_profile: int = 30
    sample_distinct_profile: int = 50
    compute_quantiles_profile: bool = True

    # Encoding limits (avoid pathological wide one-hot)
    max_one_hot_levels: int = 200

    # If any NaNs remain in W/X after imputation, drop those rows (guardrail)
    drop_rows_if_remaining_nan: bool = True


# =============================================================================
# Public node factory
# =============================================================================

def make_prepare_inference_ready_node(
    *,
    data_repo: DataRepo,
) -> CallableNodeFunc:
    def _run(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        cfg: PrepareInferenceReadyConfig = PrepareInferenceReadyConfig()
        try:
            protocol_state: ProtocolState | None =  state.get("protocol")
            if protocol_state is None:
                msg = "Protocol state is missing; cannot prepare inference-ready state."
                ConversationStateHelpers.append_ai_message(state, msg)
                return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)
            
            dataset_state: DatasetState =  state.get("dataset")
            dataset_state_id = dataset_state.get("id")
            if dataset_state_id is None:
                msg = "Dataset state or dataset id is missing; cannot prepare inference-ready state."
                ConversationStateHelpers.append_ai_message(state, msg)
                return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)
            
            df_for_copy = data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=dataset_state_id,
                limit=None,
            )
            
            if df_for_copy.empty:
                msg = "Dataset is empty; cannot prepare inference-ready state."
                ConversationStateHelpers.append_ai_message(state, msg)
                return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

            df = df_for_copy.copy()

            # 1) Apply exclusions
            df, excl_summary = _apply_exclusions(df, protocol_state["exclusions"])

            # 2) Apply treatment inclusion + normalize
            df, prepared_treatment, T_col, t_metrics, t_meta = _prepare_treatment(df, protocol_state["treatment_spec"], cfg)

            # 3) Apply outcome inclusion + normalize
            df, prepared_outcome, Y_col, y_metrics, y_meta = _prepare_outcome(df, protocol_state["outcome_spec"], cfg)

            # 4) Build W/X column lists (deterministic order = df.columns order)
            W_cols = _stable_existing_columns(df, protocol_state.get("covariates", []))
            X_cols = _stable_existing_columns(df, protocol_state.get("effect_modifiers", []))

            # Remove accidental overlaps with T/Y (hard invariant)
            W_cols = [c for c in W_cols if c not in (T_col, Y_col)]
            X_cols = [c for c in X_cols if c not in (T_col, Y_col) and c not in set(W_cols)]

            # 5) Keep only necessary columns (T/Y/W/X plus time_zero column if applicable)
            keep: List[str] = [T_col, Y_col, *W_cols, *X_cols]
            tz_keep = _maybe_time_zero_column(protocol_state, df)
            if tz_keep is not None and tz_keep not in keep:
                keep.append(tz_keep)

            df = df.loc[:, _stable_existing_columns(df, keep)].copy()

            # 6) Prepare W/X (impute + one-hot + numeric coercion)
            df, W_prepared, W_meta = _prepare_feature_block(df, cols=W_cols, role="W", cfg=cfg)
            df, X_prepared, X_meta = _prepare_feature_block(df, cols=X_cols, role="X", cfg=cfg)

            # Update authoritative lists to prepared columns
            W_cols_prepared = W_prepared
            X_cols_prepared = X_prepared

            # 7) Final NaN guard
            nan_cols = _columns_with_nan(df)
            if nan_cols:
                if cfg.drop_rows_if_remaining_nan:
                    before = int(df.shape[0])
                    df = df.dropna(axis=0, how="any") # pyright: ignore[reportUnknownMemberType]
                    after = int(df.shape[0])
                    excl_summary["rules"].append(
                        {"type": "nan_guard_drop_rows", "n_before": before, "n_after": after, "dropped_due_to_columns": nan_cols}
                    )
                else:
                    msg = f"Preparation produced NaNs in columns: {nan_cols}. Fix transforms/imputation before modeling."
                    ConversationStateHelpers.append_ai_message(state, msg)
                    return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

            if df.empty or df[T_col].nunique() < 2 or df[Y_col].nunique() < 2:
                msg = "Prepared dataset has no rows or lacks treatment/outcome variation; cannot proceed to inference-ready state required to re-run compile protocol state."
                ConversationStateHelpers.append_ai_message(state, msg)
                return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)
            
            # 8) Save prepared dataset + profile it
            new_dataset_id = uuid4()
            prepared_dataset = DatasetState()
            prepared_dataset["id"] = new_dataset_id       
            prepared_dataset["summary"] = DatasetStateHelpers.extract_dataset_summary(
                    df,
                    max_categories=cfg.max_categories_profile,
                    sample_distinct=cfg.sample_distinct_profile,
                    compute_quantiles=cfg.compute_quantiles_profile,
                    strict=True,
                )
            
            data_repo.save_csv_data(
                df=df,
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=new_dataset_id,
            )
            feature_sets = _build_feature_sets(W_cols_prepared, X_cols_prepared)
            prepared_columns = [t_meta, y_meta, *W_meta, *X_meta]
            metrics = _merge_metrics(excl_summary, t_metrics, y_metrics, df)
            ir: InferenceReadyState = cast(
                InferenceReadyState,
                {
                    "prepared_dataset": prepared_dataset,
                    "treatment": prepared_treatment,
                    "outcome": prepared_outcome,
                    "T_col": T_col,
                    "Y_cols": [Y_col],  # protocol has single Y; keep list shape for your pipeline
                    "W_cols": W_cols_prepared,
                    "X_cols": X_cols_prepared,
                    "feature_sets": feature_sets,
                    "prepared_columns": prepared_columns,
                    "exclusions_summary": excl_summary,
                    "metrics": metrics,
                },
            )

            state["inference_ready"] = ir

            msg = (
                f"Inference-ready dataset prepared: n_rows={int(df.shape[0])}, "
                f"T='{T_col}', Y='{Y_col}', W={len(W_cols_prepared)}, X={len(X_cols_prepared)}."
            )
            ConversationStateHelpers.append_ai_message(state, msg)
            return ConversationStateHelpers.set_done(state, cast(ACTION, "NONE"), msg)

        except Exception as e:
            msg = f"Prepare inference-ready failed: {e!r}"
            ConversationStateHelpers.append_ai_message(state, msg)
            # Best-effort attach error to inference_ready if possible
            try:
                state["inference_ready"] = cast(InferenceReadyState, {"error": msg})
            except Exception:
                pass
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), msg)

    return _run


# =============================================================================
# Exclusions
# =============================================================================

def _apply_exclusions(df: pd.DataFrame, rules: Sequence[ExclusionRule]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    n_before = int(df.shape[0])
    audit_rules: List[Dict[str, Any]] = []

    out = df
    for i, r in enumerate(list(rules)):
        col = r.get("column")
        op =  r.get("op")
        vals = r.get("values", [])
        reason = r.get("reason", "")

        if col not in out.columns:
            raise RuntimeError(f"exclusions[{i}].column invalid or missing in dataset: {col!r}")
        if  not reason.strip():
            raise RuntimeError(f"exclusions[{i}].reason must be non-empty string")

        before = int(out.shape[0])
        mask_exclude = _filter_mask(out[col], op, vals)
        out = out.loc[~mask_exclude].copy()
        after = int(out.shape[0])

        audit_rules.append(
            {
                "type": "exclusion",
                "index": i,
                "column": col,
                "op": op,
                "values": list(vals),
                "reason": reason,
                "n_before": before,
                "n_after": after,
                "n_removed": before - after,
            }
        )

    n_after = int(out.shape[0])
    summary: Dict[str, Any] = {"n_before": n_before, "n_after": n_after, "rules": audit_rules}
    return out, summary


def _filter_mask(s: pd.Series, op: FilterOp, values: List[str]) -> pd.Series:
    # Equality-style ops: compare normalized strings
    ss = s.astype("string")

    if op == "==":
        return ss.isin([v for v in values])
    if op == "!=":
        # missing should not match anything => keep rows with missing (do not exclude)
        return ss.notna() & (~ss.isin([v for v in values]))
    if op == "in":
        return ss.isin([v for v in values])
    if op == "not_in":
        return ss.notna() & (~ss.isin([v for v in values]))

    # Numeric comparisons: parse first value
    if not values:
        # no threshold -> exclude nothing
        return pd.Series([False] * len(s), index=s.index)

    try:
        thr = float(values[0])
    except Exception:
        # cannot parse -> exclude nothing (strict, but non-destructive)
        return pd.Series([False] * len(s), index=s.index)

    sn = pd.to_numeric(s, errors="coerce")

    if op == ">=":
        return sn.notna() & (sn >= thr)
    if op == "<=":
        return sn.notna() & (sn <= thr)
    if op == ">":
        return sn.notna() & (sn > thr)
    if op == "<":
        return sn.notna() & (sn < thr)

    # Unknown op
    raise RuntimeError(f"Unsupported FilterOp: {op!r}")


# =============================================================================
# Treatment / outcome prep
# =============================================================================

def _prepare_treatment(
    df: pd.DataFrame,
    spec: TreatmentSpec,
    cfg: PrepareInferenceReadyConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]:
    kind = spec.get("kind")
    col = spec.get("column")
    if  col not in df.columns:
        raise RuntimeError(f"Invalid treatment_spec: kind={kind!r}, column={col!r}")

    out = df

    if kind == "binary":
        tv = cast(List[str], spec.get("treated_values", []))
        cv = cast(List[str], spec.get("control_values", []))
        if not tv or not cv:
            raise RuntimeError("binary treatment_spec requires non-empty treated_values and control_values.")
        overlap = set(tv).intersection(set(cv))
        if overlap:
            raise RuntimeError(f"binary treatment_spec treated/control overlap: {sorted(overlap)}")

        s = out[col].astype("string")
        keep = s.isin(tv + cv)
        out = out.loc[keep].copy()

        # normalize to 0/1
        t01 = pd.Series([None] * len(out), index=out.index, dtype="Int64")
        t01.loc[out[col].astype("string").isin(tv)] = 1
        t01.loc[out[col].astype("string").isin(cv)] = 0

        T_col = f"{col}__T"
        out[T_col] = t01.astype("int64")
        out = out.drop(columns=[col])

        n_treated = int((out[T_col] == 1).sum())
        n_control = int((out[T_col] == 0).sum())
        treated_share = float(n_treated / (n_treated + n_control)) if (n_treated + n_control) > 0 else 0.0

        prepared_t = {"kind": "binary", "column": T_col, "treated": "1", "control": "0"}
        metrics = {"n_treated": n_treated, "n_control": n_control, "treated_share": treated_share}

        meta = _col_meta(T_col, role="T", dtype=str(out[T_col].dtype), encoding="none", imputation="drop_rows", notes="binary normalized to 0/1")
        return out, prepared_t, T_col, metrics, meta

    if kind == "continuous":
        vmin = spec.get("valid_min")
        vmax = spec.get("valid_max")

        sn = pd.to_numeric(out[col], errors="coerce")
        keep = sn.notna()
        if isinstance(vmin, (int, float)):
            keep = keep & (sn >= float(vmin))
        if isinstance(vmax, (int, float)):
            keep = keep & (sn <= float(vmax))

        out = out.loc[keep].copy()
        T_col = f"{col}__T"
        out[T_col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        out = out.drop(columns=[col])

        prepared_t: Dict[str, Any] = {"kind": "continuous", "column": T_col}
        if isinstance(vmin, (int, float)):
            prepared_t["clip_min"] = float(vmin)
        if isinstance(vmax, (int, float)):
            prepared_t["clip_max"] = float(vmax)

        metrics: Dict[str, Any] = {}
        meta = _col_meta(T_col, role="T", dtype=str(out[T_col].dtype), encoding="none", imputation="drop_rows", notes="continuous numeric coercion")
        return out, prepared_t, T_col, metrics, meta

    if kind == "categorical":
        lv = cast(List[str], spec.get("included_levels", []))
        if len(lv) < 2:
            raise RuntimeError("categorical treatment_spec requires included_levels with len>=2.")

        s = out[col].astype("string")
        keep = s.isin(lv)
        out = out.loc[keep].copy()

        # baseline = most frequent included level (tie-break lexical)
        vc = out[col].astype("string").value_counts(dropna=True)
        freq: Dict[str, int] = {str(k): int(v) for k, v in vc.items()}
        baseline = _pick_baseline(lv, freq)

        levels = [baseline] + sorted([x for x in lv if x != baseline])
        code_map = {lvl: i for i, lvl in enumerate(levels)}

        T_col = f"{col}__T"
        out[T_col] = out[col].astype("string").map(code_map).astype("int64")
        out = out.drop(columns=[col])

        prepared_t = {"kind": "categorical", "column": T_col, "levels": levels, "baseline": baseline}
        metrics = {}
        meta = _col_meta(T_col, role="T", dtype=str(out[T_col].dtype), encoding="ordinal", imputation="drop_rows", notes=f"categorical encoded; baseline='{baseline}'")
        return out, prepared_t, T_col, metrics, meta

    raise RuntimeError(f"Unsupported treatment_spec.kind: {kind!r}")


def _prepare_outcome(
    df: pd.DataFrame,
    spec: OutcomeSpec,
    cfg: PrepareInferenceReadyConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]:
    kind = spec.get("kind")
    col = spec.get("column")
    if col not in df.columns:
        raise RuntimeError(f"Invalid outcome_spec: kind={kind!r}, column={col!r}")

    out = df

    if kind == "binary":
        ev = cast(List[str], spec.get("event_values", []))
        nev = cast(List[str], spec.get("non_event_values", []))
        if not ev or not nev:
            raise RuntimeError("binary outcome_spec requires non-empty event_values and non_event_values.")
        overlap = set(ev).intersection(set(nev))
        if overlap:
            raise RuntimeError(f"binary outcome_spec event/non_event overlap: {sorted(overlap)}")

        s = out[col].astype("string")
        keep = s.isin(ev + nev)
        out = out.loc[keep].copy()

        y01 = pd.Series([None] * len(out), index=out.index, dtype="Int64")
        y01.loc[out[col].astype("string").isin(ev)] = 1
        y01.loc[out[col].astype("string").isin(nev)] = 0

        Y_col = f"{col}__Y"
        out[Y_col] = y01.astype("int64")
        out = out.drop(columns=[col])

        n_event = int((out[Y_col] == 1).sum())
        n_non_event = int((out[Y_col] == 0).sum())

        prepared_y = {"kind": "binary", "column": Y_col, "event": "1", "non_event": "0"}
        metrics = {"n_event": n_event, "n_non_event": n_non_event}

        meta = _col_meta(Y_col, role="Y", dtype=str(out[Y_col].dtype), encoding="none", imputation="drop_rows", notes="binary normalized to 0/1")
        return out, prepared_y, Y_col, metrics, meta

    if kind == "continuous":
        vmin = spec.get("valid_min")
        vmax = spec.get("valid_max")

        sn = pd.to_numeric(out[col], errors="coerce")
        keep = sn.notna()
        if isinstance(vmin, (int, float)):
            keep = keep & (sn >= float(vmin))
        if isinstance(vmax, (int, float)):
            keep = keep & (sn <= float(vmax))

        out = out.loc[keep].copy()
        Y_col = f"{col}__Y"
        out[Y_col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        out = out.drop(columns=[col])

        prepared_y: Dict[str, Any] = {"kind": "continuous", "column": Y_col}
        if isinstance(vmin, (int, float)):
            prepared_y["clip_min"] = float(vmin)
        if isinstance(vmax, (int, float)):
            prepared_y["clip_max"] = float(vmax)

        metrics: Dict[str, Any] = {}
        meta = _col_meta(Y_col, role="Y", dtype=str(out[Y_col].dtype), encoding="none", imputation="drop_rows", notes="continuous numeric coercion")
        return out, prepared_y, Y_col, metrics, meta

    if kind == "categorical":
        lv = cast(List[str], spec.get("included_levels", []))
        if len(lv) < 2:
            raise RuntimeError("categorical outcome_spec requires included_levels with len>=2.")

        s = out[col].astype("string")
        keep = s.isin(lv)
        out = out.loc[keep].copy()

        vc = out[col].astype("string").value_counts(dropna=True)
        freq: Dict[str, int] = {str(k): int(v) for k, v in vc.items()}
        baseline = _pick_baseline(lv, freq)

        levels = [baseline] + sorted([x for x in lv if x != baseline])
        code_map = {lvl: i for i, lvl in enumerate(levels)}

        Y_col = f"{col}__Y"
        out[Y_col] = out[col].astype("string").map(code_map).astype("int64")
        out = out.drop(columns=[col])

        prepared_y = {"kind": "categorical", "column": Y_col, "levels": levels, "baseline": baseline}
        metrics = {}
        meta = _col_meta(Y_col, role="Y", dtype=str(out[Y_col].dtype), encoding="ordinal", imputation="drop_rows", notes=f"categorical encoded; baseline='{baseline}'")
        return out, prepared_y, Y_col, metrics, meta

    raise RuntimeError(f"Unsupported outcome_spec.kind: {kind!r}")


def _pick_baseline(levels: List[str], freq: Dict[str, int]) -> str:
    # deterministic: max freq, tie-break lexical
    scored = sorted([(freq.get(l, 0), l) for l in levels], key=lambda x: (-x[0], x[1]))
    return scored[0][1]


# =============================================================================
# Feature prep (W/X): impute + one-hot + numeric coercion
# =============================================================================

def _prepare_feature_block(
    df: pd.DataFrame,
    *,
    cols: List[str],
    role: Literal["W", "X"],
    cfg: PrepareInferenceReadyConfig,
) -> Tuple[pd.DataFrame, List[str], List[Dict[str, Any]]]:
    if not cols:
        return df, [], []

    out = df.copy()
    produced_cols: List[str] = []
    metas: List[Dict[str, Any]] = []

    # Decide per-column by dtype
    for c in cols:
        if c not in out.columns:
            continue

        s = out[c]
        kind = _infer_kind_from_series(s)

        if kind == "NUMERIC":
            sn = pd.to_numeric(s, errors="coerce")
            med = float(sn.median()) if sn.notna().any() else 0.0
            sn = sn.fillna(med).astype("float64") # type: ignore
            out[c] = sn

            produced_cols.append(c)
            metas.append(_col_meta(c, role=role, dtype=str(out[c].dtype), encoding="none", imputation="median", notes="numeric; median-imputed"))

        else:
            # categorical/boolean/other -> stringify + mode fill + one-hot
            ss = s.astype("string")
            mode_val = None
            if ss.notna().any():
                mode_val = str(ss.mode(dropna=True).iloc[0])
            fill = mode_val if mode_val is not None else ""
            ss = ss.fillna(fill) # pyright: ignore[reportUnknownMemberType]

            levels = sorted(set([str(x) for x in ss.unique()]))
            if len(levels) > cfg.max_one_hot_levels:
                raise RuntimeError(
                    f"Column '{c}' has {len(levels)} levels > max_one_hot_levels={cfg.max_one_hot_levels}. "
                    "Reduce levels upstream or increase max_one_hot_levels."
                )

            dummy_cols = [f"{c}={lvl}" for lvl in levels]
            d = pd.get_dummies(pd.Categorical(ss, categories=levels), prefix=c, prefix_sep="=", dtype="int8")

            # Ensure full deterministic column set and order
            d = d.reindex(columns=dummy_cols, fill_value=0)

            out = out.drop(columns=[c])
            out = pd.concat([out, d], axis=1)

            produced_cols.extend(dummy_cols)
            for dc in dummy_cols:
                metas.append(_col_meta(dc, role=role, dtype=str(out[dc].dtype), encoding="none", imputation="none", notes=f"one-hot from '{c}'"))

    # Stable order for produced block: preserve creation order (already deterministic)
    return out, produced_cols, metas


def _infer_kind_from_series(s: pd.Series) -> Literal["NUMERIC", "CATEGORICAL"]:
    dt = str(s.dtype).lower()
    if any(x in dt for x in ("int", "float", "double", "numeric", "decimal")):
        return "NUMERIC"
    # bool treated as categorical for one-hot
    return "CATEGORICAL"


# =============================================================================
# Deterministic column utilities
# =============================================================================

def _stable_existing_columns(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    want = [c for c in cols if isinstance(c, str) and c.strip()]
    want_set = set(want)
    # deterministic: df.columns order
    return [c for c in df.columns if c in want_set]


def _maybe_time_zero_column(proto: ProtocolState, df: pd.DataFrame) -> Optional[str]:
    # Keep time_zero column only when specified as COLUMN and exists.
    if proto.get("time_zero_type") == "COLUMN":
        tz = proto.get("time_zero")
        if tz in df.columns:
            return tz
    return None


def _columns_with_nan(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if df[c].isna().any()]
    return cols


# =============================================================================
# Prepared column meta (match your earlier PreparedColumnMeta shape)
# =============================================================================

def _col_meta(
    name: str,
    *,
    role: Literal["T", "Y", "W", "X", "other"],
    dtype: str,
    encoding: str,
    imputation: str,
    notes: str,
) -> Dict[str, Any]:
    # missing_rate/n_unique computed later is expensive per-col; keep deterministic + cheap here.
    return {
        "name": name,
        "role": role,
        "dtype": dtype,
        "missing_rate": 0.0,
        "n_unique": 0,
        "encoding": encoding,
        "imputation": imputation,
        "notes": notes,
    }


# =============================================================================
# Feature sets + metrics
# =============================================================================

def _build_feature_sets(W_cols: List[str], X_cols: List[str]) -> Dict[str, List[str]]:
    seen: set[str] = set()
    xw: List[str] = []
    for c in [*X_cols, *W_cols]:
        if c not in seen:
            seen.add(c)
            xw.append(c)
    return {"W": list(W_cols), "X": list(X_cols), "XW": xw}


def _merge_metrics(excl_summary: Dict[str, Any], t_metrics: Dict[str, Any], y_metrics: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    m["n_rows_source"] = int(excl_summary.get("n_before", 0))
    m["n_rows_after_exclusions"] = int(excl_summary.get("n_after", 0))
    m["n_rows_final"] = int(df.shape[0])
    m.update(t_metrics)
    m.update(y_metrics)
    return m
