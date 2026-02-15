from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast
from uuid import UUID, uuid4

import pandas as pd
from pandas.api import types as ptypes

from python.domain.repo.data_repo import DataRepo
from python.workflows.state.control_state import ACTION
from python.workflows.state.dataset_state import DatasetStateHelpers
from python.workflows.state.inference_ready_state import (
    ExclusionApplicationSummary,
    InferenceReadyState,
    PreparedBinaryLabels,
    PreparedBinaryOutcome,
    PreparedCategoricalLabels,
    PreparedCategoricalOutcome,
    PreparedColumnMeta,
    PreparedContinuousMeta,
    PreparedContinuousOutcome,
    PreparedDatasetArtifact,
    PreparedDurationOutcome,
    PreparedOutcome,
    PreparedTreatment,
    PreparationMetrics,
)
from python.workflows.state.protocol_state import (
    BinaryOutcomeSpec,
    CategoricalOutcomeSpec,
    ContinuousOutcomeSpec,
    DurationOutcomeSpec,
    ExclusionRule,
    OutcomeSpec,
    ProtocolState,
    TreatmentSpec,
)
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState, ConversationStateHelpers


# ============================================================
# Public factory
# ============================================================
def make_prepare_inference_ready_node(
    data_repo: DataRepo,
) -> CallableNodeFunc:
    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        dataset_state = state.get("dataset", {})
        dataset_id = dataset_state.get("id")
        protocol = state.get("protocol")
        vstate = state.get("protocol_static_validation")

        if dataset_id is None:
            return ConversationStateHelpers.set_abort(
                state, cast(ACTION, "NONE"), "Dataset id missing; cannot prepare inference."
            )

        if protocol is None:
            return ConversationStateHelpers.set_abort(
                state, cast(ACTION, "NONE"), "Protocol missing; discuss the protocol first cannot prepare inference."
            )

        report = (vstate or {}).get("report") if vstate else None
        if report is None:
            return ConversationStateHelpers.set_pending(
                state, cast(ACTION, "NONE"), "Protocol validation report missing; cannot prepare inference."
            )

        status = report.get("status")
        if status == "FAIL":
            return ConversationStateHelpers.set_abort(
                state, cast(ACTION, "NONE"), "Protocol validation failed; inference preparation aborted."
            )

        # ----------------------------
        # Load data
        # ----------------------------
        try:
            df = data_repo.get_csv_data(
                user_id=user_id, conversation_id=conversation_id, dataset_id=dataset_id, limit=None
            )
        except Exception as e:
            err = f"Failed to load dataset for inference preparation: {type(e).__name__}: {e}"
            ConversationStateHelpers.append_ai_message(state, err)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), "Could not load dataset.")

        n_rows_source = int(df.shape[0])

        summary = dataset_state.get("summary")
        if summary is None:
            try:
                summary = DatasetStateHelpers.extract_column_profile(df, strict=True)
                dataset_state["summary"] = summary
            except Exception as e:
                err = f"Dataset profiling failed: {type(e).__name__}: {e}"
                ConversationStateHelpers.append_ai_message(state, err)
                return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), "Could not profile dataset.")

        # ----------------------------
        # Apply required NA drop (T/Y) + user exclusions
        # ----------------------------
        try:
            required_cols = _required_not_null_cols(protocol["treatment_spec"], protocol["outcome_spec"], df)
            df_filtered, excl_summary = _apply_exclusions(
                df,
                protocol.get("exclusions", []),
                required_not_null_cols=required_cols,
            )
        except Exception as e:
            err = f"Applying exclusions failed: {type(e).__name__}: {e}"
            ConversationStateHelpers.append_ai_message(state, err)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), "Exclusions application failed.")

        # We report both steps explicitly
        n_rows_after_required_drop = int(excl_summary.get("n_after_required_drop", excl_summary["n_before"]))
        n_rows_after_exclusions = int(excl_summary["n_after"])

        # ----------------------------
        # Canonicalize treatment/outcome encodings (best-effort)
        # ----------------------------
        try:
            df_prepared = df_filtered.copy(deep=False)

            treatment = _build_prepared_treatment(protocol["treatment_spec"], df_prepared)
            outcome = _build_prepared_outcome(protocol["outcome_spec"], df_prepared)

            _apply_treatment_canonicalization(df_prepared, protocol["treatment_spec"])
            _apply_outcome_canonicalization(df_prepared, protocol["outcome_spec"])
        except Exception as e:
            err = f"Canonicalization (treatment/outcome mapping) failed: {type(e).__name__}: {e}"
            ConversationStateHelpers.append_ai_message(state, err)
            return ConversationStateHelpers.set_abort(
                state, cast(ACTION, "NONE"), "Could not canonicalize treatment/outcome."
            )

        # ----------------------------
        # Build EconML conventions
        # ----------------------------
        T_col, Y_cols = _resolve_econml_cols(protocol["treatment_spec"], protocol["outcome_spec"])
        W_cols = _dedupe_preserve_order(protocol.get("covariates", []))
        X_cols = _dedupe_preserve_order(protocol.get("effect_modifiers", []))
        feature_sets = _build_feature_sets(W_cols=W_cols, X_cols=X_cols)

        # ----------------------------
        # Build prepared column metadata
        # ----------------------------
        try:
            prepared_cols = _build_prepared_columns_meta(
                df_prepared=df_prepared,
                summary=cast(Dict[str, Dict[str, Any]], summary),
                treatment_spec=protocol["treatment_spec"],
                outcome_spec=protocol["outcome_spec"],
                W_cols=W_cols,
                X_cols=X_cols,
            )
        except Exception as e:
            err = f"Prepared column metadata build failed: {type(e).__name__}: {e}"
            ConversationStateHelpers.append_ai_message(state, err)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), "Could not build prepared column metadata.")

        # ----------------------------
        # Metrics
        # ----------------------------
        metrics: PreparationMetrics = {
            "n_rows_source": n_rows_source,
            "n_rows_after_required_drop": n_rows_after_required_drop,
            "n_rows_after_exclusions": n_rows_after_exclusions,
            "n_rows_final": int(df_prepared.shape[0]),
        }

        _fill_treatment_metrics(metrics, df_prepared, protocol["treatment_spec"])
        _fill_outcome_metrics(metrics, df_prepared, protocol["outcome_spec"])
        metrics["max_missing_rate_W"] = _max_missing_rate(prepared_cols, role="W")
        metrics["max_missing_rate_X"] = _max_missing_rate(prepared_cols, role="X")

        # ----------------------------
        # Materialize prepared dataset artifact
        # ----------------------------
        prepared_artifact: Optional[PreparedDatasetArtifact] = None
        try:
            prepared_dataset_id = uuid4()
            storage_kind: Literal["DATA_REPO_CSV", "DATA_REPO_PARQUET"] = "DATA_REPO_CSV"

            data_repo.save_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=prepared_dataset_id,
                df=df_prepared,
            )

            schema_fingerprint = _schema_fingerprint(
                df=df_prepared,
                protocol=protocol,
                treatment=treatment,
                outcome=outcome,
                W_cols=W_cols,
                X_cols=X_cols,
                exclusions=protocol.get("exclusions", []),
            )

            prepared_artifact = {
                "dataset_id": prepared_dataset_id,
                "storage_kind": storage_kind,
                "schema_fingerprint": schema_fingerprint,
                "row_count": int(df_prepared.shape[0]),
                "created_from_dataset_id": dataset_id,
            }
        except Exception as e:
            ConversationStateHelpers.append_ai_message(state, f"could not materialize prepared dataset artifact. {e}")
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), f"Inference-ready state prepared.{e}")

        # ----------------------------
        # Assemble InferenceReadyState
        # ----------------------------
        inference_ready: InferenceReadyState = {
            "protocol": protocol,
            "treatment": treatment,
            "outcome": outcome,
            "T_col": T_col,
            "Y_cols": Y_cols,
            "W_cols": W_cols,
            "X_cols": X_cols,
            "feature_sets": feature_sets,
            "prepared_columns": prepared_cols,
            "exclusions_summary": excl_summary,
            "metrics": metrics,
        }
        inference_ready["prepared"] = prepared_artifact

        inference_ready["summary_text"] = _build_summary_text(
            report_status=cast(str, status),
            n_rows_source=n_rows_source,
            n_rows_after_required_drop=n_rows_after_required_drop,
            n_rows_after_exclusions=n_rows_after_exclusions,
            n_rows_final=int(df_prepared.shape[0]),
            required_not_null_cols=required_cols,
            T_col=T_col,
            Y_cols=Y_cols,
            W_cols=W_cols,
            X_cols=X_cols,
            exclusions_applied=len(protocol.get("exclusions", [])),
        )

        state["inference_ready"] = inference_ready
        ConversationStateHelpers.append_ai_message(state, inference_ready["summary_text"])
        return ConversationStateHelpers.set_done(state, cast(ACTION, "NONE"), "Inference-ready state prepared.")

    return node


# ============================================================
# Exclusions (required NA drop + user rules)
# ============================================================
ExclusionOp = Literal["==", "!=", "in", "not_in", ">=", "<=", ">", "<", "is_null", "not_null"]


def _required_not_null_cols(treatment_spec: TreatmentSpec, outcome_spec: OutcomeSpec, df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    tcol = cast(str, treatment_spec["column"])
    cols.append(tcol)

    if outcome_spec["kind"] == "duration":
        y = cast(Any, outcome_spec)
        cols.append(cast(str, y["event_column"]))
        cols.append(cast(str, y["duration_column"]))
    else:
        ycol = cast(str, cast(Any, outcome_spec)["column"])
        cols.append(ycol)

    # ensure exist; protocol should already be validated, but fail loudly if not
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Required treatment/outcome columns missing in dataset: {missing}")

    # dedupe preserve order
    seen: set[str] = set()
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _apply_exclusions(
    df: pd.DataFrame,
    exclusions: List[ExclusionRule],
    *,
    required_not_null_cols: List[str],
) -> Tuple[pd.DataFrame, ExclusionApplicationSummary]:
    n_before = int(df.shape[0])
    rules_audit: List[Dict[str, Any]] = []

    out = df

    # (1) ALWAYS drop true-missing in required T/Y columns first
    before_req = int(out.shape[0])
    out = out.dropna(subset=required_not_null_cols)
    after_req = int(out.shape[0])
    rules_audit.append(
        {
            "kind": "AUTO_DROP_REQUIRED_NA",
            "columns": list(required_not_null_cols),
            "n_before": before_req,
            "n_after": after_req,
            "n_removed": before_req - after_req,
        }
    )

    # (2) Apply user exclusions (rows to REMOVE)
    for r in exclusions or []:
        col = cast(str, r["column"])
        op = cast(ExclusionOp, r["op"])
        vals = cast(List[str], r.get("values", []) or [])
        reason = cast(str, r.get("reason", ""))

        if col not in out.columns:
            raise ValueError(f"Exclusion rule references missing column '{col}'.")

        before = int(out.shape[0])
        mask_keep = _mask_keep_for_rule(out[col], op, vals)
        out = out.loc[mask_keep].copy(deep=False)
        after = int(out.shape[0])

        rules_audit.append(
            {
                "kind": "USER_RULE",
                "column": col,
                "op": op,
                "values": list(vals),
                "reason": reason,
                "n_before": before,
                "n_after": after,
                "n_removed": before - after,
            }
        )

    summary: Dict[str, Any] = {
        "n_before": n_before,
        "n_after_required_drop": after_req,
        "n_after": int(out.shape[0]),
        "rules": rules_audit,
    }
    return out, cast(ExclusionApplicationSummary, summary)


def _mask_keep_for_rule(series: pd.Series, op: ExclusionOp, values: List[str]) -> pd.Series:
    # Exclusion rule defines rows to REMOVE; keep is the inverse.
    exclude_mask = _mask_exclude_for_rule(series, op, values)
    return ~exclude_mask


def _mask_exclude_for_rule(series: pd.Series, op: ExclusionOp, values: List[str]) -> pd.Series:
    s = series

    # Null checks: exclude missing or exclude non-missing
    if op == "is_null":
        return s.isna()
    if op == "not_null":
        return ~s.isna()

    # For other ops, empty values => exclude nothing
    cleaned_vals = [str(v) for v in (values or []) if str(v).strip() != ""]
    if not cleaned_vals and op in ("==", "!=", "in", "not_in"):
        return pd.Series([False] * len(s), index=s.index)

    coerced_vals = _coerce_values_for_series(s, cleaned_vals)

    if op in ("==", "in"):
        return s.isin(coerced_vals)

    if op in ("!=", "not_in"):
        return ~s.isin(coerced_vals)

    # Comparisons: exclude rows matching the comparison predicate
    if op in (">", ">=", "<", "<="):
        if not coerced_vals:
            return pd.Series([False] * len(s), index=s.index)

        x, v0 = _coerce_series_and_scalar_for_comparison(s, coerced_vals[0])

        if op == ">":
            return x > v0
        if op == ">=":
            return x >= v0
        if op == "<":
            return x < v0
        return x <= v0

    raise ValueError(f"Unsupported exclusion op '{op}'.")


def _coerce_series_and_scalar_for_comparison(series: pd.Series, v0: Any) -> Tuple[pd.Series, Any]:
    # numeric columns
    if ptypes.is_numeric_dtype(series):
        x = pd.to_numeric(series, errors="coerce")
        try:
            return x, float(v0)
        except Exception:
            raise ValueError(f"Comparison threshold '{v0}' is not numeric for numeric column '{series.name}'.")

    # datetime columns
    if ptypes.is_datetime64_any_dtype(series):
        x = pd.to_datetime(series, errors="coerce")
        v = pd.to_datetime(v0, errors="coerce")
        if pd.isna(v):
            raise ValueError(f"Comparison threshold '{v0}' is not a valid datetime for column '{series.name}'.")
        return x, v

    # object columns: best-effort numeric coercion (common case)
    x_num = pd.to_numeric(series, errors="coerce")
    if int(x_num.notna().sum()) > 0:
        try:
            return x_num, float(v0)
        except Exception:
            raise ValueError(f"Comparison threshold '{v0}' is not numeric for column '{series.name}'.")

    # object columns: best-effort datetime coercion
    x_dt = pd.to_datetime(series, errors="coerce")
    if int(x_dt.notna().sum()) > 0:
        v = pd.to_datetime(v0, errors="coerce")
        if pd.isna(v):
            raise ValueError(f"Comparison threshold '{v0}' is not a valid datetime for column '{series.name}'.")
        return x_dt, v

    raise ValueError(
        f"Cannot apply comparison operator on non-numeric/non-datetime column '{series.name}' (dtype={series.dtype})."
    )


def _coerce_values_for_series(series: pd.Series, values: List[str]) -> List[Any]:
    if not values:
        return []

    # numeric
    if ptypes.is_numeric_dtype(series):
        out: List[Any] = []
        for v in values:
            try:
                out.append(float(v))
            except Exception:
                out.append(str(v))
        return out

    # datetime-like
    if ptypes.is_datetime64_any_dtype(series):
        out_dt: List[Any] = []
        for v in values:
            try:
                out_dt.append(pd.to_datetime(v))
            except Exception:
                out_dt.append(str(v))
        return out_dt

    # boolean-like
    if ptypes.is_bool_dtype(series):
        out_b: List[Any] = []
        for v in values:
            vv = str(v).strip().lower()
            if vv in ("true", "1", "yes", "y"):
                out_b.append(True)
            elif vv in ("false", "0", "no", "n"):
                out_b.append(False)
            else:
                out_b.append(str(v))
        return out_b

    # default string
    return [str(v) for v in values]


# ============================================================
# Remaining helpers (unchanged except summary_text signature)
# ============================================================
def _dedupe_preserve_order(cols: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for c in cols:
        c2 = str(c).strip()
        if not c2 or c2 in seen:
            continue
        seen.add(c2)
        out.append(c2)
    return out


def _build_feature_sets(*, W_cols: List[str], X_cols: List[str]) -> Dict[str, List[str]]:
    xw = _dedupe_preserve_order(list(W_cols) + list(X_cols))
    return {"W": list(W_cols), "X": list(X_cols), "XW": xw}


def _resolve_econml_cols(treatment_spec: TreatmentSpec, outcome_spec: OutcomeSpec) -> Tuple[str, List[str]]:
    T_col = cast(str, treatment_spec["column"])

    if outcome_spec["kind"] == "duration":
        y = cast(Any, outcome_spec)
        return T_col, [cast(str, y["event_column"]), cast(str, y["duration_column"])]

    return T_col, [cast(str, cast(Any, outcome_spec)["column"])]


def _build_prepared_treatment(spec: TreatmentSpec, df: pd.DataFrame) -> PreparedTreatment:
    if spec["kind"] == "binary":
        s = cast(Any, spec)
        labels_binary: PreparedBinaryLabels = {
            "treated": cast(str, s["treated"]),
            "control": cast(str, s["control"]),
            "value_map": _build_binary_value_map(
                treated=cast(str, s["treated"]),
                control=cast(str, s["control"]),
                treated_aliases=cast(Sequence[str], s.get("treated_aliases", [])),
                control_aliases=cast(Sequence[str], s.get("control_aliases", [])),
            ),
        }
        return {"kind": "binary", "column": cast(str, s["column"]), "labels": labels_binary}

    if spec["kind"] == "continuous":
        s = cast(Any, spec)
        numeric: PreparedContinuousMeta = {}
        if "unit" in s:
            numeric["unit"] = cast(str, s["unit"])
        if "transform" in s:
            numeric["transform"] = cast(Any, s["transform"])
        if "clip_min" in s:
            numeric["clip_min"] = cast(float, s["clip_min"])
        if "clip_max" in s:
            numeric["clip_max"] = cast(float, s["clip_max"])
        return {"kind": "continuous", "column": cast(str, s["column"]), "numeric": numeric}

    s = cast(Any, spec)
    col = cast(str, s["column"])
    baseline = s.get("baseline") or (_most_frequent_level(df[col]) if col in df.columns else None)
    labels: PreparedCategoricalLabels = {
        "levels": list(cast(List[str], s["levels"])),
        "baseline": cast(str, baseline if baseline is not None else (s["levels"][0] if s["levels"] else "")),
        "value_map": {lvl: lvl for lvl in cast(List[str], s["levels"])},
    }
    return {"kind": "categorical", "column": col, "labels": labels}


def _build_prepared_outcome(spec: OutcomeSpec, df: pd.DataFrame) -> PreparedOutcome:
    kind = spec["kind"]

    if kind == "binary":
        s = cast(BinaryOutcomeSpec, spec)
        binary: PreparedBinaryOutcome = {
            "column": s["column"],
            "event": s["event"],
            "non_event": s["non_event"],
            "value_map": _build_binary_value_map(
                treated=s["event"],
                control=s["non_event"],
                treated_aliases=cast(Sequence[str], s.get("event_aliases", [])),
                control_aliases=cast(Sequence[str], s.get("non_event_aliases", [])),
            ),
        }
        return {"kind": "binary", "binary": binary}

    if kind == "continuous":
        s = cast(ContinuousOutcomeSpec, spec)
        cont: PreparedContinuousOutcome = {"column": s["column"]}
        if "unit" in s:
            cont["unit"] = s["unit"]
        if "transform" in s:
            cont["transform"] = cast(Any, s["transform"])
        if "clip_min" in s:
            cont["clip_min"] = cast(float, s["clip_min"])
        if "clip_max" in s:
            cont["clip_max"] = cast(float, s["clip_max"])
        return {"kind": "continuous", "continuous": cont}

    if kind == "categorical":
        s = cast(CategoricalOutcomeSpec, spec)
        col = s["column"]
        baseline = s.get("baseline") or (_most_frequent_level(df[col]) if col in df.columns else None)
        cat: PreparedCategoricalOutcome = {
            "column": col,
            "levels": list(s["levels"]),
            "baseline": cast(str, baseline if baseline is not None else (s["levels"][0] if s["levels"] else "")),
            "value_map": {lvl: lvl for lvl in s["levels"]},
        }
        return {"kind": "categorical", "categorical": cat}

    s = cast(DurationOutcomeSpec, spec)
    dur: PreparedDurationOutcome = {
        "duration_column": s["duration_column"],
        "event_column": s["event_column"],
        "event_value": s["event_value"],
        "censor_value": s["censor_value"],
        "value_map": _build_binary_value_map(
            treated=s["event_value"],
            control=s["censor_value"],
            treated_aliases=cast(Sequence[str], s.get("event_aliases", [])),
            control_aliases=cast(Sequence[str], s.get("censor_aliases", [])),
        ),
    }
    return {"kind": "duration", "duration": dur}


def _build_binary_value_map(
    *,
    treated: str,
    control: str,
    treated_aliases: Sequence[str] | None,
    control_aliases: Sequence[str] | None,
) -> Dict[str, str]:
    m: Dict[str, str] = {}
    m[str(treated)] = str(treated)
    m[str(control)] = str(control)
    for a in (treated_aliases or []):
        m[str(a)] = str(treated)
    for a in (control_aliases or []):
        m[str(a)] = str(control)
    return m


def _apply_treatment_canonicalization(df: pd.DataFrame, spec: TreatmentSpec) -> None:
    col = cast(str, spec["column"])
    if col not in df.columns:
        raise ValueError(f"Treatment column '{col}' not found in dataset.")

    if spec["kind"] == "binary":
        s = cast(Any, spec)
        vmap = _build_binary_value_map(
            treated=cast(str, s["treated"]),
            control=cast(str, s["control"]),
            treated_aliases=cast(Sequence[str], s.get("treated_aliases", [])),
            control_aliases=cast(Sequence[str], s.get("control_aliases", [])),
        )
        df[col] = _map_series_best_effort(df[col], vmap)


def _apply_outcome_canonicalization(df: pd.DataFrame, spec: OutcomeSpec) -> None:
    kind = spec["kind"]

    if kind == "binary":
        s = cast(BinaryOutcomeSpec, spec)
        col = s["column"]
        if col not in df.columns:
            raise ValueError(f"Outcome column '{col}' not found in dataset.")
        vmap = _build_binary_value_map(
            treated=s["event"],
            control=s["non_event"],
            treated_aliases=cast(Sequence[str], s.get("event_aliases", [])),
            control_aliases=cast(Sequence[str], s.get("non_event_aliases", [])),
        )
        df[col] = _map_series_best_effort(df[col], vmap)
        return

    if kind == "duration":
        s = cast(DurationOutcomeSpec, spec)
        ecol = s["event_column"]
        if ecol not in df.columns:
            raise ValueError(f"Duration event column '{ecol}' not found in dataset.")
        vmap = _build_binary_value_map(
            treated=s["event_value"],
            control=s["censor_value"],
            treated_aliases=cast(Sequence[str], s.get("event_aliases", [])),
            control_aliases=cast(Sequence[str], s.get("censor_aliases", [])),
        )
        df[ecol] = _map_series_best_effort(df[ecol], vmap)
        return


def _map_series_best_effort(series: pd.Series, value_map: Dict[str, str]) -> pd.Series:
    def _map_one(x: Any) -> Any:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return x
        if x in value_map:  # type: ignore[operator]
            return value_map[cast(str, x)]
        sx = str(x)
        if sx in value_map:
            return value_map[sx]
        return x

    return series.map(_map_one)


def _most_frequent_level(series: pd.Series) -> Optional[str]:
    try:
        vc = series.value_counts(dropna=True)
        if len(vc) == 0:
            return None
        return str(vc.index[0])
    except Exception:
        return None


def _build_prepared_columns_meta(
    *,
    df_prepared: pd.DataFrame,
    summary: Dict[str, Dict[str, Any]],
    treatment_spec: TreatmentSpec,
    outcome_spec: OutcomeSpec,
    W_cols: List[str],
    X_cols: List[str],
) -> List[PreparedColumnMeta]:
    cols = list(df_prepared.columns)
    tcol = cast(str, treatment_spec["column"])
    y_roles = _outcome_roles_map(outcome_spec)

    metas: List[PreparedColumnMeta] = []
    for c in cols:
        c_str = str(c)
        prof = summary.get(c_str, {})

        role: str = "other"
        if c_str == tcol:
            role = "T"
        elif c_str in y_roles:
            role = y_roles[c_str]
        elif c_str in W_cols:
            role = "W"
        elif c_str in X_cols:
            role = "X"

        dtype = str(df_prepared[c].dtype)
        missing_rate = _safe_float(prof.get("missing_rate"), default=_compute_missing_rate(df_prepared[c]))
        n_unique = _safe_int(prof.get("distinct_count"), default=_compute_nunique(df_prepared[c]))

        inferred_kind = str(prof.get("inferred_kind") or "").upper()
        encoding = "none"
        if role in ("W", "X"):
            if (
                inferred_kind in ("CATEGORICAL", "BOOLEAN")
                or ptypes.is_object_dtype(df_prepared[c])
                or ptypes.is_bool_dtype(df_prepared[c])
            ):
                encoding = "one_hot"

        metas.append(
            {
                "name": c_str,
                "role": cast(Any, role),
                "dtype": dtype,
                "missing_rate": float(missing_rate),
                "n_unique": int(n_unique),
                "encoding": cast(Any, encoding),
                "imputation": "none",
            }
        )

    return metas


def _outcome_roles_map(outcome_spec: OutcomeSpec) -> Dict[str, Literal["Y", "Y_event", "Y_duration"]]:
    if outcome_spec["kind"] == "duration":
        y = cast(Any, outcome_spec)
        return {cast(str, y["event_column"]): "Y_event", cast(str, y["duration_column"]): "Y_duration"}
    col = cast(str, cast(Any, outcome_spec)["column"])
    return {col: "Y"}


def _safe_float(v: Any, *, default: float) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, *, default: int) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _compute_missing_rate(series: pd.Series) -> float:
    try:
        return float(series.isna().mean())
    except Exception:
        return 0.0


def _compute_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except Exception:
        return 0


def _max_missing_rate(prepared_cols: List[PreparedColumnMeta], *, role: str) -> float:
    vals = [float(c["missing_rate"]) for c in prepared_cols if c.get("role") == role]
    return max(vals) if vals else 0.0


def _fill_treatment_metrics(metrics: PreparationMetrics, df: pd.DataFrame, spec: TreatmentSpec) -> None:
    if spec["kind"] != "binary":
        return
    s = cast(Any, spec)
    col = cast(str, s["column"])
    if col not in df.columns:
        return
    treated_val = cast(str, s["treated"])
    control_val = cast(str, s["control"])
    try:
        vc = df[col].value_counts(dropna=True)
        n_treated = int(vc.get(treated_val, 0))
        n_control = int(vc.get(control_val, 0))
        metrics["n_treated"] = n_treated
        metrics["n_control"] = n_control
        denom = n_treated + n_control
        metrics["treated_share"] = float(n_treated / denom) if denom > 0 else 0.0
    except Exception:
        return


def _fill_outcome_metrics(metrics: PreparationMetrics, df: pd.DataFrame, spec: OutcomeSpec) -> None:
    if spec["kind"] == "binary":
        s = cast(BinaryOutcomeSpec, spec)
        col = s["column"]
        if col not in df.columns:
            return
        try:
            vc = df[col].value_counts(dropna=True)
            metrics["n_event"] = int(vc.get(s["event"], 0))
            metrics["n_non_event"] = int(vc.get(s["non_event"], 0))
        except Exception:
            return
        return

    if spec["kind"] == "duration":
        s = cast(DurationOutcomeSpec, spec)
        col = s["event_column"]
        if col not in df.columns:
            return
        try:
            vc = df[col].value_counts(dropna=True)
            metrics["n_event"] = int(vc.get(s["event_value"], 0))
            metrics["n_censor"] = int(vc.get(s["censor_value"], 0))
        except Exception:
            return


def _schema_fingerprint(
    *,
    df: pd.DataFrame,
    protocol: ProtocolState,
    treatment: PreparedTreatment,
    outcome: PreparedOutcome,
    W_cols: List[str],
    X_cols: List[str],
    exclusions: List[ExclusionRule],
) -> str:
    payload = {
        "cols": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "protocol_core": {
            "experiment_type": protocol.get("experiment_type"),
            "time_zero_type": protocol.get("time_zero_type"),
            "time_zero": protocol.get("time_zero"),
        },
        "treatment": treatment,
        "outcome": outcome,
        "W_cols": W_cols,
        "X_cols": X_cols,
        "exclusions": exclusions,
    }
    b = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _build_summary_text(
    *,
    report_status: str,
    n_rows_source: int,
    n_rows_after_required_drop: int,
    n_rows_after_exclusions: int,
    n_rows_final: int,
    required_not_null_cols: List[str],
    T_col: str,
    Y_cols: List[str],
    W_cols: List[str],
    X_cols: List[str],
    exclusions_applied: int,
) -> str:
    return (
        f"Inference-ready prepared ({report_status}). "
        f"Rows: {n_rows_source} -> {n_rows_after_required_drop} after dropping NA in required columns {required_not_null_cols}; "
        f"-> {n_rows_after_exclusions} after exclusions ({exclusions_applied} rule(s)); "
        f"-> {n_rows_final} final. "
        f"T={T_col}; Y={Y_cols}; |W|={len(W_cols)}; |X|={len(X_cols)}."
    )