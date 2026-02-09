from __future__ import annotations

import hashlib
import json
from typing import Any,  Dict, List, Literal, Optional, Sequence, Tuple, cast
from uuid import UUID, uuid4

import pandas as pd
from pandas.api import types as ptypes


from python.workflows.state.control_state import ACTION
from python.workflows.state.dataset_state import  DatasetStateHelpers
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
    TreatmentSpec,
)
from python.workflows.state.protocol_state import ProtocolState
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState, ConversationStateHelpers
from python.domain.repo.data_repo import DataRepo



# ============================================================
# Public factory
# ============================================================
def make_prepare_inference_ready_node(
    data_repo: DataRepo,
) -> CallableNodeFunc:

    def node(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
        # ----------------------------
        # Preconditions
        # ----------------------------
        dataset_state =  state.get("dataset", {})
        dataset_id = dataset_state.get("id")
        protocol =  state.get("protocol")
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

        # PASS or WARN => proceed
        # ----------------------------
        # Load data
        # ----------------------------
        try:
            df = data_repo.get_csv_data(user_id=user_id, conversation_id=conversation_id, dataset_id=dataset_id, limit=None)
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
        # Apply exclusions (eligibility)
        # ----------------------------
        try:
            df_filtered, excl_summary = _apply_exclusions(df, protocol.get("exclusions", []))
        except Exception as e:
            err = f"Applying exclusions failed: {type(e).__name__}: {e}"
            ConversationStateHelpers.append_ai_message(state, err)
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), "Exclusions application failed.")

        n_rows_after_exclusions = int(df_filtered.shape[0])

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
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), "Could not canonicalize treatment/outcome.")

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
                summary= summary,
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
            "n_rows_after_exclusions": n_rows_after_exclusions,
            "n_rows_final": int(df_prepared.shape[0]),
        }

        # binary T counts if applicable
        _fill_treatment_metrics(metrics, df_prepared, protocol["treatment_spec"])
        # binary/duration event counts if applicable
        _fill_outcome_metrics(metrics, df_prepared, protocol["outcome_spec"])

        metrics["max_missing_rate_W"] = _max_missing_rate(prepared_cols, role="W")
        metrics["max_missing_rate_X"] = _max_missing_rate(prepared_cols, role="X")

        # ----------------------------
        # Materialize prepared dataset artifact (best-effort)
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
                "created_from_dataset_id":  dataset_id,
            }
        except Exception as e:
            ConversationStateHelpers.append_ai_message(
                state,
                f"could not materialize prepared dataset artifact. {e}"
            )
            return ConversationStateHelpers.set_abort(state, cast(ACTION, "NONE"), f"Inference-ready state prepared.{e}")

        # ----------------------------
        # Assemble InferenceReadyState
        # ----------------------------
        inference_ready: InferenceReadyState = {
            "source_dataset_id":  dataset_id,
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

        # Optional human-readable summary
        inference_ready["summary_text"] = _build_summary_text(
            report_status=cast(str, status),
            n_rows_source=n_rows_source,
            n_rows_final=int(df_prepared.shape[0]),
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
# Helpers
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
    T_col = treatment_spec["column"]

    if outcome_spec["kind"] == "duration":
        y =  outcome_spec
        # convention: [event, duration]
        return T_col, [y["event_column"], y["duration_column"]]

    # all other outcomes are single-column
    return T_col, [cast(Any, outcome_spec)["column"]]


def _apply_exclusions(df: pd.DataFrame, exclusions: List[ExclusionRule]) -> Tuple[pd.DataFrame, ExclusionApplicationSummary]:
    n_before = int(df.shape[0])
    rules_audit: List[Dict[str, Any]] = []

    out = df
    for r in exclusions or []:
        col = r["column"]
        op = r["op"]
        vals = r.get("values", []) or []
        reason = r.get("reason", "")

        if col not in out.columns:
            raise ValueError(f"Exclusion rule references missing column '{col}'.")

        before = int(out.shape[0])
        mask_keep = _mask_keep_for_rule(out[col], op, vals)
        out = out.loc[mask_keep].copy(deep=False)
        after = int(out.shape[0])

        rules_audit.append(
            {
                "column": col,
                "op": op,
                "values": list(vals),
                "reason": reason,
                "n_before": before,
                "n_after": after,
                "n_removed": before - after,
            }
        )

    return out, {"n_before": n_before, "n_after": int(out.shape[0]), "rules": rules_audit}


def _mask_keep_for_rule(series: pd.Series, op: str, values: List[str]) -> pd.Series:
    s = series
    # null checks do not need values
    if op == "is_null":
        return s.isna()
    if op == "not_null":
        return ~s.isna()

    coerced_vals = _coerce_values_for_series(s, values)

    if op == "==":
        return ~s.isin(coerced_vals) if len(coerced_vals) > 0 else pd.Series([True] * len(s), index=s.index)
    if op == "!=":
        return s.isin(coerced_vals) if len(coerced_vals) > 0 else pd.Series([True] * len(s), index=s.index)
    if op == "in":
        return ~s.isin(coerced_vals)
    if op == "not_in":
        return s.isin(coerced_vals)

    # comparisons: use first value
    if op in (">", ">=", "<", "<="):
        if not coerced_vals:
            return pd.Series([True] * len(s), index=s.index)
        v0 = coerced_vals[0]
        # For comparisons, define "keep" as NOT matching the exclusion condition.
        if op == ">":
            return ~(s > v0)
        if op == ">=":
            return ~(s >= v0)
        if op == "<":
            return ~(s < v0)
        if op == "<=":
            return ~(s <= v0)

    raise ValueError(f"Unsupported exclusion op '{op}'.")


def _coerce_values_for_series(series: pd.Series, values: List[str]) -> List[Any]:
    vals = [v for v in (values or []) if str(v).strip() != ""]
    if not vals:
        return []

    # numeric
    if ptypes.is_numeric_dtype(series):
        out: List[Any] = []
        for v in vals:
            try:
                out.append(float(v))
            except Exception:
                # fallback to string comparison if numeric cast fails
                out.append(str(v))
        return out

    # datetime-like
    if ptypes.is_datetime64_any_dtype(series):
        out = []
        for v in vals:
            try:
                out.append(pd.to_datetime(v))
            except Exception:
                out.append(str(v))
        return out

    # boolean-like
    if ptypes.is_bool_dtype(series):
        out = []
        for v in vals:
            vv = str(v).strip().lower()
            if vv in ("true", "1", "yes", "y"):
                out.append(True)
            elif vv in ("false", "0", "no", "n"):
                out.append(False)
            else:
                out.append(str(v))
        return out

    # default string
    return [str(v) for v in vals]


def _build_prepared_treatment(spec: TreatmentSpec, df: pd.DataFrame) -> PreparedTreatment:
    if spec["kind"] == "binary":
        s =  spec
        labels_binary: PreparedBinaryLabels = {
            "treated": s["treated"],
            "control": s["control"],
            "value_map": _build_binary_value_map(
                treated=s["treated"],
                control=s["control"],
                treated_aliases=s.get("treated_aliases", []),
                control_aliases=s.get("control_aliases", []),
            ),
        }
        return {"kind": "binary", "column": s["column"], "labels": labels_binary}

    if spec["kind"] == "continuous":
        s =  spec
        numeric: PreparedContinuousMeta = {}
        if "unit" in s:
            numeric["unit"] = s["unit"]
        if "transform" in s:
            numeric["transform"] = cast(Any, s["transform"])
        if "clip_min" in s:
            numeric["clip_min"] =  s["clip_min"]
        if "clip_max" in s:
            numeric["clip_max"] =  s["clip_max"]
        return {"kind": "continuous", "column": s["column"], "numeric": numeric}

    # categorical
    s =  spec
    col = s["column"]
    baseline = s.get("baseline") or _most_frequent_level(df[col]) if col in df.columns else None
    labels: PreparedCategoricalLabels = {
        "levels": list(s["levels"]),
        "baseline":  baseline if baseline is not None else (s["levels"][0] if s["levels"] else ""),
        "value_map": {lvl: lvl for lvl in s["levels"]},
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
                treated_aliases=s.get("event_aliases", []),
                control_aliases=s.get("non_event_aliases", []),
            ),
        }
        return {"kind": "binary", "binary": binary}

    if kind == "continuous":
        s = cast(ContinuousOutcomeSpec, spec)
        cont: PreparedContinuousOutcome = {"column": s["column"]}
        if "unit" in s:
            cont["unit"] =  s["unit"]
        if "transform" in s:
            cont["transform"] = cast(Any, s["transform"])
        if "clip_min" in s:
            cont["clip_min"] =  s["clip_min"]
        if "clip_max" in s:
            cont["clip_max"] =  s["clip_max"]
        return {"kind": "continuous", "continuous": cont}

    if kind == "categorical":
        s = cast(CategoricalOutcomeSpec, spec)
        col = s["column"]
        baseline = s.get("baseline") or _most_frequent_level(df[col]) if col in df.columns else None
        cat: PreparedCategoricalOutcome = {
            "column": col,
            "levels": list(s["levels"]),
            "baseline":  baseline if baseline is not None else (s["levels"][0] if s["levels"] else ""),
            "value_map": {lvl: lvl for lvl in s["levels"]},
        }
        return {"kind": "categorical", "categorical": cat}

    # duration
    s = cast(DurationOutcomeSpec, spec)
    dur: PreparedDurationOutcome = {
        "duration_column": s["duration_column"],
        "event_column": s["event_column"],
        "event_value": s["event_value"],
        "censor_value": s["censor_value"],
        "value_map": _build_binary_value_map(
            treated=s["event_value"],
            control=s["censor_value"],
            treated_aliases=s.get("event_aliases", []),
            control_aliases=s.get("censor_aliases", []),
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
    # canonical
    m[str(treated)] = str(treated)
    m[str(control)] = str(control)
    # aliases
    for a in (treated_aliases or []):
        m[str(a)] = str(treated)
    for a in (control_aliases or []):
        m[str(a)] = str(control)
    return m


def _apply_treatment_canonicalization(df: pd.DataFrame, spec: TreatmentSpec) -> None:
    col = spec["column"]
    if col not in df.columns:
        raise ValueError(f"Treatment column '{col}' not found in dataset.")

    if spec["kind"] == "binary":
        s =  spec
        vmap = _build_binary_value_map(
            treated=s["treated"],
            control=s["control"],
            treated_aliases=s.get("treated_aliases", []),
            control_aliases=s.get("control_aliases", []),
        )
        df[col] = _map_series_best_effort(df[col], vmap)

    # categorical baseline/levels already constrain; we do not remap unless you add alias support
    # continuous: no remap


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
            treated_aliases=s.get("event_aliases", []),
            control_aliases=s.get("non_event_aliases", []),
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
            treated_aliases=s.get("event_aliases", []),
            control_aliases=s.get("censor_aliases", []),
        )
        df[ecol] = _map_series_best_effort(df[ecol], vmap)
        return

    # continuous/categorical: no remap by default


def _map_series_best_effort(series: pd.Series, value_map: Dict[str, str]) -> pd.Series:
    """
    Map values using:
      - exact key match
      - fallback to stringified key match
    Unmapped values remain unchanged.
    """
    def _map_one(x: Any) -> Any:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return x
        # exact
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
        if  len(vc) == 0:
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

    tcol = treatment_spec["column"]
    y_roles = _outcome_roles_map(outcome_spec)

    metas: List[PreparedColumnMeta] = []
    for c in cols:
        c_str = str(c)
        prof = summary.get(c_str, {})

        role = "other"
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
            if inferred_kind in ("CATEGORICAL", "BOOLEAN") or ptypes.is_object_dtype(df_prepared[c]) or ptypes.is_bool_dtype(df_prepared[c]):
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
        y =  outcome_spec
        return {y["event_column"]: "Y_event", y["duration_column"]: "Y_duration"}
    # single outcome column
    col = cast(Any, outcome_spec)["column"]
    return {cast(str, col): "Y"}


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
    s =  spec
    col = s["column"]
    if col not in df.columns:
        return
    treated_val = s["treated"]
    control_val = s["control"]
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
        s = spec
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
        s =  spec
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
    payload = { # pyright: ignore[reportUnknownVariableType]
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
    n_rows_final: int,
    T_col: str,
    Y_cols: List[str],
    W_cols: List[str],
    X_cols: List[str],
    exclusions_applied: int,
) -> str:
    return (
        f"Inference-ready prepared ({report_status}). "
        f"Rows: {n_rows_source} -> {n_rows_final} after exclusions ({exclusions_applied} rule(s)). "
        f"T={T_col}; Y={Y_cols}; |W|={len(W_cols)}; |X|={len(X_cols)}. "
    )
