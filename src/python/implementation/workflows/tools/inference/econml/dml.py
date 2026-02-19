from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast
from uuid import UUID

import pandas as pd
from econml.dml import DML  # pyright: ignore[reportMissingTypeStubs]
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression  # pyright: ignore[reportMissingTypeStubs]
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.workflows.state.inference_ready_state import InferenceReadyState
from python.workflows.tools.inference.causal_inference import CausalInference
from python.workflows.tools.inference.econml.utils import (
    build_from_registry,
    covariates_or_none,
    invalid,
    json_safe,
    need,
    resolve_col,
    resolve_cols,
    unsupported,
    validate_kwargs,
)
from python.workflows.tools.inference.models.causal_command import CausalCommand
from python.workflows.tools.inference.models.causal_result import CausalResult

DML_FQCN = "econml.dml.DML"

_DML_SUPPORTED: Dict[str, type] = {
    "sklearn.linear_model.LinearRegression": LinearRegression,
    "sklearn.linear_model.Lasso": Lasso,
    "sklearn.linear_model.ElasticNet": ElasticNet,
    "sklearn.linear_model.LogisticRegression": LogisticRegression,
    "sklearn.ensemble.RandomForestRegressor": RandomForestRegressor,
    "sklearn.ensemble.RandomForestClassifier": RandomForestClassifier,
    "econml.sklearn_extensions.linear_model.StatsModelsLinearRegression": StatsModelsLinearRegression,
}

_DML_LINEAR_FINAL_ALLOWED: set[str] = {
    "sklearn.linear_model.LinearRegression",
    "sklearn.linear_model.Lasso",
    "sklearn.linear_model.ElasticNet",
    "econml.sklearn_extensions.linear_model.StatsModelsLinearRegression",
}


# =============================================================================
# InferenceReadyState compatibility helpers (old + new shapes)
# =============================================================================

def _as_str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, tuple):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def _get_nested(d: Any, *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _ir_prepared_artifact(ir: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(ir, dict):
        return None
    # new shape (preferred)
    for k in ("prepared_dataset", "prepared_dataset_artifact", "prepared"):
        v = ir.get(k)
        if isinstance(v, dict):
            return cast(Dict[str, Any], v)
    return None


def _ir_dataset_id(ir: Any) -> Optional[Any]:
    art = _ir_prepared_artifact(ir)
    if isinstance(art, dict):
        return art.get("dataset_id")
    return None


def _ir_schema_fingerprint(ir: Any) -> Optional[Any]:
    art = _ir_prepared_artifact(ir)
    if isinstance(art, dict):
        return art.get("schema_fingerprint")
    return None


def _ir_treatment_kind(ir: Any) -> str:
    v = _get_nested(ir, "treatment", "kind")
    if isinstance(v, str):
        return v
    # legacy fallback
    v2 = _get_nested(ir, "treatment_kind")
    return str(v2) if isinstance(v2, str) else ""


def _ir_outcome_kind(ir: Any) -> str:
    v = _get_nested(ir, "outcome", "kind")
    if isinstance(v, str):
        return v
    # legacy fallback
    v2 = _get_nested(ir, "outcome_kind")
    return str(v2) if isinstance(v2, str) else ""


def _ir_T_col(ir: Any) -> Optional[str]:
    if not isinstance(ir, dict):
        return None

    # legacy top-level
    v = ir.get("T_col")
    if isinstance(v, str) and v.strip():
        return v.strip()

    # new-ish nested
    for key in ("T_col", "col", "column", "name"):
        vv = _get_nested(ir, "treatment", key)
        if isinstance(vv, str) and vv.strip():
            return vv.strip()

    # sometimes stored under treatment.prepared.*
    for key in ("T_col", "col", "column", "name"):
        vv = _get_nested(ir, "treatment", "prepared", key)
        if isinstance(vv, str) and vv.strip():
            return vv.strip()

    return None


def _ir_Y_cols(ir: Any) -> List[str]:
    if not isinstance(ir, dict):
        return []

    # legacy top-level
    top = ir.get("Y_cols")
    out = _as_str_list(top)
    if out:
        return out

    # nested outcome
    for key in ("Y_cols", "cols", "columns"):
        vv = _get_nested(ir, "outcome", key)
        out2 = _as_str_list(vv)
        if out2:
            return out2

    # single-col outcome
    for key in ("Y_col", "col", "column", "name"):
        vv = _get_nested(ir, "outcome", key)
        out3 = _as_str_list(vv)
        if out3:
            return out3

    # sometimes stored under outcome.prepared.*
    for key in ("Y_cols", "cols", "columns"):
        vv = _get_nested(ir, "outcome", "prepared", key)
        out4 = _as_str_list(vv)
        if out4:
            return out4
    for key in ("Y_col", "col", "column", "name"):
        vv = _get_nested(ir, "outcome", "prepared", key)
        out5 = _as_str_list(vv)
        if out5:
            return out5

    return []


def _ir_X_cols(ir: Any) -> List[str]:
    if not isinstance(ir, dict):
        return []

    # legacy top-level
    out = _as_str_list(ir.get("X_cols"))
    if out:
        return out

    # common naming in your protocol (effect modifiers)
    for keyspace in ("effect_modifiers", "effect_modifiers_state", "X", "x"):
        v = _get_nested(ir, keyspace, "cols")
        out2 = _as_str_list(v)
        if out2:
            return out2
        v = _get_nested(ir, keyspace, "columns")
        out3 = _as_str_list(v)
        if out3:
            return out3

    # sometimes flattened
    for key in ("effect_modifiers_cols", "x_cols"):
        out4 = _as_str_list(ir.get(key))
        if out4:
            return out4

    return []


def _ir_W_cols(ir: Any) -> List[str]:
    if not isinstance(ir, dict):
        return []

    # legacy top-level
    out = _as_str_list(ir.get("W_cols"))
    if out:
        return out

    # common naming in your protocol (covariates)
    for keyspace in ("covariates", "covariates_state", "W", "w"):
        v = _get_nested(ir, keyspace, "cols")
        out2 = _as_str_list(v)
        if out2:
            return out2
        v = _get_nested(ir, keyspace, "columns")
        out3 = _as_str_list(v)
        if out3:
            return out3

    # sometimes flattened
    for key in ("covariates_cols", "w_cols"):
        out4 = _as_str_list(ir.get(key))
        if out4:
            return out4

    return []


def _count_missing(x: Any) -> int:
    try:
        if isinstance(x, pd.DataFrame):
            return int(x.isna().sum().sum())
        if isinstance(x, pd.Series):
            return int(x.isna().sum())
    except Exception:
        return 0
    return 0


@dataclass(frozen=True)
class EconMLDMLInference(CausalInference):
    """
    EconML DML adapter compatible with evolving InferenceReadyState shapes.
    Contract:
    - Negotiation/execution only use `options.*` paths.
    - Column resolution always comes from IR (never user-specified column names in options).
    """

    data_repo: DataRepo
    models_repo: ModelsRepo

    # -------------------------
    # capability advertisement
    # -------------------------
    def get_info(self, estimator_fqcn: str) -> Dict[str, Any]:
        if estimator_fqcn != DML_FQCN:
            return {"status": "UNSUPPORTED", "estimator_fqcn": estimator_fqcn, "supports_cmds": []}
        return {"estimator_fqcn": DML_FQCN, "supports_cmds": ["FIT", "EFFECT", "INTERVAL"]}

    # -------------------------
    # negotiation contract (knobs only; options.* only)
    # -------------------------
    def get_input_requirements(self, *, cmd: str, ir: InferenceReadyState) -> Dict[str, Any]:
        if _ir_outcome_kind(ir) == "duration":
            return {"status": "UNSUPPORTED", "reason": "DML does not support duration outcomes."}

        nuisance_choices: List[Any] = ["auto", *sorted(_DML_SUPPORTED.keys())]
        final_choices: List[Any] = sorted(_DML_LINEAR_FINAL_ALLOWED)
        feature_choices: List[Any] = ["X", "W", "XW", None]

        if cmd == "FIT":
            return {
                "optional_user": [
                    {"path": "options.init.model_y", "prompt": "Outcome nuisance model", "choices": nuisance_choices, "default": "auto"},
                    {"path": "options.init.model_t", "prompt": "Treatment nuisance model", "choices": nuisance_choices, "default": "auto"},
                    {
                        "path": "options.init.model_final",
                        "prompt": "Final model (must be linear for DML correctness)",
                        "choices": final_choices,
                        "default": {"name": "sklearn.linear_model.LinearRegression", "kwargs": {"fit_intercept": False}},
                    },
                    {"path": "options.init.cv", "prompt": "Cross-fitting folds", "default": 2},
                    {"path": "options.fit.inference", "prompt": "Inference (needed for intervals)", "choices": ["auto", "bootstrap", None], "default": "auto"},
                    {"path": "options.feature_set_key", "prompt": "Which prepared feature set to use", "choices": feature_choices, "default": None},
                ]
            }

        if cmd == "EFFECT":
            req: Dict[str, Any] = {
                "optional_user": [
                    {"path": "options.effect.Xq", "prompt": "Optional query rows Xq (array-like). If omitted, use prepared X.", "default": None},
                    {"path": "options.feature_set_key", "prompt": "Which prepared feature set to use", "choices": feature_choices, "default": None},
                ]
            }
            if _ir_treatment_kind(ir) != "binary":
                req["optional_user"] += [
                    {"path": "options.effect.T0", "prompt": "Baseline treatment value T0", "default": None},
                    {"path": "options.effect.T1", "prompt": "Target treatment value T1", "default": None},
                ]
            return req

        if cmd == "INTERVAL":
            req2: Dict[str, Any] = {
                "optional_user": [
                    {"path": "options.interval.Xq", "prompt": "Optional query rows Xq (array-like). If omitted, use prepared X.", "default": None},
                    {"path": "options.interval.alpha", "prompt": "alpha", "default": 0.05},
                    {"path": "options.feature_set_key", "prompt": "Which prepared feature set to use", "choices": feature_choices, "default": None},
                ]
            }
            if _ir_treatment_kind(ir) != "binary":
                req2["optional_user"] += [
                    {"path": "options.interval.T0", "prompt": "Baseline treatment value T0", "default": None},
                    {"path": "options.interval.T1", "prompt": "Target treatment value T1", "default": None},
                ]
            return req2

        return {"status": "UNSUPPORTED", "reason": f"Unknown cmd: {cmd}"}

    def get_output_schema(self, *, cmd: str) -> Dict[str, Any]:
        if cmd == "FIT":
            return {"outputs": {"model_id": "uuid"}}
        if cmd == "EFFECT":
            return {"outputs": {"effect": "array-like"}}
        if cmd == "INTERVAL":
            return {"outputs": {"lb": "array-like", "ub": "array-like", "alpha": "float"}}
        return {"outputs": {}}

    # -------------------------
    # execution (runtime ctx passed in)
    # -------------------------
    def execute(
        self,
        command: CausalCommand,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
        ir: InferenceReadyState,
    ) -> CausalResult:
        if command.estimator_fqcn != DML_FQCN:
            return CausalResult(status="UNSUPPORTED", issues=[unsupported("estimator_fqcn", f"Unsupported: {command.estimator_fqcn}")])

        if _ir_outcome_kind(ir) == "duration":
            return CausalResult(status="UNSUPPORTED", issues=[unsupported("outcome.kind", "Duration outcomes not supported by DML.")])

        if command.cmd == "FIT":
            return self._fit(command, user_id=user_id, conversation_id=conversation_id, model_id=model_id, ir=ir)
        if command.cmd == "EFFECT":
            return self._effect(command, user_id=user_id, conversation_id=conversation_id, model_id=model_id, ir=ir)
        if command.cmd == "INTERVAL":
            return self._interval(command, user_id=user_id, conversation_id=conversation_id, model_id=model_id, ir=ir)

        return CausalResult(status="UNSUPPORTED", issues=[unsupported("cmd", f"Unsupported cmd: {command.cmd}")])

    # -------------------------
    # internals
    # -------------------------
    def _load_prepared_df(self, *, user_id: UUID, conversation_id: UUID, ir: InferenceReadyState) -> pd.DataFrame:
        dataset_id = _ir_dataset_id(ir)
        if dataset_id is None:
            raise ValueError("InferenceReadyState missing prepared dataset artifact (dataset_id).")

        return self.data_repo.get_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
            limit=None,
        )

    def _choose_feature_set(self, *, ir: InferenceReadyState, key: Optional[str]) -> Tuple[List[str], List[str]]:
        X_cols = _ir_X_cols(ir)
        W_cols = _ir_W_cols(ir)

        if key is None:
            if X_cols and W_cols:
                return X_cols, W_cols
            if X_cols:
                return X_cols, []
            if W_cols:
                return [], W_cols
            return [], []

        if key not in ("X", "W", "XW"):
            raise ValueError("feature_set_key must be one of: X, W, XW")

        if key == "X":
            return X_cols, []
        if key == "W":
            return [], W_cols
        return X_cols, W_cols

    def _derive_discrete_flags(self, *, ir: InferenceReadyState) -> Tuple[bool, bool]:
        discrete_treatment = _ir_treatment_kind(ir) in ("binary", "categorical")
        discrete_outcome = _ir_outcome_kind(ir) in ("binary", "categorical")
        return discrete_outcome, discrete_treatment

    # -------------------------
    # FIT
    # -------------------------
    def _fit(
        self,
        command: CausalCommand,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
        ir: InferenceReadyState,
    ) -> CausalResult:
        try:
            df = self._load_prepared_df(user_id=user_id, conversation_id=conversation_id, ir=ir)
        except Exception as e:
            return CausalResult(status="ERROR", issues=[invalid("prepared_dataset.dataset_id", f"Failed to load prepared dataset: {e}")])

        opts: Dict[str, Any] = command.options or {}
        init_raw: Dict[str, Any] = cast(Dict[str, Any], opts.get("init") or {})
        fit_raw: Dict[str, Any] = cast(Dict[str, Any], opts.get("fit") or {})

        init_allowed = [
            "model_y",
            "model_t",
            "model_final",
            "featurizer",
            "treatment_featurizer",
            "fit_cate_intercept",
            "discrete_outcome",
            "discrete_treatment",
            "categories",
            "cv",
            "mc_iters",
            "mc_agg",
            "random_state",
            "allow_missing",
            "use_ray",
            "ray_remote_func_options",
        ]
        fit_allowed = ["cache_values", "inference", "sample_weight", "freq_weight", "sample_var", "groups"]

        init_kwargs, issues = validate_kwargs(provided=init_raw, allowed=init_allowed, path="options.init")
        if issues:
            return CausalResult(status="INVALID", issues=issues)

        fit_kwargs, issues = validate_kwargs(provided=fit_raw, allowed=fit_allowed, path="options.fit")
        if issues:
            return CausalResult(status="INVALID", issues=issues)

        # Columns ONLY from IR
        try:
            t_col = _ir_T_col(ir)
            if not t_col:
                return CausalResult(status="INVALID", issues=[invalid("ir.treatment", "Missing treatment column in InferenceReadyState.")])

            ycols = _ir_Y_cols(ir)
            if not ycols:
                return CausalResult(status="INVALID", issues=[invalid("ir.outcome", "Missing outcome column(s) in InferenceReadyState.")])

            T = resolve_col(df, t_col, role="T")
            Y = resolve_cols(df, ycols, role="Y") if len(ycols) > 1 else resolve_col(df, ycols[0], role="Y")

            feature_key = cast(Optional[str], opts.get("feature_set_key"))
            X_cols, W_cols = self._choose_feature_set(ir=ir, key=feature_key)
            X = covariates_or_none(df, X_cols)
            W = covariates_or_none(df, W_cols)
        except Exception as e:
            return CausalResult(status="INVALID", issues=[invalid("state.columns", f"Failed to resolve columns from IR: {e}")])

        # Optional pre-check: fail fast on missing values (common cause of DML fit failure)
        missing = _count_missing(T) + _count_missing(Y) + _count_missing(X) + _count_missing(W)
        if missing > 0 and not bool(init_kwargs.get("allow_missing", False)):
            return CausalResult(
                status="INVALID",
                issues=[
                    invalid(
                        "prepared_dataset",
                        f"Input contains missing values (total NaNs across Y/T/X/W = {missing}).",
                        fix="Ensure inference-ready preparation removes/imputes NaNs in the prepared dataset before FIT.",
                    )
                ],
            )

        # Defaults from IR
        discrete_outcome, discrete_treatment = self._derive_discrete_flags(ir=ir)
        init_kwargs.setdefault("discrete_outcome", discrete_outcome)
        init_kwargs.setdefault("discrete_treatment", discrete_treatment)
        init_kwargs.setdefault("model_y", "auto")
        init_kwargs.setdefault("model_t", "auto")
        init_kwargs.setdefault("model_final", {"name": "sklearn.linear_model.LinearRegression", "kwargs": {"fit_intercept": False}})
        init_kwargs.setdefault("cv", 2)

        model_y, issue = build_from_registry(
            _DML_SUPPORTED,
            init_kwargs.get("model_y"),
            role_path="options.init.model_y",
            allow_auto=True,
            default="auto",
        )
        if issue:
            return CausalResult(status="UNSUPPORTED", issues=[issue])

        model_t, issue = build_from_registry(
            _DML_SUPPORTED,
            init_kwargs.get("model_t"),
            role_path="options.init.model_t",
            allow_auto=True,
            default="auto",
        )
        if issue:
            return CausalResult(status="UNSUPPORTED", issues=[issue])

        model_final, issue = build_from_registry(
            _DML_SUPPORTED,
            init_kwargs.get("model_final"),
            role_path="options.init.model_final",
            allow_auto=False,
            default={"name": "sklearn.linear_model.LinearRegression", "kwargs": {"fit_intercept": False}},
            allow_names=_DML_LINEAR_FINAL_ALLOWED,
        )
        if issue:
            return CausalResult(status="UNSUPPORTED", issues=[issue])

        init_kwargs["model_y"] = model_y
        init_kwargs["model_t"] = model_t
        init_kwargs["model_final"] = model_final

        fit_kwargs.setdefault("inference", "auto")

        try:
            est = DML(**init_kwargs)
            est.fit(Y, T, X=X, W=W, **fit_kwargs)  # pyright: ignore[reportUnknownMemberType]
        except Exception as e:
            return CausalResult(status="ERROR", issues=[invalid("fit", f"DML fit failed: {e}")])

        meta: Dict[str, Any] = {
            "estimator_fqcn": DML_FQCN,
            "prepared_dataset_id": str(_ir_dataset_id(ir)) if _ir_dataset_id(ir) is not None else None,
            "schema_fingerprint": _ir_schema_fingerprint(ir),
            "T_col": t_col,
            "Y_cols": list(ycols),
            "X_cols_used": X.columns.tolist() if isinstance(X, pd.DataFrame) else [],
            "W_cols_used": W.columns.tolist() if isinstance(W, pd.DataFrame) else [],
            "feature_set_key": opts.get("feature_set_key"),
            "init": json_safe(init_raw),
            "fit": json_safe(fit_kwargs),
        }

        try:
            self.models_repo.save_model(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
                model=est,
                metadata=meta,
            )
        except Exception as e:
            return CausalResult(status="ERROR", issues=[invalid("models_repo.save_model", f"Persist failed: {e}")])

        return CausalResult(status="OK", model_id=model_id, outputs={"model_id": str(model_id)})

    # -------------------------
    # EFFECT (options.* only)
    # -------------------------
    def _effect(
        self,
        command: CausalCommand,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
        ir: InferenceReadyState,
    ) -> CausalResult:
        rec = self.models_repo.load_model(user_id=user_id, conversation_id=conversation_id, model_id=model_id)
        if rec is None:
            return CausalResult(status="INVALID", issues=[invalid("model_id", f"Model not found: {model_id}")])

        est = rec.model
        opts: Dict[str, Any] = command.options or {}
        effect_opts: Dict[str, Any] = cast(Dict[str, Any], opts.get("effect") or {})

        # For non-binary treatments EconML needs a contrast (T0 -> T1).
        if _ir_treatment_kind(ir) == "binary":
            T0, T1 = effect_opts.get("T0", 0), effect_opts.get("T1", 1)
        else:
            if "T0" not in effect_opts or "T1" not in effect_opts:
                return CausalResult(
                    status="NEEDS_INPUT",
                    issues=[need("options.effect.T0", "Need T0 and T1 for non-binary treatment.", required=["options.effect.T0", "options.effect.T1"])],
                )
            T0, T1 = effect_opts["T0"], effect_opts["T1"]  # pyright: ignore[reportConstantRedefinition]

        Xq: Any = effect_opts.get("Xq")
        if Xq is None:
            try:
                df = self._load_prepared_df(user_id=user_id, conversation_id=conversation_id, ir=ir)
                feature_key = cast(Optional[str], opts.get("feature_set_key"))
                X_cols, _ = self._choose_feature_set(ir=ir, key=feature_key)
                Xq = covariates_or_none(df, X_cols)
            except Exception:
                Xq = None

        try:
            tau = est.effect(Xq, T0=T0, T1=T1)
        except Exception as e:
            return CausalResult(status="ERROR", issues=[invalid("effect", f"effect failed: {e}")])

        return CausalResult(status="OK", model_id=model_id, outputs={"effect": json_safe(tau)})

    # -------------------------
    # INTERVAL (options.* only)
    # -------------------------
    def _interval(
        self,
        command: CausalCommand,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
        ir: InferenceReadyState,
    ) -> CausalResult:
        rec = self.models_repo.load_model(user_id=user_id, conversation_id=conversation_id, model_id=model_id)
        if rec is None:
            return CausalResult(status="INVALID", issues=[invalid("model_id", f"Model not found: {model_id}")])

        est = rec.model
        opts: Dict[str, Any] = command.options or {}
        interval_opts: Dict[str, Any] = cast(Dict[str, Any], opts.get("interval") or {})
        alpha = float(interval_opts.get("alpha", 0.05))

        if _ir_treatment_kind(ir) == "binary":
            T0, T1 = interval_opts.get("T0", 0), interval_opts.get("T1", 1)
        else:
            if "T0" not in interval_opts or "T1" not in interval_opts:
                return CausalResult(
                    status="NEEDS_INPUT",
                    issues=[need("options.interval.T0", "Need T0 and T1 for non-binary treatment.", required=["options.interval.T0", "options.interval.T1"])],
                )
            T0, T1 = interval_opts["T0"], interval_opts["T1"]  # pyright: ignore[reportConstantRedefinition]

        Xq: Any = interval_opts.get("Xq")
        if Xq is None:
            try:
                df = self._load_prepared_df(user_id=user_id, conversation_id=conversation_id, ir=ir)
                feature_key = cast(Optional[str], opts.get("feature_set_key"))
                X_cols, _ = self._choose_feature_set(ir=ir, key=feature_key)
                Xq = covariates_or_none(df, X_cols)
            except Exception:
                Xq = None

        try:
            lb, ub = est.effect_interval(Xq, T0=T0, T1=T1, alpha=alpha)
        except Exception as e:
            return CausalResult(
                status="INVALID",
                issues=[
                    invalid(
                        "interval",
                        f"effect_interval failed: {e}",
                        fix="Re-fit with options.fit.inference='auto' or 'bootstrap' (not None).",
                    )
                ],
            )

        return CausalResult(status="OK", model_id=model_id, outputs={"lb": json_safe(lb), "ub": json_safe(ub), "alpha": alpha})
