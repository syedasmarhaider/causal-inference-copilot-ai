from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast
from uuid import UUID, uuid4

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from econml.dml import DML 

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.workflows.state.inference_ready_state import InferenceReadyState
from python.workflows.tools.inference.models.causal_command import CausalCommand
from python.workflows.tools.inference.models.causal_result import CausalResult
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

DML_FQCN = "econml.dml.DML"

# Adapter-local registry (NO global assumptions in shared utils)
_DML_SUPPORTED: Dict[str, type] = {
    # linear
    "sklearn.linear_model.LinearRegression": LinearRegression,
    "sklearn.linear_model.Lasso": Lasso,
    "sklearn.linear_model.ElasticNet": ElasticNet,
    "sklearn.linear_model.LogisticRegression": LogisticRegression,
    # forests
    "sklearn.ensemble.RandomForestRegressor": RandomForestRegressor,
    "sklearn.ensemble.RandomForestClassifier": RandomForestClassifier,
    # econml linear
    "econml.sklearn_extensions.linear_model.StatsModelsLinearRegression": StatsModelsLinearRegression,
}

# DML-specific correctness constraint: final stage must be linear
_DML_LINEAR_FINAL_ALLOWED: set[str] = {
    "sklearn.linear_model.LinearRegression",
    "sklearn.linear_model.Lasso",
    "sklearn.linear_model.ElasticNet",
    "econml.sklearn_extensions.linear_model.StatsModelsLinearRegression",
}


@dataclass(frozen=True)
class EconMLDMLAdapter:
    data_repo: DataRepo
    models_repo: ModelsRepo

    # ---------------------------------
    # minimal capability advertisement
    # ---------------------------------
    def get_info(self) -> Dict[str, Any]:
        return {"estimator_fqcn": DML_FQCN, "supports_cmds": ["FIT", "EFFECT", "INTERVAL"]}

    # ---------------------------------
    # what the LLM should ask user for
    # (ONLY knobs; columns come from IR)
    # ---------------------------------
    def get_user_input_requirements(self, *, cmd: str, ir: InferenceReadyState) -> Dict[str, Any]:
        if ir["outcome"]["kind"] == "duration":
            return {"status": "UNSUPPORTED", "reason": "DML adapter does not support duration outcomes."}

        nuisance_choices = ["auto", *sorted(_DML_SUPPORTED.keys())]
        final_choices = sorted(_DML_LINEAR_FINAL_ALLOWED)

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
                    {"path": "options.fit.inference", "prompt": "Inference for intervals", "choices": ["auto", "bootstrap", None], "default": "auto"},
                    {"path": "options.feature_set_key", "prompt": "Which prepared feature set to use", "choices": ["X", "W", "XW", None], "default": None},
                    # keep any other econml knobs behind an escape hatch if you want:
                    # {"path": "options.init.<other>", ...}
                ]
            }

        if cmd in ("EFFECT", "INTERVAL"):
            optional_user: List[Dict[str, Any]] = [
                {"path": "inputs.Xq", "prompt": "Optional query rows Xq (array-like). If omitted, uses prepared X from state."}
            ]
            if ir["treatment"]["kind"] != "binary":
                optional_user += [
                    {"path": f"options.{cmd.lower()}.T0", "prompt": "Baseline treatment value T0"},
                    {"path": f"options.{cmd.lower()}.T1", "prompt": "Target treatment value T1"},
                ]
            if cmd == "INTERVAL":
                optional_user.append({"path": "options.interval.alpha", "prompt": "alpha", "default": 0.05})

            return {"ask_user": [{"path": "options.model_id", "prompt": "model_id"}], "optional_user": optional_user}

        return {"status": "UNSUPPORTED", "reason": f"Unknown cmd: {cmd}"}

    # ---------------------------------
    # output contract (lean)
    # ---------------------------------
    def get_output_schema(self, *, cmd: str) -> Dict[str, Any]:
        if cmd == "FIT":
            return {"outputs": {"model_id": "uuid"}}
        if cmd == "EFFECT":
            return {"outputs": {"effect": "array-like", "T0": "any", "T1": "any"}}
        if cmd == "INTERVAL":
            return {"outputs": {"lb": "array-like", "ub": "array-like", "alpha": "float"}}
        return {"outputs": {}}

    # ---------------------------------
    # execution
    # ---------------------------------
    def execute(self, command: CausalCommand, *, user_id: UUID, conversation_id: UUID, ir: InferenceReadyState) -> CausalResult:
        if command.estimator_fqcn != DML_FQCN:
            return CausalResult(status="UNSUPPORTED", issues=[unsupported("estimator_fqcn", f"Unsupported: {command.estimator_fqcn}")])

        if ir["outcome"]["kind"] == "duration":
            return CausalResult(status="UNSUPPORTED", issues=[unsupported("outcome.kind", "Duration outcomes not supported by DML adapter.")])

        if command.cmd == "FIT":
            return self._fit(command, user_id=user_id, conversation_id=conversation_id, ir=ir)
        if command.cmd == "EFFECT":
            return self._effect(command, user_id=user_id, conversation_id=conversation_id, ir=ir)
        if command.cmd == "INTERVAL":
            return self._interval(command, user_id=user_id, conversation_id=conversation_id, ir=ir)

        return CausalResult(status="UNSUPPORTED", issues=[unsupported("cmd", f"Unsupported cmd: {command.cmd}")])

    # -------------------------
    # internals
    # -------------------------
    def _load_prepared_df(self, *, user_id: UUID, conversation_id: UUID, ir: InferenceReadyState) -> pd.DataFrame:
        prepared = ir.get("prepared")
        if not prepared:
            raise ValueError("InferenceReadyState.prepared missing (not READY).")
        return self.data_repo.get_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=prepared["dataset_id"],
            limit=None,
        )

    def _choose_feature_set(self, ir: InferenceReadyState, key: Optional[str]) -> tuple[List[str], List[str]]:
        if key is None:
            # default heuristic
            if ir["X_cols"] and ir["W_cols"]:
                return ir["X_cols"], ir["W_cols"]
            if ir["X_cols"]:
                return ir["X_cols"], []
            if ir["W_cols"]:
                return [], ir["W_cols"]
            return [], []

        if key not in ("X", "W", "XW"):
            raise ValueError("feature_set_key must be one of: X, W, XW")

        if key == "X":
            return ir["X_cols"], []
        if key == "W":
            return [], ir["W_cols"]
        return ir["X_cols"], ir["W_cols"]

    def _derive_discrete_flags(self, ir: InferenceReadyState) -> tuple[bool, bool]:
        discrete_treatment = ir["treatment"]["kind"] in ("binary", "categorical")
        discrete_outcome = ir["outcome"]["kind"] in ("binary", "categorical")
        return discrete_outcome, discrete_treatment

    # -------------------------
    # FIT
    # -------------------------
    def _fit(self, command: CausalCommand, *, user_id: UUID, conversation_id: UUID, ir: InferenceReadyState) -> CausalResult:
        try:
            df = self._load_prepared_df(user_id=user_id, conversation_id=conversation_id, ir=ir)
        except Exception as e:
            return CausalResult(status="ERROR", issues=[invalid("prepared.dataset_id", f"Failed to load prepared dataset: {e}")])

        opts = command.options or {}
        init_raw = cast(Dict[str, Any], opts.get("init") or {})
        fit_raw = cast(Dict[str, Any], opts.get("fit") or {})

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

        # Columns come ONLY from IR
        try:
            T = resolve_col(df, ir["T_col"], role="T")

            ycols = ir["Y_cols"]
            if not ycols:
                return CausalResult(status="INVALID", issues=[invalid("ir.Y_cols", "Y_cols is empty.")])

            Y = resolve_cols(df, ycols, role="Y") if len(ycols) > 1 else resolve_col(df, ycols[0], role="Y")

            feature_key = cast(Optional[str], opts.get("feature_set_key"))
            X_cols, W_cols = self._choose_feature_set(ir, feature_key)
            X = covariates_or_none(df, X_cols)
            W = covariates_or_none(df, W_cols)
        except Exception as e:
            return CausalResult(status="INVALID", issues=[invalid("state.columns", f"Failed to resolve columns from state: {e}")])

        # Defaults derived from IR
        discrete_outcome, discrete_treatment = self._derive_discrete_flags(ir)
        init_kwargs.setdefault("discrete_outcome", discrete_outcome)
        init_kwargs.setdefault("discrete_treatment", discrete_treatment)

        init_kwargs.setdefault("model_y", "auto")
        init_kwargs.setdefault("model_t", "auto")
        init_kwargs.setdefault("model_final", {"name": "sklearn.linear_model.LinearRegression", "kwargs": {"fit_intercept": False}})
        init_kwargs.setdefault("cv", 2)

        # Build estimators (adapter-local registry)
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
            allow_names=_DML_LINEAR_FINAL_ALLOWED,  # DML-specific rule stays in adapter
        )
        if issue:
            return CausalResult(status="UNSUPPORTED", issues=[issue])

        init_kwargs["model_y"] = model_y
        init_kwargs["model_t"] = model_t
        init_kwargs["model_final"] = model_final

        # Intervals need inference; default enabled
        fit_kwargs.setdefault("inference", "auto")

        # Optional weights/groups: if user gave column names, they should have resolved earlier.
        # Here we only pass through provided scalar/arrays if any.
        try:
            est = DML(**init_kwargs)
            est.fit(Y, T, X=X, W=W, **fit_kwargs)
        except Exception as e:
            return CausalResult(status="ERROR", issues=[invalid("fit", f"DML fit failed: {e}")])

        model_id = cast(Optional[UUID], opts.get("model_id")) or uuid4()
        meta: Dict[str, Any] = {
            "estimator_fqcn": DML_FQCN,
            "prepared_dataset_id": str(ir["prepared"]["dataset_id"]) if ir.get("prepared") else None,
            "schema_fingerprint": ir["prepared"]["schema_fingerprint"] if ir.get("prepared") else None,
            "T_col": ir["T_col"],
            "Y_cols": list(ir["Y_cols"]),
            "X_cols_used": X.columns.tolist() if X is not None else [],
            "W_cols_used": W.columns.tolist() if W is not None else [],
            "init": json_safe(init_raw),
            "fit": json_safe(fit_kwargs),
        }

        try:
            self.models_repo.save_model(user_id=user_id, conversation_id=conversation_id, model_id=model_id, model=est, metadata=meta)
        except Exception as e:
            return CausalResult(status="ERROR", issues=[invalid("models_repo.save_model", f"Persist failed: {e}")])

        return CausalResult(status="OK", model_id=model_id, outputs={"model_id": str(model_id)})

    # -------------------------
    # EFFECT
    # -------------------------
    def _effect(self, command: CausalCommand, *, user_id: UUID, conversation_id: UUID, ir: InferenceReadyState) -> CausalResult:
        opts = command.options or {}
        model_id = cast(Optional[UUID], opts.get("model_id"))
        if model_id is None:
            return CausalResult(status="NEEDS_INPUT", issues=[need("options.model_id", "model_id is required", required=["options.model_id"])])

        rec = self.models_repo.load_model(user_id=user_id, conversation_id=conversation_id, model_id=model_id)
        if rec is None:
            return CausalResult(status="INVALID", issues=[invalid("options.model_id", f"Model not found: {model_id}")])

        est = rec.model
        effect_opts = cast(Dict[str, Any], opts.get("effect") or {})

        if ir["treatment"]["kind"] == "binary":
            T0 = effect_opts.get("T0", 0)
            T1 = effect_opts.get("T1", 1)
        else:
            if "T0" not in effect_opts or "T1" not in effect_opts:
                return CausalResult(
                    status="NEEDS_INPUT",
                    issues=[need("options.effect.T0/T1", "T0 and T1 required for non-binary treatment.", required=["options.effect.T0", "options.effect.T1"])],
                )
            T0, T1 = effect_opts["T0"], effect_opts["T1"]

        Xq = (command.inputs or {}).get("Xq")
        if Xq is None:
            # fall back to prepared X from IR
            try:
                df = self._load_prepared_df(user_id=user_id, conversation_id=conversation_id, ir=ir)
                feature_key = cast(Optional[str], opts.get("feature_set_key"))
                X_cols, _ = self._choose_feature_set(ir, feature_key)
                Xq = covariates_or_none(df, X_cols)
            except Exception:
                Xq = None

        try:
            tau = est.effect(Xq, T0=T0, T1=T1)
        except Exception as e:
            return CausalResult(status="ERROR", issues=[invalid("effect", f"effect failed: {e}")])

        return CausalResult(status="OK", model_id=model_id, outputs={"effect": json_safe(tau), "T0": json_safe(T0), "T1": json_safe(T1)})

    # -------------------------
    # INTERVAL
    # -------------------------
    def _interval(self, command: CausalCommand, *, user_id: UUID, conversation_id: UUID, ir: InferenceReadyState) -> CausalResult:
        opts = command.options or {}
        model_id = cast(Optional[UUID], opts.get("model_id"))
        if model_id is None:
            return CausalResult(status="NEEDS_INPUT", issues=[need("options.model_id", "model_id is required", required=["options.model_id"])])

        rec = self.models_repo.load_model(user_id=user_id, conversation_id=conversation_id, model_id=model_id)
        if rec is None:
            return CausalResult(status="INVALID", issues=[invalid("options.model_id", f"Model not found: {model_id}")])

        est = rec.model
        interval_opts = cast(Dict[str, Any], opts.get("interval") or {})
        alpha = float(interval_opts.get("alpha", 0.05))

        if ir["treatment"]["kind"] == "binary":
            T0 = interval_opts.get("T0", 0)
            T1 = interval_opts.get("T1", 1)
        else:
            if "T0" not in interval_opts or "T1" not in interval_opts:
                return CausalResult(
                    status="NEEDS_INPUT",
                    issues=[need("options.interval.T0/T1", "T0 and T1 required for non-binary treatment.", required=["options.interval.T0", "options.interval.T1"])],
                )
            T0, T1 = interval_opts["T0"], interval_opts["T1"]

        Xq = (command.inputs or {}).get("Xq")
        if Xq is None:
            try:
                df = self._load_prepared_df(user_id=user_id, conversation_id=conversation_id, ir=ir)
                feature_key = cast(Optional[str], opts.get("feature_set_key"))
                X_cols, _ = self._choose_feature_set(ir, feature_key)
                Xq = covariates_or_none(df, X_cols)
            except Exception:
                Xq = None

        try:
            lb, ub = est.effect_interval(Xq, T0=T0, T1=T1, alpha=alpha)
        except Exception as e:
            return CausalResult(
                status="INVALID",
                issues=[invalid("interval", f"effect_interval failed: {e}", fix="Re-fit with options.fit.inference='auto' or 'bootstrap' (not None).")],
            )

        return CausalResult(status="OK", model_id=model_id, outputs={"lb": json_safe(lb), "ub": json_safe(ub), "alpha": alpha})
