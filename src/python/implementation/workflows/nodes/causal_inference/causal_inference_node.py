from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.causal_inference.causal_inference_deps import CausalInferenceDeps
from python.implementation.workflows.nodes.causal_inference.causal_inference_prompts import (
    CATE_GENERAL_PROMPT,
    CATE_INCLUSION_PROMPT,
    CATE_SUMMARY_PROMPT,
    CAUSAL_INFERENCE_ATE_SUMMARY_SYSTEM_PROMPT,
    CAUSAL_INFERENCE_ATE_SUMMARY_USER_PROMPT_TEMPLATE,
    INVALID_PLAN_MESSAGE_PROMPT,
)
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import CausalInferenceState


from python.implementation.workflows.tools.causal.causal_command import (
    ATECommand,
    ATEInputsModel,
    ATEResult,
    ATESuccess,
    CATECommand,
    CATEInputs,
    CATEModelResult,
    CATEResult,
    CATESuccess,
    CommandFailure,
)
from python.implementation.workflows.tools.causal.causal_model import CausalModel
from python.implementation.workflows.tools.causal.causal_model_factory_tool import CausalModelFactoryTool
from python.implementation.workflows.tools.causal.causal_spec import (
    BinaryOutcomeSpecModel,
    BinaryTreatmentSpecModel,
    CausalSpec,
    ContinuousOutcomeSpecModel,
)

from python.implementation.workflows.tools.causal.encoding_plan import TransformPlan
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.tools.data_processing.data_processing_tool import (
    ALLOWED_OPS,
    SCALAR_OPS,
    SET_OPS,
    DataProcessingTool,
    IncExcRuleModel,
    InclusionPlanModel,
)
from python.implementation.workflows.tools.data_profiling.causal_data_profiling_tool import CausalDataProfilingTool
from python.implementation.workflows.tools.data_profiling.plots.model import CohortCate, GraphImage
from python.implementation.workflows.utils.validation import ValidationIssueModel

# ============================================================
# Node
# ============================================================

@dataclass(frozen=True, slots=True)
class CausalInferenceNode(Node):
    llm: LLMService
    data_repo: DataRepo

    @property
    def name(self) -> str:
        return CausalInferenceState.NAME

    @classmethod
    def get_info(cls) -> str:
        return "Computes ATE from the trained causal model and answers clinician questions (ATE + CATE)."

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        tool_factory: ToolFactory,
        previous_state_dependencies: Any,
        messages_history: Optional[Sequence[ChatMessage]],
    ) -> State:
        if not isinstance(state, CausalInferenceState):
            raise ValueError(f"{self.name}: invalid state (got {type(state).__name__})")

        deps = CausalInferenceDeps.from_loaded(previous_state_dependencies)

        causal_specs = deps.compile_protocol.payload.causal_specs
        assert causal_specs is not None, "CausalSpecs is required in CompileProtocolState payload"

        trained_model_id = getattr(deps.model_train.payload, "trained_model_id", None)
        assert trained_model_id is not None, "trained_model_id is required in ModelTrainState payload"

        clean_dataset_id = getattr(deps.clean_protocol.payload, "clean_dataset_id", None)
        assert clean_dataset_id is not None, "clean_dataset_id is required in CleanProtocolState payload"

        selected = deps.model_selection.payload.confirmed_model_selection
        assert selected is not None, "Confirmed model selection is required in ModelSelectionState payload"

        selected_model_fqcn = selected.selected_model
        assert selected_model_fqcn is not None, "selected_model (fqcn) is required"

        data_summary = deps.clean_protocol.payload.summary
        assert data_summary is not None, "dataset_summary is required in CleanProtocolState"

        order_X = deps.model_train.payload.order_X
        order_W = deps.model_train.payload.order_W
        assert order_X is not None, "order_X is required in ModelTrainState payload"
        assert order_W is not None, "order_W is required in ModelTrainState payload"

        # Resolve model
        mf_raw = tool_factory.get_tool(CausalModelFactoryTool.NAME)
        model_factory = cast(CausalModelFactoryTool, mf_raw)
        model = model_factory.resolve(selected_model_fqcn)
        assert model is not None, f"Model factory could not resolve model for fqcn: {selected_model_fqcn}"

        # Tools
        dp_raw = tool_factory.get_tool(DataProcessingTool.NAME)
        data_processing_tool = cast(DataProcessingTool, dp_raw)

        # Optional tool (kept for parity with your existing wiring)
        _prof_raw = tool_factory.get_tool(CausalDataProfilingTool.NAME)
        _data_profiling_tool = cast(CausalDataProfilingTool, _prof_raw)  # currently unused here

        # Context for ATE summarizer prompt
        context: dict[str, Any] = {
            "selected_model": selected_model_fqcn,
            "causal_specs": causal_specs.model_dump(mode="json"),
            "dataset_summary": data_summary.model_dump(mode="json"),
        }

        last_8 = messages_history[-8:] if messages_history else None

        # ---------------------------
        # ATE: compute once (idempotent)
        # ---------------------------
        if state.payload.ate_result_raw_json_str is None:

            cmd = ATECommand(
                model_name=selected_model_fqcn,
                dataset_id=clean_dataset_id,
                run_id=uuid4(),
                data_summary=data_summary,
                transformation_plan=deps.model_train.payload.column_transformation_plan,
                causal_specs=causal_specs,
                fitted_model_id=trained_model_id,
                order_X=order_X,
                order_W=order_W,
                inputs=ATEInputsModel(),
                options={},
            )

            logging.warning(
                f"Executing ATECommand: model={selected_model_fqcn} dataset_id={clean_dataset_id} fitted_model_id={trained_model_id}"
            )
            res = model.execute(user_id=user_id, conversation_id=conversation_id, command=cmd)
            logging.warning(f"ATECommand executed with result: {res}")

            if not isinstance(res, ATEResult):
                raise TypeError(f"Expected ATEResult from model.execute, got {type(res).__name__}")

            match res:
                case ATESuccess():
                    ate_json = _serialize_result_to_json_str(res.ate)
                    warnings = res.warnings if hasattr(res, "warnings") else []
                    summary_out = self.llm.generate(
                        system_prompt=CAUSAL_INFERENCE_ATE_SUMMARY_SYSTEM_PROMPT,
                        user_prompt=CAUSAL_INFERENCE_ATE_SUMMARY_USER_PROMPT_TEMPLATE.format(
                            context_json=_dumps(context),
                            ate_result_json=ate_json,
                            warnings_json=_dumps(warnings),
                        ),
                        config=LLMConfig(temperature=0.2, model="basic"),
                        history=last_8,
                    ).content.strip()

                    return CausalInferenceState(
                        payload=state.payload.model_copy(
                            update={
                                "ate_result_raw_json_str": ate_json,
                                "ate_inference_error": None,
                                "should_abort": False,
                                "abort_error_message": None,
                                "message": summary_out,
                            }
                        )
                    )

                case CommandFailure():
                    msg = f"ATE computation failed: {res.error.message}"
                    return CausalInferenceState(
                        payload=state.payload.model_copy(
                            update={
                                "ate_result_raw_json_str": None,
                                "error": res.error.message,
                                "should_abort": True,
                                "message": msg,
                            }
                        )
                    )

                case _:
                    raise TypeError(f"Unhandled ATEResult type: {type(res).__name__}")

        # ---------------------------
        # If ATE already exists, interpret user request: answer-from-context OR compute CATE
        # ---------------------------
        return _process_cate_question(
            llm=self.llm,
            data_repo=self.data_repo,
            user_id=user_id,
            conversation_id=conversation_id,
            ate_model_output_json_str=state.payload.ate_result_raw_json_str,
            messages_history=messages_history,
            current_state= state,
            causal_specs=causal_specs,
            clean_dataset_id=clean_dataset_id,
            data_summary=data_summary,
            transformation_plan=deps.model_train.payload.column_transformation_plan,
            selected_model_fqcn=selected_model_fqcn,
            trained_model_id=trained_model_id,
            order_X=order_X,
            order_W=order_W,
            model=model,
            data_processing_tool=data_processing_tool,
            data_profiling_tool=_data_profiling_tool,
        )





# ============================================================
# Small utilities
# ============================================================

def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def _serialize_result_to_json_str(res: Any) -> str:
    if hasattr(res, "model_dump"):
        return _dumps(res.model_dump(mode="json"))
    if isinstance(res, dict):
        return _dumps(res)
    try:
        import dataclasses
        if dataclasses.is_dataclass(res) and not isinstance(res, type):
            return _dumps(dataclasses.asdict(res))
    except Exception:
        pass
    return _dumps({"repr": repr(res)})


def _last_user_text(history: Optional[Sequence[ChatMessage]]) -> str:
    if not history:
        return ""
    for msg in reversed(history):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    return ""


def _extract_cols_data(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    cols_list = [str(c) for c in cols]
    missing = [c for c in cols_list if c not in df.columns]
    if missing:
        raise KeyError(
            f"Requested columns not found in df: {missing}. Available columns: {list(df.columns)}"
        )
    return df.loc[:, cols_list].copy()


# ============================================================
# Intent router payload (tight + reliable)
# ============================================================

class _CateIntentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prev_context_relevant: bool
    answer: str = ""

    @model_validator(mode="after")
    def _normalize(self) -> "_CateIntentPayload":
        if not self.prev_context_relevant:
            self.answer = ""
        return self


# ============================================================
# Inclusion plan semantic validation (X-only + operator/value shape)
# ============================================================

def _validate_inclusion_plan_semantic(
    *,
    plan: InclusionPlanModel,
    effect_modifiers: Sequence[str],
) -> List[ValidationIssueModel]:
    issues: List[ValidationIssueModel] = []
    allowed_x = {str(c) for c in effect_modifiers}

    for c_idx, cohort in enumerate(plan.rules):
        gk = str(cohort.group_key)

        for r_idx, r in enumerate(cohort.inclusion_rules):
            col = str(r.column)

            if col not in allowed_x:
                issues.append(
                    ValidationIssueModel(
                        severity="FAIL",
                        message=f"[group {gk}] Inclusion rule column '{col}' is not an effect modifier (X).",
                        evidence={
                            "group_index": c_idx,
                            "rule_index": r_idx,
                            "column": col,
                            "op": r.op,
                            "values": r.values,
                            "allowed_effect_modifiers": sorted(allowed_x),
                        },
                        fix_hint="Use only columns from protocol.effect_modifiers (X).",
                    )
                )

            if r.op in ("==", ">=", "<=", ">", "<"):
                if len(r.values) != 1:
                    issues.append(
                        ValidationIssueModel(
                            severity="FAIL",
                            message=f"[group {gk}] Rule on '{col}' with op '{r.op}' requires exactly 1 value; got {len(r.values)}.",
                            evidence={
                                "group_index": c_idx,
                                "rule_index": r_idx,
                                "column": col,
                                "op": r.op,
                                "values": r.values,
                            },
                            fix_hint="For scalar ops (==, >=, <=, >, <), set values=[single_value].",
                        )
                    )
            elif r.op in ("in", "not_in"):
                if len(r.values) < 1:
                    issues.append(
                        ValidationIssueModel(
                            severity="FAIL",
                            message=f"[group {gk}] Rule on '{col}' with op '{r.op}' requires a non-empty values list.",
                            evidence={
                                "group_index": c_idx,
                                "rule_index": r_idx,
                                "column": col,
                                "op": r.op,
                                "values": r.values,
                            },
                            fix_hint="For membership ops (in, not_in), set values=[v1, v2, ...].",
                        )
                    )
            else:
                issues.append(
                    ValidationIssueModel(
                        severity="FAIL",
                        message=f"[group {gk}] Unsupported operator '{r.op}' in rule for '{col}'.",
                        evidence={
                            "group_index": c_idx,
                            "rule_index": r_idx,
                            "column": col,
                            "op": r.op,
                            "values": r.values,
                        },
                        fix_hint="Use one of: ==, in, not_in, >=, <=, >, <",
                    )
                )

    return issues


# ============================================================
# CATE post-processing: raw if n<=5 else stats (cate + intervals + inference)
# ============================================================

def _to_1d_float(arr: Any) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        a = arr # pyright: ignore[reportUnknownVariableType]
    elif isinstance(arr, (list, tuple)):
        try:
            a = np.asarray(arr, dtype=float)
        except Exception:
            return None
    else:
        return None
    if a.ndim == 0:
        a = a.reshape(1) # pyright: ignore[reportUnknownVariableType]
    return a.astype(float, copy=False).ravel()


def _stats(arr: np.ndarray) -> Dict[str, Any]:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return {"n": 0}
    q = np.percentile(a, [10, 25, 50, 75, 90])
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(q[2]),
        "std": float(np.std(a, ddof=1)) if a.size >= 2 else 0.0,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p10": float(q[0]),
        "p25": float(q[1]),
        "p75": float(q[3]),
        "p90": float(q[4]),
        "frac_positive": float(np.mean(a > 0.0)),
        "frac_negative": float(np.mean(a < 0.0)),
    }


def _interval_stats(lower: Optional[np.ndarray], upper: Optional[np.ndarray]) -> Dict[str, Any]:
    if lower is None or upper is None:
        return {"available": False}
    if lower.shape != upper.shape:
        return {"available": False, "reason": "shape_mismatch"}
    width = upper - lower
    mask = np.isfinite(lower) & np.isfinite(upper)
    if not np.any(mask):
        return {"available": False, "reason": "no_finite"}
    l = lower[mask]
    u = upper[mask]
    w = width[mask]
    crosses0 = (l <= 0.0) & (u >= 0.0)
    return {
        "available": True,
        "n": int(l.size),
        "mean_width": float(np.mean(w)),
        "median_width": float(np.median(w)),
        "frac_crosses_zero": float(np.mean(crosses0)),
        "mean_lower": float(np.mean(l)),
        "mean_upper": float(np.mean(u)),
    }


def _extract_effect_fields(effect_obj: Dict[CATEModelResult, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Any]:
    """
    Supports dict-like payloads with keys:
      - "cate"
      - "cate_interval" (tuple/list of (lower, upper) or dict {"lower":..., "upper":...})
      - "cate_inference"
    """
    cate = None
    lo = None
    hi = None
    inf = None
    logging.warning("effect_obj type=%s", type(effect_obj))
    logging.warning("effect_obj class=%s", type(effect_obj).__name__)
    cate_raw = effect_obj["cate"]
    cate = _to_1d_float(cate_raw)
    
    if effect_obj["cate_interval"] is not None:
        interval_raw = effect_obj["cate_interval"]
        lo = _to_1d_float(interval_raw[0])
        hi = _to_1d_float(interval_raw[1])
    if effect_obj["cate_inference"] is not None:
        inf = effect_obj["cate_inference"]
        
    return cate, lo, hi, inf


def _make_llm_cate_payload_for_group(
    *,
    group_key: str,
    inclusion_rules: Sequence[IncExcRuleModel],
    is_counterfactual: bool,
    t0: Any,
    t1: Any,
    outcome_kind: str,
    effect_obj: Dict[CATEModelResult, Any],
) -> Dict[str, Any]:
    cate, lo, hi, inf = _extract_effect_fields(effect_obj)

    rules_compact = [ # pyright: ignore[reportUnknownVariableType]
        {"column": str(r.column), "op": r.op, "values": list(r.values)}
        for r in inclusion_rules
    ]

    out: Dict[str, Any] = {
        "group_key": group_key,
        "is_counterfactual": bool(is_counterfactual),
        "contrast": {"t0": t0, "t1": t1},
        "outcome_kind": outcome_kind,
        "inclusion_rules": rules_compact,
        "cate": None,
        "cate_interval": None,
        "cate_inference": None,
    }

    if cate is None:
        out["cate"] = {"available": False}
        if inf is not None:
            out["cate_inference"] = inf
        return out

    n = int(cate.size)
    if n <= 5:
        out["cate"] = {"available": True, "n": n, "values": [float(x) for x in cate.tolist()]}
        if lo is not None and hi is not None and lo.size == n and hi.size == n:
            out["cate_interval"] = {
                "available": True,
                "pairs": [{"lower": float(a), "upper": float(b)} for a, b in zip(lo.tolist(), hi.tolist())],
            }
        else:
            out["cate_interval"] = _interval_stats(lo, hi)
    else:
        out["cate"] = {"available": True, "summary": _stats(cate)}
        out["cate_interval"] = _interval_stats(lo, hi)

    if inf is not None:
        out["cate_inference"] = inf

    return out


# ============================================================
# CATE processing (uses YOUR DataProcessingTool for filtering)
# ============================================================

def _apply_rules_with_tool(
    *,
    tool: DataProcessingTool,
    df: pd.DataFrame,
    rules: Sequence[IncExcRuleModel],
) -> pd.DataFrame:
    """
    Use your existing DataProcessingTool (do not reimplement filtering logic).
    Support common parameter names robustly.
    """
    try:
        return tool.apply_inclusion_rules(df=df, rules=rules)  # type: ignore[arg-type]
    except TypeError:
        # some implementations prefer inclusion_rules=
        return tool.apply_inclusion_rules(df=df, inclusion_rules=rules)  # type: ignore[call-arg]


def _binary_t0_t1(causal_specs: CausalSpec, *, is_counterfactual: bool) -> Tuple[Any, Any]:
    t = causal_specs.treatment_spec
    if not isinstance(t, BinaryTreatmentSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"Binary treatment required, got {type(t).__name__}")
    treated = t.treated
    control = t.control
    if is_counterfactual:
        # reverse direction: "no treatment vs treatment" -> swap
        return treated, control  # (t0, t1) swapped relative to normal
    return control, treated


def _process_cate_question(
    *,
    llm: LLMService,
    data_repo: DataRepo,
    user_id: UUID,
    conversation_id: UUID,
    ate_model_output_json_str: str,
    messages_history: Optional[Sequence[ChatMessage]],
    current_state: CausalInferenceState,
    causal_specs: CausalSpec,
    clean_dataset_id: UUID,
    data_summary: DatasetSummaryModel,
    transformation_plan: Optional[TransformPlan],
    selected_model_fqcn: str,
    trained_model_id: UUID,
    order_X: List[str],
    order_W: List[str],
    model: CausalModel,
    data_processing_tool: DataProcessingTool,
    data_profiling_tool: CausalDataProfilingTool,
) -> CausalInferenceState:
    last_8 = messages_history[-8:] if messages_history else None
    last_4_messages = messages_history[-4:] if messages_history else None

    # ---------------------------
    # 1) Context router (answer from history/ATE if possible)
    # ---------------------------
    intent = llm.generate_json(
        schema=_CateIntentPayload,
        user_prompt=CATE_GENERAL_PROMPT+
            f"\n ATESummary:\n{ate_model_output_json_str}\n",
        system_prompt=None,
        config=LLMConfig(temperature=0.2, model="basic"),
        history=last_8,
        max_attempts=3,
    )

    if intent.prev_context_relevant and intent.answer.strip():
        return CausalInferenceState(
            payload=current_state.payload.model_copy(update={"message": intent.answer.strip()})
        )

    # ---------------------------
    # 2) Build inclusion plan (multi-cohort, X-only)
    # ---------------------------
    user_q = _last_user_text(messages_history)
    error_message: Optional[str] = None
    plan: Optional[InclusionPlanModel] = None
    
    effect_modifiers_summary = _filter_dataset_summary_to_effect_modifiers(summary=data_summary, effect_modifiers=causal_specs.effect_modifiers)
    logging.warning(f"Effect modifiers summary for prompt: {effect_modifiers_summary.model_dump_json()}")
    plot_cohorts: List[CohortCate] = [] 
    for attempt in range(3):
        plan = llm.generate_json(
            schema=InclusionPlanModel,
            system_prompt=None,
            user_prompt=(
                CATE_INCLUSION_PROMPT
                +f"\n\nEffect modifiers summary (only these columns can be used for cohort definitions):\n{effect_modifiers_summary.model_dump_json()}\n"
                +f"Effect modifiers columns: {', '.join(causal_specs.effect_modifiers)}\n"
                +f"\n\nUSER_QUESTION:\n{user_q}\n"
                + (f"\nPrevious error message:\n{error_message}\n" if error_message else "")
            ),
            config=LLMConfig(temperature=0.1, model="pro"),
            history=last_4_messages,
            max_attempts=3,
        )

        issues = _validate_inclusion_plan_semantic(plan=plan, effect_modifiers=causal_specs.effect_modifiers)
        if not issues or (len(issues) == 0 and len(plan.rules) > 0):
            break

        logging.warning(f"Invalid inclusion plan (attempt {attempt + 1}): {plan}")
        error_message = (
            "Your inclusion plan has the following issues or plan rules are empty:\n"
            + "\n".join(f"- {i.message}" for i in issues)
            + "\nFix them and output JSON only in the required schema."
        )  
    
    logging.warning(f"Inclusion plan after validation attempts: {plan}")
    is_valid, log = _validate_inclusion_plan(plan, effect_modifiers=causal_specs.effect_modifiers)
    if not is_valid:
        logging.warning(f"Final inclusion plan is invalid: {log}")
        invalid_plan_message = _invalid_plan_message(
            llm=llm,
            model_name="basic",
            effect_modifiers_summary=effect_modifiers_summary,
            effect_modifiers=causal_specs.effect_modifiers
        )
                                                     
        return CausalInferenceState(
            payload=current_state.payload.model_copy(
                update={
                    "message": invalid_plan_message, 
                }
            )
        )
        
    # ---------------------------
    # 3) Load X dataframe and compute CATE per cohort
    # ---------------------------
    df = data_repo.get_csv_data(
        user_id=user_id,
        conversation_id=conversation_id,
        dataset_id=clean_dataset_id,
    )
    df_x = _extract_cols_data(df=df, cols=order_X)

    outcome_kind = "unknown"
    if isinstance(causal_specs.outcome_spec, BinaryOutcomeSpecModel):
        outcome_kind = "binary"
    elif isinstance(causal_specs.outcome_spec, ContinuousOutcomeSpecModel): # pyright: ignore[reportUnnecessaryIsInstance]
        outcome_kind = "continuous"

    group_payloads: List[Dict[str, Any]] = []
    non_empty_any = False

    for cohort in plan.rules: # pyright: ignore[reportOptionalMemberAccess]
        gk = str(cohort.group_key)
        cohort_df = _apply_rules_with_tool(
            tool=data_processing_tool,
            df=df_x,
            rules=cohort.inclusion_rules,
        )

        if cohort_df.empty:
            group_payloads.append(
                {
                    "group_key": gk,
                    "is_counterfactual": bool(cohort.is_counterfactual),
                    "inclusion_rules": [
                        {"column": str(r.column), "op": r.op, "values": list(r.values)}
                        for r in cohort.inclusion_rules
                    ],
                    "empty": True,
                    "message": "No rows matched this cohort’s inclusion rules.",
                }
            )
            continue

        non_empty_any = True

        # Decide t0/t1 for THIS cohort (binary treatment; swap if counterfactual)
        t0, t1 = _binary_t0_t1(causal_specs, is_counterfactual=bool(cohort.is_counterfactual))

        cate_inputs = CATEInputs(x_rows=cohort_df, t0=t0, t1=t1)

        cmd = CATECommand(
            model_name=selected_model_fqcn,
            dataset_id=clean_dataset_id,
            run_id=uuid4(),
            data_summary=data_summary,
            transformation_plan=transformation_plan,
            causal_specs=causal_specs,
            fitted_model_id=trained_model_id,
            order_X=order_X,
            order_W=order_W,
            inputs=cate_inputs,
            options={},
        )

        res = model.execute(
            user_id=user_id,
            conversation_id=conversation_id,
            command=cmd,
        )

        if not isinstance(res, CATEResult):
            raise TypeError(f"Expected CATEResult from model.execute, got {type(res).__name__}")
        
        match res:
            case CATESuccess():           
                group_payloads.append(
                    _make_llm_cate_payload_for_group(
                        group_key=gk,
                        inclusion_rules=cohort.inclusion_rules,
                        is_counterfactual=bool(cohort.is_counterfactual),
                        t0=t0,
                        t1=t1,
                        outcome_kind=outcome_kind,
                        effect_obj=res.effects,
                    )
                )
                cate_arr, lo_arr, hi_arr, _ = _extract_effect_fields(res.effects)
                if cate_arr is not None and cate_arr.size > 0:
                    plot_cohorts.append(
                        CohortCate(
                            group_key=gk,
                            cate=cate_arr,
                            lower=lo_arr,
                            upper=hi_arr,
                        )
                )

            case CommandFailure():
                group_payloads.append(
                    {
                        "group_key": gk,
                        "is_counterfactual": bool(cohort.is_counterfactual),
                        "inclusion_rules": [
                            {"column": str(r.column), "op": r.op, "values": list(r.values)}
                            for r in cohort.inclusion_rules
                        ],
                        "error": f"CATE computation failed: {res.error.message}",
                    }
                )

            case _:
                raise TypeError(f"Unhandled CATEResult type: {type(res).__name__}")

    if not non_empty_any:
        return CausalInferenceState(
            payload=current_state.payload.model_copy(
                update={
                    "message": (
                        "After applying your cohort rules, I found **0 rows** for all cohorts.\n"
                        "Please loosen the inclusion rules or choose effect-modifier values that exist in the data.\n"
                        f"Allowed effect modifiers (X): {', '.join(causal_specs.effect_modifiers)}"
                    )
                }
            )
        )
    
    graphs: List[GraphImage] = []
    artifacts: Optional[List[UUID]] = []
    if len(plot_cohorts) > 0:
        graphs.append(data_profiling_tool.plot_cate_distribution(plot_cohorts, causal_specs))
    if len(plot_cohorts) == 1:
        graphs.append(data_profiling_tool.plot_cate_sorted_curve(plot_cohorts, causal_specs))
    else:
        graphs.append(data_profiling_tool.plot_cate_forest_mean_ci(plot_cohorts, causal_specs))

    # Persist artifacts (you said you'll wire return processing)
    for g in graphs:
        aid = uuid4()
        artifacts.append(aid)
        data_repo.save_artifact(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=aid,
            content=g.content,
            mime=g.mime,
        )
    # ---------------------------
    # 4) LLM summarization (clinician-friendly)
    #    - For n<=5 we included raw values
    #    - For n>5 we included robust stats
    # ---------------------------
    llm_payload = { # pyright: ignore[reportUnknownVariableType]
        "selected_model": selected_model_fqcn,
        "outcome_kind": outcome_kind,
        "treatment_column": getattr(causal_specs.treatment_spec, "column", None),
        "cohorts": group_payloads,
        # Optional: useful for context, but not huge
        "ate_summary_raw": ate_model_output_json_str,
    }

    answer = llm.generate(
        system_prompt=CATE_SUMMARY_PROMPT,
        user_prompt=_dumps(llm_payload),
        config=LLMConfig(temperature=0.2, model="basic"),
        history=last_8,
    ).content.strip()

    return CausalInferenceState(
        payload=current_state.payload.model_copy(update={"message": answer,
                                                         "artifacts": (current_state.payload.artifacts or []) + (artifacts if artifacts else [])
                                                         }),
        current_artifact_ids=artifacts
    )



def _validate_inclusion_plan(
    plan: Optional[InclusionPlanModel],
    *,
    effect_modifiers: Sequence[str],
    require_rules_per_cohort: bool = True,   # comparisons should be True
) -> Tuple[bool, str]:
    if plan is None:
        return False, "plan is None"
    if not plan.rules:
        return False, "plan.rules is empty"

    allowed_cols = {str(c).strip() for c in effect_modifiers}
    if not allowed_cols:
        return False, "effect_modifiers is empty"

    for cohort in plan.rules:
        gk = str(cohort.group_key).strip()
        if not gk:
            return False, "missing/empty group_key"

        rules = cohort.inclusion_rules or []
        if require_rules_per_cohort and len(rules) == 0:
            return False, f"group '{gk}': inclusion_rules is empty"

        for rule in rules:
            col = str(rule.column).strip()
            op = str(rule.op).strip()

            if col not in allowed_cols:
                return False, f"group '{gk}': column '{col}' not in effect modifiers (X={sorted(allowed_cols)})"

            if op not in ALLOWED_OPS:
                return False, f"group '{gk}': unsupported op '{op}'"

            nvals = len(rule.values)
            if op in SCALAR_OPS and nvals != 1:
                return False, f"group '{gk}': op '{op}' requires exactly 1 value (got {nvals})"
            if op in SET_OPS and nvals < 1:
                return False, f"group '{gk}': op '{op}' requires at least 1 value"

    return True, ""


def _filter_dataset_summary_to_effect_modifiers(
    summary: DatasetSummaryModel,
    effect_modifiers: Sequence[str],
    *,
    strict: bool = True,
) -> DatasetSummaryModel:
    """
    Return a DatasetSummaryModel containing only column profiles whose names are in `effect_modifiers`.

    - Preserves original profile order (df.columns order).
    - Updates n_rows consistently (keeps original summary.n_rows).
    - strict=True raises if any requested effect modifier is missing from summary.profiles.
    """
    wanted = [str(c) for c in effect_modifiers]
    wanted_set = set(wanted)

    # Keep deterministic order: follow summary.profiles order
    kept = [p for p in summary.profiles if str(p.name) in wanted_set]

    if strict:
        present = {str(p.name) for p in summary.profiles}
        missing = [c for c in wanted if c not in present]
        if missing:
            raise KeyError(
                f"Effect modifier columns missing from DatasetSummaryModel.profiles: {missing}. "
                f"Available: {sorted(present)}"
            )

    # Rebuild via model_dump to avoid any mutation / union quirks
    return DatasetSummaryModel.model_validate(
        {
            "n_rows": int(summary.n_rows),
            "profiles": [p.model_dump(mode="python") for p in kept],
        }
    )   


def _invalid_plan_message(
    *,
    llm: LLMService,
    model_name: str,
    effect_modifiers: Sequence[str],
    effect_modifiers_summary: DatasetSummaryModel,
    history: Optional[Sequence[ChatMessage]] = None,
) -> str:
    last_user_message = _last_user_text(history)
    cols = [str(c) for c in effect_modifiers]
    return llm.generate(
        system_prompt=None,
        user_prompt=INVALID_PLAN_MESSAGE_PROMPT +
            f"\nEffect modifiers summary (only these columns can be used for cohort definitions):\n{effect_modifiers_summary.model_dump_json()}\n"
            +f"Effect modifiers columns: {', '.join(cols)}\n"
            +f"\nUser question that led to invalid plan:\n{last_user_message}\n",
        config=LLMConfig(temperature=0.7, model="basic"),
        history=None,
 
    ).content.strip()        