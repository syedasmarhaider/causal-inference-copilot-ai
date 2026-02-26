from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any, ClassVar, Dict, List, Literal, Optional, Sequence, Tuple, cast
from uuid import UUID, uuid4

import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, ConfigDict, Field

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory

from python.implementation.workflows.nodes.compile_protocol.protocol_specs import ProtocolSpec
from python.implementation.workflows.nodes.transform_protocol.transform_protcol_plan import (
    TransformPlanModel,
    validate_plan_against_df_columns,
)
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_deps import TransformProtocolDeps
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_encoding import apply_encoding_plan
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_prompts import (
    build_hard_validation_system_prompt,
    build_transform_plan_system_prompt,
    build_transform_plan_user_prompt_template,
    build_user_friendly_message_for_transform_protocol_system_prompt,
    get_transform_protocol_node_info,
)
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_state import (
    MaxAttempt,
    TransformProtocolPayloadModel,
    TransformProtocolState,
)
from python.implementation.workflows.tools.data.data_profiling_tool import (
    DatasetProfilingStateTool,
    DatasetSummaryModel,
)
from python.implementation.workflows.utils.validation import ValidationIssueModel


# =============================================================================
# Issue helpers
# =============================================================================
def _is_fail(issue: ValidationIssueModel) -> bool:
    return issue.severity == "FAIL"


def _fail(
    message: str,
    evidence: Optional[dict[str, Any]] = None,
    fix_hint: Optional[str] = None,
) -> ValidationIssueModel:
    return ValidationIssueModel(severity="FAIL", message=message, evidence=evidence or {}, fix_hint=fix_hint)

# =============================================================================
# LLM messages
# =============================================================================
def _llm_stage_message(
    *,
    llm: LLMService,
    stage: str,
    ok: bool,
    payload: dict[str, Any],
) -> str:
    system = (
        "You are a progress reporter for a data transformation pipeline.\n"
        "Write clear and comprehensive status update for the user.\n"
        "Rules:\n"
        "- Start with a one-line headline.\n"
        "- Be concrete (counts, column names) when present.\n"
        "- If failure, say what failed and what will happen next.\n"
        "- Do NOT mention internal class names.\n"
    )
    user = json.dumps(
        {"stage": stage, "ok": ok, "context": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return llm.generate(
        system_prompt=system,
        user_prompt=user,
        config=LLMConfig(temperature=0.2),
        history=None,
    ).content


def get_message_for_hard_validation_issue(llm: LLMService, issues: List[ValidationIssueModel]) -> str:
    return llm.generate(
        system_prompt="You are an assistant for generating user-friendly error messages.",
        user_prompt=build_hard_validation_system_prompt().format(
            validation_issues_json=json.dumps(
                [i.model_dump(mode="json") for i in issues],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        config=LLMConfig(temperature=1.0),
        history=None,
    ).content


# =============================================================================
# Minimal post-transform validation (covariates + effect modifiers only)
# =============================================================================
def validate_covariates_and_effect_modifiers_numeric_only(
    *,
    df_after: pd.DataFrame,
    protocol_after: ProtocolSpec,
) -> List[ValidationIssueModel]:
    issues: List[ValidationIssueModel] = []

    cols = list(protocol_after.covariates) + list(protocol_after.effect_modifiers)

    missing = sorted([c for c in cols if c not in df_after.columns])
    if missing:
        issues.append(
            _fail(
                "Missing covariate/effect-modifier columns in transformed dataframe.",
                evidence={
                    "missing_columns": missing,
                    "expected_covariates": list(protocol_after.covariates),
                    "expected_effect_modifiers": list(protocol_after.effect_modifiers),
                },
                fix_hint="Update the transform plan so it produces these columns.",
            )
        )
        return issues

    non_numeric = [{"column": c, "dtype": str(df_after[c].dtype)} for c in cols if not is_numeric_dtype(df_after[c])]
    if non_numeric:
        issues.append(
            _fail(
                "Some covariates/effect modifiers are not numeric after transformation.",
                evidence={"non_numeric_columns": non_numeric},
                fix_hint="Encode these covariates/effect modifiers to numeric (e.g., one-hot / ordinal / to_numeric).",
            )
        )

    return issues


# =============================================================================
# PLAN: deterministic checks (planner defects -> auto replan)
# =============================================================================
def validate_plan_against_protocol_controls_only(
    *,
    plan: TransformPlanModel,
    protocol: ProtocolSpec,
    df_columns: Sequence[str],
) -> List[ValidationIssueModel]:
    """
    Deterministic plan constraints:
      - Plan columns exist in df (basic referential integrity)
      - Plan covers ALL covariates + effect modifiers
      - Plan must NOT include treatment or outcome
      - Plan must NOT include columns outside covariates/effect modifiers
    """
    issues: List[ValidationIssueModel] = []

    # 1) planned columns exist
    issues.extend(validate_plan_against_df_columns(plan=plan, df_columns=df_columns, require_full_coverage=False))

    t_col = protocol.treatment_spec.column
    y_col = protocol.outcome_spec.column

    controls = set(protocol.covariates) | set(protocol.effect_modifiers)

    plan_cols = [cp.column for cp in plan.columns]
    plan_set = set(plan_cols)

    # 2) forbid T/Y
    forbidden = sorted([c for c in plan_cols if c in (t_col, y_col)])
    if forbidden:
        issues.append(
            _fail(
                "Transform plan must not include treatment or outcome columns.",
                evidence={"forbidden_columns": forbidden, "treatment_column": t_col, "outcome_column": y_col},
                fix_hint="Remove treatment/outcome from the plan; only encode covariates/effect modifiers.",
            )
        )

    # 3) forbid non-controls
    illegal = sorted([c for c in plan_cols if c not in controls and c not in (t_col, y_col)])
    if illegal:
        issues.append(
            _fail(
                "Transform plan includes columns that are not covariates/effect modifiers.",
                evidence={"illegal_plan_columns": illegal, "allowed_controls": sorted(controls)},
                fix_hint="Only include protocol covariates/effect modifiers in the plan.",
            )
        )

    # 4) require full coverage of controls
    missing_controls = sorted([c for c in controls if c not in plan_set])
    if missing_controls:
        issues.append(
            _fail(
                "Transform plan does not cover all covariates/effect modifiers.",
                evidence={"missing_controls": missing_controls, "plan_columns": sorted(plan_set)},
                fix_hint="Add a plan item for each missing covariate/effect modifier. Use no_encoding_identity if already numeric.",
            )
        )

    return issues


# =============================================================================
# APPLY: apply plan + lineage (raw -> produced columns)
# =============================================================================
def apply_plan_with_lineage_or_raise(
    *,
    df: pd.DataFrame,
    plan: TransformPlanModel,
) -> Tuple[pd.DataFrame, List[ValidationIssueModel], Dict[str, List[str]]]:
    cur: Optional[pd.DataFrame] = df
    all_issues: List[ValidationIssueModel] = []
    raw_to_outputs: Dict[str, List[str]] = {}

    for cp in plan.columns:
        assert cur is not None
        before_cols = set(cur.columns.tolist())

        cur, step_issues = apply_encoding_plan(df=cur, plan=cp)

        # step_issues is List[ValidationIssueModel]
        if step_issues:
            all_issues.extend([step_issues])

        after_cols = set(cur.columns.tolist())
        added = sorted(after_cols - before_cols)

        if added:
            raw_to_outputs[cp.column] = added
        elif cp.column in after_cols:
            raw_to_outputs[cp.column] = [cp.column]
        else:
            raw_to_outputs[cp.column] = []

    assert cur is not None
    return cur, all_issues, raw_to_outputs


def build_protocol_after_from_lineage(
    *,
    protocol_before: ProtocolSpec,
    df_after: pd.DataFrame,
    raw_to_outputs: Dict[str, List[str]],
) -> ProtocolSpec:
    """
    Keep treatment/outcome unchanged.
    Expand covariates/effect_modifiers to produced feature columns.
    """
    df_cols = set(df_after.columns.tolist())
    t_col = protocol_before.treatment_spec.column
    y_col = protocol_before.outcome_spec.column

    def _expand(raw_list: List[str]) -> List[str]:
        out: List[str] = []
        for raw in raw_list:
            outs = raw_to_outputs.get(raw, [raw])
            out.extend(outs)

        cleaned = [c for c in out if c not in (t_col, y_col) and c in df_cols]

        # de-dup preserve order
        seen: set[str] = set()
        final: List[str] = []
        for c in cleaned:
            if c not in seen:
                seen.add(c)
                final.append(c)
        return final

    return protocol_before.model_copy(
        update={
            "covariates": _expand(list(protocol_before.covariates)),
            "effect_modifiers": _expand(list(protocol_before.effect_modifiers)),
        }
    )


# =============================================================================
# Negotiation: deterministic decision parsing via LLM->Pydantic (no heuristics)
# =============================================================================
NegotiationAction = str  # kept simple; enforced by Pydantic Literal below


class NegotiationDecisionModel(BaseModel):
    """
    action:
      - approve: accept current plan and proceed
      - change: user wants modifications; replan using feedback
      - question: user is asking; respond and stay in NEGOTIATE
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["approve", "change", "question"] = Field(...)
    feedback: Optional[str] = None


def parse_negotiation_decision(
    *,
    llm: LLMService,
    protocol: ProtocolSpec,
    plan: TransformPlanModel,
    issues: List[ValidationIssueModel],
    user_text: str,
) -> NegotiationDecisionModel:
    system = (
        "You are a strict parser for a clinician negotiation step.\n"
        "Task: classify the user's last message into one of:\n"
        "- approve: they accept the plan and want to proceed.\n"
        "- change: they want modifications to the plan.\n"
        "- question: they are asking a question / need clarification.\n"
        "\n"
        "Return ONLY valid JSON for NegotiationDecisionModel.\n"
        "Put the user's requested change or question text into 'feedback'.\n"
    )
    user = json.dumps(
        {
            "user_text": user_text,
            "context": {
                "protocol": protocol.model_dump(mode="json"),
                "plan": plan.model_dump(mode="json"),
                "issues": [i.model_dump(mode="json") for i in issues],
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return llm.generate_json(
        schema=NegotiationDecisionModel,
        system_prompt=system,
        user_prompt=user,
        config=LLMConfig(temperature=0.0),
        history=None,
        max_attempts=2,
    )


def answer_negotiation_question(
    *,
    llm: LLMService,
    protocol: ProtocolSpec,
    plan: TransformPlanModel,
    issues: List[ValidationIssueModel],
    question_text: str,
    messages_history: Optional[Sequence[ChatMessage]],
) -> str:
    system = build_user_friendly_message_for_transform_protocol_system_prompt().format(
        causal_transformation_summary="CAUSAL_TRANSFORMATION_SUMMARY"
    )
    user = json.dumps(
        {
            "instruction": "Answer the clinician question about the transformation plan in simple language. "
                           "End by asking whether to proceed or what to change.",
            "question": question_text,
            "protocol": protocol.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "issues": [i.model_dump(mode="json") for i in issues],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    history = list(messages_history[-12:]) if messages_history else None
    return llm.generate(
        system_prompt=system,
        user_prompt=user,
        config=LLMConfig(temperature=0.7),
        history=history,
    ).content


def build_plan_review_message(
    *,
    llm: LLMService,
    protocol: ProtocolSpec,
    plan: TransformPlanModel,
    dataset_summary: DatasetSummaryModel,
    messages_history: Optional[Sequence[ChatMessage]],
) -> str:
    system = build_user_friendly_message_for_transform_protocol_system_prompt().format(
        causal_transformation_summary="CAUSAL_TRANSFORMATION_SUMMARY"
    )
    user = json.dumps(
        {
            "instruction": (
                "Present this transformation plan to a clinician for approval. "
                "Be explicit: only covariates/effect modifiers are encoded; treatment/outcome unchanged. "
                "Ask them to either approve or describe changes."
            ),
            "protocol": protocol.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "dataset_summary": dataset_summary.model_dump(mode="json"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    history = list(messages_history[-12:]) if messages_history else None
    return llm.generate(
        system_prompt=system,
        user_prompt=user,
        config=LLMConfig(temperature=0.7),
        history=history,
    ).content


# =============================================================================
# Repair context + resets
# =============================================================================
def _make_repair_context_json(
    *,
    attempt: int,
    stage: str,
    issues: List[ValidationIssueModel],
    extra: Optional[dict[str, Any]] = None,
) -> str:
    fails = [x for x in issues if x.severity == "FAIL"]
    sample = (fails or issues)[:10]
    payload: dict[str, Any] = {
        "attempt": attempt,
        "stage": stage,
        "n_issues": len(issues),
        "issues_sample": [x.model_dump(mode="json") for x in sample],
        "instruction": "Produce a corrected TransformPlanModel. Output JSON only.",
    }
    if extra:
        payload["extra"] = extra
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _reset_to_plan(
    *,
    cleaned_dataset_id: UUID,
    cleaned_dataset_summary: DatasetSummaryModel,
    cleaned_dataset_validation_issues: List[ValidationIssueModel],
    issues: List[ValidationIssueModel],
    attempt_next: int,
    repair_context_json: str,
    user_message: str,
    protocol: ProtocolSpec,
) -> TransformProtocolState:
    return TransformProtocolState(
        TransformProtocolPayloadModel(
            needs_user_input=False,
            stage="PLAN",
            attempt=attempt_next,
            repair_context_json=repair_context_json,
            transform_protocol_plan=None,
            transformed_dataset_id=None,
            transformed_spec=None,
            cleaned_dataset_id=cleaned_dataset_id,
            protoctol_spec=protocol,
            cleaned_dataset_summary=cleaned_dataset_summary,
            cleaned_dataset_validation_issues=cleaned_dataset_validation_issues,
            transformation_issues=issues,
            user_message=user_message,
        )
    )


def _last_user_text(messages_history: Optional[Sequence[ChatMessage]]) -> str:
    if not messages_history:
        return ""
    for m in reversed(messages_history):
        if getattr(m, "role", None) == "user":
            return (m.content or "").strip()
    return ""


# =============================================================================
# PLAN generation
# =============================================================================
def get_transform_encoding_plan(
    *,
    llm: LLMService,
    protocol: ProtocolSpec,
    dataset_summary: DatasetSummaryModel,
    repair_context_json: Optional[str],
) -> TransformPlanModel:
    system_prompt = build_transform_plan_system_prompt()
    user_prompt = build_transform_plan_user_prompt_template().format(
        protocol_json=json.dumps(protocol.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        summary_json=json.dumps(dataset_summary.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )

    history: Optional[List[ChatMessage]] = None
    if repair_context_json:
        history = [ChatMessage(role="system", content="REPAIR_CONTEXT=" + repair_context_json)]

    return llm.generate_json(
        schema=TransformPlanModel,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=LLMConfig(temperature=0.4),
        history=history,
        max_attempts=2,
    )


# =============================================================================
# Node
# =============================================================================
@dataclass(frozen=True)
class TransformProtocolNode(Node):
    NAME: ClassVar[str] = TransformProtocolState.NAME

    llm: LLMService
    data_repo: DataRepo
    model_name: str

    profiling_max_categories: int = 50
    profiling_sample_distinct: int = 50

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_transform_protocol_node_info()

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        tool_factory: ToolFactory,
        previous_state_dependencies: Mapping[str, State],
        messages_history: Optional[Sequence[ChatMessage]],
    ) -> State:
        # -----------------------------
        # Deps
        # -----------------------------
        data_profiling_tool = cast(DatasetProfilingStateTool, tool_factory.get_tool("DATA_PROFILING_TOOL"))
        deps = TransformProtocolDeps.from_loaded(previous_state_dependencies)

        clean_state = deps.clean_protocol
        compile_state = deps.compile_protocol
        validate_clean_protocol = deps.validate_cleaned_protocol

        clean_dataset_validation_issues = validate_clean_protocol.payload.issues
        clean_dataset_id = clean_state.payload.clean_dataset_id
        assert clean_dataset_id is not None

        protocol = compile_state.payload.protocol
        assert protocol is not None

        # -----------------------------
        # Load cleaned df
        # -----------------------------
        try:
            df_clean: pd.DataFrame = self.data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=clean_dataset_id,
            )
        except Exception as e:  # noqa: BLE001
            issues = clean_dataset_validation_issues + [
                _fail("Failed to read cleaned dataset.", {"error": str(e), "type": type(e).__name__})
            ]
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    stage="PLAN",
                    attempt=1,
                    repair_context_json=None,
                    transformation_issues=issues,
                    cleaned_dataset_id=clean_dataset_id,
                    user_message="Failed to read the cleaned dataset. Please check previous steps and try again.",
                )
            )

        # -----------------------------
        # Profile (PLAN input)
        # -----------------------------
        try:
            clean_dataset_summary: DatasetSummaryModel = data_profiling_tool.extract_dataset_summary(
                df_clean,
                max_categories=self.profiling_max_categories,
                sample_distinct=self.profiling_sample_distinct,
                compute_quantiles=True,
                strict=True,
            )
        except Exception as e:  # noqa: BLE001
            issues = clean_dataset_validation_issues + [
                _fail("Failed to profile cleaned dataset.", {"error": str(e), "type": type(e).__name__})
            ]
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    stage="PLAN",
                    attempt=1,
                    repair_context_json=None,
                    transformation_issues=issues,
                    cleaned_dataset_id=clean_dataset_id,
                    user_message="Failed to profile the cleaned dataset. Please check previous steps and try again.",
                )
            )

        # -----------------------------
        # Stage routing
        # -----------------------------
        prev_payload: Optional[TransformProtocolPayloadModel] = state.payload if isinstance(state, TransformProtocolState) else None
        stage = prev_payload.stage if prev_payload and prev_payload.stage else "PLAN"
        attempt = prev_payload.attempt if prev_payload and prev_payload.attempt else 1
        repair_context_json = prev_payload.repair_context_json if prev_payload and prev_payload.repair_context_json else None

        # Hard stop
        if attempt > MaxAttempt:
            issues = [_fail("Maximum transformation attempts exceeded.", {"max_attempts": MaxAttempt})]
            msg = get_message_for_hard_validation_issue(self.llm, issues)
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    stage="PLAN",
                    attempt=attempt,
                    repair_context_json=repair_context_json,
                    transformation_issues=issues,
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    protoctol_spec=protocol,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    user_message=msg,
                )
            )

        # =============================
        # STAGE: PLAN
        # =============================
        if stage == "PLAN":
            plan = get_transform_encoding_plan(
                llm=self.llm,
                protocol=protocol,
                dataset_summary=clean_dataset_summary,
                repair_context_json=repair_context_json,
            )

            plan_issues = validate_plan_against_protocol_controls_only(
                plan=plan,
                protocol=protocol,
                df_columns=df_clean.columns.tolist(),
            )

            # planner defects -> auto replan (no user negotiation)
            if plan_issues:
                repair_json = _make_repair_context_json(attempt=attempt, stage="PLAN_SANITY_FAIL", issues=plan_issues)
                msg = _llm_stage_message(
                    llm=self.llm,
                    stage="PLAN",
                    ok=False,
                    payload={"attempt": attempt, "n_issues": len(plan_issues), "auto_action": "replan"},
                )
                return _reset_to_plan(
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    issues=plan_issues,
                    protocol=protocol,
                    attempt_next=attempt + 1,
                    repair_context_json=repair_json,
                    user_message=msg,
                )

            # valid plan -> NEGOTIATE
            negotiate_msg = build_plan_review_message(
                llm=self.llm,
                protocol=protocol,
                plan=plan,
                dataset_summary=clean_dataset_summary,
                messages_history=messages_history,
            )
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    stage="NEGOTIATE",
                    attempt=attempt,
                    needs_user_input=True,
                    repair_context_json=None,
                    transform_protocol_plan=plan,
                    transformed_dataset_id=None,
                    transformed_spec=None,
                    protoctol_spec=protocol,
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    transformation_issues=[],
                    user_message=negotiate_msg,
                )
            )

        # =============================
        # STAGE: NEGOTIATE
        # =============================
        if stage == "NEGOTIATE":
            if not prev_payload:
                # fallback
                issues = [_fail("Missing previous payload in NEGOTIATE; restarting planning.")]
                repair_json = _make_repair_context_json(attempt=attempt, stage="NEGOTIATE_MISSING_PAYLOAD", issues=issues)
                msg = _llm_stage_message(llm=self.llm, stage="NEGOTIATE", ok=False, payload={"attempt": attempt, "auto_action": "replan"})
                return _reset_to_plan(
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    issues=issues,
                    protocol=protocol,
                    attempt_next=attempt + 1,
                    repair_context_json=repair_json,
                    user_message=msg,
                )

            plan = prev_payload.transform_protocol_plan
            if plan is None:
                issues = [_fail("Missing plan in NEGOTIATE stage; restarting planning.")]
                repair_json = _make_repair_context_json(attempt=attempt, stage="NEGOTIATE_MISSING_PLAN", issues=issues)
                msg = _llm_stage_message(llm=self.llm, stage="NEGOTIATE", ok=False, payload={"attempt": attempt, "auto_action": "replan"})
                return _reset_to_plan(
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    issues=issues,
                    protocol=protocol,
                    attempt_next=attempt + 1,
                    repair_context_json=repair_json,
                    user_message=msg,
                )

            last_user = _last_user_text(messages_history)
            if not last_user:
                # no new user input yet; keep state as-is
                return TransformProtocolState(prev_payload)

            prior_issues = list(prev_payload.transformation_issues)

            decision = parse_negotiation_decision(
                llm=self.llm,
                protocol=protocol,
                plan=plan,
                issues=prior_issues,
                user_text=last_user,
            )

            if decision.action == "question":
                reply = answer_negotiation_question(
                    llm=self.llm,
                    protocol=protocol,
                    plan=plan,
                    issues=prior_issues,
                    question_text=decision.feedback or last_user,
                    messages_history=messages_history,
                )
                return TransformProtocolState(
                    TransformProtocolPayloadModel(
                        stage="NEGOTIATE",
                        attempt=attempt,
                        needs_user_input=True,
                        repair_context_json=prev_payload.repair_context_json,
                        transform_protocol_plan=plan,
                        transformed_dataset_id=None,
                        transformed_spec=None,
                        cleaned_dataset_id=clean_dataset_id,
                        protoctol_spec=protocol,
                        cleaned_dataset_summary=clean_dataset_summary,
                        cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                        transformation_issues=prior_issues,
                        user_message=reply,
                    )
                )

            if decision.action == "approve":
                msg = _llm_stage_message(
                    llm=self.llm,
                    stage="NEGOTIATE",
                    ok=True,
                    payload={"attempt": attempt, "decision": "approved", "next_stage": "APPLY"},
                )
                return TransformProtocolState(
                    TransformProtocolPayloadModel(
                        stage="APPLY",
                        attempt=attempt,
                        needs_user_input=False,
                        repair_context_json=None,
                        transform_protocol_plan=plan,
                        transformed_dataset_id=None,
                        transformed_spec=None,
                        protoctol_spec=protocol,
                        cleaned_dataset_id=clean_dataset_id,
                        cleaned_dataset_summary=clean_dataset_summary,
                        cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                        transformation_issues=[],
                        user_message=msg,
                    )
                )

            # change -> replan with user feedback (works both for plan-review and apply-failure negotiation)
            repair_json = _make_repair_context_json(
                attempt=attempt,
                stage="NEGOTIATE_USER_FEEDBACK",
                issues=prior_issues,
                extra={"user_feedback": decision.feedback or last_user},
            )
            msg = _llm_stage_message(
                llm=self.llm,
                stage="NEGOTIATE",
                ok=False,
                payload={"attempt": attempt, "decision": "replan", "next_stage": "PLAN"},
            )
            return _reset_to_plan(
                cleaned_dataset_id=clean_dataset_id,
                cleaned_dataset_summary=clean_dataset_summary,
                cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                issues=prior_issues,
                protocol=protocol,
                attempt_next=attempt + 1,
                repair_context_json=repair_json,
                user_message=msg,
            )

        # =============================
        # STAGE: APPLY
        # =============================
        if stage == "APPLY":
            plan = getattr(prev_payload, "transform_protocol_plan", None) if prev_payload else None
            if plan is None:
                issues = [_fail("Missing transformation plan in APPLY stage; restarting planning.")]
                repair_json = _make_repair_context_json(attempt=attempt, stage="APPLY_MISSING_PLAN", issues=issues)
                msg = _llm_stage_message(llm=self.llm, stage="APPLY", ok=False, payload={"attempt": attempt, "auto_action": "replan"})
                return _reset_to_plan(
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    issues=issues,
                    protocol=protocol,
                    attempt_next=attempt + 1,
                    repair_context_json=repair_json,
                    user_message=msg,
                )

            try:
                df_transformed, apply_issues, raw_to_outputs = apply_plan_with_lineage_or_raise(df=df_clean, plan=plan)
            except Exception as e:  # noqa: BLE001
                fail_issue = _fail(
                    "Failed to apply encoding plan.",
                    evidence={"type": type(e).__name__, "error": str(e)},
                    fix_hint="Describe how you want to adjust encoding for the problematic columns.",
                )
                msg = get_message_for_hard_validation_issue(self.llm, [fail_issue])
                return TransformProtocolState(
                    TransformProtocolPayloadModel(
                        stage="NEGOTIATE",
                        attempt=attempt,
                        needs_user_input=True,
                        repair_context_json=_make_repair_context_json(attempt=attempt, stage="APPLY_EXCEPTION", issues=[fail_issue]),
                        transform_protocol_plan=plan,
                        transformed_dataset_id=None,
                        transformed_spec=None,
                        cleaned_dataset_id=clean_dataset_id,
                        protoctol_spec=protocol,
                        cleaned_dataset_summary=clean_dataset_summary,
                        cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                        transformation_issues=[fail_issue],
                        user_message=msg,
                    )
                )

            apply_fail = [x for x in apply_issues if _is_fail(x)]
            if apply_fail:
                msg = get_message_for_hard_validation_issue(self.llm, apply_fail)
                return TransformProtocolState(
                    TransformProtocolPayloadModel(
                        stage="NEGOTIATE",
                        attempt=attempt,
                        needs_user_input=True,
                        repair_context_json=_make_repair_context_json(attempt=attempt, stage="APPLY_FAIL_ISSUES", issues=apply_fail),
                        transform_protocol_plan=plan,
                        transformed_dataset_id=None,
                        transformed_spec=None,
                        protoctol_spec=protocol,
                        cleaned_dataset_id=clean_dataset_id,
                        cleaned_dataset_summary=clean_dataset_summary,
                        cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                        transformation_issues=apply_fail,
                        user_message=msg,
                    )
                )

            # persist transformed df
            new_transformed_dataset_id = uuid4()
            self.data_repo.save_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=new_transformed_dataset_id,
                df=df_transformed,
            )

            # build updated ProtocolSpec (covariates/effect_modifiers expanded)
            protocol_after = build_protocol_after_from_lineage(
                protocol_before=protocol,
                df_after=df_transformed,
                raw_to_outputs=raw_to_outputs,
            )

            msg = _llm_stage_message(
                llm=self.llm,
                stage="APPLY",
                ok=True,
                payload={
                    "attempt": attempt,
                    "before_shape": [int(df_clean.shape[0]), int(df_clean.shape[1])],
                    "after_shape": [int(df_transformed.shape[0]), int(df_transformed.shape[1])],
                    "next_stage": "VALIDATE",
                },
            )
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    stage="VALIDATE",
                    attempt=attempt,
                    needs_user_input=False,
                    repair_context_json=None,
                    transform_protocol_plan=plan,
                    protoctol_spec=protocol,
                    transformed_dataset_id=new_transformed_dataset_id,
                    transformed_spec=protocol_after,  # ProtocolSpec (updated covariates/effect_modifiers)
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    transformation_issues=apply_issues,  # keep WARNs for visibility
                    user_message=msg,
                )
            )

        # =============================
        # STAGE: VALIDATE (minimal, no negotiation)
        # =============================
        if stage == "VALIDATE":
            plan = getattr(prev_payload, "transform_protocol_plan", None) if prev_payload else None
            transformed_dataset_id = getattr(prev_payload, "transformed_dataset_id", None) if prev_payload else None
            protocol_after = getattr(prev_payload, "transformed_spec", None) if prev_payload else None

            if plan is None or transformed_dataset_id is None or protocol_after is None:
                issues = [_fail("Missing inputs in VALIDATE stage; restarting planning.")]
                repair_json = _make_repair_context_json(attempt=attempt, stage="VALIDATE_MISSING_INPUTS", issues=issues)
                msg = _llm_stage_message(llm=self.llm, stage="VALIDATE", ok=False, payload={"attempt": attempt, "auto_action": "replan"})
                return _reset_to_plan(
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    issues=issues,
                    protocol=protocol,
                    attempt_next=attempt + 1,
                    repair_context_json=repair_json,
                    user_message=msg,
                )

            df_transformed = self.data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=transformed_dataset_id,
            )

            suite_issues = validate_covariates_and_effect_modifiers_numeric_only(
                df_after=df_transformed,
                protocol_after=protocol_after,
            )

            if any(_is_fail(x) for x in suite_issues):
                # deterministic -> auto replan (no negotiation)
                repair_json = _make_repair_context_json(attempt=attempt, stage="VALIDATE_NUMERIC_FAIL", issues=suite_issues)
                msg = _llm_stage_message(
                    llm=self.llm,
                    stage="VALIDATE",
                    ok=False,
                    payload={"attempt": attempt, "n_issues": len(suite_issues), "auto_action": "replan"},
                )
                return _reset_to_plan(
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    issues=suite_issues,
                    protocol=protocol,
                    attempt_next=attempt + 1,
                    repair_context_json=repair_json,
                    user_message=msg,
                )

            msg = _llm_stage_message(
                llm=self.llm,
                stage="VALIDATE",
                ok=True,
                payload={"attempt": attempt, "next_stage": "DONE"},
            )
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    stage="DONE",
                    attempt=attempt,
                    needs_user_input=False,
                    repair_context_json=None,
                    transform_protocol_plan=plan,
                    transformed_dataset_id=transformed_dataset_id,
                    transformed_spec=protocol_after,
                    protoctol_spec=protocol,
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    transformation_issues=(list(prev_payload.transformation_issues) if prev_payload else []) + suite_issues,
                    user_message=msg,
                )
            )

        return state