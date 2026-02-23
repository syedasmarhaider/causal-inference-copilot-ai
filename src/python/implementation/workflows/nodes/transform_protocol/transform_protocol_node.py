from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any, ClassVar, List, Optional, Sequence, Tuple, cast
from uuid import UUID, uuid4

import pandas as pd

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
    build_transformed_protocol_user_prompt_template,
    get_transform_protocol_node_info,
)
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_specs import TransformedProtocolSpec
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_state import (
    TransformProtocolPayloadModel,
    TransformProtocolState,
    TransformStage,
)
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_validation import (
    validate_binary_and_one_hot_invariants,
    validate_constant_or_near_constant_controls,
    validate_dimensionality_caps,
    validate_encoding_postconditions,
    validate_id_like_features_in_controls,
    validate_input_columns_exist_and_are_unambiguous,
    validate_model_inputs_are_numeric_dtypes,
    validate_treatment_outcome_domains_by_kind,
)
from python.implementation.workflows.tools.data.data_profiling_tool import (
    DatasetProfilingStateTool,
    DatasetSummaryModel,
)
from python.implementation.workflows.utils.validation import ValidationIssueModel



# =============================================================================
# Policy
# =============================================================================
@dataclass(frozen=True)
class TransformValidationPolicy:
    allow_bool_inputs: bool = True
    binary_allowed_values: tuple[int, int] = (0, 1)
    min_variance: float = 1e-12
    duration_min_value: float = 0.0

    value_tol: float = 1e-9
    check_one_hot_group_row_sums: bool = True
    allow_zero_sum_rows: bool = True

    uniqueness_ratio_threshold: float = 0.98
    max_allowed_id_like: int = 0

    controls_min_variance: float = 1e-12
    max_constant_allowed: Optional[int] = None
    skip_binary_one_hot_in_constant_check: bool = True

    max_total_features: Optional[int] = 5000
    max_w_features: Optional[int] = None
    max_x_features: Optional[int] = None
    max_features_per_source_raw: Optional[int] = None

    minmax_tol: float = 1e-6
    zscore_warn_abs: float = 10.0
    zscore_fail_abs: Optional[float] = None


def _is_fail(issue: ValidationIssueModel) -> bool:
    return issue.severity == "FAIL"


def _fail(message: str, evidence: Optional[dict[str, Any]] = None, fix_hint: Optional[str] = None) -> ValidationIssueModel:
    return ValidationIssueModel(severity="FAIL", message=message, evidence=evidence or {}, fix_hint=fix_hint)

# =============================================================================
# LLM stage messages (separate system prompts as requested)
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
        "Write a short, clear status update for the user.\n"
        "Rules:\n"
        "- Start with a one-line headline.\n"
        "- Then 3-6 bullet points.\n"
        "- Be concrete (counts, column names) when present.\n"
        "- If failure, say what failed and what will happen next.\n"
        "- Do NOT mention internal class names.\n"
    )
    user = json.dumps(
        {
            "stage": stage,
            "ok": ok,
            "context": payload,
        },
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


def get_message_for_hard_validation_issue(llm: LLMService, issue: List[ValidationIssueModel]) -> str:
    return llm.generate(
        system_prompt="You are an assistant for generating user-friendly error messages.",
        user_prompt=build_hard_validation_system_prompt().format(
            validation_issues_json=json.dumps(
                [i.model_dump(mode="json") for i in issue],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        config=LLMConfig(temperature=1.0),
        history=None,
    ).content


# =============================================================================
# PLAN / APPLY / VALIDATE helpers
# =============================================================================
def get_transform_encoding_plan(
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


def protocol_spec_to_transformed_spec(
    llm: LLMService,
    protocol: ProtocolSpec,
    transformed_df: pd.DataFrame,
) -> TransformedProtocolSpec:
    user_prompt = build_transformed_protocol_user_prompt_template().format(
        protocol_json=json.dumps(protocol.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        df_after_columns_json=json.dumps(transformed_df.columns.tolist(), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )
    return llm.generate_json(
        schema=TransformedProtocolSpec,
        system_prompt=get_transform_protocol_node_info(),
        user_prompt=user_prompt,
        config=LLMConfig(temperature=0.2),
        history=None,
        max_attempts=2,
    )


def apply_plan_or_raise(
    *,
    df: pd.DataFrame,
    plan: TransformPlanModel,
) -> Tuple[pd.DataFrame, List[ValidationIssueModel]]:
    """
    APPLY stage:
    - apply_encoding_plan may throw exceptions for real apply failures
    - we still collect non-fatal issues (WARN) if returned
    """
    cur: Optional[pd.DataFrame] = df
    all_issues: List[ValidationIssueModel] = []

    for _, decision in enumerate(plan.columns, start=1):

        # contract: may raise
        cur, step_issues = apply_encoding_plan(df=cur, plan=decision)
        if step_issues:
             all_issues.extend([step_issues])
             
    assert cur is not None
    return cur, all_issues


def run_transform_validations(
    *,
    df_after: pd.DataFrame,
    spec: TransformedProtocolSpec,
    policy: Optional[TransformValidationPolicy] = None,
    fail_fast: bool = False,
) -> List[ValidationIssueModel]:
    pol = policy or TransformValidationPolicy()
    issues: List[ValidationIssueModel] = []

    class _StopValidation(Exception):
        pass

    def _extend(new_issues: List[ValidationIssueModel]) -> None:
        nonlocal issues
        if not new_issues:
            return
        issues.extend(new_issues)
        if fail_fast and any(_is_fail(x) for x in new_issues):
            raise _StopValidation()

    try:
        _extend(validate_input_columns_exist_and_are_unambiguous(df_after=df_after, spec=spec))
        _extend(validate_model_inputs_are_numeric_dtypes(df_after=df_after, spec=spec, allow_bool=pol.allow_bool_inputs))
        _extend(
            validate_treatment_outcome_domains_by_kind(
                df_after=df_after,
                spec=spec,
                binary_allowed_values=pol.binary_allowed_values,
                min_variance=pol.min_variance,
                duration_min_value=pol.duration_min_value,
            )
        )
        _extend(
            validate_binary_and_one_hot_invariants(
                df_after=df_after,
                spec=spec,
                value_tol=pol.value_tol,
                check_one_hot_group_row_sums=pol.check_one_hot_group_row_sums,
                allow_zero_sum_rows=pol.allow_zero_sum_rows,
            )
        )
        _extend(
            validate_id_like_features_in_controls(
                df_after=df_after,
                spec=spec,
                uniqueness_ratio_threshold=pol.uniqueness_ratio_threshold,
                max_allowed_id_like=pol.max_allowed_id_like,
            )
        )
        _extend(
            validate_constant_or_near_constant_controls(
                df_after=df_after,
                spec=spec,
                min_variance=pol.controls_min_variance,
                max_constant_allowed=pol.max_constant_allowed,
                skip_binary_one_hot=pol.skip_binary_one_hot_in_constant_check,
            )
        )
        _extend(
            validate_dimensionality_caps(
                df_after=df_after,
                spec=spec,
                max_total_features=pol.max_total_features,
                max_w_features=pol.max_w_features,
                max_x_features=pol.max_x_features,
                max_features_per_source_raw=pol.max_features_per_source_raw,
            )
        )
        _extend(
            validate_encoding_postconditions(
                df_after=df_after,
                spec=spec,
                minmax_tol=pol.minmax_tol,
                zscore_warn_abs=pol.zscore_warn_abs,
                zscore_fail_abs=pol.zscore_fail_abs,
            )
        )
    except _StopValidation:
        pass

    return issues


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
    prev: TransformProtocolPayloadModel,
    cleaned_dataset_id: UUID,
    cleaned_dataset_summary: DatasetSummaryModel,
    cleaned_dataset_validation_issues: List[ValidationIssueModel],
    issues: List[ValidationIssueModel],
    attempt_next: int,
    repair_context_json: str,
    user_message: str,
) -> TransformProtocolState:
    # Set plan/apply/spec null + stage PLAN
    return TransformProtocolState(
        TransformProtocolPayloadModel(
            stage="PLAN",
            attempt=attempt_next,
            repair_context_json=repair_context_json,
            transform_protocol_plan=None,
            transformed_dataset_id=None,
            transformed_spec=None,
            cleaned_dataset_id=cleaned_dataset_id,
            cleaned_dataset_summary=cleaned_dataset_summary,
            cleaned_dataset_validation_issues=cleaned_dataset_validation_issues,
            transformation_issues=issues,
            user_message=user_message,
        )
    )


# =============================================================================
# Node (single stage per run)
# =============================================================================
@dataclass(frozen=True)
class TransformProtocolNode(Node):
    NAME: ClassVar[str] = TransformProtocolState.NAME

    llm: LLMService
    data_repo: DataRepo
    model_name: str
    max_attempts: int = 5

    profiling_max_categories: int = 50
    profiling_sample_distinct: int = 50

    fail_fast_apply: bool = False  # not used; apply errors are exceptions
    fail_fast_validate: bool = False
    validation_policy: Optional[TransformValidationPolicy] = None

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
        # Load deps / protocol / cleaned id
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
        # Read cleaned df (needed for all stages)
        # -----------------------------
        try:
            df_clean: pd.DataFrame = self.data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=clean_dataset_id,
            )
        except Exception as e:  # noqa: BLE001
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    stage="PLAN",
                    attempt=1,
                    repair_context_json=None,
                    transformation_issues=clean_dataset_validation_issues + [_fail(f"Failed to load cleaned dataset: {e}")],
                    user_message="Failed to read the cleaned dataset. Please check the previous steps and try again.",
                )
            )

        # -----------------------------
        # Profile (only truly needed for PLAN; but cheap enough / or cache in payload if you want later)
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
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    stage="PLAN",
                    attempt=1,
                    repair_context_json=None,
                    transformation_issues=clean_dataset_validation_issues + [_fail(f"Failed to profile cleaned dataset: {e}")],
                    user_message="Failed to profile the cleaned dataset. Please check the previous steps and try again.",
                )
            )

        # -----------------------------
        # Determine current stage from incoming state (if any)
        # -----------------------------
        prev_payload: Optional[TransformProtocolPayloadModel] = None
        if isinstance(state, TransformProtocolState):
            prev_payload = state.payload

        stage: TransformStage = cast(TransformStage, getattr(prev_payload, "stage", "PLAN"))
        attempt: int = int(getattr(prev_payload, "attempt", 1))
        repair_context_json: Optional[str] = getattr(prev_payload, "repair_context_json", None)

        # hard stop on attempts
        if attempt > self.max_attempts:
            issues = [_fail("Maximum transformation attempts exceeded.", {"max_attempts": self.max_attempts})]
            msg = get_message_for_hard_validation_issue(self.llm, issues)
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    stage="PLAN",
                    attempt=attempt,
                    repair_context_json=repair_context_json,
                    transformation_issues=issues,
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

            plan_issues = validate_plan_against_df_columns(
                plan=plan,
                df_columns=df_clean.columns.tolist(),
                require_full_coverage=True,
            )

            if plan_issues:
                repair_json = _make_repair_context_json(attempt=attempt, stage="PLAN_VALIDATION", issues=plan_issues)
                msg = _llm_stage_message(
                    llm=self.llm,
                    stage="PLAN",
                    ok=False,
                    payload={"attempt": attempt, "issues": [x.model_dump(mode="json") for x in plan_issues]},
                )
                return TransformProtocolState(
                    TransformProtocolPayloadModel(
                        stage="PLAN",
                        attempt=attempt + 1,
                        repair_context_json=repair_json,
                        transform_protocol_plan=None,
                        transformed_dataset_id=None,
                        transformed_spec=None,
                        cleaned_dataset_id=clean_dataset_id,
                        cleaned_dataset_summary=clean_dataset_summary,
                        cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                        transformation_issues=plan_issues,
                        user_message=msg,
                    )
                )

            msg = _llm_stage_message(
                llm=self.llm,
                stage="PLAN",
                ok=True,
                payload={
                    "attempt": attempt,
                    "n_plan_columns": len(plan.columns),
                    "plan_preview": [c.model_dump(mode="json") for c in plan.columns[:10]],
                    "next_stage": "APPLY",
                },
            )
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    stage="APPLY",
                    attempt=attempt,
                    repair_context_json=None,  # clear repair context on success
                    transform_protocol_plan=plan,
                    transformed_dataset_id=None,
                    transformed_spec=None,
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    transformation_issues=[],
                    user_message=msg,
                )
            )

        # =============================
        # STAGE: APPLY
        # =============================
        if stage == "APPLY":
            plan = getattr(prev_payload, "transform_protocol_plan", None) if prev_payload else None
            if plan is None:
                issues = [_fail("Missing transformation plan in APPLY stage; restarting planning.")]
                repair_json = _make_repair_context_json(attempt=attempt, stage="APPLY_MISSING_PLAN", issues=issues)
                msg = _llm_stage_message(llm=self.llm, stage="APPLY", ok=False, payload={"attempt": attempt, "error": "missing_plan"})
                return _reset_to_plan(
                    prev=prev_payload or TransformProtocolPayloadModel(),
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    issues=issues,
                    attempt_next=attempt + 1,
                    repair_context_json=repair_json,
                    user_message=msg,
                )

            try:
                df_transformed, apply_issues = apply_plan_or_raise(df=df_clean, plan=plan)
            except Exception as e:  # noqa: BLE001
                issues = [_fail("Failed to apply encoding plan (exception). Restarting from planning.", {"error": str(e), "type": type(e).__name__})]
                repair_json = _make_repair_context_json(
                    attempt=attempt,
                    stage="APPLY_EXCEPTION",
                    issues=issues,
                    extra={"exception_type": type(e).__name__, "exception": str(e)},
                )
                msg = _llm_stage_message(
                    llm=self.llm,
                    stage="APPLY",
                    ok=False,
                    payload={"attempt": attempt, "exception_type": type(e).__name__, "exception": str(e), "next_stage": "PLAN"},
                )
                return _reset_to_plan(
                    prev=prev_payload or TransformProtocolPayloadModel(),
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    issues=issues,
                    attempt_next=attempt + 1,
                    repair_context_json=repair_json,
                    user_message=msg,
                )

            # Persist transformed df
            new_transformed_dataset_id = uuid4()
            self.data_repo.save_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=new_transformed_dataset_id,
                df=df_transformed,
            )

            msg = _llm_stage_message(
                llm=self.llm,
                stage="APPLY",
                ok=True,
                payload={
                    "attempt": attempt,
                    "before_shape": [int(df_clean.shape[0]), int(df_clean.shape[1])],
                    "after_shape": [int(df_transformed.shape[0]), int(df_transformed.shape[1])],
                    "n_apply_issues": len(apply_issues),
                    "apply_issues_sample": [x.model_dump(mode="json") for x in apply_issues[:10]],
                    "next_stage": "VALIDATE",
                },
            )
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    stage="VALIDATE",
                    attempt=attempt,
                    repair_context_json=None,
                    transform_protocol_plan=plan,
                    transformed_dataset_id=new_transformed_dataset_id,
                    transformed_spec=None,
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    transformation_issues=apply_issues,  # keep warnings for visibility
                    user_message=msg,
                )
            )

        # =============================
        # STAGE: VALIDATE
        # =============================
        if stage == "VALIDATE":
            plan = getattr(prev_payload, "transform_protocol_plan", None) if prev_payload else None
            transformed_dataset_id = getattr(prev_payload, "transformed_dataset_id", None) if prev_payload else None

            if plan is None or transformed_dataset_id is None:
                issues = [_fail("Missing plan or transformed dataset in VALIDATE stage; restarting planning.")]
                repair_json = _make_repair_context_json(attempt=attempt, stage="VALIDATE_MISSING_INPUTS", issues=issues)
                msg = _llm_stage_message(llm=self.llm, stage="VALIDATE", ok=False, payload={"attempt": attempt, "error": "missing_inputs"})
                return _reset_to_plan(
                    prev=prev_payload or TransformProtocolPayloadModel(),
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    issues=issues,
                    attempt_next=attempt + 1,
                    repair_context_json=repair_json,
                    user_message=msg,
                )

            df_transformed = self.data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=transformed_dataset_id,
            )

            transformed_spec = protocol_spec_to_transformed_spec(
                llm=self.llm,
                protocol=protocol,
                transformed_df=df_transformed,
            )

            suite_issues = run_transform_validations(
                df_after=df_transformed,
                spec=transformed_spec,
                policy=self.validation_policy,
                fail_fast=self.fail_fast_validate,
            )

            if any(_is_fail(x) for x in suite_issues):
                repair_json = _make_repair_context_json(attempt=attempt, stage="VALIDATION_FAIL", issues=suite_issues)
                msg = _llm_stage_message(
                    llm=self.llm,
                    stage="VALIDATE",
                    ok=False,
                    payload={"attempt": attempt, "n_issues": len(suite_issues), "issues": [x.model_dump(mode="json") for x in suite_issues[:15]]},
                )
                return _reset_to_plan(
                    prev=prev_payload or TransformProtocolPayloadModel(),
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    issues=suite_issues,
                    attempt_next=attempt + 1,
                    repair_context_json=repair_json,
                    user_message=msg,
                )

            msg = _llm_stage_message(
                llm=self.llm,
                stage="VALIDATE",
                ok=True,
                payload={
                    "attempt": attempt,
                    "shape": [int(df_transformed.shape[0]), int(df_transformed.shape[1])],
                    "n_warnings": sum(1 for x in suite_issues if x.severity == "WARN"),
                    "warnings_sample": [x.model_dump(mode="json") for x in suite_issues[:10]],
                    "next_stage": "DONE",
                },
            )
            return TransformProtocolState(
                TransformProtocolPayloadModel(
                    stage="DONE",
                    attempt=attempt,
                    repair_context_json=None,
                    transform_protocol_plan=plan,
                    transformed_dataset_id=transformed_dataset_id,
                    transformed_spec=transformed_spec,
                    cleaned_dataset_id=clean_dataset_id,
                    cleaned_dataset_summary=clean_dataset_summary,
                    cleaned_dataset_validation_issues=clean_dataset_validation_issues,
                    transformation_issues=suite_issues,
                    user_message=msg,
                )
            )

        # =============================
        # STAGE: DONE
        # =============================
        return state