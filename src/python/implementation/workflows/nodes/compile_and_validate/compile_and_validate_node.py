from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, ClassVar, Literal, cast
from uuid import UUID

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.validation import ValidationIssueModel
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.ochestrator_state import ReadOnlyOchestratorState
from python.domain.workflows.node_state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_deps import (
    CompileAndValidateDeps,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_prompts import (
    get_compile_and_validate_node_info,
    get_compile_causal_spec_prompt,
    get_compile_review_decision_prompt,
    get_compile_review_summary_prompt,
    get_compile_transformation_plan_prompt,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_state import (
    CompileAndValidatePayloadModel,
    CompileAndValidateState,
)
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.encoding.encoding_plan_tool import (
    EncodingPlanTool,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.specs.causal_specs_tool import (
    CausalSpecsTool,
)
from python.implementation.workflows.tools.causal.validation.validation_backdoor_tool import (
    ValidationBackdoorTool,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.utils.utils import safe_err

log = get_logger(__name__)


class _ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["confirm", "revise", "clarify"]
    assistant_message: str = Field(..., min_length=1)


class _ReviewSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    assistant_message: str = Field(..., min_length=1)


class CompileAndValidateNode(Node):
    NAME: ClassVar[str] = CompileAndValidateState.NAME

    def __init__(
        self,
        *,
        llm: LLMService,
        data_repo: DataRepo,
        tool_factory: ToolFactory,
    ) -> None:
        self._llm = llm
        self._data_repo = data_repo
        causal_specs_raw = tool_factory.get_tool(CausalSpecsTool.NAME)
        encoding_plan_raw = tool_factory.get_tool(EncodingPlanTool.NAME)
        validation_raw = tool_factory.get_tool(ValidationBackdoorTool.NAME)
        self._causal_specs_tool = cast(CausalSpecsTool, causal_specs_raw)
        self._encoding_plan_tool = cast(EncodingPlanTool, encoding_plan_raw)
        self._validation_tool = cast(ValidationBackdoorTool, validation_raw)

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_compile_and_validate_node_info()

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        readonly_orchestrator_state: ReadOnlyOchestratorState,
        messages_history: Sequence[ChatMessage] | None,
        state: State,
    ) -> State:
        if not isinstance(state, CompileAndValidateState):
            raise TypeError(
                f"{self.name}: expected CompileAndValidateState, got {type(state).__name__}"
            )

        deps = CompileAndValidateDeps.from_loaded(readonly_orchestrator_state)
        payload = self._bind_payload_to_dataset(
            payload=state.payload.model_copy(deep=True),
            dataset_id=deps.dataset_id,
        )
        latest_user_message = _latest_user_message(messages_history)

        if payload.phase == "REVIEW_READY":
            return self._handle_review_response(
                payload=payload, latest_user_message=latest_user_message
            )

        if payload.phase == "CONFIRMED":
            return CompileAndValidateState(payload)

        return self._compile_and_validate(
            user_id=user_id,
            conversation_id=conversation_id,
            payload=payload,
            dataset_id=deps.dataset_id,
            dataset_summary=deps.dataset_summary,
            protocol_discussion=deps.protocol_discussion,
            messages_history=messages_history,
        )

    @staticmethod
    def _bind_payload_to_dataset(
        *,
        payload: CompileAndValidatePayloadModel,
        dataset_id: UUID,
    ) -> CompileAndValidatePayloadModel:
        if payload.dataset_id == dataset_id:
            return payload

        if payload.dataset_id is None and payload.phase == "INIT":
            return payload.bind_dataset(dataset_id)

        return payload.reset_for_recompile(dataset_id=dataset_id)

    def _compile_and_validate(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        payload: CompileAndValidatePayloadModel,
        dataset_id: UUID,
        dataset_summary: DatasetSummaryModel,
        protocol_discussion: str,
        messages_history: Sequence[ChatMessage] | None,
    ) -> CompileAndValidateState:
        history = list(messages_history[-4:]) if messages_history else None
        context_payload: dict[str, Any] = {
            "protocol_discussion": protocol_discussion,
            "dataset_summary": dataset_summary.model_dump(mode="json"),
        }

        try:
            causal_schema = self._causal_specs_tool.build_backdoor_schema(
                data_summary=dataset_summary,
            )
            causal_spec = self._llm.generate_json(
                schema=causal_schema,
                system_prompt=get_compile_causal_spec_prompt(),
                user_prompt=json.dumps(context_payload, ensure_ascii=False),
                config=LLMConfig(model="pro", temperature=0.1),
                history=history,
                max_attempts=3,
            )
            causal_spec = self._causal_specs_tool.post_validate_backdoor_spec(
                causal_spec=causal_spec,
                data_summary=dataset_summary,
            )
        except Exception as e:
            return self._failed_state(
                payload=payload,
                issues=[
                    _fail_issue(
                        message="Causal specification compilation failed.",
                        evidence={"error": repr(e)},
                        fix_hint=(
                            "Clarify the confirmed protocol so treatment, outcome, study design, "
                            "and baseline features are explicit and grounded in the dataset."
                        ),
                    )
                ],
                assistant_message=(
                    "I could not compile the confirmed protocol into a valid causal specification. "
                    "Please revise the confirmed protocol details before proceeding."
                ),
                error_message=f"causal specification compilation failed: {safe_err(e)}",
            )

        if not causal_spec.covariates and not causal_spec.effect_modifiers:
            return self._failed_state(
                payload=payload,
                issues=[
                    _fail_issue(
                        message="Compiled protocol has no covariates or effect modifiers for baseline adjustment.",
                        evidence={"experiment_type": causal_spec.experiment_type},
                        fix_hint=(
                            "Add baseline covariates or effect modifiers before preparing an "
                            "inference-ready causal specification."
                        ),
                    )
                ],
                assistant_message=(
                    "I compiled the protocol, but it does not contain any baseline adjustment "
                    "features. Please add covariates or effect modifiers before we proceed."
                ),
                error_message="compiled protocol has no adjustment columns",
            )

        try:
            plan_schema = self._encoding_plan_tool.build_encoding_schema(
                data_summary=dataset_summary,
                covariate_columns=causal_spec.covariates,
                effect_modifier_columns=causal_spec.effect_modifiers,
            )
            transform_plan = self._llm.generate_json(
                schema=plan_schema,
                system_prompt=get_compile_transformation_plan_prompt(),
                user_prompt=json.dumps(
                    {
                        **context_payload,
                        "causal_spec": causal_spec.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                ),
                config=LLMConfig(model="pro", temperature=0.1),
                history=history,
                max_attempts=3,
            )
            transform_plan = self._encoding_plan_tool.post_validate_encoding_plan(
                plan=transform_plan,
                data_summary=dataset_summary,
                covariate_columns=causal_spec.covariates,
                effect_modifier_columns=causal_spec.effect_modifiers,
            )
        except Exception as e:
            return self._failed_state(
                payload=payload,
                issues=[
                    _fail_issue(
                        message="Transformation-plan compilation failed.",
                        evidence={"error": repr(e)},
                        fix_hint=(
                            "Revise the protocol roles or preprocessing assumptions so each "
                            "baseline feature has a valid grounded encoding."
                        ),
                    )
                ],
                assistant_message=(
                    "I compiled the clinical protocol, but the preprocessing plan is not valid yet. "
                    "Please revise the protocol or preprocessing assumptions before proceeding."
                ),
                error_message=f"transformation plan compilation failed: {safe_err(e)}",
            )

        try:
            dataframe = self._data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=dataset_id,
                limit=None,
            )
            scope_issues = _validate_dataset_protocol_scope_columns(
                dataframe=dataframe,
                causal_spec=causal_spec,
            )
            validation_report = self._validation_tool.validate(
                causal_spec=causal_spec,
                dataframe=dataframe,
                transform_plan=transform_plan,
            )
            inference_ready = InferenceReadyCausalSpec(
                causal_spec=causal_spec,
                transformation_plan=transform_plan,
                data_summary=dataset_summary,
            )
        except Exception as e:
            return self._failed_state(
                payload=payload,
                issues=[
                    _fail_issue(
                        message="Compilation succeeded but final validation failed unexpectedly.",
                        evidence={"error": repr(e)},
                        fix_hint="Review the compiled protocol, transform plan, and active dataset for inconsistencies.",
                    )
                ],
                assistant_message=(
                    "I compiled the protocol, but the final validation step failed unexpectedly. "
                    "Please review the protocol and dataset assumptions before proceeding."
                ),
                error_message=f"final validation failed unexpectedly: {safe_err(e)}",
            )

        issues = [*scope_issues, *validation_report.issues]
        if any(issue.severity == "FAIL" for issue in issues):
            return self._failed_state(
                payload=payload,
                issues=issues,
                assistant_message=_build_blocking_user_message(
                    causal_spec=causal_spec,
                    issues=issues,
                ),
                error_message="blocking validation issues prevent confirmation",
            )

        assistant_message = self._build_review_summary_message(
            protocol_discussion=protocol_discussion,
            causal_spec=causal_spec,
            transform_plan=transform_plan,
            issues=issues,
            messages_history=messages_history,
        )
        return CompileAndValidateState(
            payload.model_copy(
                update={
                    "compiled_causal_spec": causal_spec,
                    "transformation_plan": transform_plan,
                    "inference_ready_causal_spec": inference_ready,
                    "validation_issues": issues,
                    "phase": "REVIEW_READY",
                    "assistant_message": assistant_message,
                    "system_message": None,
                    "error_message": None,
                }
            )
        )

    def _failed_state(
        self,
        *,
        payload: CompileAndValidatePayloadModel,
        issues: list[ValidationIssueModel],
        assistant_message: str,
        error_message: str,
    ) -> CompileAndValidateState:
        return CompileAndValidateState(
            payload.model_copy(
                update={
                    "compiled_causal_spec": None,
                    "transformation_plan": None,
                    "inference_ready_causal_spec": None,
                    "validation_issues": issues,
                    "phase": "FAILED",
                    "assistant_message": assistant_message,
                    "system_message": _build_blocking_system_message(issues),
                    "error_message": error_message,
                }
            )
        )

    def _handle_review_response(
        self,
        *,
        payload: CompileAndValidatePayloadModel,
        latest_user_message: str | None,
    ) -> CompileAndValidateState:
        if not latest_user_message:
            return CompileAndValidateState(payload)

        decision = self._llm.generate_json(
            schema=_ReviewDecision,
            system_prompt=get_compile_review_decision_prompt(),
            user_prompt=json.dumps(
                {
                    "compiled_causal_spec": (
                        None
                        if payload.compiled_causal_spec is None
                        else payload.compiled_causal_spec.model_dump(mode="json")
                    ),
                    "transformation_plan": (
                        None
                        if payload.transformation_plan is None
                        else payload.transformation_plan.model_dump(mode="json")
                    ),
                    "validation_issues": [
                        issue.model_dump(mode="json") for issue in payload.validation_issues
                    ],
                    "latest_user_message": latest_user_message,
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="basic", temperature=0.0),
            history=None,
            max_attempts=3,
        )

        if decision.action == "confirm":
            return CompileAndValidateState(
                payload.model_copy(
                    update={
                        "phase": "CONFIRMED",
                        "assistant_message": decision.assistant_message,
                        "system_message": None,
                        "error_message": None,
                    }
                )
            )

        if decision.action == "revise":
            return self._failed_state(
                payload=payload,
                issues=payload.validation_issues,
                assistant_message=decision.assistant_message,
                error_message="user rejected the compiled protocol review",
            )

        return CompileAndValidateState(
            payload.model_copy(
                update={
                    "assistant_message": decision.assistant_message,
                    "system_message": None,
                    "error_message": None,
                }
            )
        )

    def _build_review_summary_message(
        self,
        *,
        protocol_discussion: str,
        causal_spec: CausalSpec,
        transform_plan: TransformPlan,
        issues: Sequence[ValidationIssueModel],
        messages_history: Sequence[ChatMessage] | None,
    ) -> str:
        history = list(messages_history[-4:]) if messages_history else None
        context_payload = {
            "protocol_discussion": protocol_discussion,
            "compiled_causal_spec": causal_spec.model_dump(mode="json"),
            "transformation_plan": transform_plan.model_dump(mode="json"),
            "validation_issues": [issue.model_dump(mode="json") for issue in issues],
        }

        try:
            review_summary = self._llm.generate_json(
                schema=_ReviewSummary,
                system_prompt=get_compile_review_summary_prompt(),
                user_prompt=json.dumps(context_payload, ensure_ascii=False),
                config=LLMConfig(model="basic", temperature=0.2),
                history=history,
                max_attempts=2,
            )
            return review_summary.assistant_message
        except Exception as exc:
            log.exception("COMPILE_AND_VALIDATE review summary failed", error=safe_err(exc))
            return _build_review_user_message(
                causal_spec=causal_spec,
                transform_plan=transform_plan,
                issues=issues,
            )


def _latest_user_message(messages_history: Sequence[ChatMessage] | None) -> str | None:
    if not messages_history:
        return None
    for message in reversed(messages_history):
        if message.role != "user":
            continue
        content = message.content.strip()
        if content:
            return content
    return None


def _fail_issue(
    *,
    message: str,
    evidence: dict[str, Any],
    fix_hint: str | None,
) -> ValidationIssueModel:
    return ValidationIssueModel(
        severity="FAIL",
        message=message,
        evidence=evidence,
        fix_hint=fix_hint,
    )


def _validate_dataset_protocol_scope_columns(
    *,
    dataframe: pd.DataFrame,
    causal_spec: CausalSpec,
) -> list[ValidationIssueModel]:
    allowed_columns = {
        str(causal_spec.treatment_spec.column),
        str(causal_spec.outcome_spec.column),
        *(str(column) for column in causal_spec.covariates),
        *(str(column) for column in causal_spec.effect_modifiers),
    }
    extra_columns = sorted(
        str(column) for column in dataframe.columns if str(column) not in allowed_columns
    )
    if not extra_columns:
        return []
    return [
        _fail_issue(
            message="Cleaned dataset contains columns outside the confirmed protocol scope.",
            evidence={
                "extra_columns": extra_columns,
                "allowed_columns": sorted(allowed_columns),
            },
            fix_hint=(
                "Keep only treatment, outcome, covariates, and effect modifiers in the "
                "final cleaned working dataset before confirmation."
            ),
        )
    ]


def _build_blocking_system_message(issues: Sequence[ValidationIssueModel]) -> str:
    lines = [
        "COMPILE_AND_VALIDATE_BLOCKED",
        "Blocking issues were found while compiling or validating the confirmed protocol.",
    ]
    for issue in issues:
        lines.append(f"- {issue.severity}: {issue.message}")
        if (
            issue.message
            == "Cleaned dataset contains columns outside the confirmed protocol scope."
        ):
            extra_columns = issue.evidence.get("extra_columns")
            allowed_columns = issue.evidence.get("allowed_columns")
            if extra_columns:
                lines.append(f"  extra_columns={extra_columns}")
            if allowed_columns:
                lines.append(f"  allowed_columns={allowed_columns}")
    return "\n".join(lines)


def _build_blocking_user_message(
    *,
    causal_spec: CausalSpec,
    issues: Sequence[ValidationIssueModel],
) -> str:
    scope_issue = next(
        (
            issue
            for issue in issues
            if issue.message
            == "Cleaned dataset contains columns outside the confirmed protocol scope."
        ),
        None,
    )
    main_lines = [
        "I compiled the confirmed clinical protocol, but it is not ready for confirmation because there are blocking issues.",
        "",
        f"Treatment: {causal_spec.treatment_spec.column}",
        f"Outcome: {causal_spec.outcome_spec.column}",
        "",
        "Main blocking issues:",
    ]
    for issue in issues[:5]:
        if issue.severity != "FAIL":
            continue
        main_lines.append(f"- {issue.message}")
        if issue is scope_issue:
            extra_columns = issue.evidence.get("extra_columns")
            if extra_columns:
                main_lines.append(
                    f"  Extra columns currently present: {', '.join(map(str, extra_columns))}"
                )
    main_lines.append("")
    if scope_issue is not None:
        main_lines.append(
            "The final cleaned working dataset should contain only treatment, outcome, covariates, and effect modifiers."
        )
        main_lines.append(
            "If any of the extra columns are intentionally still needed, tell me exactly which columns they are and whether they should remain in the protocol scope."
        )
        main_lines.append("Otherwise, remove those extra columns and then rerun this step.")
    else:
        main_lines.append(
            "Please revise the protocol or the cleaned dataset assumptions before we continue."
        )
    return "\n".join(main_lines)


def _build_review_user_message(
    *,
    causal_spec: CausalSpec,
    transform_plan: TransformPlan,
    issues: Sequence[ValidationIssueModel],
) -> str:
    treatment = causal_spec.treatment_spec
    outcome = causal_spec.outcome_spec
    plan_lines = [
        f"- {column.column}: {column.encoding.preset}" for column in transform_plan.columns
    ]
    warning_lines = _build_warning_review_lines(issues)
    lines = [
        "I compiled the confirmed protocol into a candidate causal specification and baseline transformation plan.",
        "",
        (
            f"The current treatment definition is {treatment.column} "
            f"({treatment.control} vs {treatment.treated})."
        ),
        f"The outcome is {outcome.column} ({outcome.kind}).",
        (
            "Baseline covariates: "
            f"{', '.join(causal_spec.covariates) if causal_spec.covariates else 'None'}."
        ),
        (
            "Effect modifiers: "
            f"{', '.join(causal_spec.effect_modifiers) if causal_spec.effect_modifiers else 'None'}."
        ),
        "",
        "Planned baseline transformations:",
        *plan_lines,
    ]
    if warning_lines:
        lines.extend(
            [
                "",
                "Points to review before confirmation:",
                *warning_lines,
            ]
        )
    else:
        lines.extend(
            [
                "",
                "I found no blocking validation issues and no additional warnings that need discussion before confirmation.",
            ]
        )
    lines.extend(
        [
            "",
            "If this matches your clinical intent, please confirm this compiled setup. "
            "If not, tell me exactly what should change in the treatment, outcome, covariates, "
            "effect modifiers, or planned encodings.",
        ]
    )
    return "\n".join(lines)


def _build_warning_review_lines(issues: Sequence[ValidationIssueModel]) -> list[str]:
    grouped: dict[tuple[str, str | None], list[str]] = {}
    ordered_keys: list[tuple[str, str | None]] = []

    for issue in issues:
        if issue.severity != "WARN":
            continue
        key = (issue.message, issue.fix_hint)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        column_name = issue.evidence.get("column")
        if isinstance(column_name, str) and column_name not in grouped[key]:
            grouped[key].append(column_name)

    lines: list[str] = []
    for message, fix_hint in ordered_keys:
        columns = grouped[(message, fix_hint)]
        if columns:
            lines.append(f"- {message} Columns: {', '.join(columns)}.")
        else:
            lines.append(f"- {message}")
        if fix_hint:
            lines.append(f"  Review point: {fix_hint}")
    return lines


__all__ = ["CompileAndValidateNode"]
