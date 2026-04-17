from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, ClassVar, Literal, cast
from uuid import UUID

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.validation import (
    ValidationIssueModel,
    ValidationStatus,
)
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.data_validation.data_validation_deps import (
    DataValidationDeps,
)
from python.implementation.workflows.nodes.data_validation.data_validation_prompts import (
    data_validation_node_info,
    data_validation_review_decision_prompt,
    data_validation_review_summary_prompt,
)
from python.implementation.workflows.nodes.data_validation.data_validation_state import (
    DataValidationPayloadModel,
    DataValidationState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.validation.validation_backdoor_tool import (
    ValidationBackdoorTool,
)
from python.implementation.workflows.utils.utils import safe_err

log = get_app_logger(__name__, component="data_validation_node", log_type="node")


class _ReviewSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    assistant_message: str = Field(..., min_length=1)


class _ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["confirm", "revise", "clarify"]
    assistant_message: str = Field(..., min_length=1)


class DataValidationNode(Node):
    NAME: ClassVar[str] = DataValidationState.NAME

    def __init__(
        self,
        *,
        data_repo: DataRepo,
        llm: LLMService,
        tools_factory: ToolFactory,
    ) -> None:
        self._data_repo = data_repo
        self._llm = llm
        self._validation_tool = cast(
            ValidationBackdoorTool,
            tools_factory.get_tool(ValidationBackdoorTool.NAME),
        )

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return data_validation_node_info()

    def run(
        self,
        *,
        request: NodeRequest,
    ) -> NodeExecutionResult:
        if not isinstance(request.node_state, DataValidationState):
            raise TypeError(
                f"{self.name}: expected DataValidationState, got "
                f"{type(request.node_state).__name__}"
            )

        payload = request.node_state.payload.model_copy(deep=True)
        deps = DataValidationDeps.from_request(request)

        if deps.dataset_id is None:
            return self._needs_data_result(
                request=request,
                user_message=(
                    "I need a compiled working dataset before I can run causal validation."
                ),
            )

        if deps.causal_spec is None or deps.transformation_plan is None:
            return self._needs_input_result(
                request=request,
                payload=DataValidationPayloadModel(),
                user_message=(
                    "I need a compiled causal specification and transformation plan "
                    "before I can run validation."
                ),
            )

        payload, sources_changed = self._bind_payload_to_sources(
            payload=payload,
            dataset_id=deps.dataset_id,
            causal_spec=deps.causal_spec,
            transformation_plan=deps.transformation_plan,
        )

        latest_user_message = _latest_user_message(request.read_only_messages_history)

        if payload.phase == "REVIEW_READY":
            if not self._review_payload_complete(payload):
                log.warning(
                    "data validation review payload incomplete; revalidating",
                    conversation_id=str(request.conversation_id),
                    source_dataset_id=str(deps.dataset_id),
                )
            else:
                if latest_user_message is None:
                    return self._needs_input_result(
                        request=request,
                        payload=payload,
                        user_message=payload.assistant_message
                        or "Please confirm the validation result.",
                    )
                return self._handle_review_response(
                    request=request,
                    payload=payload,
                    latest_user_message=latest_user_message,
                )

        if payload.phase == "CONFIRMED" and not sources_changed:
            return self._done_result(
                request=request,
                payload=payload,
                user_message=payload.assistant_message
                or "The validation result is already confirmed.",
            )

        if payload.phase == "FAILED" and not sources_changed:
            return self._aborted_result(
                request=request,
                payload=payload,
                user_message=payload.assistant_message
                or "Validation is blocked and needs upstream revision.",
            )

        try:
            dataframe = self._data_repo.get_csv_data(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                dataset_id=deps.dataset_id,
                limit=None,
            )
        except Exception as exc:
            log.exception(
                "failed to load data validation dataset",
                dataset_id=str(deps.dataset_id),
                error=safe_err(exc),
            )
            return self._needs_data_result(
                request=request,
                user_message=(
                    "I could not load the compiled dataset for validation. Please "
                    "recompile or reselect the dataset and try again."
                ),
            )

        return self._validate(
            request=request,
            payload=payload,
            dataframe=dataframe,
            dataset_id=deps.dataset_id,
            causal_spec=deps.causal_spec,
            transformation_plan=deps.transformation_plan,
            sources_changed=sources_changed,
        )

    def _bind_payload_to_sources(
        self,
        *,
        payload: DataValidationPayloadModel,
        dataset_id: UUID,
        causal_spec: CausalSpec,
        transformation_plan: TransformPlan,
    ) -> tuple[DataValidationPayloadModel, bool]:
        if (
            payload.source_dataset_id == dataset_id
            and _model_json_equal(payload.source_causal_spec, causal_spec)
            and _model_json_equal(payload.source_transformation_plan, transformation_plan)
        ):
            return payload, False

        if (
            payload.source_dataset_id is None
            and payload.source_causal_spec is None
            and payload.source_transformation_plan is None
            and payload.phase == "INIT"
        ):
            return payload.bind_sources(
                dataset_id=dataset_id,
                causal_spec=causal_spec,
                transformation_plan=transformation_plan,
            ), False

        return payload.reset_for_revalidation(
            dataset_id=dataset_id,
            causal_spec=causal_spec,
            transformation_plan=transformation_plan,
        ), True

    def _validate(
        self,
        *,
        request: NodeRequest,
        payload: DataValidationPayloadModel,
        dataframe: pd.DataFrame,
        dataset_id: UUID,
        causal_spec: CausalSpec,
        transformation_plan: TransformPlan,
        sources_changed: bool,
    ) -> NodeExecutionResult:
        try:
            scope_issues = _validate_dataset_protocol_scope_columns(
                dataframe=dataframe,
                causal_spec=causal_spec,
            )
            validation_report = self._validation_tool.validate(
                causal_spec=causal_spec,
                dataframe=dataframe,
                transform_plan=transformation_plan,
            )
        except Exception as exc:
            log.exception("data validation failed unexpectedly", error=safe_err(exc))
            return self._failed_result(
                request=request,
                payload=payload,
                issues=[
                    _fail_issue(
                        message="Validation failed unexpectedly.",
                        evidence={"error": repr(exc)},
                        fix_hint=(
                            "Review the compiled dataset, causal specification, and "
                            "transformation plan for inconsistencies."
                        ),
                    )
                ],
                user_message=(
                    "I could not finish validation for the compiled dataset and accepted "
                    "causal setup. Please review the dataset, causal specification, and "
                    "transformation plan."
                ),
                error_message=f"validation failed unexpectedly: {safe_err(exc)}",
            )

        issues = [*scope_issues, *validation_report.issues]
        validation_status = _validation_status(issues)

        if validation_status == "FAIL":
            return self._failed_result(
                request=request,
                payload=payload,
                issues=issues,
                user_message=_build_blocking_user_message(
                    causal_spec=causal_spec,
                    issues=issues,
                ),
                error_message="blocking validation issues prevent confirmation",
            )

        try:
            review_message = self._build_review_summary_message(
                causal_spec=causal_spec,
                transformation_plan=transformation_plan,
                validation_status=validation_status,
                issues=issues,
                messages_history=request.read_only_messages_history,
            )
        except Exception as exc:
            log.exception("data validation review summary failed", error=safe_err(exc))
            review_message = _build_validation_review_fallback(
                causal_spec=causal_spec,
                issues=issues,
            )

        if sources_changed:
            review_message = (
                "The compiled dataset or accepted causal setup changed, so I reran "
                f"validation. {review_message}"
            )

        review_payload = payload.model_copy(
            update={
                "source_dataset_id": dataset_id,
                "source_causal_spec": causal_spec,
                "source_transformation_plan": transformation_plan,
                "validation_issues": issues,
                "validation_status": validation_status,
                "phase": "REVIEW_READY",
                "assistant_message": review_message,
                "system_message": None,
                "error_message": None,
            }
        )
        return self._needs_input_result(
            request=request,
            payload=review_payload,
            user_message=review_message,
        )

    def _build_review_summary_message(
        self,
        *,
        causal_spec: CausalSpec,
        transformation_plan: TransformPlan,
        validation_status: ValidationStatus,
        issues: Sequence[ValidationIssueModel],
        messages_history: Sequence[ChatMessage] | None,
    ) -> str:
        history = list(messages_history[-4:]) if messages_history else None
        review_summary = self._llm.generate_json(
            schema=_ReviewSummary,
            system_prompt=data_validation_review_summary_prompt(),
            user_prompt=json.dumps(
                {
                    "compiled_causal_spec": causal_spec.model_dump(mode="json"),
                    "transformation_plan": transformation_plan.model_dump(mode="json"),
                    "validation_status": validation_status,
                    "validation_issues": [
                        issue.model_dump(mode="json", exclude_none=True) for issue in issues
                    ],
                },
                ensure_ascii=False,
            ),
            config=LLMConfig(model="basic", temperature=0.2),
            history=history,
            max_attempts=2,
        )
        return review_summary.assistant_message

    def _review_payload_complete(self, payload: DataValidationPayloadModel) -> bool:
        return (
            payload.source_dataset_id is not None
            and payload.source_causal_spec is not None
            and payload.source_transformation_plan is not None
            and payload.validation_status is not None
        )

    def _handle_review_response(
        self,
        *,
        request: NodeRequest,
        payload: DataValidationPayloadModel,
        latest_user_message: str,
    ) -> NodeExecutionResult:
        if not self._review_payload_complete(payload):
            return self._failed_result(
                request=request,
                payload=DataValidationPayloadModel(),
                issues=[
                    _fail_issue(
                        message="Stored validation review state is incomplete.",
                        evidence={},
                        fix_hint=(
                            "Rerun validation from the current compiled dataset and "
                            "accepted causal setup."
                        ),
                    )
                ],
                user_message=(
                    "The stored validation review state is incomplete, so validation needs "
                    "to be rerun from the current dataset and causal setup."
                ),
                error_message="review payload incomplete",
            )

        decision = self._llm.generate_json(
            schema=_ReviewDecision,
            system_prompt=data_validation_review_decision_prompt(),
            user_prompt=json.dumps(
                {
                    "compiled_causal_spec": payload.source_causal_spec.model_dump(
                        mode="json"
                    ),
                    "transformation_plan": payload.source_transformation_plan.model_dump(
                        mode="json"
                    ),
                    "validation_status": payload.validation_status,
                    "validation_issues": [
                        issue.model_dump(mode="json", exclude_none=True)
                        for issue in payload.validation_issues
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
            request.orchestrator_state.set(
                request.node_state.name(),
                {
                    "validation_issues": payload.validation_issues,
                    "is_validated": True,
                },
            )
            confirmed_payload = payload.model_copy(
                update={
                    "phase": "CONFIRMED",
                    "assistant_message": decision.assistant_message,
                    "system_message": None,
                    "error_message": None,
                }
            )
            return self._done_result(
                request=request,
                payload=confirmed_payload,
                user_message=decision.assistant_message,
            )

        if decision.action == "revise":
            failed_payload = payload.model_copy(
                update={
                    "phase": "FAILED",
                    "assistant_message": decision.assistant_message,
                    "system_message": "DATA_VALIDATION_REVISE_REQUESTED",
                    "error_message": "user rejected validation review",
                }
            )
            return self._aborted_result(
                request=request,
                payload=failed_payload,
                user_message=decision.assistant_message,
            )

        review_payload = payload.model_copy(
            update={
                "assistant_message": decision.assistant_message,
                "system_message": None,
                "error_message": None,
            }
        )
        return self._needs_input_result(
            request=request,
            payload=review_payload,
            user_message=decision.assistant_message,
        )

    def _needs_input_result(
        self,
        *,
        request: NodeRequest,
        payload: DataValidationPayloadModel,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=DataValidationState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_INPUT",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _needs_data_result(
        self,
        *,
        request: NodeRequest,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=DataValidationState.init_empty(),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_DATA",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _done_result(
        self,
        *,
        request: NodeRequest,
        payload: DataValidationPayloadModel,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=DataValidationState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="DONE",
            action="NONE",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _aborted_result(
        self,
        *,
        request: NodeRequest,
        payload: DataValidationPayloadModel,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=DataValidationState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="ABORTED",
            action="NONE",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _failed_result(
        self,
        *,
        request: NodeRequest,
        payload: DataValidationPayloadModel,
        issues: list[ValidationIssueModel],
        user_message: str,
        error_message: str,
    ) -> NodeExecutionResult:
        failed_payload = payload.model_copy(
            update={
                "validation_issues": issues,
                "validation_status": _validation_status(issues),
                "phase": "FAILED",
                "assistant_message": user_message,
                "system_message": _build_blocking_system_message(issues),
                "error_message": error_message,
            }
        )
        return self._aborted_result(
            request=request,
            payload=failed_payload,
            user_message=user_message,
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


def _validation_status(issues: Sequence[ValidationIssueModel]) -> ValidationStatus:
    if any(issue.severity == "FAIL" for issue in issues):
        return "FAIL"
    if any(issue.severity == "WARN" for issue in issues):
        return "WARN"
    return "PASS"


def _model_json_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if hasattr(left, "model_dump") and hasattr(right, "model_dump"):
        return left.model_dump(mode="json") == right.model_dump(mode="json")
    return left == right


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
            message="Scoped dataset contains columns outside the confirmed protocol scope.",
            evidence={
                "extra_columns": extra_columns,
                "allowed_columns": sorted(allowed_columns),
            },
            fix_hint=(
                "Keep only treatment, outcome, covariates, and effect modifiers in the "
                "compiled validation dataset."
            ),
        )
    ]


def _build_blocking_system_message(issues: Sequence[ValidationIssueModel]) -> str:
    lines = [
        "DATA_VALIDATION_BLOCKED",
        "Blocking issues were found while validating the compiled dataset and accepted setup.",
    ]
    for issue in issues:
        if issue.severity != "FAIL":
            continue
        lines.append(f"- {issue.severity}: {issue.message}")
    return "\n".join(lines)


def _build_blocking_user_message(
    *,
    causal_spec: CausalSpec,
    issues: Sequence[ValidationIssueModel],
) -> str:
    lines = [
        "Validation found hard errors, so I cannot ask for confirmation.",
        "",
        f"Treatment: {causal_spec.treatment_spec.column}",
        f"Outcome: {causal_spec.outcome_spec.column}",
        "",
        "Hard errors:",
    ]
    for issue in issues:
        if issue.severity != "FAIL":
            continue
        lines.append(f"- {issue.message}")
        if issue.fix_hint:
            lines.append(f"  What to fix: {issue.fix_hint}")
    return "\n".join(lines)


def _build_validation_review_fallback(
    *,
    causal_spec: CausalSpec,
    issues: Sequence[ValidationIssueModel],
) -> str:
    warning_lines: list[str] = []
    for issue in issues:
        if issue.severity != "WARN":
            continue
        warning_lines.append(f"- {issue.message}")
        if issue.fix_hint:
            warning_lines.append(f"  Review point: {issue.fix_hint}")

    lines = [
        "Validation finished without hard failures.",
        "",
        f"Treatment: {causal_spec.treatment_spec.column}",
        f"Outcome: {causal_spec.outcome_spec.column}",
    ]
    if warning_lines:
        lines.extend(
            [
                "",
                "Warnings to review before confirmation:",
                *warning_lines,
            ]
        )
    else:
        lines.extend(
            [
                "",
                "I found no warnings that require discussion before confirmation.",
            ]
        )
    lines.extend(
        [
            "",
            "Reply to confirm this validation result, or tell me exactly what should change.",
        ]
    )
    return "\n".join(lines)
