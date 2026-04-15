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
from python.implementation.workflows.nodes.compile_and_validate.validate_deps import (
    ValidateDeps,
)
from python.implementation.workflows.nodes.compile_and_validate.validate_prompts import (
    get_validate_node_info,
    get_validate_review_decision_prompt,
)
from python.implementation.workflows.nodes.compile_and_validate.validate_state import (
    ValidatePayloadModel,
    ValidateState,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.validation.validation_backdoor_tool import (
    ValidationBackdoorTool,
)
from python.implementation.workflows.utils.utils import safe_err

log = get_logger(__name__)


class _ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["confirm", "revise", "clarify"]
    assistant_message: str = Field(..., min_length=1)


class ValidateNode(Node):
    NAME: ClassVar[str] = "VALIDATE"

    def __init__(
        self,
        *,
        llm: LLMService,
        data_repo: DataRepo,
        tool_factory: ToolFactory,
    ) -> None:
        self._llm = llm
        self._data_repo = data_repo
        validation_raw = tool_factory.get_tool(ValidationBackdoorTool.NAME)
        self._validation_tool = cast(ValidationBackdoorTool, validation_raw)

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_validate_node_info()

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        readonly_orchestrator_state: ReadOnlyOchestratorState,
        messages_history: Sequence[ChatMessage] | None,
        state: State,
    ) -> State:
        if not isinstance(state, ValidateState):
            raise TypeError(
                f"{self.name}: expected ValidateState, got {type(state).__name__}"
            )

        deps = ValidateDeps.from_loaded(readonly_orchestrator_state)
        payload = self._bind_payload_to_dataset(
            payload=state.payload.model_copy(deep=True),
            dataset_id=deps.dataset_id,
        )
        latest_user_message = _latest_user_message(messages_history)

        if payload.phase == "REVIEW_READY":
            return self._handle_review_response(
                payload=payload,
                latest_user_message=latest_user_message,
                deps=deps,
            )

        if payload.phase == "CONFIRMED":
            return ValidateState(payload)

        return self._validate(
            user_id=user_id,
            conversation_id=conversation_id,
            payload=payload,
            deps=deps,
        )

    @staticmethod
    def _bind_payload_to_dataset(
        *,
        payload: ValidatePayloadModel,
        dataset_id: UUID,
    ) -> ValidatePayloadModel:
        if payload.dataset_id == dataset_id:
            return payload

        if payload.dataset_id is None and payload.phase == "INIT":
            return payload.bind_dataset(dataset_id)

        return payload.reset_for_recompile(dataset_id=dataset_id)

    def _validate(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        payload: ValidatePayloadModel,
        deps: ValidateDeps,
    ) -> ValidateState:
        try:
            dataframe = self._data_repo.get_csv_data(
                user_id=user_id,
                conversation_id=conversation_id,
                dataset_id=deps.dataset_id,
                limit=None,
            )
            scope_issues = _validate_dataset_protocol_scope_columns(
                dataframe=dataframe,
                causal_spec=deps.causal_spec,
            )
            validation_report = self._validation_tool.validate(
                causal_spec=deps.causal_spec,
                dataframe=dataframe,
                transform_plan=deps.data_transformation_plan,
            )
        except Exception as exc:
            return self._failed_state(
                payload=payload,
                issues=[
                    _fail_issue(
                        message="Validation failed unexpectedly.",
                        evidence={"error": repr(exc)},
                        fix_hint=(
                            "Review the scoped dataset, the causal specification, and the "
                            "transformation plan for inconsistencies."
                        ),
                    )
                ],
                assistant_message=(
                    "I could not finish validation for the accepted dataset and compiled setup. "
                    "Please review the dataset, causal specification, and transformation plan."
                ),
                error_message=f"validation failed unexpectedly: {safe_err(exc)}",
            )

        issues = [*scope_issues, *validation_report.issues]
        if any(issue.severity == "FAIL" for issue in issues):
            return self._failed_state(
                payload=payload,
                issues=issues,
                assistant_message=_build_blocking_user_message(
                    causal_spec=deps.causal_spec,
                    issues=issues,
                ),
                error_message="blocking validation issues prevent confirmation",
            )

        assistant_message = _build_validation_review_message(
            causal_spec=deps.causal_spec,
            issues=issues,
        )
        return ValidateState(
            payload.model_copy(
                update={
                    "dataset_id": deps.dataset_id,
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
        payload: ValidatePayloadModel,
        issues: list[ValidationIssueModel],
        assistant_message: str,
        error_message: str,
    ) -> ValidateState:
        return ValidateState(
            payload.model_copy(
                update={
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
        payload: ValidatePayloadModel,
        latest_user_message: str | None,
        deps: ValidateDeps,
    ) -> ValidateState:
        if not latest_user_message:
            return ValidateState(payload)

        decision = self._llm.generate_json(
            schema=_ReviewDecision,
            system_prompt=get_validate_review_decision_prompt(),
            user_prompt=json.dumps(
                {
                    "compiled_causal_spec": deps.causal_spec.model_dump(mode="json"),
                    "transformation_plan": deps.data_transformation_plan.model_dump(
                        mode="json"
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
            return ValidateState(
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
                error_message="user rejected validation review",
            )

        return ValidateState(
            payload.model_copy(
                update={
                    "assistant_message": decision.assistant_message,
                    "system_message": None,
                    "error_message": None,
                }
            )
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
            message="Scoped dataset contains columns outside the confirmed protocol scope.",
            evidence={
                "extra_columns": extra_columns,
                "allowed_columns": sorted(allowed_columns),
            },
            fix_hint=(
                "Keep only treatment, outcome, covariates, and effect modifiers in the "
                "scoped validation dataset."
            ),
        )
    ]


def _build_blocking_system_message(issues: Sequence[ValidationIssueModel]) -> str:
    lines = [
        "VALIDATE_BLOCKED",
        "Blocking issues were found while validating the accepted dataset and compiled setup.",
    ]
    for issue in issues:
        lines.append(f"- {issue.severity}: {issue.message}")
    return "\n".join(lines)


def _build_blocking_user_message(
    *,
    causal_spec: CausalSpec,
    issues: Sequence[ValidationIssueModel],
) -> str:
    lines = [
        "Validation found blocking issues, so the dataset cannot be frozen yet.",
        "",
        f"Treatment: {causal_spec.treatment_spec.column}",
        f"Outcome: {causal_spec.outcome_spec.column}",
        "",
        "Blocking issues:",
    ]
    for issue in issues:
        if issue.severity != "FAIL":
            continue
        lines.append(f"- {issue.message}")
    lines.append("")
    lines.append("Please revise the dataset, compiled causal specification, or transformation plan.")
    return "\n".join(lines)


def _build_validation_review_message(
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
                "Warnings to review before final confirmation:",
                *warning_lines,
            ]
        )
    else:
        lines.extend(
            [
                "",
                "I found no warnings that require discussion before final confirmation.",
            ]
        )
    lines.extend(
        [
            "",
            "Reply to confirm this validation result, or tell me exactly what should change.",
        ]
    )
    return "\n".join(lines)


CompileAndValidateNode = ValidateNode


__all__ = ["CompileAndValidateNode", "ValidateNode"]
