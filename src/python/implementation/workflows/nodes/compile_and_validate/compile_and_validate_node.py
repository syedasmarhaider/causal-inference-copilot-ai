from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast
from uuid import UUID

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_deps import (
    CompileAndValidateDeps,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_prompts import (
    get_compile_and_validate_node_info,
    get_compile_causal_spec_prompt,
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
from python.implementation.workflows.utils.utils import safe_err
from python.implementation.workflows.utils.validation import ValidationIssueModel

log = get_logger(__name__)

_AFFIRM_RE = re.compile(
    r"\b(yes|yep|yeah|confirm|confirmed|approve|approved|accept|accepted|proceed|looks good|go ahead)\b",
    re.IGNORECASE,
)
_REJECT_RE = re.compile(
    r"\b(no|reject|rejected|wrong|change|changes|revise|revise it|modify|fix|not correct|do not|don't)\b",
    re.IGNORECASE,
)


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
        previous_state_dependencies: Mapping[str, Any],
        messages_history: Sequence[ChatMessage] | None,
        state: State,
    ) -> State:
        if not isinstance(state, CompileAndValidateState):
            raise TypeError(f"{self.name}: expected CompileAndValidateState, got {type(state).__name__}")

        deps = CompileAndValidateDeps.from_loaded(previous_state_dependencies)
        payload = self._bind_payload(state=state, deps=deps)
        latest_user_message = _latest_user_message(messages_history)

        if payload.phase == "REVIEW_READY":
            return self._handle_review_response(payload=payload, latest_user_message=latest_user_message)

        if payload.phase == "CONFIRMED":
            return CompileAndValidateState(payload)

        return self._compile_and_validate(
            user_id=user_id,
            conversation_id=conversation_id,
            payload=payload,
            messages_history=messages_history,
        )

    def _bind_payload(
        self,
        *,
        state: CompileAndValidateState,
        deps: CompileAndValidateDeps,
    ) -> CompileAndValidatePayloadModel:
        payload = state.payload.model_copy(deep=True)
        dataset_changed = payload.dataset_id is not None and payload.dataset_id != deps.dataset_id
        discussion_changed = (
            bool(payload.protocol_discussion.strip())
            and payload.protocol_discussion.strip() != deps.protocol_discussion.strip()
        )
        should_reset = dataset_changed or discussion_changed or payload.phase == "INIT"

        updates: dict[str, Any] = {
            "dataset_id": deps.dataset_id,
            "dataset_summary": deps.dataset_summary,
            "protocol_discussion": deps.protocol_discussion,
        }
        if should_reset:
            updates.update(
                {
                    "compiled_causal_spec": None,
                    "transformation_plan": None,
                    "inference_ready_causal_spec": None,
                    "validation_issues": [],
                    "phase": "INIT",
                    "assistant_message": None,
                    "system_message": None,
                    "error_message": None,
                }
            )
        return payload.model_copy(update=updates)

    def _compile_and_validate(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        payload: CompileAndValidatePayloadModel,
        messages_history: Sequence[ChatMessage] | None,
    ) -> CompileAndValidateState:
        history = list(messages_history[-4:]) if messages_history else None
        context_payload = {
            "protocol_discussion": payload.protocol_discussion,
            "dataset_summary": payload.dataset_summary.model_dump(mode="json")
            if payload.dataset_summary is not None
            else None,
        }

        try:
            causal_schema = self._causal_specs_tool.build_backdoor_schema(
                data_summary=payload.dataset_summary,
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
                data_summary=payload.dataset_summary,
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
                data_summary=payload.dataset_summary,
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
                data_summary=payload.dataset_summary,
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
                dataset_id=payload.dataset_id,
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
                data_summary=payload.dataset_summary,
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

        assistant_message = _build_review_user_message(
            causal_spec=causal_spec,
            transform_plan=transform_plan,
            issues=issues,
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

        if _is_affirmative(latest_user_message):
            return CompileAndValidateState(
                payload.model_copy(
                    update={
                        "phase": "CONFIRMED",
                        "assistant_message": (
                            "The compiled causal specification, transformation plan, and "
                            "validation review are now confirmed. We can proceed with this setup."
                        ),
                        "system_message": None,
                        "error_message": None,
                    }
                )
            )

        if _is_rejection(latest_user_message):
            return self._failed_state(
                payload=payload,
                issues=payload.validation_issues,
                assistant_message=(
                    "The compiled protocol review was not confirmed. Please go back and revise "
                    "the protocol or dataset assumptions before we continue."
                ),
                error_message="user rejected the compiled protocol review",
            )

        return CompileAndValidateState(
            payload.model_copy(
                update={
                    "assistant_message": (
                        f"{payload.assistant_message or ''}\n\n"
                        "Please reply clearly with confirmation if this compiled setup is acceptable, "
                        "or state what must change."
                    ).strip(),
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


def _is_affirmative(text: str) -> bool:
    stripped = text.strip()
    return bool(_AFFIRM_RE.search(stripped)) and not bool(_REJECT_RE.search(stripped))


def _is_rejection(text: str) -> bool:
    return bool(_REJECT_RE.search(text.strip()))


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
    dataframe,
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
        if issue.message == "Cleaned dataset contains columns outside the confirmed protocol scope.":
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
            if issue.message == "Cleaned dataset contains columns outside the confirmed protocol scope."
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
                main_lines.append(f"  Extra columns currently present: {', '.join(map(str, extra_columns))}")
    main_lines.append("")
    if scope_issue is not None:
        main_lines.append(
            "The final cleaned working dataset should contain only treatment, outcome, covariates, and effect modifiers."
        )
        main_lines.append(
            "If any of the extra columns are intentionally still needed, tell me exactly which columns they are and whether they should remain in the protocol scope."
        )
        main_lines.append(
            "Otherwise, remove those extra columns and then rerun this step."
        )
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
    warning_lines = [f"- {issue.message}" for issue in issues if issue.severity == "WARN"]
    plan_lines = [
        f"- {column.column}: {column.encoding.preset}"
        for column in transform_plan.columns
    ]
    lines = [
        "I compiled the confirmed protocol into a causal specification and a baseline transformation plan.",
        "",
        f"Treatment: {treatment.column} ({treatment.control} vs {treatment.treated})",
        f"Outcome: {outcome.column} ({outcome.kind})",
        f"Covariates: {', '.join(causal_spec.covariates) if causal_spec.covariates else 'None'}",
        f"Effect modifiers: {', '.join(causal_spec.effect_modifiers) if causal_spec.effect_modifiers else 'None'}",
        "",
        "Planned baseline transformations:",
        *plan_lines,
    ]
    if warning_lines:
        lines.extend(
            [
                "",
                "Validation warnings to review:",
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
            "If something is wrong, tell me what should change.",
        ]
    )
    return "\n".join(lines)


__all__ = ["CompileAndValidateNode"]
