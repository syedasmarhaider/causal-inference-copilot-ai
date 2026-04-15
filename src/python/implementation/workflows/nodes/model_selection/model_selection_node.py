from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.validation import ValidationIssueModel, ValidationStatus
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.model_selection.mode_selection_state import (
    ConfirmedModelSelectionPayload,
    ModelRecommendationModel,
    ModelSelectionPayload,
    ModelSelectionState,
)
from python.implementation.workflows.nodes.model_selection.model_selection_deps import (
    ModelSelectionDeps,
)
from python.implementation.workflows.nodes.model_selection.model_selection_prompts import (
    model_selection_node_info,
    model_selection_recommender_system_prompt,
    model_selection_recommender_user_prompt,
    model_selection_review_decision_prompt,
    model_selection_review_decision_user_prompt,
)
from python.implementation.workflows.tools.causal.inference.causal_model_factory_tool import (
    CausalModelFactoryTool,
)
from python.implementation.workflows.tools.causal.inference.econml.models_meta import (
    get_model_display_labels,
)
from python.implementation.workflows.utils.utils import safe_err

log = get_app_logger(__name__, component="model_selection_node", log_type="node")


class _RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    estimator_fqcn: str
    best_when: str = Field(..., min_length=1)
    why: str = Field(..., min_length=1)
    tradeoffs: str | None = None


class _ModelShortlist(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    recommendations: list[_RecommendationItem] = Field(..., min_length=3, max_length=3)
    user_message: str = Field(..., min_length=1)


class _ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    selected_model: str | None = None
    assistant_message: str = Field(..., min_length=1)


class ModelSelectionNode(Node):
    NAME: ClassVar[str] = ModelSelectionState.NAME

    def __init__(
        self,
        *,
        llm: LLMService,
        tools_factory: ToolFactory,
    ) -> None:
        self._llm = llm
        factory_raw = tools_factory.get_tool(CausalModelFactoryTool.NAME)
        self._model_factory = cast(CausalModelFactoryTool, factory_raw)

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return model_selection_node_info()

    def run(
        self,
        *,
        request: NodeRequest,
    ) -> NodeExecutionResult:
        if not isinstance(request.node_state, ModelSelectionState):
            raise TypeError(
                f"{self.name}: expected ModelSelectionState, got "
                f"{type(request.node_state).__name__}"
            )

        payload = request.node_state.payload.model_copy(deep=True)
        deps = ModelSelectionDeps.from_request(request)

        if deps.causal_spec is None or deps.transformation_plan is None:
            return self._needs_input_result(
                request=request,
                payload=ModelSelectionPayload(),
                user_message=(
                    "I need a confirmed compiled causal specification and transformation "
                    "plan before I can recommend models."
                ),
            )

        if deps.validation_status is None:
            return self._needs_input_result(
                request=request,
                payload=ModelSelectionPayload(),
                user_message=(
                    "I need a confirmed validation result before I can recommend models."
                ),
            )

        payload, sources_changed = self._bind_payload_to_sources(
            payload=payload,
            deps=deps,
        )

        if deps.validation_status == "FAIL":
            return self._failed_result(
                request=request,
                payload=payload,
                user_message=_build_validation_blocked_message(deps.validation_issues),
                error_message="validation failed; model selection blocked",
            )

        latest_user_message = _latest_user_message(request.read_only_messages_history)

        if payload.phase == "REVIEW_READY":
            if not payload.recommendations:
                log.warning(
                    "model selection review payload missing recommendations; regenerating",
                    conversation_id=str(request.conversation_id),
                )
            else:
                if latest_user_message is None:
                    return self._needs_input_result(
                        request=request,
                        payload=payload,
                        user_message=payload.assistant_message
                        or "Please confirm one of the shortlisted models.",
                    )
                return self._handle_review_response(
                    request=request,
                    payload=payload,
                    deps=deps,
                    latest_user_message=latest_user_message,
                )

        if payload.phase == "CONFIRMED" and not sources_changed:
            return self._done_result(
                request=request,
                payload=payload,
                user_message=payload.assistant_message or "The model selection is already confirmed.",
            )

        if payload.phase == "FAILED" and not sources_changed:
            return self._aborted_result(
                request=request,
                payload=payload,
                user_message=payload.assistant_message
                or "Model selection is blocked until the upstream setup changes.",
            )

        return self._generate_shortlist(
            request=request,
            payload=payload,
            deps=deps,
            sources_changed=sources_changed,
        )

    def _bind_payload_to_sources(
        self,
        *,
        payload: ModelSelectionPayload,
        deps: ModelSelectionDeps,
    ) -> tuple[ModelSelectionPayload, bool]:
        if (
            payload.source_dataset_id == deps.dataset_id
            and _model_json_equal(payload.source_causal_spec, deps.causal_spec)
            and _model_json_equal(
                payload.source_transformation_plan,
                deps.transformation_plan,
            )
            and _issues_json_equal(payload.source_validation_issues, deps.validation_issues)
            and payload.source_validation_status == deps.validation_status
        ):
            return payload, False

        if (
            payload.source_dataset_id is None
            and payload.source_causal_spec is None
            and payload.source_transformation_plan is None
            and not payload.source_validation_issues
            and payload.source_validation_status is None
            and payload.phase == "INIT"
        ):
            return payload.bind_sources(
                dataset_id=deps.dataset_id,
                causal_spec=deps.causal_spec,
                transformation_plan=deps.transformation_plan,
                validation_issues=deps.validation_issues,
                validation_status=deps.validation_status,
            ), False

        return payload.reset_for_reselection(
            dataset_id=deps.dataset_id,
            causal_spec=deps.causal_spec,
            transformation_plan=deps.transformation_plan,
            validation_issues=deps.validation_issues,
            validation_status=deps.validation_status,
        ), True

    def _generate_shortlist(
        self,
        *,
        request: NodeRequest,
        payload: ModelSelectionPayload,
        deps: ModelSelectionDeps,
        sources_changed: bool,
    ) -> NodeExecutionResult:
        history = (
            list(request.read_only_messages_history[-5:])
            if request.read_only_messages_history
            else None
        )
        model_catalog = _build_supported_model_catalog(
            supported_estimators=self._model_factory.supported_estimators(),
            estimators_info=self._model_factory.get_all_esimators_info(),
        )
        selection_context = _build_selection_context(deps=deps)

        try:
            shortlist = self._llm.generate_json(
                schema=_ModelShortlist,
                system_prompt=model_selection_recommender_system_prompt(),
                user_prompt=model_selection_recommender_user_prompt(
                    supported_models_json=_dumps(model_catalog),
                    selection_context_json=_dumps(selection_context),
                ),
                config=LLMConfig(model="pro", temperature=0.4),
                history=history,
                max_attempts=3,
            )
        except Exception as exc:
            log.exception("model selection shortlist generation failed", error=safe_err(exc))
            return self._needs_input_result(
                request=request,
                payload=payload.reset_for_reselection(
                    dataset_id=deps.dataset_id,
                    causal_spec=deps.causal_spec,
                    transformation_plan=deps.transformation_plan,
                    validation_issues=deps.validation_issues,
                    validation_status=deps.validation_status,
                ),
                user_message=(
                    "I could not generate a valid shortlist of supported models from the "
                    "current setup. Please try again."
                ),
            )

        recommendations = _build_structured_recommendations(
            shortlist=shortlist,
            model_catalog=model_catalog,
        )
        if recommendations is None:
            return self._needs_input_result(
                request=request,
                payload=payload.reset_for_reselection(
                    dataset_id=deps.dataset_id,
                    causal_spec=deps.causal_spec,
                    transformation_plan=deps.transformation_plan,
                    validation_issues=deps.validation_issues,
                    validation_status=deps.validation_status,
                ),
                user_message=(
                    "I could not turn the shortlist into supported model options. Please try again."
                ),
            )

        review_message = _format_shortlist_message(
            recommendations=recommendations,
            user_message=shortlist.user_message,
        )
        if sources_changed:
            review_message = (
                "The compiled setup or validation result changed, so I refreshed the model "
                f"recommendations. {review_message}"
            )

        review_payload = payload.model_copy(
            update={
                "source_dataset_id": deps.dataset_id,
                "source_causal_spec": deps.causal_spec,
                "source_transformation_plan": deps.transformation_plan,
                "source_validation_issues": list(deps.validation_issues),
                "source_validation_status": deps.validation_status,
                "recommendations": recommendations,
                "confirmed_model_selection": None,
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

    def _handle_review_response(
        self,
        *,
        request: NodeRequest,
        payload: ModelSelectionPayload,
        deps: ModelSelectionDeps,
        latest_user_message: str,
    ) -> NodeExecutionResult:
        selection_context = _build_selection_context(deps=deps)
        decision = self._llm.generate_json(
            schema=_ReviewDecision,
            system_prompt=model_selection_review_decision_prompt(),
            user_prompt=model_selection_review_decision_user_prompt(
                recommended_options_json=_dumps(
                    [
                        recommendation.model_dump(mode="json")
                        for recommendation in payload.recommendations
                    ]
                ),
                selection_context_json=_dumps(selection_context),
                latest_user_message=latest_user_message,
            ),
            config=LLMConfig(model="basic", temperature=0.1),
            history=None,
            max_attempts=3,
        )

        if decision.selected_model is None:
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

        selected_recommendation = next(
            (
                recommendation
                for recommendation in payload.recommendations
                if recommendation.estimator_fqcn == decision.selected_model
            ),
            None,
        )
        if selected_recommendation is None:
            return self._needs_input_result(
                request=request,
                payload=payload,
                user_message=(
                    "That choice does not match the shortlisted supported models. Please pick "
                    "one of the presented options."
                ),
            )

        confirmed_selection = ConfirmedModelSelectionPayload(
            selected_model=selected_recommendation.estimator_fqcn,
            selected_model_display_label=selected_recommendation.display_label,
            reasoning=decision.assistant_message,
        )
        request.orchestrator_state.set(
            request.node_state.name(),
            {
                "selected_model": confirmed_selection.selected_model,
                "selected_model_display_label": confirmed_selection.selected_model_display_label,
                "selection_reasoning": confirmed_selection.reasoning,
            },
        )

        confirmed_payload = payload.model_copy(
            update={
                "confirmed_model_selection": confirmed_selection,
                "phase": "CONFIRMED",
                "assistant_message": (
                    f"Confirmed model selection: {selected_recommendation.display_label}. "
                    f"{decision.assistant_message}"
                ),
                "system_message": None,
                "error_message": None,
            }
        )
        return self._done_result(
            request=request,
            payload=confirmed_payload,
            user_message=confirmed_payload.assistant_message or selected_recommendation.display_label,
        )

    def _needs_input_result(
        self,
        *,
        request: NodeRequest,
        payload: ModelSelectionPayload,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=ModelSelectionState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="PENDING",
            action="NEEDS_INPUT",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _done_result(
        self,
        *,
        request: NodeRequest,
        payload: ModelSelectionPayload,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=ModelSelectionState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="DONE",
            action="NONE",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _aborted_result(
        self,
        *,
        request: NodeRequest,
        payload: ModelSelectionPayload,
        user_message: str,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=ModelSelectionState(payload),
            new_orchestrator_state=request.orchestrator_state,
            status="ABORTED",
            action="NONE",
            response_messages=[ChatMessage(role="assistant", content=user_message)],
        )

    def _failed_result(
        self,
        *,
        request: NodeRequest,
        payload: ModelSelectionPayload,
        user_message: str,
        error_message: str,
    ) -> NodeExecutionResult:
        failed_payload = payload.model_copy(
            update={
                "phase": "FAILED",
                "assistant_message": user_message,
                "system_message": "MODEL_SELECTION_BLOCKED",
                "error_message": error_message,
            }
        )
        return self._aborted_result(
            request=request,
            payload=failed_payload,
            user_message=user_message,
        )


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)


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


def _build_selection_context(*, deps: ModelSelectionDeps) -> dict[str, Any]:
    return {
        "compiled_dataset_summary": (
            None
            if deps.dataset_summary is None
            else deps.dataset_summary.model_dump(mode="json", exclude_none=True)
        ),
        "causal_spec": deps.causal_spec.model_dump(mode="json", exclude_none=True),
        "transformation_plan": deps.transformation_plan.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "validation_status": deps.validation_status,
        "validation_issues": [
            issue.model_dump(mode="json", exclude_none=True)
            for issue in deps.validation_issues
        ],
    }


def _build_supported_model_catalog(
    *,
    supported_estimators: Sequence[str],
    estimators_info: Mapping[str, str],
) -> list[dict[str, str]]:
    catalog: list[dict[str, str]] = []
    for fqcn in supported_estimators:
        display_name, family_label = get_model_display_labels(fqcn)
        catalog.append(
            {
                "estimator_fqcn": fqcn,
                "display_label": f"{display_name} ({family_label})",
                "display_name": display_name,
                "family_label": family_label,
                "model_info": estimators_info.get(fqcn, ""),
            }
        )
    return catalog


def _build_structured_recommendations(
    *,
    shortlist: _ModelShortlist,
    model_catalog: Sequence[Mapping[str, str]],
) -> list[ModelRecommendationModel] | None:
    by_fqcn = {
        str(entry["estimator_fqcn"]): str(entry["display_label"]) for entry in model_catalog
    }
    recommendations: list[ModelRecommendationModel] = []
    for item in shortlist.recommendations:
        display_label = by_fqcn.get(item.estimator_fqcn)
        if display_label is None:
            return None
        recommendations.append(
            ModelRecommendationModel(
                estimator_fqcn=item.estimator_fqcn,
                display_label=display_label,
                best_when=item.best_when,
                why=item.why,
                tradeoffs=item.tradeoffs,
            )
        )

    if len({item.estimator_fqcn for item in recommendations}) != 3:
        return None
    return recommendations


def _format_shortlist_message(
    *,
    recommendations: Sequence[ModelRecommendationModel],
    user_message: str,
) -> str:
    lines = [user_message.strip(), ""]
    for index, recommendation in enumerate(recommendations, start=1):
        lines.append(f"Option {index}: {recommendation.display_label}")
        lines.append(f"- Best when: {recommendation.best_when}")
        lines.append(f"- Why: {recommendation.why}")
        if recommendation.tradeoffs:
            lines.append(f"- Trade-offs: {recommendation.tradeoffs}")
        lines.append("")
    lines.append(
        "Tell me which option you want to confirm, or tell me what tradeoff matters most."
    )
    return "\n".join(lines).strip()


def _build_validation_blocked_message(
    issues: Sequence[ValidationIssueModel],
) -> str:
    fail_lines = [issue.message for issue in issues if issue.severity == "FAIL"]
    if fail_lines:
        return (
            "Model selection cannot proceed because validation still has blocking issues: "
            + "; ".join(fail_lines)
        )
    return (
        "Model selection cannot proceed because the latest validation result is marked as failed."
    )


def _model_json_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if hasattr(left, "model_dump") and hasattr(right, "model_dump"):
        return left.model_dump(mode="json") == right.model_dump(mode="json")
    return left == right


def _issues_json_equal(
    left: Sequence[ValidationIssueModel],
    right: Sequence[ValidationIssueModel],
) -> bool:
    return [
        issue.model_dump(mode="json", exclude_none=True) for issue in left
    ] == [
        issue.model_dump(mode="json", exclude_none=True) for issue in right
    ]


__all__ = ["ModelSelectionNode"]
