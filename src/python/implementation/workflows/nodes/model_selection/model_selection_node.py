from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
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
    get_model_selection_freezed_answer_prompt,
    MODEL_SELECTION_NEGOTIATOR_SYSTEM_PROMPT,
    MODEL_SELECTION_NEGOTIATOR_USER_PROMPT_TEMPLATE,
    MODEL_SELECTION_RECOMMENDER_SYSTEM_PROMPT,
    MODEL_SELECTION_RECOMMENDER_USER_PROMPT_TEMPLATE,
    get_model_selection_node_info,
)
from python.implementation.workflows.tools.causal.inference.causal_model_factory_tool import (
    CausalModelFactoryTool,
)
from python.implementation.workflows.tools.causal.inference.econml.models_meta import (
    get_model_display_labels,
)


class _RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    estimator_fqcn: str
    best_when: str = Field(..., min_length=1)
    why: str = Field(..., min_length=1)
    tradeoffs: str | None = None


class _ModelShortlist(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    recommendations: list[_RecommendationItem] = Field(..., min_length=3, max_length=3)
    clinician_message: str = Field(..., min_length=1)


class ModelSelectionNode(Node):
    NAME: ClassVar[str] = ModelSelectionState.NAME

    def __init__(
        self,
        *,
        llm: LLMService,
        tool_factory: ToolFactory,
    ) -> None:
        self._llm = llm
        factory_raw = tool_factory.get_tool(CausalModelFactoryTool.NAME)
        self._model_factory = cast(CausalModelFactoryTool, factory_raw)

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_model_selection_node_info()

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        previous_state_dependencies: Mapping[str, State],
        messages_history: Sequence[ChatMessage] | None,
    ) -> State:
        _ = user_id
        _ = conversation_id
        if not isinstance(state, ModelSelectionState):
            raise TypeError(f"{self.name}: expected ModelSelectionState, got {type(state).__name__}")

        deps = ModelSelectionDeps.from_loaded(previous_state_dependencies)
        history = list(messages_history[-5:]) if messages_history else None

        model_catalog = _build_supported_model_catalog(
            supported_estimators=self._model_factory.supported_estimators(),
            estimators_info=self._model_factory.get_all_esimators_info(),
        )
        selection_context = _build_selection_context(deps=deps)
        latest_user_message = _latest_user_message(messages_history)

        if state.payload.freezed:
            return self._answer_freezed_question(
                state=state,
                selection_context=selection_context,
                latest_user_message=latest_user_message,
                history=history,
            )

        if not state.payload.recommendations:
            shortlist = self._llm.generate_json(
                schema=_ModelShortlist,
                system_prompt=MODEL_SELECTION_RECOMMENDER_SYSTEM_PROMPT,
                user_prompt=MODEL_SELECTION_RECOMMENDER_USER_PROMPT_TEMPLATE.format(
                    supported_models_json=_dumps(model_catalog),
                    selection_context_json=_dumps(selection_context),
                ),
                config=LLMConfig(model="pro", temperature=0.4),
                history=history,
                max_attempts=3,
            )

            recommendations = _build_structured_recommendations(
                shortlist=shortlist,
                model_catalog=model_catalog,
            )
            if recommendations is None:
                return ModelSelectionState(
                    ModelSelectionPayload(
                        assistant_message=(
                            "I could not generate a valid shortlist of supported models. "
                            "Please try again."
                        ),
                    )
                )

            return ModelSelectionState(
                ModelSelectionPayload(
                    recommendations=recommendations,
                    assistant_message=_format_shortlist_message(
                        recommendations=recommendations,
                        clinician_message=shortlist.clinician_message,
                    ),
                )
            )

        decision = self._llm.generate_json(
            schema=ConfirmedModelSelectionPayload,
            system_prompt=MODEL_SELECTION_NEGOTIATOR_SYSTEM_PROMPT,
            user_prompt=MODEL_SELECTION_NEGOTIATOR_USER_PROMPT_TEMPLATE.format(
                recommended_options_json=_dumps(
                    [recommendation.model_dump(mode="json") for recommendation in state.payload.recommendations]
                ),
                selection_context_json=_dumps(selection_context),
            ),
            config=LLMConfig(model="basic", temperature=0.2),
            history=history,
            max_attempts=3,
        )

        if decision.selected_model is None:
            return ModelSelectionState(
                state.payload.model_copy(
                    update={
                        "assistant_message": decision.reasoning
                        or "Please tell me which option you prefer, or what tradeoff matters most.",
                    }
                )
            )

        if not any(
            recommendation.estimator_fqcn == decision.selected_model
            for recommendation in state.payload.recommendations
        ):
            return ModelSelectionState(
                state.payload.model_copy(
                    update={
                        "assistant_message": (
                            "That model choice does not match the shortlisted supported options. "
                            "Please choose one of the presented options."
                        )
                    }
                )
            )

        selected_label = next(
            recommendation.display_label
            for recommendation in state.payload.recommendations
            if recommendation.estimator_fqcn == decision.selected_model
        )
        return ModelSelectionState(
            state.payload.model_copy(
                update={
                    "confirmed_model_selection": decision,
                    "assistant_message": (
                        f"Confirmed model selection: {selected_label}. "
                        f"{decision.reasoning or 'Next I will use this model for training and effect estimation.'}"
                    ),
                    "error_message": None,
                }
            )
        )

    def _answer_freezed_question(
        self,
        *,
        state: ModelSelectionState,
        selection_context: Mapping[str, Any],
        latest_user_message: str | None,
        history: Sequence[ChatMessage] | None,
    ) -> ModelSelectionState:
        if not latest_user_message:
            return state

        try:
            assistant_message = self._llm.generate(
                system_prompt=get_model_selection_freezed_answer_prompt(),
                user_prompt=_dumps(
                    {
                        "recommendations": [
                            recommendation.model_dump(mode="json")
                            for recommendation in state.payload.recommendations
                        ],
                        "confirmed_model_selection": None
                        if state.payload.confirmed_model_selection is None
                        else state.payload.confirmed_model_selection.model_dump(mode="json"),
                        "selection_context": dict(selection_context),
                        "latest_user_message": latest_user_message,
                    }
                ),
                config=LLMConfig(model="basic", temperature=0.2),
                history=history,
            ).content.strip()
        except Exception:
            assistant_message = (
                "This model-selection state is frozen. I can answer read-only questions about "
                "the shortlisted options and the confirmed selection, but I could not answer "
                "that question right now."
            )

        return ModelSelectionState(
            state.payload.model_copy(
                update={
                    "assistant_message": assistant_message,
                    "error_message": None,
                }
            )
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
    causal_spec = deps.inference_ready_spec.causal_spec
    column_types = [
        {
            "name": str(profile.name),
            "inferred_kind": str(profile.inferred_kind),
        }
        for profile in deps.inference_ready_spec.data_summary.profiles
    ]
    return {
        "treatment": {
            "column": str(causal_spec.treatment_spec.column),
            "kind": str(causal_spec.treatment_spec.kind),
            "treated": str(causal_spec.treatment_spec.treated),
            "control": str(causal_spec.treatment_spec.control),
        },
        "outcome": {
            "column": str(causal_spec.outcome_spec.column),
            "kind": str(causal_spec.outcome_spec.kind),
        },
        "experiment_type": str(causal_spec.experiment_type),
        "covariates": list(causal_spec.covariates),
        "effect_modifiers": list(causal_spec.effect_modifiers),
        "column_types": column_types,
        "validation_warnings": [
            {
                "message": issue.message,
                "fix_hint": issue.fix_hint,
            }
            for issue in deps.validation_warnings
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
        str(entry["estimator_fqcn"]): str(entry["display_label"])
        for entry in model_catalog
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
    clinician_message: str,
) -> str:
    lines = [clinician_message.strip(), ""]
    for index, recommendation in enumerate(recommendations, start=1):
        lines.append(f"Option {index}: {recommendation.display_label}")
        lines.append(f"- Best when: {recommendation.best_when}")
        lines.append(f"- Why: {recommendation.why}")
        if recommendation.tradeoffs:
            lines.append(f"- Trade-offs: {recommendation.tradeoffs}")
        lines.append("")
    lines.append("Tell me which option fits your clinical goal best, or what tradeoff matters most.")
    return "\n".join(lines).strip()
