from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.model_selection.mode_selection_state import (
    ConfirmedModelSelectionPayload,
    ModelSelectionPayload,
    ModelSelectionState,
)
from python.implementation.workflows.nodes.model_selection.model_selection_deps import (
    ModelSelectionDeps,
)
from python.implementation.workflows.nodes.model_selection.model_selection_prompts import (
    MODEL_SELECTION_NEGOTIATOR_SYSTEM_PROMPT,
    MODEL_SELECTION_NEGOTIATOR_USER_PROMPT_TEMPLATE,
    MODEL_SELECTION_RECOMMENDER_SYSTEM_PROMPT,
    MODEL_SELECTION_RECOMMENDER_USER_PROMPT_TEMPLATE,
    get_model_selection_node_info,
)
from python.implementation.workflows.tools.causal.causal_model_factory_tool import (
    CausalModelFactoryTool,
)


# ----------------------------
# LLM schema for call 1
# ----------------------------
class _RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    estimator_fqcn: str
    title: str
    best_when: str
    why: str
    tradeoffs: str | None = None


class _ModelShortlist(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    recommendations: list[_RecommendationItem] = Field(..., min_length=3, max_length=3)
    clinician_message: str


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _safe_model_dump(x: Any) -> Any:
    if x is None:
        return None
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    return x

def _build_context(*, deps: ModelSelectionDeps) -> dict[str, Any]:
    causal_specs = deps.compiled_causal_spec
    validation_issues = deps.validation_errors
    summary = deps.clean_dataset_summary
    
    treatment_spec = causal_specs.treatment_spec
    outcome_spec = causal_specs.outcome_spec
    covariates = causal_specs.covariates
    effect_modifiers = causal_specs.effect_modifiers
    experiment_type = causal_specs.experiment_type

    return {
    
        "treatment_spec": _safe_model_dump(treatment_spec),
        "outcome_spec": _safe_model_dump(outcome_spec),
        "covariates": _safe_model_dump(covariates),
        "effect_modifiers": _safe_model_dump(effect_modifiers),
        "experiment_type": _safe_model_dump(experiment_type),
        "summary": _safe_model_dump(summary),
        "validate_clean_protocol": {
            "issues": [_safe_model_dump(issue) for issue in validation_issues],
        },
    }


def _format_shortlist_message(shortlist: _ModelShortlist) -> str:
    lines: list[str] = []
    lines.append("")
    for i, rec in enumerate(shortlist.recommendations, start=1):
        lines.append(f"Option {i}: {rec.title}")
        lines.append(f"- Best when: {rec.best_when}")
        lines.append(f"- Why: {rec.why}")
        if rec.tradeoffs:
            lines.append(f"- Trade-offs: {rec.tradeoffs}")
        lines.append(f"- Internal model id: {rec.estimator_fqcn}")
        lines.append("")
    return "\n".join(lines).strip()


@dataclass(frozen=True, slots=True)
class ModelSelectionNode(Node):
    llm: LLMService


    @property
    def name(self) -> str:
        return ModelSelectionState.NAME

    @classmethod
    def get_info(cls) -> str:
        return get_model_selection_node_info()

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        tool_factory: ToolFactory,
        previous_state_dependencies: Any,  # Mapping[str, State] (kept Any to match your ABC signature)
        messages_history: Sequence[ChatMessage] | None
    ) -> State:
        if not isinstance(state, ModelSelectionState):
            raise ValueError(f"{self.name}: invalid state (got {type(state).__name__})")

        # deps
        deps = ModelSelectionDeps.from_loaded(previous_state_dependencies)
        context = _build_context(deps=deps)
        last_5_messages = messages_history[-5:] if messages_history else None
        
        ci_tool_factory_raw = tool_factory.get_tool(CausalModelFactoryTool.NAME)
        ci_tool_factory = cast(CausalModelFactoryTool, ci_tool_factory_raw)
        supported_estimators = ci_tool_factory.supported_estimators()
        supported_estimators_info = ci_tool_factory.get_all_esimators_info()
        
        
        # ============================================================
        # CALL 1: generate shortlist if missing (skip if already exists)
        # ============================================================
        if not state.payload.system_choice_message:
            user_prompt = MODEL_SELECTION_RECOMMENDER_USER_PROMPT_TEMPLATE.format(
                supported_estimators_json=_dumps(  supported_estimators),
                estimators_info_json=_dumps(  supported_estimators_info),
                context_json=_dumps(context))

            shortlist = self.llm.generate_json(
                    schema=_ModelShortlist,
                    system_prompt=MODEL_SELECTION_RECOMMENDER_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    config= LLMConfig(temperature=1.0, model="pro"),
                    history=last_5_messages,
                    max_attempts=3,
                )

            if len(shortlist.recommendations) != 3:
                    return ModelSelectionState(
                         ModelSelectionPayload(
                            error=f"LLM returned {len(shortlist.recommendations)} recommendations, but exactly 3 are required.",
                            system_choice_message=None,
                            confirmed_model_selection=None,
                            message="Selection error: LLM returned an incorrect number of recommendations. I will retry. Please wait a moment..."
                       )
                    )
                
            for rec in shortlist.recommendations:
                    if rec.estimator_fqcn not in supported_estimators:
                        return ModelSelectionState(
                            ModelSelectionPayload(
                                error=f"LLM recommended model '{rec.estimator_fqcn}' which is not supported by the system.",
                                system_choice_message=None,
                                confirmed_model_selection=None,
                                message="Selection error: LLM recommended an unsupported model. I will retry. Please wait a moment..."
                            )
                        )

            return ModelSelectionState(
                ModelSelectionPayload(
                    error=None,
                    system_choice_message=_format_shortlist_message(shortlist),
                    confirmed_model_selection=None,
                    message=shortlist.clinician_message,
                )
            )

        # ============================================================
        # CALL 2: negotiate/confirm using user's reply (if present)
        # ============================================================
      
        negotiator_user_prompt = MODEL_SELECTION_NEGOTIATOR_USER_PROMPT_TEMPLATE.format(
            recommended_message=state.payload.system_choice_message or "",
            supported_estimators_json=_dumps(supported_estimators),
            estimators_info_json=_dumps(supported_estimators_info),
            context_json=_dumps(context),
        )
        
        decision = self.llm.generate_json(
                schema=ConfirmedModelSelectionPayload,
                system_prompt=MODEL_SELECTION_NEGOTIATOR_SYSTEM_PROMPT,
                user_prompt=negotiator_user_prompt,
                config= LLMConfig(temperature=0.2, model="basic"),
                history=last_5_messages,
                max_attempts=3,
            )
      
 

        # Not final: selected_model is null -> ask follow-up (stored in system_choice_message)
        if not decision.selected_model:
            payload = state.payload.model_copy(
                update={
                    "confirmed_model_selection": None,
                    "error": None,
                    "message": decision.reasoning 
                }
            )
            return ModelSelectionState(payload=payload)

        # Final selection: confirm via tool
        selected = decision.selected_model.strip()
        if not ci_tool_factory.has_estimator(selected):
            payload = state.payload.model_copy(
                update={
                    "confirmed_model_selection": None,
                    "message": "Sorry but the selected model is not recognized. Please choose one of the recommended options.",
                }
            )
            return ModelSelectionState(payload=payload)

        final_msg = (
            f"Confirmed model selection: {selected}\n"
            "Next, I will fit this model and estimate the treatment effect."
        ).strip()

        payload = state.payload.model_copy(
            update={
                "confirmed_model_selection": decision,
                "error": None,
                "message": final_msg,
            }
        )
        return ModelSelectionState(payload=payload)