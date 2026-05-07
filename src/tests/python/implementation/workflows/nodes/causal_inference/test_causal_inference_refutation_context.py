from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from python.domain.models.models import ChatMessage
from python.domain.service.llm_service import LLMConfig, LLMResponse, LLMService
from python.implementation.workflows.nodes.causal_inference.causal_inference_node import (
    _summarize_cate,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec


@dataclass
class _CapturingLLM(LLMService):
    user_prompts: list[str] = field(default_factory=list)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
    ) -> LLMResponse:
        _ = system_prompt, config, history
        self.user_prompts.append(user_prompt)
        return LLMResponse(content="CATE summary.")

    def generate_json(
        self,
        *,
        schema: type[BaseModel],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> BaseModel:
        _ = schema, system_prompt, user_prompt, config, history, max_attempts
        raise AssertionError("generate_json should not be called")


def test_cate_summary_prompt_includes_saved_negative_control_refutation_summary() -> None:
    causal_spec = CausalSpec.model_validate(
        {
            "treatment_spec": {
                "kind": "binary",
                "column": "treatment",
                "treated": "drug",
                "control": "control",
            },
            "outcome_spec": {
                "kind": "continuous",
                "column": "outcome",
                "unit": "score",
            },
            "negative_control_outcome": {
                "kind": "continuous",
                "column": "negative_control",
                "unit": "score",
            },
            "covariates": ["age"],
            "effect_modifiers": ["sex"],
            "experiment_type": "OBSERVATIONAL",
            "id_col": "patient_id",
        }
    )
    refutation_summary: dict[str, Any] = {
        "status": "COMPLETED",
        "primary_model_id": "primary-id",
        "negative_control_model_id": "negative-id",
        "comparison": {"n_rows": 4, "mean_abs_negative_control_to_primary_ratio": 0.05},
    }
    llm = _CapturingLLM()

    message = _summarize_cate(
        llm=llm,
        selected_model="econml.dml.LinearDML",
        causal_spec=causal_spec,
        cate_payload={
            "request_summary": "Estimate subgroup treatment effects.",
            "cohorts": [],
            "non_effect_modifier_filter_columns": [],
            "effect_modifier_columns": ["sex"],
        },
        negative_control_refutation_summary=refutation_summary,
        history=[],
    )

    assert message == "CATE summary."
    assert len(llm.user_prompts) == 1
    assert "negative_control_refutation_summary" in llm.user_prompts[0]
    assert "COMPLETED" in llm.user_prompts[0]
    assert "negative-id" in llm.user_prompts[0]
