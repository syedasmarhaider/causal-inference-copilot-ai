from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest

from python.domain.models.errors import StateDependencyError
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_state import (
    CompileAndValidatePayloadModel,
    CompileAndValidateState,
)
from python.implementation.workflows.nodes.model_selection.mode_selection_state import (
    ConfirmedModelSelectionPayload,
    ModelRecommendationModel,
    ModelSelectionPayload,
    ModelSelectionState,
)
from python.implementation.workflows.nodes.model_selection.model_selection_deps import (
    ModelSelectionDeps,
)
from python.implementation.workflows.nodes.model_selection.model_selection_node import (
    ModelSelectionNode,
)
from python.implementation.workflows.nodes.model_selection.model_selection_prompts import (
    get_model_selection_node_info,
    get_model_selection_freezed_answer_prompt,
)
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import (
    TransformPlan,
)
from python.implementation.workflows.tools.causal.inference.causal_model_factory_tool import (
    CausalModelFactoryTool,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.domain.models.validation import ValidationIssueModel


def _build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "treatment": ["drug", "control", "drug", "control"],
            "outcome": [1.0, 0.0, 0.5, 0.2],
            "age": [60, 55, 70, 48],
            "sex": ["F", "M", "F", "M"],
        }
    )


def _build_inference_ready_spec() -> InferenceReadyCausalSpec:
    df = _build_dataframe()
    summary = DatasetProfilingTool().extract_dataset_summary(
        df,
        max_categories=10,
        sample_distinct=10,
        compute_quantiles=False,
        strict=True,
    )
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
            "covariates": ["age"],
            "effect_modifiers": ["sex"],
            "experiment_type": "OBSERVATIONAL",
        }
    )
    plan = TransformPlan.model_validate(
        {
            "columns": [
                {
                    "column": "age",
                    "role": "covariate",
                    "encoding": {"preset": "num_standard"},
                },
                {
                    "column": "sex",
                    "role": "effect_modifier",
                    "encoding": {
                        "preset": "map_binary",
                        "mapping": {"F": 0.0, "M": 1.0},
                        "allow_unknown": False,
                        "missing": "error",
                    },
                },
            ]
        }
    )
    return InferenceReadyCausalSpec(
        causal_spec=causal_spec,
        transformation_plan=plan,
        data_summary=summary,
    )


def _compile_state(*, warnings: list[ValidationIssueModel] | None = None) -> CompileAndValidateState:
    spec = _build_inference_ready_spec()
    return CompileAndValidateState(
        CompileAndValidatePayloadModel(
            dataset_id=uuid4(),
            dataset_summary=spec.data_summary,
            protocol_discussion="Confirmed protocol discussion",
            compiled_causal_spec=spec.causal_spec,
            transformation_plan=spec.transformation_plan,
            inference_ready_causal_spec=spec,
            validation_issues=warnings or [],
            phase="CONFIRMED",
            assistant_message="Confirmed compile review",
        )
    )


@dataclass
class _FakeLLM:
    json_outputs: list[object] = field(default_factory=list)
    generate_outputs: list[object] = field(default_factory=list)
    generate_json_calls: list[dict[str, object]] = field(default_factory=list)
    generate_calls: list[dict[str, object]] = field(default_factory=list)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
    ) -> LLMResponse:
        self.generate_calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "config": config,
                "history": history,
            }
        )
        if not self.generate_outputs:
            raise AssertionError("unexpected generate call")
        next_output = self.generate_outputs.pop(0)
        if isinstance(next_output, Exception):
            raise next_output
        if isinstance(next_output, LLMResponse):
            return next_output
        return LLMResponse(content=str(next_output))

    def generate_json(
        self,
        *,
        schema: type[Any],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> Any:
        self.generate_json_calls.append(
            {
                "schema": schema,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "config": config,
                "history": history,
                "max_attempts": max_attempts,
            }
        )
        if not self.json_outputs:
            raise AssertionError("unexpected generate_json call")
        next_output = self.json_outputs.pop(0)
        if isinstance(next_output, Exception):
            raise next_output
        if isinstance(next_output, dict):
            return schema.model_validate(next_output)
        return next_output


@dataclass
class _FakeModelFactory:
    supported: list[str]
    info: dict[str, str]

    def supported_estimators(self) -> list[str]:
        return list(self.supported)

    def get_all_esimators_info(self) -> dict[str, str]:
        return dict(self.info)

    def has_estimator(self, estimator_fqcn: str) -> bool:
        return estimator_fqcn in self.supported


@dataclass
class _FakeToolFactory(ToolFactory):
    model_factory: Any

    def get_tool_names(self) -> list[str]:
        return [CausalModelFactoryTool.NAME]

    def get_tool_info(self, name: str) -> str:
        raise NotImplementedError

    def get_tools_info(self) -> dict[str, str]:
        raise NotImplementedError

    def has_tool(self, name: str) -> bool:
        return name == CausalModelFactoryTool.NAME

    def get_tool(self, name: str) -> Any:
        if name != CausalModelFactoryTool.NAME:
            raise KeyError(name)
        return self.model_factory


def _supported_models() -> tuple[list[str], dict[str, str]]:
    supported = [
        "econml.dml.CausalForestDML",
        "econml.dml.LinearDML",
        "econml.dr.LinearDRLearner",
    ]
    info = {model: f"Info for {model}" for model in supported}
    return supported, info


def test_model_selection_info_and_state_roundtrip() -> None:
    assert "confirmed inference-ready causal specification" in get_model_selection_node_info().lower()
    assert "read-only clinician questions" in get_model_selection_freezed_answer_prompt().lower()

    state = ModelSelectionState.init_empty()
    assert state.status() == "PENDING"
    assert state.messages()[0].role == "assistant"

    confirmed = ModelSelectionState(
        ModelSelectionPayload(
            recommendations=[],
            confirmed_model_selection=ConfirmedModelSelectionPayload(
                selected_model="econml.dml.CausalForestDML",
                reasoning="Best fit for subgroup heterogeneity.",
            ),
            assistant_message="Confirmed.",
        )
    )
    assert confirmed.status() == "DONE"
    restored = ModelSelectionState.from_json_dict(confirmed.to_json_dict())
    assert restored.payload.model_dump(mode="json") == confirmed.payload.model_dump(mode="json")

    confirmed.set_status_freez()
    assert confirmed.payload.freezed is True
    assert confirmed.status() == "FREEZED"
    assert "model selection is freezed" in confirmed.messages()[0].content.lower()


def test_model_selection_deps_require_confirmed_compile_and_extract_warn_only() -> None:
    warning = ValidationIssueModel(
        severity="WARN",
        message="Minor overlap concern",
        evidence={},
        fix_hint=None,
    )
    deps = ModelSelectionDeps.from_loaded(
        {CompileAndValidateState.NAME: _compile_state(warnings=[warning])}
    )
    assert deps.inference_ready_spec.causal_spec.treatment_spec.column == "treatment"
    assert [issue.message for issue in deps.validation_warnings] == ["Minor overlap concern"]

    with pytest.raises(StateDependencyError):
        ModelSelectionDeps.from_loaded(
            {
                CompileAndValidateState.NAME: CompileAndValidateState(
                    _compile_state().payload.model_copy(update={"phase": "REVIEW_READY"})
                )
            }
        )


def test_model_selection_first_run_builds_clinician_friendly_shortlist() -> None:
    supported, info = _supported_models()
    llm = _FakeLLM(
        json_outputs=[
            {
                "recommendations": [
                    {
                        "estimator_fqcn": "econml.dml.CausalForestDML",
                        "best_when": "You expect treatment effects to vary across patient subgroups.",
                        "why": "It handles flexible subgroup patterns well.",
                        "tradeoffs": "Less transparent than the simplest linear options.",
                    },
                    {
                        "estimator_fqcn": "econml.dr.LinearDRLearner",
                        "best_when": "You want a clearer baseline estimate with straightforward interpretation.",
                        "why": "It is often easier to explain clinically.",
                        "tradeoffs": "May miss more complex heterogeneity.",
                    },
                    {
                        "estimator_fqcn": "econml.dml.LinearDML",
                        "best_when": "You want an adjusted baseline estimate with a structured linear form.",
                        "why": "It balances adjustment with interpretability.",
                        "tradeoffs": "Still less flexible than forest-style options.",
                    },
                ],
                "clinician_message": "Here are three reasonable model choices based on the confirmed protocol and reviewed warnings.",
            }
        ]
    )
    node = ModelSelectionNode(
        llm=llm,
        tool_factory=_FakeToolFactory(_FakeModelFactory(supported=supported, info=info)),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=ModelSelectionState.init_empty(),
        previous_state_dependencies={CompileAndValidateState.NAME: _compile_state()},
        messages_history=[ChatMessage(role="user", content="Recommend the best models.")],
    )

    assert result.status() == "PENDING"
    assert result.payload.recommendations
    assert result.payload.assistant_message is not None
    assert "Flexible Heterogeneity Model (Causal Forest Model)" in result.payload.assistant_message
    assert "Clinically Transparent Baseline Model (Doubly Robust Linear Model)" in result.payload.assistant_message
    assert "econml." not in result.payload.assistant_message

    user_prompt = str(llm.generate_json_calls[0]["user_prompt"])
    assert '"column_types"' in user_prompt
    assert '"name": "treatment"' in user_prompt
    assert '"inferred_kind": "CATEGORICAL"' in user_prompt
    assert '"validation_warnings"' in user_prompt
    assert '"summary"' not in user_prompt
    assert "top_categories" not in user_prompt


def test_model_selection_second_run_confirms_user_choice() -> None:
    supported, info = _supported_models()
    llm = _FakeLLM(
        json_outputs=[
            {
                "selected_model": "econml.dml.CausalForestDML",
                "reasoning": "This best matches your focus on subgroup differences.",
            }
        ]
    )
    node = ModelSelectionNode(
        llm=llm,
        tool_factory=_FakeToolFactory(_FakeModelFactory(supported=supported, info=info)),
    )
    state = ModelSelectionState(
        ModelSelectionPayload(
            recommendations=[
                ModelRecommendationModel(
                    estimator_fqcn="econml.dml.CausalForestDML",
                    display_label="Flexible Heterogeneity Model (Causal Forest Model)",
                    best_when="Subgroup variation matters.",
                    why="Flexible subgroup effects.",
                    tradeoffs="Less transparent.",
                ),
                ModelRecommendationModel(
                    estimator_fqcn="econml.dr.LinearDRLearner",
                    display_label="Clinically Transparent Baseline Model (Doubly Robust Linear Model)",
                    best_when="Interpretability matters.",
                    why="Simple baseline estimate.",
                    tradeoffs="Less flexible.",
                ),
                ModelRecommendationModel(
                    estimator_fqcn="econml.dml.LinearDML",
                    display_label="Adjusted Baseline Effect Model (Linear Double Machine Learning Model)",
                    best_when="Adjusted linear estimate is acceptable.",
                    why="Balanced choice.",
                    tradeoffs="Less flexible than forest.",
                ),
            ],
            assistant_message="Choose one option.",
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=state,
        previous_state_dependencies={CompileAndValidateState.NAME: _compile_state()},
        messages_history=[ChatMessage(role="user", content="I want the heterogeneity option.")],
    )

    assert result.status() == "DONE"
    assert result.payload.confirmed_model_selection is not None
    assert result.payload.confirmed_model_selection.selected_model == "econml.dml.CausalForestDML"
    assert "Flexible Heterogeneity Model (Causal Forest Model)" in (result.payload.assistant_message or "")


def test_model_selection_second_run_keeps_pending_when_user_is_unclear() -> None:
    supported, info = _supported_models()
    llm = _FakeLLM(
        json_outputs=[
            {
                "selected_model": None,
                "reasoning": "Do you want the most flexible subgroup model or the most interpretable baseline model?",
            }
        ]
    )
    node = ModelSelectionNode(
        llm=llm,
        tool_factory=_FakeToolFactory(_FakeModelFactory(supported=supported, info=info)),
    )
    state = ModelSelectionState(
        ModelSelectionPayload(
            recommendations=[
                ModelRecommendationModel(
                    estimator_fqcn="econml.dml.CausalForestDML",
                    display_label="Flexible Heterogeneity Model (Causal Forest Model)",
                    best_when="Subgroup variation matters.",
                    why="Flexible subgroup effects.",
                    tradeoffs="Less transparent.",
                ),
                ModelRecommendationModel(
                    estimator_fqcn="econml.dr.LinearDRLearner",
                    display_label="Clinically Transparent Baseline Model (Doubly Robust Linear Model)",
                    best_when="Interpretability matters.",
                    why="Simple baseline estimate.",
                    tradeoffs="Less flexible.",
                ),
                ModelRecommendationModel(
                    estimator_fqcn="econml.dml.LinearDML",
                    display_label="Adjusted Baseline Effect Model (Linear Double Machine Learning Model)",
                    best_when="Adjusted linear estimate is acceptable.",
                    why="Balanced choice.",
                    tradeoffs="Less flexible than forest.",
                ),
            ],
            assistant_message="Choose one option.",
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=state,
        previous_state_dependencies={CompileAndValidateState.NAME: _compile_state()},
        messages_history=[ChatMessage(role="user", content="I am not sure.")],
    )

    assert result.status() == "PENDING"
    assert result.payload.confirmed_model_selection is None
    assert result.payload.assistant_message == "Do you want the most flexible subgroup model or the most interpretable baseline model?"


def test_model_selection_freezed_flag_answers_read_only_questions() -> None:
    supported, info = _supported_models()
    llm = _FakeLLM(
        generate_outputs=[
            "The confirmed model favors subgroup heterogeneity over maximal interpretability."
        ]
    )
    node = ModelSelectionNode(
        llm=llm,
        tool_factory=_FakeToolFactory(_FakeModelFactory(supported=supported, info=info)),
    )
    state = ModelSelectionState(
        ModelSelectionPayload(
            recommendations=[
                ModelRecommendationModel(
                    estimator_fqcn="econml.dml.CausalForestDML",
                    display_label="Flexible Heterogeneity Model (Causal Forest Model)",
                    best_when="Subgroup variation matters.",
                    why="Flexible subgroup effects.",
                    tradeoffs="Less transparent.",
                ),
                ModelRecommendationModel(
                    estimator_fqcn="econml.dr.LinearDRLearner",
                    display_label="Clinically Transparent Baseline Model (Doubly Robust Linear Model)",
                    best_when="Interpretability matters.",
                    why="Simple baseline estimate.",
                    tradeoffs="Less flexible.",
                ),
                ModelRecommendationModel(
                    estimator_fqcn="econml.dml.LinearDML",
                    display_label="Adjusted Baseline Effect Model (Linear Double Machine Learning Model)",
                    best_when="Adjusted linear estimate is acceptable.",
                    why="Balanced choice.",
                    tradeoffs="Less flexible than forest.",
                ),
            ],
            confirmed_model_selection=ConfirmedModelSelectionPayload(
                selected_model="econml.dml.CausalForestDML",
                reasoning="Best fit for heterogeneity.",
            ),
            freezed=True,
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=state,
        previous_state_dependencies={CompileAndValidateState.NAME: _compile_state()},
        messages_history=[ChatMessage(role="user", content="Why was this model selected?")],
    )

    assert result.status() == "FREEZED"
    assert result.payload.freezed is True
    assert result.payload.assistant_message == (
        "The confirmed model favors subgroup heterogeneity over maximal interpretability."
    )
    assert len(llm.generate_calls) == 1
    assert len(llm.generate_json_calls) == 0
