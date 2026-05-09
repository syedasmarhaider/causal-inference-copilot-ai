from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pandas as pd

from python.domain.service.llm_service import ChatMessage, LLMConfig
from python.domain.workflows.node import NodeRequest
from python.domain.workflows.ochestrator_state import OchestratorState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import (
    ProtocolDiscussionNode,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    CausalSpecDraft,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
    DatasetSummaryModel,
)


def _summary_for_df(df: pd.DataFrame) -> DatasetSummaryModel:
    return DatasetProfilingTool().extract_dataset_summary(
        df,
        max_categories=10,
        sample_distinct=10,
        compute_quantiles=False,
        strict=True,
    )


class _FakeOrchestratorState(OchestratorState):
    def __init__(self, *, dataset_summary: DatasetSummaryModel) -> None:
        self._values: dict[str, Any] = {
            "working_dataset_id": uuid4(),
            "latest_dataset_summary": dataset_summary,
        }
        self.set_calls: list[tuple[str, dict[str, Any]]] = []

    def name(self) -> str:
        return "FAKE_ORCHESTRATOR"

    def get_update_counter(self) -> int:
        return 0

    def set_update_counter(self, value: int) -> None:
        del value

    def get(self, key: str) -> Any:
        return self._values.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.set_calls.append((key, value))
        if "protocol_discussion" in value or "protocol_cleaning_instructions" in value:
            raise AssertionError("protocol node must not write legacy protocol artifacts")
        self._values.update(value)

    def get_current_node_name(self) -> str:
        return "PROTOCOL_DISCUSSION"

    def get_current_node_companion_names(self, node_name: str) -> list[str]:
        del node_name
        return []

    def get_completed_and_last_pending_nodes(self) -> list[str]:
        return []

    def rocover_failure(self, current_failed_node: str) -> None:
        del current_failed_node

    def get_forward_states_after_node(self, node_name: str) -> list[str]:
        del node_name
        return []

    def roll_back_to_state(self, state_name: str) -> None:
        del state_name

    def get_working_dataset_id_and_frozen_status(self) -> tuple[Any, bool]:
        return self._values.get("working_dataset_id"), False

    def get_ochestration_prompt(self) -> str:
        return ""

    def to_json_dict(self) -> dict[str, Any]:
        return dict(self._values)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> _FakeOrchestratorState:
        instance = cls(dataset_summary=payload["latest_dataset_summary"])
        instance._values = dict(payload)
        return instance

    @classmethod
    def init_empty(cls) -> _FakeOrchestratorState:
        raise NotImplementedError


@dataclass
class _FakeLLM:
    json_outputs: list[Any] = field(default_factory=list)
    generate_json_calls: list[dict[str, Any]] = field(default_factory=list)

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
        next_output = self.json_outputs.pop(0)
        if isinstance(next_output, dict):
            return schema.model_validate(next_output)
        return next_output


def _request(
    *,
    dataset_summary: DatasetSummaryModel,
    llm: _FakeLLM,
    user_message: str = "Use RXASP as treatment and DIED as outcome.",
) -> tuple[ProtocolDiscussionNode, NodeRequest, _FakeOrchestratorState]:
    node = ProtocolDiscussionNode(llm=llm)
    orchestrator_state = _FakeOrchestratorState(dataset_summary=dataset_summary)
    request = NodeRequest(
        user_id=uuid4(),
        conversation_id=uuid4(),
        node_state=ProtocolDiscussionState.init_empty(),
        orchestrator_state=orchestrator_state,
        read_only_messages_history=[ChatMessage(role="user", content=user_message)],
    )
    return node, request, orchestrator_state


def _complete_draft(**overrides: Any) -> dict[str, Any]:
    draft = {
        "treatment_column": "RXASP",
        "outcome_column": "DIED",
        "covariates": ["AGE"],
        "effect_modifiers": ["SEX"],
        "target_population": "all rows",
        "study_type": "OBSERVATIONAL",
        "negative_control_outcome": None,
        "time_zero": "baseline treatment decision at cohort entry",
    }
    draft.update(overrides)
    return draft


def test_protocol_discussion_updates_structured_draft_from_user_answer() -> None:
    summary = _summary_for_df(
        pd.DataFrame(
            {
                "RXASP": ["Y", "N"],
                "DIED": [1, 0],
                "AGE": [70, 65],
                "SEX": ["F", "M"],
            }
        )
    )
    llm = _FakeLLM(
        json_outputs=[
            {
                "draft": _complete_draft(target_population=None, time_zero=None),
                "next_action": "continue",
                "assistant_message": "I added the treatment and outcome. What target population and time zero should anchor the target trial?",
            }
        ]
    )
    node, request, _ = _request(dataset_summary=summary, llm=llm)

    result = node.run(request=request)
    prompt_payload = json.loads(llm.generate_json_calls[0]["user_prompt"])

    assert result.status == "PENDING"
    assert result.action == "NEEDS_INPUT"
    assert result.new_node_state.payload.draft.treatment_column == "RXASP"
    assert result.new_node_state.payload.draft.outcome_column == "DIED"
    assert "current_draft" in prompt_payload
    assert "dataset_column_names" in prompt_payload


def test_protocol_discussion_reports_selected_column_structure_and_missingness() -> None:
    summary = _summary_for_df(
        pd.DataFrame(
            {
                "RXASP": ["Y", "N", "Y"],
                "DIED": [1, 0, 1],
                "AGE": [70.0, None, 65.0],
                "SEX": ["F", "M", "Unknown"],
            }
        )
    )
    llm = _FakeLLM(
        json_outputs=[
            {
                "draft": _complete_draft(),
                "next_action": "continue",
                "assistant_message": "The draft is filled. Please confirm if this is the causal draft you want.",
            }
        ]
    )
    node, request, _ = _request(dataset_summary=summary, llm=llm)

    result = node.run(request=request)
    message = result.response_messages[0].content

    assert "Column structure:" in message
    assert "RXASP as treatment" in message
    assert "AGE as covariate" in message
    assert "1 missing" in message
    assert "Don't worry, we can figure this out in the next step." in message
    assert "plausible" in message.lower() or "risky" in message.lower()


def test_protocol_discussion_gives_population_filter_command_without_filtering() -> None:
    summary = _summary_for_df(
        pd.DataFrame(
            {
                "RXASP": ["Y", "N"],
                "DIED": [1, 0],
                "age": [20, 17],
            }
        )
    )
    llm = _FakeLLM(
        json_outputs=[
            {
                "draft": _complete_draft(
                    covariates=[],
                    effect_modifiers=[],
                    target_population="patients where age >= 18",
                ),
                "next_action": "continue",
                "assistant_message": "I saved that as the target population.",
            }
        ]
    )
    node, request, orchestrator_state = _request(dataset_summary=summary, llm=llm)

    result = node.run(request=request)

    assert result.status == "PENDING"
    assert "Target population can stay as draft text" in result.response_messages[0].content
    assert "update dataset and filter rows where age >= 18" in result.response_messages[0].content
    assert orchestrator_state.set_calls == []


def test_protocol_discussion_blocks_missing_selected_columns_with_update_command() -> None:
    summary = _summary_for_df(
        pd.DataFrame(
            {
                "RXASP": ["Y", "N"],
                "DIED": [1, 0],
                "AGE": [70, 65],
            }
        )
    )
    llm = _FakeLLM(
        json_outputs=[
            {
                "draft": _complete_draft(treatment_column="treated", effect_modifiers=[]),
                "next_action": "confirm",
                "assistant_message": "I can accept this draft.",
            }
        ]
    )
    node, request, orchestrator_state = _request(dataset_summary=summary, llm=llm)

    result = node.run(request=request)
    message = result.response_messages[0].content

    assert result.status == "PENDING"
    assert result.action == "NEEDS_INPUT"
    assert "treated (treatment)" in message
    assert "not current dataset columns" in message
    assert "update dataset and create treated" in message
    assert orchestrator_state.set_calls == []


def test_protocol_discussion_accepts_valid_draft_and_writes_only_causal_spec_draft() -> None:
    summary = _summary_for_df(
        pd.DataFrame(
            {
                "RXASP": ["Y", "N"],
                "DIED": [1, 0],
                "AGE": [70, 65],
                "SEX": ["F", "M"],
                "NEGCTRL": [0, 1],
            }
        )
    )
    llm = _FakeLLM(
        json_outputs=[
            {
                "draft": _complete_draft(negative_control_outcome="NEGCTRL"),
                "next_action": "confirm",
                "assistant_message": "Accepted. I stored the causal draft.",
            }
        ]
    )
    node, request, orchestrator_state = _request(
        dataset_summary=summary,
        llm=llm,
        user_message="confirm",
    )

    result = node.run(request=request)

    assert result.status == "DONE"
    assert result.action == "NONE"
    assert len(orchestrator_state.set_calls) == 1
    _, payload = orchestrator_state.set_calls[0]
    assert list(payload.keys()) == ["causal_spec_draft"]
    draft = payload["causal_spec_draft"]
    assert isinstance(draft, CausalSpecDraft)
    assert draft.treatment_column == "RXASP"
    assert draft.outcome_column == "DIED"
    assert draft.negative_control_outcome == "NEGCTRL"
    assert draft.target_population == "all rows"
    assert draft.study_type == "OBSERVATIONAL"
    assert draft.time_zero == "baseline treatment decision at cohort entry"


def test_protocol_discussion_blocks_negative_control_role_conflict() -> None:
    summary = _summary_for_df(
        pd.DataFrame(
            {
                "RXASP": ["Y", "N"],
                "DIED": [1, 0],
                "AGE": [70, 65],
            }
        )
    )
    llm = _FakeLLM(
        json_outputs=[
            {
                "draft": _complete_draft(
                    covariates=["AGE"],
                    effect_modifiers=[],
                    negative_control_outcome="AGE",
                ),
                "next_action": "confirm",
                "assistant_message": "I can accept this draft.",
            }
        ]
    )
    node, request, orchestrator_state = _request(dataset_summary=summary, llm=llm)

    result = node.run(request=request)

    assert result.status == "PENDING"
    assert "cannot be the negative-control outcome" in result.response_messages[0].content
    assert orchestrator_state.set_calls == []
