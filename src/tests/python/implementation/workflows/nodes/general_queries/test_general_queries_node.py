from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from python.domain.service.llm_service import ChatMessage, LLMConfig
from python.domain.workflows.node import NodeRequest
from python.domain.workflows.ochestrator_state import OchestratorState
from python.implementation.workflows.nodes.general_queries.general_queries_node import (
    GeneralQueriesNode,
)
from python.implementation.workflows.nodes.general_queries.general_queries_state import (
    GeneralQueriesState,
)


@dataclass
class _FakeColumnProfile:
    name: str


@dataclass
class _FakeDatasetSummary:
    n_rows: int
    profiles: list[_FakeColumnProfile]


@dataclass
class _FakeDraft:
    treatment_column: str
    outcome_column: str
    covariates: list[str]
    effect_modifiers: list[str]


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


class _FakeOrchestratorState(OchestratorState):
    def __init__(
        self,
        *,
        values: dict[str, Any],
        current_node_name: str,
        companion_names: list[str] | None = None,
    ) -> None:
        self._values = dict(values)
        self._current_node_name = current_node_name
        self._companion_names = list(companion_names or [])

    def name(self) -> str:
        return "FAKE_ORCHESTRATOR"

    def get_update_counter(self) -> int:
        return int(self._values.get("update_counter", 0))

    def set_update_counter(self, value: int) -> None:
        self._values["update_counter"] = value

    def get(self, key: str) -> Any:
        if key == "working_dataset_id":
            dataset_ids = self._values.get("working_dataset_ids") or []
            if not dataset_ids:
                return None
            return dataset_ids[-1]
        return self._values.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._values[key] = value

    def to_json_dict(self) -> dict[str, Any]:
        return dict(self._values)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "_FakeOrchestratorState":
        return cls(values=payload, current_node_name="UNKNOWN")

    @classmethod
    def init_empty(cls) -> "_FakeOrchestratorState":
        return cls(values={}, current_node_name="UNKNOWN")

    def get_current_node_name(self) -> str:
        return self._current_node_name

    def get_current_node_companion_names(self, node_name: str) -> list[str]:
        assert node_name == self._current_node_name
        return list(self._companion_names)

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
        return self.get("working_dataset_id"), False

    def get_ochestration_prompt(self) -> str:
        return ""


def _dataset_summary() -> _FakeDatasetSummary:
    return _FakeDatasetSummary(
        n_rows=60,
        profiles=[
            _FakeColumnProfile(name="age"),
            _FakeColumnProfile(name="isex"),
            _FakeColumnProfile(name="treatment"),
            _FakeColumnProfile(name="outcome"),
        ],
    )


def _causal_draft() -> _FakeDraft:
    return _FakeDraft(
        treatment_column="treatment",
        outcome_column="outcome",
        covariates=["age"],
        effect_modifiers=["isex"],
    )


def _base_orchestrator_state() -> _FakeOrchestratorState:
    dataset_id = str(uuid4())
    return _FakeOrchestratorState(
        current_node_name="DATA_COMPILATION",
        companion_names=["DATA_MANUPULATION", "DATA_STATISTICS"],
        values={
            "working_dataset_ids": [dataset_id],
            "latest_dataset_summary": _dataset_summary(),
            "protocol_discussion": "Estimate whether treatment changes outcome.",
            "protocol_cleaning_instructions": "Use grounded cleaning only.",
            "causal_spec_draft": _causal_draft(),
            "causal_spec": None,
            "data_transformation_plan": None,
            "working_dataset_frozen": False,
            "validation_issues": [],
            "is_validated": False,
            "selected_model": None,
            "selection_reasoning": None,
            "trained_model_id": None,
            "training_warnings": [],
        },
    )


def test_general_queries_node_returns_done_and_summarizes_current_stage() -> None:
    llm = _FakeLLM(json_outputs=[{"assistant_message": "Here is the current workflow summary."}])
    node = GeneralQueriesNode(llm=llm)
    orchestrator_state = _base_orchestrator_state()

    result = node.run(
        request=NodeRequest(
            user_id=uuid4(),
            conversation_id=uuid4(),
            node_state=GeneralQueriesState.init_empty(),
            orchestrator_state=orchestrator_state,
            read_only_messages_history=[ChatMessage(role="user", content="where are we now?")],
        )
    )

    assert result.status == "DONE"
    assert result.action == "NONE"
    assert result.response_messages is not None
    assert result.response_messages[0].content == "Here is the current workflow summary."

    prompt = llm.generate_json_calls[0]["user_prompt"]
    assert "Next required node: DATA_COMPILATION" in prompt
    assert "Stage 2 — Protocol discussion and causal draft accepted" in prompt
    assert "Stage 3 — Compilation, transformation planning, and validation." in prompt
    assert "Not fully accepted yet." in prompt


def test_general_queries_node_reports_accepted_stage3_and_next_model_selection() -> None:
    llm = _FakeLLM(json_outputs=[{"assistant_message": "Accepted stage summary."}])
    node = GeneralQueriesNode(llm=llm)
    orchestrator_state = _base_orchestrator_state()
    orchestrator_state._current_node_name = "MODEL_SELECTION"
    orchestrator_state._companion_names = ["DATA_STATISTICS"]
    orchestrator_state._values.update(
        {
            "causal_spec": object(),
            "data_transformation_plan": object(),
            "working_dataset_frozen": True,
            "validation_issues": [],
            "is_validated": True,
        }
    )

    result = node.run(
        request=NodeRequest(
            user_id=uuid4(),
            conversation_id=uuid4(),
            node_state=GeneralQueriesState.init_empty(),
            orchestrator_state=orchestrator_state,
            read_only_messages_history=[ChatMessage(role="user", content="what is done already?")],
        )
    )

    assert result.status == "DONE"
    assert result.action == "NONE"

    prompt = llm.generate_json_calls[0]["user_prompt"]
    assert "Next required node: MODEL_SELECTION" in prompt
    assert "Stage 3 — Compilation, transformation planning, and validation accepted." in prompt
    assert "Frozen dataset: yes" in prompt
