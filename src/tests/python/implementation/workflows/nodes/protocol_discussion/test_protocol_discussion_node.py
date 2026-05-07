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
    _DiscussionDecisionModel,
    _discussion_with_confirmed_unknown_category_decision,
    _identifier_column_candidates,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
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

    def name(self) -> str:
        return "FAKE_ORCHESTRATOR"

    def get_update_counter(self) -> int:
        return int(self._values.get("update_counter", 0))

    def set_update_counter(self, value: int) -> None:
        self._values["update_counter"] = value

    def get(self, key: str) -> Any:
        return self._values.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._values[key] = value

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
    def from_json_dict(cls, payload: dict[str, Any]) -> "_FakeOrchestratorState":
        instance = cls(dataset_summary=payload["latest_dataset_summary"])
        instance._values = dict(payload)
        return instance

    @classmethod
    def init_empty(cls) -> "_FakeOrchestratorState":
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
) -> tuple[ProtocolDiscussionNode, NodeRequest]:
    node = ProtocolDiscussionNode(llm=llm)
    request = NodeRequest(
        user_id=uuid4(),
        conversation_id=uuid4(),
        node_state=ProtocolDiscussionState.init_empty(),
        orchestrator_state=_FakeOrchestratorState(dataset_summary=dataset_summary),
        read_only_messages_history=[ChatMessage(role="user", content="Please start the protocol.")],
    )
    return node, request


def test_protocol_discussion_node_passes_identifier_candidate_in_update_payload() -> None:
    summary = _summary_for_df(
        pd.DataFrame(
            {
                "patient_id": ["p1", "p2"],
                "treatment": ["drug", "control"],
                "outcome": [1.0, 0.0],
            }
        )
    )
    llm = _FakeLLM(
        json_outputs=[
            _DiscussionDecisionModel(
                discussion="Updated protocol discussion",
                next_action="continue",
                assistant_message="Please confirm the suggested identifier column or correct it.",
                dataset_change_request=None,
            )
        ]
    )
    node, request = _request(dataset_summary=summary, llm=llm)

    result = node.run(request=request)
    update_payload = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))

    assert result.status == "PENDING"
    assert result.action == "NEEDS_INPUT"
    assert update_payload["identifier_column_candidates"] == ["patient_id"]
    assert update_payload["suggested_identifier_column"] == "patient_id"


def test_protocol_discussion_node_uses_empty_identifier_suggestions_when_none_exist() -> None:
    summary = _summary_for_df(
        pd.DataFrame(
            {
                "age": [50, 60],
                "sex": ["f", "m"],
                "treatment": ["drug", "control"],
                "outcome": [1.0, 0.0],
            }
        )
    )
    llm = _FakeLLM(
        json_outputs=[
            _DiscussionDecisionModel(
                discussion="Updated protocol discussion",
                next_action="continue",
                assistant_message="If there is no real identifier column, we can use auto_id.",
                dataset_change_request=None,
            )
        ]
    )
    node, request = _request(dataset_summary=summary, llm=llm)

    _ = node.run(request=request)
    update_payload = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))

    assert update_payload["identifier_column_candidates"] == []
    assert update_payload["suggested_identifier_column"] is None


def test_identifier_column_candidates_preserve_deterministic_order() -> None:
    summary = _summary_for_df(
        pd.DataFrame(
            {
                "encounter_id": ["e1", "e2"],
                "patient_id": ["p1", "p2"],
                "visit_id": ["v1", "v2"],
                "row_id": ["r1", "r2"],
                "treatment": ["drug", "control"],
                "outcome": [1.0, 0.0],
            }
        )
    )

    assert _identifier_column_candidates(summary) == [
        "encounter_id",
        "patient_id",
        "visit_id",
    ]


def test_fallback_review_summary_includes_identifier_line_from_q17() -> None:
    discussion = "\n".join(
        [
            "1) Causal question: What is the effect of treatment on outcome?",
            "14) Treatment/outcome data-quality decisions: Keep grounded values only.",
            "15) Baseline feature preparation decisions: None.",
            "16) Negative-control outcome (optional): null.",
            "17) Identifier column (optional): use auto_id.",
        ]
    )

    summary = ProtocolDiscussionNode._fallback_review_summary(discussion)

    assert "Identifier handling" in summary
    assert "auto_id" in summary
    assert "Please confirm this protocol" in summary


def test_confirmed_unknown_category_blocker_is_written_to_discussion() -> None:
    discussion = "\n".join(
        [
            "1) Causal question: What is the effect of treatment on outcome?",
            "15) Baseline feature preparation decisions: UNCLEAR",
        ]
    )
    updated = _discussion_with_confirmed_unknown_category_decision(
        protocol_discussion=discussion,
        previous_assistant_message=(
            "I need confirmation for selected effect modifiers with Unknown values. "
            "Should Unknown be kept as its own category?"
        ),
        latest_user_message="yes I confirm that",
    )

    assert "Keep Unknown and unknown-like categories as their own category" in updated
