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
    _discussion_with_confirmed_unknown_category_decision,
    _DiscussionDecisionModel,
    _identifier_column_candidates,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionPayloadModel,
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
    generate_outputs: list[Any] = field(default_factory=list)
    generate_json_calls: list[dict[str, Any]] = field(default_factory=list)
    generate_calls: list[dict[str, Any]] = field(default_factory=list)

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

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
    ) -> ChatMessage:
        self.generate_calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "config": config,
                "history": history,
            }
        )
        if self.generate_outputs:
            next_output = self.generate_outputs.pop(0)
            if isinstance(next_output, str):
                return ChatMessage(role="assistant", content=next_output)
            return next_output
        return ChatMessage(role="assistant", content="Please resolve the protocol blocker.")


@dataclass
class _FakeDataRepo:
    dataframe: pd.DataFrame

    def get_csv_data(self, **_: Any) -> pd.DataFrame:
        return self.dataframe.head(1)


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


def test_protocol_discussion_confirmation_blocker_message_is_not_truncated() -> None:
    dataframe = pd.DataFrame(
        {
            "RXASP": ["Y", "N", "Y"],
            "DIED": ["Y", "N", "N"],
            "race": ["white", "black", "other"],
        }
    )
    summary = _summary_for_df(dataframe)
    discussion = "\n".join(
        [
            "1) Causal question: What is the effect of aspirin allocation on death?",
            "6) Treatment/exposure definition: RXASP, treated Y, control N.",
            "8) Outcome specification: DIED.",
            "11) Effect modifiers / heterogeneity features (X, optional): race.",
            "14) Treatment/outcome data-quality decisions: Treatment and outcome are complete; use as-is.",
            "15) Baseline feature preparation decisions: UNCLEAR",
            "16) Negative-control outcome (optional): null.",
            "17) Identifier column (optional): auto_id.",
        ]
    )
    long_blocker_message = (
        "I have reviewed your protocol draft and the dataset metadata. "
        "We have one clarification needed regarding your baseline features before we can finalize "
        "the causal specification.\n\n"
        "Issue: Handling of the 'other' category in the 'race' variable\n\n"
        "You have identified race as an effect modifier. The dataset shows three categories for "
        'this column: "white", "black", and "other". In causal modeling, we must have a '
        "deterministic cleaning decision for this category before compilation can continue. "
        "Please confirm whether to keep 'other' as its own level, merge it into another group, "
        "or use another explicit handling rule."
    )
    llm = _FakeLLM(
        json_outputs=[
            {"action": "confirm", "assistant_message": "Confirmed."},
            {
                "id_col": "auto_id",
                "treatment_column": "RXASP",
                "outcome_column": "DIED",
                "negative_control_outcome": None,
                "covariates": [],
                "effect_modifiers": ["race"],
            },
        ],
        generate_outputs=[long_blocker_message],
    )
    node = ProtocolDiscussionNode(
        llm=llm,  # type: ignore[arg-type]
        data_repo=_FakeDataRepo(dataframe=dataframe),  # type: ignore[arg-type]
    )
    orchestrator_state = _FakeOrchestratorState(dataset_summary=summary)
    request = NodeRequest(
        user_id=uuid4(),
        conversation_id=uuid4(),
        node_state=ProtocolDiscussionState(
            ProtocolDiscussionPayloadModel(
                dataset_id=orchestrator_state.get("working_dataset_id"),
                discussion=discussion,
                phase="REVIEW_READY",
                pending_dataset_change_request="Keep confirmed protocol columns only.",
                assistant_message="Please confirm this protocol.",
            )
        ),
        orchestrator_state=orchestrator_state,
        read_only_messages_history=[ChatMessage(role="user", content="yes confirm")],
    )

    result = node.run(request=request)

    assert result.status == "PENDING"
    assert result.action == "NEEDS_INPUT"
    assert result.response_messages[0].content == long_blocker_message
    assert result.response_messages[0].content.endswith("explicit handling rule.")
    assert len(result.response_messages[0].content) > 500


def test_protocol_discussion_blocks_negative_control_effect_modifier_conflict() -> None:
    summary = _summary_for_df(
        pd.DataFrame(
            {
                "RXASP": ["Y", "N"],
                "DIED": ["Y", "N"],
                "RSBP": [140, 160],
                "AGE": [72, 67],
            }
        )
    )
    discussion = "\n".join(
        [
            "1) Causal question: What is the effect of aspirin allocation on death?",
            "6) Treatment/exposure definition: RXASP, treated Y, control N.",
            "8) Outcome specification: DIED.",
            "11) Effect modifiers / heterogeneity features (X, optional): RSBP and AGE.",
            "14) Treatment/outcome data-quality decisions: Treatment and outcome are complete; use as-is.",
            "15) Baseline feature preparation decisions: Baseline features are complete; use as-is.",
            "16) Negative-control outcome (optional): RSBP.",
            "17) Identifier column (optional): auto_id.",
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _DiscussionDecisionModel(
                discussion=discussion,
                next_action="confirm",
                assistant_message="Please review this protocol.",
                dataset_change_request="Keep confirmed protocol columns only.",
            ),
            {
                "id_col": "auto_id",
                "treatment_column": "RXASP",
                "outcome_column": "DIED",
                "negative_control_outcome": None,
                "covariates": [],
                "effect_modifiers": ["RSBP", "AGE"],
            },
        ]
    )
    node, request = _request(dataset_summary=summary, llm=llm)

    result = node.run(request=request)
    message = result.response_messages[0].content

    assert result.status == "PENDING"
    assert result.action == "NEEDS_INPUT"
    assert result.new_node_state.payload.phase == "DISCUSSING"
    assert "RSBP" in message
    assert "negative-control outcome" in message
    assert "effect modifier" in message
    assert "choose one role" in message
    assert len(llm.generate_json_calls) == 2


def test_protocol_discussion_allows_distinct_negative_control_outcome() -> None:
    summary = _summary_for_df(
        pd.DataFrame(
            {
                "RXASP": ["Y", "N"],
                "DIED": ["Y", "N"],
                "RSBP": [140, 160],
                "NEGCTRL": [0.1, 0.2],
            }
        )
    )
    discussion = "\n".join(
        [
            "1) Causal question: What is the effect of aspirin allocation on death?",
            "6) Treatment/exposure definition: RXASP, treated Y, control N.",
            "8) Outcome specification: DIED.",
            "11) Effect modifiers / heterogeneity features (X, optional): RSBP.",
            "14) Treatment/outcome data-quality decisions: Treatment and outcome are complete; use as-is.",
            "15) Baseline feature preparation decisions: Baseline features are complete; use as-is.",
            "16) Negative-control outcome (optional): NEGCTRL.",
            "17) Identifier column (optional): auto_id.",
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _DiscussionDecisionModel(
                discussion=discussion,
                next_action="confirm",
                assistant_message="Please review this protocol.",
                dataset_change_request="Keep confirmed protocol columns only.",
            ),
            {
                "id_col": "auto_id",
                "treatment_column": "RXASP",
                "outcome_column": "DIED",
                "negative_control_outcome": "NEGCTRL",
                "covariates": [],
                "effect_modifiers": ["RSBP"],
            },
            {"assistant_message": "Review ready."},
        ]
    )
    node, request = _request(dataset_summary=summary, llm=llm)

    result = node.run(request=request)

    assert result.status == "PENDING"
    assert result.action == "NEEDS_INPUT"
    assert result.new_node_state.payload.phase == "REVIEW_READY"
    assert result.response_messages[0].content == "Review ready."


def test_protocol_discussion_allows_null_negative_control_outcome() -> None:
    summary = _summary_for_df(
        pd.DataFrame(
            {
                "RXASP": ["Y", "N"],
                "DIED": ["Y", "N"],
                "RSBP": [140, 160],
            }
        )
    )
    discussion = "\n".join(
        [
            "1) Causal question: What is the effect of aspirin allocation on death?",
            "6) Treatment/exposure definition: RXASP, treated Y, control N.",
            "8) Outcome specification: DIED.",
            "11) Effect modifiers / heterogeneity features (X, optional): RSBP.",
            "14) Treatment/outcome data-quality decisions: Treatment and outcome are complete; use as-is.",
            "15) Baseline feature preparation decisions: Baseline features are complete; use as-is.",
            "16) Negative-control outcome (optional): null.",
            "17) Identifier column (optional): auto_id.",
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _DiscussionDecisionModel(
                discussion=discussion,
                next_action="confirm",
                assistant_message="Please review this protocol.",
                dataset_change_request="Keep confirmed protocol columns only.",
            ),
            {
                "id_col": "auto_id",
                "treatment_column": "RXASP",
                "outcome_column": "DIED",
                "negative_control_outcome": None,
                "covariates": [],
                "effect_modifiers": ["RSBP"],
            },
            {"assistant_message": "Review ready without negative control."},
        ]
    )
    node, request = _request(dataset_summary=summary, llm=llm)

    result = node.run(request=request)

    assert result.status == "PENDING"
    assert result.action == "NEEDS_INPUT"
    assert result.new_node_state.payload.phase == "REVIEW_READY"
    assert result.response_messages[0].content == "Review ready without negative control."
