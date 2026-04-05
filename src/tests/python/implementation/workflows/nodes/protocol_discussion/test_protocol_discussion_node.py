from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest

from python.domain.models.errors import StateDependencyError
from python.domain.service.llm_service import ChatMessage, LLMConfig
from python.implementation.workflows.nodes.dataset.dataset_state import (
    DatasetIterationModel,
    DatasetPayloadModel,
    DatasetState,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_deps import (
    ProtocolDiscussionDeps,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import (
    ProtocolDiscussionNode,
    _DiscussionDecisionModel,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
    get_protocol_discussion_update_prompt,
    get_questions,
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


def _dataset_state(*, dataset_id: UUID, summary: DatasetSummaryModel | None) -> DatasetState:
    return DatasetState(
        DatasetPayloadModel(
            dataset_iterations=[DatasetIterationModel(dataset_id=dataset_id, summary=summary)],
        )
    )


@dataclass
class _FakeLLM:
    json_outputs: list[object] = field(default_factory=list)
    generate_json_calls: list[dict[str, object]] = field(default_factory=list)

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
        assert isinstance(next_output, schema)
        return next_output


def _messages(state: ProtocolDiscussionState) -> list[ChatMessage]:
    return list(state.messages())


def test_discussion_decision_model_requires_dataset_change_request_on_confirm() -> None:
    with pytest.raises(ValueError, match="dataset_change_request is required"):
        _DiscussionDecisionModel(
            discussion="Protocol text",
            next_action="confirm",
            assistant_message="Confirmed.",
            dataset_change_request=None,
        )


def test_discussion_decision_model_forbids_dataset_change_request_on_continue() -> None:
    with pytest.raises(ValueError, match="must be null unless next_action=confirm"):
        _DiscussionDecisionModel(
            discussion="Protocol text",
            next_action="continue",
            assistant_message="Need more detail.",
            dataset_change_request="Do not allow this here.",
        )


def test_protocol_discussion_state_init_empty_is_instantiable() -> None:
    state = ProtocolDiscussionState.init_empty()

    assert isinstance(state, ProtocolDiscussionState)
    assert state.payload.dataset_id is None
    assert state.payload.dataset_summary is None
    assert state.payload.discussion == ""
    assert state.payload.phase == "DISCUSSING"


def test_protocol_discussion_state_status_messages_error_and_roundtrip() -> None:
    summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0]}))
    discussing = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=uuid4(),
            dataset_summary=summary,
            discussion="Q/A",
            phase="DISCUSSING",
            assistant_message="Need more info",
        )
    )
    confirmed = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=uuid4(),
            dataset_summary=summary,
            discussion="Locked",
            phase="CONFIRMED",
            assistant_message="Confirmed",
            system_message="System handoff",
        )
    )
    assert discussing.status() == "PENDING"
    assert _messages(discussing) == [ChatMessage(role="assistant", content="Need more info")]
    assert discussing.error() is None

    assert confirmed.status() == "DONE"
    assert _messages(confirmed) == [
        ChatMessage(role="system", content="System handoff"),
        ChatMessage(role="assistant", content="Confirmed"),
    ]
    assert confirmed.error() is None

    restored = ProtocolDiscussionState.from_json_dict(confirmed.to_json_dict())
    assert restored.payload.model_dump(mode="json") == confirmed.payload.model_dump(mode="json")


def test_protocol_discussion_update_prompt_contract_mentions_confirm_only_and_cleaning_requirements() -> None:
    prompt = get_protocol_discussion_update_prompt()

    assert '"next_action": "continue" | "confirm"' in prompt
    assert "abort" not in prompt.lower()
    assert "instruct normalization to exactly two canonical values" in prompt
    assert "handling of unexpected labels as grounded by the discussion" in prompt
    assert "The request is for a downstream data-changing stage." in prompt


def test_protocol_discussion_deps_returns_latest_dataset_revision() -> None:
    first_summary = _summary_for_df(pd.DataFrame({"a": [1]}))
    latest_summary = _summary_for_df(pd.DataFrame({"b": [2]}))
    first_id = uuid4()
    latest_id = uuid4()
    dataset_state = DatasetState(
        DatasetPayloadModel(
            dataset_iterations=[
                DatasetIterationModel(dataset_id=first_id, summary=first_summary),
                DatasetIterationModel(dataset_id=latest_id, summary=latest_summary),
            ]
        )
    )

    deps = ProtocolDiscussionDeps.from_loaded({DatasetState.NAME: dataset_state})

    assert deps.dataset_id == latest_id
    assert deps.dataset_summary.model_dump(mode="json") == latest_summary.model_dump(mode="json")


def test_protocol_discussion_deps_fails_when_dataset_state_missing() -> None:
    with pytest.raises(StateDependencyError):
        ProtocolDiscussionDeps.from_loaded({})


def test_protocol_discussion_deps_fails_when_latest_summary_missing() -> None:
    dataset_state = _dataset_state(dataset_id=uuid4(), summary=None)

    with pytest.raises(StateDependencyError):
        ProtocolDiscussionDeps.from_loaded({DatasetState.NAME: dataset_state})


def test_protocol_discussion_node_first_run_initializes_from_latest_dataset_revision() -> None:
    dataset_id = uuid4()
    summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0]}))
    llm = _FakeLLM(
        json_outputs=[
            _DiscussionDecisionModel(
                discussion="Updated discussion against current dataset",
                next_action="continue",
                assistant_message="Need one more clarification before we confirm the protocol.",
                dataset_change_request=None,
            )
        ],
    )
    node = ProtocolDiscussionNode(llm=llm)

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary)},
        messages_history=[ChatMessage(role="user", content="Here is the protocol.")],
        state=ProtocolDiscussionState.init_empty(),
    )

    assert isinstance(result, ProtocolDiscussionState)
    assert result.payload.dataset_id == dataset_id
    assert result.payload.dataset_summary is not None
    assert result.payload.discussion == "Updated discussion against current dataset"
    assert result.payload.phase == "DISCUSSING"
    assert result.status() == "PENDING"
    assert result.payload.assistant_message == "Need one more clarification before we confirm the protocol."
    assert result.payload.system_message is None


def test_protocol_discussion_node_without_user_message_returns_initial_prompt_without_llm_call() -> None:
    dataset_id = uuid4()
    summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0]}))
    llm = _FakeLLM()
    node = ProtocolDiscussionNode(llm=llm)

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary)},
        messages_history=None,
        state=ProtocolDiscussionState.init_empty(),
    )

    assert result.payload.phase == "DISCUSSING"
    assert result.payload.assistant_message is not None
    assert "Let’s define the protocol carefully" in result.payload.assistant_message
    assert result.payload.system_message is None
    assert llm.generate_json_calls == []


def test_protocol_discussion_node_confirms_discussion_and_emits_cleaning_system_message() -> None:
    dataset_id = uuid4()
    summary = _summary_for_df(
        pd.DataFrame(
            {
                "treatment": ["drug", "control"],
                "outcome": [1.0, 0.0],
                "age": [50, 60],
                "sex": ["f", "m"],
            }
        )
    )
    llm = _FakeLLM(
        json_outputs=[
            _DiscussionDecisionModel(
                discussion="Confirmed protocol discussion",
                next_action="confirm",
                assistant_message="The protocol discussion is now confirmed. I will hand off the required data cleaning and normalization steps next.",
                dataset_change_request=(
                    "This is a data-changing request. Preserve treatment, outcome, age, and sex. "
                    "Normalize treatment to exactly two canonical values: drug and control. "
                    "Normalize binary outcome to exactly two canonical values. "
                    "Filter rows outside the confirmed cohort eligibility only when the rule is grounded."
                ),
            )
        ],
    )
    node = ProtocolDiscussionNode(llm=llm)
    state = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=dataset_id,
            dataset_summary=summary,
            discussion="Existing discussion",
            phase="DISCUSSING",
            assistant_message="Old prompt",
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary)},
        messages_history=[ChatMessage(role="user", content="Yes, I confirm this protocol.")],
        state=state,
    )

    assert isinstance(result, ProtocolDiscussionState)
    assert result.payload.discussion == "Confirmed protocol discussion"
    assert result.payload.phase == "CONFIRMED"
    assert result.status() == "DONE"
    assert result.payload.system_message is not None
    assert result.payload.system_message.startswith("PROTOCOL_DISCUSSION_CONFIRMED")
    assert "This is a data-changing request." in result.payload.system_message
    assert "Normalize treatment to exactly two canonical values" in result.payload.system_message
    assert _messages(result) == [
        ChatMessage(role="system", content=result.payload.system_message),
        ChatMessage(role="assistant", content=result.payload.assistant_message or ""),
    ]


def test_protocol_discussion_node_unchanged_dataset_preserves_discussion_source_and_continues() -> None:
    dataset_id = uuid4()
    summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0]}))
    llm = _FakeLLM(
        json_outputs=[
            _DiscussionDecisionModel(
                discussion="Refined discussion",
                next_action="continue",
                assistant_message="I updated the protocol discussion. One remaining issue is separating covariates from effect modifiers clearly.",
                dataset_change_request=None,
            )
        ],
    )
    node = ProtocolDiscussionNode(llm=llm)
    state = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=dataset_id,
            dataset_summary=summary,
            discussion="Existing discussion",
            phase="DISCUSSING",
            assistant_message="Old prompt",
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary)},
        messages_history=[ChatMessage(role="user", content="The covariates should be age and sex.")],
        state=state,
    )

    update_payload = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))
    assert update_payload["protocol_discussion"] == "Existing discussion"
    assert isinstance(result, ProtocolDiscussionState)
    assert result.payload.discussion == "Refined discussion"
    assert result.payload.phase == "DISCUSSING"
    assert result.status() == "PENDING"
    assert result.payload.assistant_message.startswith("I updated the protocol discussion.")
    assert result.error() is None


def test_protocol_discussion_node_changed_dataset_resets_discussion_and_prefixes_message() -> None:
    old_dataset_id = uuid4()
    new_dataset_id = uuid4()
    old_summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0]}))
    new_summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0], "age": [50]}))
    llm = _FakeLLM(
        json_outputs=[
            _DiscussionDecisionModel(
                discussion="Restarted discussion",
                next_action="continue",
                assistant_message="Please answer the updated dataset-specific protocol questions.",
                dataset_change_request=None,
            )
        ],
    )
    node = ProtocolDiscussionNode(llm=llm)
    state = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=old_dataset_id,
            dataset_summary=old_summary,
            discussion="Stale discussion about old data",
            phase="CONFIRMED",
            assistant_message="Old ready message",
            system_message="Old system message",
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={DatasetState.NAME: _dataset_state(dataset_id=new_dataset_id, summary=new_summary)},
        messages_history=[ChatMessage(role="user", content="Use the latest dataset now.")],
        state=state,
    )

    update_payload = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))
    assert update_payload["protocol_discussion"] == "\n".join(get_questions())
    assert isinstance(result, ProtocolDiscussionState)
    assert result.payload.dataset_id == new_dataset_id
    assert result.payload.discussion == "Restarted discussion"
    assert result.payload.phase == "DISCUSSING"
    assert result.status() == "PENDING"
    assert result.payload.assistant_message is not None
    assert result.payload.assistant_message.startswith(
        "The active dataset changed, so I reset protocol discussion against the latest data."
    )
    assert result.payload.system_message is None


def test_protocol_discussion_node_changed_dataset_without_user_message_resets_and_returns_prompt() -> None:
    old_dataset_id = uuid4()
    new_dataset_id = uuid4()
    old_summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0]}))
    new_summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0], "age": [50]}))
    node = ProtocolDiscussionNode(llm=_FakeLLM())
    state = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=old_dataset_id,
            dataset_summary=old_summary,
            discussion="Stale discussion",
            phase="CONFIRMED",
            assistant_message="Old ready message",
            system_message="Old system message",
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={DatasetState.NAME: _dataset_state(dataset_id=new_dataset_id, summary=new_summary)},
        messages_history=None,
        state=state,
    )

    assert result.payload.dataset_id == new_dataset_id
    assert result.payload.phase == "DISCUSSING"
    assert result.payload.system_message is None
    assert result.payload.assistant_message is not None
    assert result.payload.assistant_message.startswith(
        "The active dataset changed, so I reset protocol discussion against the latest data."
    )


def test_protocol_discussion_node_update_failure_returns_pending_retry_state() -> None:
    dataset_id = uuid4()
    summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0]}))
    llm = _FakeLLM(json_outputs=[RuntimeError("llm exploded")])
    node = ProtocolDiscussionNode(llm=llm)
    state = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=dataset_id,
            dataset_summary=summary,
            discussion="Existing discussion",
            phase="DISCUSSING",
            assistant_message="Old",
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary)},
        messages_history=[ChatMessage(role="user", content="Continue")],
        state=state,
    )

    assert isinstance(result, ProtocolDiscussionState)
    assert result.payload.dataset_id == dataset_id
    assert result.payload.discussion == "Existing discussion"
    assert result.payload.phase == "DISCUSSING"
    assert result.payload.assistant_message == "Protocol discussion update failed. Please try again."
    assert result.status() == "PENDING"


def test_protocol_discussion_node_infeasible_case_stays_pending_with_explanation() -> None:
    dataset_id = uuid4()
    summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0]}))
    llm = _FakeLLM(
        json_outputs=[
            _DiscussionDecisionModel(
                discussion="Discussion updated with infeasibility note",
                next_action="continue",
                assistant_message=(
                    "I’m sorry, but this protocol cannot proceed as a survival-style design with the "
                    "current dataset because there is no grounded time support. If you want to continue, "
                    "either switch to a defensible snapshot design or provide time variables that support follow-up."
                ),
                dataset_change_request=None,
            )
        ],
    )
    node = ProtocolDiscussionNode(llm=llm)
    state = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=dataset_id,
            dataset_summary=summary,
            discussion="Existing discussion",
            phase="DISCUSSING",
            assistant_message="Old",
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary)},
        messages_history=[ChatMessage(role="user", content="Proceed even though survival time is unavailable.")],
        state=state,
    )

    assert isinstance(result, ProtocolDiscussionState)
    assert result.payload.phase == "DISCUSSING"
    assert result.status() == "PENDING"
    assert "I’m sorry" in (result.payload.assistant_message or "")
    assert "cannot proceed" in (result.payload.assistant_message or "")
    assert result.error() is None
