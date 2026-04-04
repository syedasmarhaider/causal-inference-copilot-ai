from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest

from python.domain.models.errors import NodeExecutionError, StateDependencyError
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.domain.workflows.state import StateMessage
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
    _DiscussionAndGateModel,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_prompts import (
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
    generate_outputs: list[object] = field(default_factory=list)
    generate_json_calls: list[dict[str, object]] = field(default_factory=list)
    generate_calls: list[dict[str, object]] = field(default_factory=list)

    def generate_json(
        self,
        *,
        schema: type[_DiscussionAndGateModel],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> _DiscussionAndGateModel:
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
        assert isinstance(next_output, _DiscussionAndGateModel)
        return next_output

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


@dataclass
class _FakeToolFactory:
    profiling_tool: DatasetProfilingTool
    calls: list[str] = field(default_factory=list)

    def get_tool(self, name: str) -> object:
        self.calls.append(name)
        if name == DatasetProfilingTool.NAME:
            return self.profiling_tool
        raise KeyError(name)


def test_protocol_discussion_state_init_empty_is_instantiable() -> None:
    state = ProtocolDiscussionState.init_empty()

    assert isinstance(state, ProtocolDiscussionState)
    assert state.payload.dataset_id is None
    assert state.payload.dataset_summary is None
    assert state.payload.discussion == ""
    assert state.payload.readiness == "PENDING"


def test_protocol_discussion_state_status_message_error_and_roundtrip() -> None:
    summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0]}))
    pending = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=uuid4(),
            dataset_summary=summary,
            discussion="Q/A",
            readiness="PENDING",
            node_message="Need more info",
        )
    )
    ready = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=uuid4(),
            dataset_summary=summary,
            discussion="Locked",
            readiness="READY",
            node_message="Ready",
        )
    )
    abort = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=uuid4(),
            dataset_summary=summary,
            discussion="Broken",
            readiness="ABORT",
            node_message="Cannot proceed",
            error_message="Cannot proceed",
        )
    )

    assert pending.status == "PENDING"
    assert pending.message == StateMessage(txt_message="Need more info", action="NEEDS_INPUT")
    assert pending.error is None

    assert ready.status == "DONE"
    assert ready.message == StateMessage(txt_message="Ready", action="NONE")

    assert abort.status == "ABORTED"
    assert isinstance(abort.error, NodeExecutionError)
    assert abort.error.error == "Cannot proceed"

    restored = ProtocolDiscussionState.from_json_dict(ready.to_json_dict())
    assert restored.payload.model_dump(mode="json") == ready.payload.model_dump(mode="json")


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
            _DiscussionAndGateModel(
                protocol_discussion="Updated discussion against current dataset",
                readiness="READY",
            )
        ],
        generate_outputs=["Need one more clarification."],
    )
    node = ProtocolDiscussionNode(llm=llm)

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        tool_factory=_FakeToolFactory(profiling_tool=DatasetProfilingTool()),
        previous_state_dependencies={DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary)},
        messages_history=[ChatMessage(role="user", content="Here is the protocol.")],
        state=ProtocolDiscussionState.init_empty(),
    )

    assert isinstance(result, ProtocolDiscussionState)
    assert result.payload.dataset_id == dataset_id
    assert result.payload.dataset_summary is not None
    assert result.payload.discussion == "Updated discussion against current dataset"
    assert result.payload.readiness == "PENDING"
    assert result.status == "PENDING"
    assert result.payload.node_message == "Need one more clarification."


def test_protocol_discussion_node_unchanged_dataset_preserves_discussion_source_and_allows_ready() -> None:
    dataset_id = uuid4()
    summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0]}))
    llm = _FakeLLM(
        json_outputs=[_DiscussionAndGateModel(protocol_discussion="Refined discussion", readiness="READY")],
        generate_outputs=["Ready to proceed."],
    )
    node = ProtocolDiscussionNode(llm=llm)
    state = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=dataset_id,
            dataset_summary=summary,
            discussion="Existing discussion",
            readiness="PENDING",
            node_message="Old prompt",
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        tool_factory=_FakeToolFactory(profiling_tool=DatasetProfilingTool()),
        previous_state_dependencies={DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary)},
        messages_history=[ChatMessage(role="user", content="Yes, this protocol is correct.")],
        state=state,
    )

    update_payload = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))
    assert update_payload["protocol_discussion"] == "Existing discussion"
    assert isinstance(result, ProtocolDiscussionState)
    assert result.payload.discussion == "Refined discussion"
    assert result.payload.readiness == "READY"
    assert result.status == "DONE"
    assert result.payload.node_message == "Ready to proceed."
    assert result.error is None


def test_protocol_discussion_node_changed_dataset_resets_discussion_and_forces_pending() -> None:
    old_dataset_id = uuid4()
    new_dataset_id = uuid4()
    old_summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0]}))
    new_summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0], "age": [50]}))
    llm = _FakeLLM(
        json_outputs=[_DiscussionAndGateModel(protocol_discussion="Restarted discussion", readiness="READY")],
        generate_outputs=["Please answer the updated dataset-specific questions."],
    )
    node = ProtocolDiscussionNode(llm=llm)
    state = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=old_dataset_id,
            dataset_summary=old_summary,
            discussion="Stale discussion about old data",
            readiness="READY",
            node_message="Old ready message",
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        tool_factory=_FakeToolFactory(profiling_tool=DatasetProfilingTool()),
        previous_state_dependencies={DatasetState.NAME: _dataset_state(dataset_id=new_dataset_id, summary=new_summary)},
        messages_history=[ChatMessage(role="user", content="Use the latest dataset now.")],
        state=state,
    )

    update_payload = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))
    assert update_payload["protocol_discussion"] == "\n".join(get_questions())
    assert isinstance(result, ProtocolDiscussionState)
    assert result.payload.dataset_id == new_dataset_id
    assert result.payload.discussion == "Restarted discussion"
    assert result.payload.readiness == "PENDING"
    assert result.status == "PENDING"
    assert result.payload.node_message is not None
    assert result.payload.node_message.startswith(
        "The active dataset changed, so I reset protocol discussion against the latest data."
    )


def test_protocol_discussion_node_gate_failure_returns_retry_state() -> None:
    dataset_id = uuid4()
    summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0]}))
    llm = _FakeLLM(json_outputs=[RuntimeError("llm exploded")])
    node = ProtocolDiscussionNode(llm=llm)
    state = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=dataset_id,
            dataset_summary=summary,
            discussion="Existing discussion",
            readiness="PENDING",
            node_message="Old",
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        tool_factory=_FakeToolFactory(profiling_tool=DatasetProfilingTool()),
        previous_state_dependencies={DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary)},
        messages_history=[ChatMessage(role="user", content="Continue")],
        state=state,
    )

    assert isinstance(result, ProtocolDiscussionState)
    assert result.payload.dataset_id == dataset_id
    assert result.payload.discussion == "Existing discussion"
    assert result.payload.readiness == "PENDING"
    assert result.payload.node_message == "Protocol discussion update failed. Retrying..."
    assert result.payload.error_message is not None
    assert "llm exploded" in result.payload.error_message
    assert result.status == "ABORTED"


def test_protocol_discussion_node_abort_sets_error() -> None:
    dataset_id = uuid4()
    summary = _summary_for_df(pd.DataFrame({"treatment": ["drug"], "outcome": [1.0]}))
    llm = _FakeLLM(
        json_outputs=[_DiscussionAndGateModel(protocol_discussion="Aborted discussion", readiness="ABORT")],
        generate_outputs=["This protocol cannot proceed."],
    )
    node = ProtocolDiscussionNode(llm=llm)
    state = ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=dataset_id,
            dataset_summary=summary,
            discussion="Existing discussion",
            readiness="PENDING",
            node_message="Old",
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        tool_factory=_FakeToolFactory(profiling_tool=DatasetProfilingTool()),
        previous_state_dependencies={DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary)},
        messages_history=[ChatMessage(role="user", content="Proceed even though it is impossible.")],
        state=state,
    )

    assert isinstance(result, ProtocolDiscussionState)
    assert result.payload.readiness == "ABORT"
    assert result.status == "ABORTED"
    assert result.payload.error_message == "This protocol cannot proceed."
    assert isinstance(result.error, NodeExecutionError)
    assert result.error.error == "This protocol cannot proceed."
