from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

from python.domain.models.errors import NodeExecutionError
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.domain.workflows.route import NextDecision
from python.domain.workflows.state import State, StateMessage
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import (
    CleanProtocolState,
)
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainState
from python.implementation.workflows.nodes.noop_done.noop_done_state import NoopDoneState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)
from python.implementation.workflows.router.llm_assisted_router import (
    LLMAssistedRouterRouter,
    build_state_classes_by_name,
    get_node_name_with_description,
    init_next_state_names,
)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class _FakeState(State):
    state_name: str
    state_status: str
    state_error: NodeExecutionError | None = None

    @property
    def name(self) -> str:
        return self.state_name

    @property
    def status(self) -> str:
        return self.state_status

    @property
    def message(self) -> StateMessage:
        return StateMessage(txt_message="router-test", action="NONE")

    @property
    def error(self) -> NodeExecutionError | None:
        return self.state_error

    def pre_required_states_names(self) -> Sequence[str]:
        return []

    def to_json_dict(self) -> dict[str, Any]:
        return {"name": self.state_name, "status": self.state_status}

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> _FakeState:
        return cls(
            state_name=str(payload.get("name", "UNKNOWN")),
            state_status=str(payload.get("status", "PENDING")),
        )

    @classmethod
    def init_empty(cls) -> _FakeState:
        return cls(state_name="EMPTY", state_status="PENDING")


@dataclass
class _FakeLLM:
    decisions: list[NextDecision] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
    ) -> LLMResponse:
        raise AssertionError("generate should not be called")

    def generate_json(
        self,
        *,
        schema: type[T],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> T:
        self.calls.append(
            {
                "schema": schema,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "config": config,
                "history": list(history or []),
                "max_attempts": max_attempts,
            }
        )
        if not self.decisions:
            raise AssertionError("No fake decision configured")
        return self.decisions.pop(0)  # type: ignore[return-value]


def _messages(n: int) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=f"m-{i}") for i in range(n)]


def test_router_initial_and_done_names() -> None:
    router = LLMAssistedRouterRouter(llm=_FakeLLM())

    assert router.get_initial_state_name() == LoadDatasetState.NAME
    assert router.get_done_state_name() == NoopDoneState.NAME


def test_get_next_state_names_returns_remaining_chain() -> None:
    router = LLMAssistedRouterRouter(llm=_FakeLLM())

    chain = router.get_next_state_names(ProtocolDiscussionState.NAME)

    assert chain[0] == CleanProtocolState.NAME
    assert chain[-1] == NoopDoneState.NAME
    assert NoopDoneState.NAME not in router.get_next_state_names(NoopDoneState.NAME)


def test_decide_next_handles_none_pending_and_done_states() -> None:
    router = LLMAssistedRouterRouter(llm=_FakeLLM())

    none_decision = router.decide_next(current_state=None, messages_history=[])
    assert none_decision.state_name == LoadDatasetState.NAME

    pending_state = _FakeState(state_name=ProtocolDiscussionState.NAME, state_status="PENDING")
    pending_decision = router.decide_next(current_state=pending_state, messages_history=[])
    assert pending_decision.state_name == ProtocolDiscussionState.NAME

    done_state = _FakeState(state_name=ProtocolDiscussionState.NAME, state_status="DONE")
    done_decision = router.decide_next(current_state=done_state, messages_history=[])
    assert done_decision.state_name == CleanProtocolState.NAME

    final_done = _FakeState(state_name=NoopDoneState.NAME, state_status="DONE")
    final_decision = router.decide_next(current_state=final_done, messages_history=[])
    assert final_decision.state_name == NoopDoneState.NAME


def test_decide_next_raises_when_done_state_has_no_mapping() -> None:
    router = LLMAssistedRouterRouter(llm=_FakeLLM())
    router._next_state_names_map = {NoopDoneState.NAME: None}  # noqa: SLF001 - explicit branch test

    with pytest.raises(ValueError, match=r"no next state defined"):
        router.decide_next(
            current_state=_FakeState(state_name=ProtocolDiscussionState.NAME, state_status="DONE"),
            messages_history=[],
        )


def test_decide_next_aborted_uses_llm_and_truncates_history() -> None:
    llm = _FakeLLM(
        decisions=[
            NextDecision(
                state_name=CleanProtocolState.NAME,
                router_message_for_node="retry from clean protocol",
            )
        ]
    )
    router = LLMAssistedRouterRouter(llm=llm)

    aborted = _FakeState(
        state_name=ModelTrainState.NAME,
        state_status="ABORTED",
        state_error=NodeExecutionError(ModelTrainState.NAME, "fit failed"),
    )

    decision = router.decide_next(current_state=aborted, messages_history=_messages(15))

    assert decision.state_name == CleanProtocolState.NAME
    assert decision.delete_next_states_names == router.get_next_state_names(CleanProtocolState.NAME)
    assert len(llm.calls) == 1
    assert len(llm.calls[0]["history"]) == 10
    assert llm.calls[0]["schema"] is NextDecision
    assert llm.calls[0]["config"].model == "basic"


def test_decide_next_aborted_rejects_empty_choice() -> None:
    llm = _FakeLLM(decisions=[NextDecision(state_name="", router_message_for_node="no choice")])
    router = LLMAssistedRouterRouter(llm=llm)

    aborted = _FakeState(state_name=ModelTrainState.NAME, state_status="ABORTED")

    with pytest.raises(ValueError, match=r"empty/null"):
        router.decide_next(current_state=aborted, messages_history=[])


def test_decide_next_aborted_rejects_invalid_choice() -> None:
    llm = _FakeLLM(decisions=[NextDecision(state_name="NOT_A_STATE", router_message_for_node="bad")])
    router = LLMAssistedRouterRouter(llm=llm)

    aborted = _FakeState(state_name=ModelTrainState.NAME, state_status="ABORTED")

    with pytest.raises(ValueError, match=r"selected invalid state"):
        router.decide_next(current_state=aborted, messages_history=[])


def test_decide_next_aborted_rejects_non_previous_choice() -> None:
    llm = _FakeLLM(decisions=[NextDecision(state_name=NoopDoneState.NAME, router_message_for_node="bad")])
    router = LLMAssistedRouterRouter(llm=llm)

    aborted = _FakeState(state_name=ModelTrainState.NAME, state_status="ABORTED")

    with pytest.raises(ValueError, match=r"not a previous state"):
        router.decide_next(current_state=aborted, messages_history=[])


def test_decide_next_raises_on_unexpected_status() -> None:
    router = LLMAssistedRouterRouter(llm=_FakeLLM())

    weird = _FakeState(state_name=ProtocolDiscussionState.NAME, state_status="UNKNOWN")

    with pytest.raises(ValueError, match=r"unexpected status"):
        router.decide_next(current_state=weird, messages_history=[])


def test_state_and_node_maps_are_well_formed() -> None:
    state_map = build_state_classes_by_name()
    next_map = init_next_state_names()
    node_info_map = get_node_name_with_description()

    assert LoadDatasetState.NAME in state_map
    assert NoopDoneState.NAME in state_map
    assert next_map[NoopDoneState.NAME] is None
    assert next_map[LoadDatasetState.NAME] == ProtocolDiscussionState.NAME
    assert set(next_map.keys()).issubset(set(node_info_map.keys()))
