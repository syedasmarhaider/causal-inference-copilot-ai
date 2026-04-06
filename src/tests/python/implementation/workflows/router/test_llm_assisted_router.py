from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

import pytest
from pydantic import BaseModel

from python.domain.models.errors import NodeExecutionError
from python.domain.models.models import ChatMessage
from python.domain.service.llm_service import LLMConfig, LLMResponse
from python.domain.workflows.route import NextDecision
from python.domain.workflows.state import Action, State
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import (
    CausalInferenceState,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_state import (
    CompileAndValidateState,
)
from python.implementation.workflows.nodes.dataset.dataset_state import DatasetState
from python.implementation.workflows.nodes.model_selection.mode_selection_state import (
    ModelSelectionState,
)
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
    recoverable_states_map,
    states_can_call_other_states_during_execution_map,
)

T = TypeVar("T", bound=BaseModel)


@dataclass
class _FakeState(State):
    state_name: str
    state_status: str
    state_messages: Sequence[ChatMessage] = field(default_factory=tuple)
    state_error: NodeExecutionError | None = None

    def name(self) -> str:
        return self.state_name

    def status(self) -> str:
        return self.state_status

    def action(self) -> Action:
        return "NONE"

    def set_status_freez(self) -> None:
        self.state_status = "FREEZED"

    def set_status_pending(self) -> None:
        self.state_status = "PENDING"

    def messages(self) -> Sequence[ChatMessage]:
        return self.state_messages

    def error(self) -> NodeExecutionError | None:
        return self.state_error

    def pre_required_states_names(self) -> Sequence[str]:
        return ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "state_name": self.state_name,
            "state_status": self.state_status,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> _FakeState:
        return cls(
            state_name=str(payload["state_name"]),
            state_status=str(payload["state_status"]),
        )

    @classmethod
    def init_empty(cls) -> _FakeState:
        return cls(state_name="EMPTY", state_status="PENDING")


class _LegacyLikeState:
    def __init__(
        self,
        *,
        state_name: str,
        state_status: str,
        state_messages: Sequence[ChatMessage] = (),
        state_error: NodeExecutionError | None = None,
    ) -> None:
        self._state_name = state_name
        self._state_status = state_status
        self._state_messages = state_messages
        self._state_error = state_error

    @property
    def name(self) -> str:
        return self._state_name

    @property
    def status(self) -> str:
        return self._state_status

    @property
    def messages(self) -> Sequence[ChatMessage]:
        return self._state_messages

    @property
    def error(self) -> NodeExecutionError | None:
        return self._state_error


@dataclass
class _FakeLLM:
    scripted_results: list[Any] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
    ) -> LLMResponse:
        raise AssertionError("generate should not be called by the router")

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
                "history": history,
                "max_attempts": max_attempts,
            }
        )
        if not self.scripted_results:
            raise AssertionError("No scripted LLM result configured")

        result = self.scripted_results.pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, schema):
            return result
        return schema.model_validate(result)


def _messages(*items: tuple[str, str]) -> list[ChatMessage]:
    return [ChatMessage(role=role, content=content) for role, content in items]


def test_router_metadata_and_chain_helpers() -> None:
    router = LLMAssistedRouterRouter(llm=_FakeLLM())

    assert router.get_initial_state_name() == DatasetState.NAME
    assert router.get_done_state_name() == NoopDoneState.NAME
    assert router.get_next_state_names(ProtocolDiscussionState.NAME) == [
        DatasetState.NAME,
        CompileAndValidateState.NAME,
        ModelSelectionState.NAME,
        ModelTrainState.NAME,
        CausalInferenceState.NAME,
        NoopDoneState.NAME,
    ]
    assert router.get_next_state_names(NoopDoneState.NAME) == []


def test_registry_helpers_are_well_formed() -> None:
    state_map = build_state_classes_by_name()
    next_map = init_next_state_names()
    callable_map = states_can_call_other_states_during_execution_map()
    recoverable_map = recoverable_states_map()
    node_info_map = get_node_name_with_description()

    assert set(state_map) == {
        DatasetState.NAME,
        ProtocolDiscussionState.NAME,
        CompileAndValidateState.NAME,
        ModelSelectionState.NAME,
        ModelTrainState.NAME,
        CausalInferenceState.NAME,
        NoopDoneState.NAME,
    }
    assert next_map[DatasetState.NAME] == CompileAndValidateState.NAME
    assert [item.name for item in callable_map[ModelSelectionState.NAME]] == [
        DatasetState.NAME,
        CompileAndValidateState.NAME,
    ]
    assert recoverable_map[ModelTrainState.NAME] == (
        ProtocolDiscussionState.NAME,
        ModelSelectionState.NAME,
    )
    assert set(next_map).issubset(set(node_info_map))
    assert set(state_map).issubset(set(node_info_map))


def test_decide_next_routes_none_to_dataset_without_llm() -> None:
    llm = _FakeLLM()
    router = LLMAssistedRouterRouter(llm=llm)

    decision = router.decide_next(current_state=None, messages_history=[])

    assert decision == NextDecision(state_name=DatasetState.NAME)
    assert llm.calls == []


def test_decide_next_done_and_freezed_are_deterministic_without_llm() -> None:
    llm = _FakeLLM()
    router = LLMAssistedRouterRouter(llm=llm)

    assert router.decide_next(
        current_state=_FakeState(state_name=ProtocolDiscussionState.NAME, state_status="DONE"),
        messages_history=_messages(("user", "continue")),
    ) == NextDecision(state_name=DatasetState.NAME)
    assert router.decide_next(
        current_state=_FakeState(state_name=DatasetState.NAME, state_status="FREEZED"),
        messages_history=_messages(("user", "continue")),
    ) == NextDecision(state_name=CompileAndValidateState.NAME)
    assert router.decide_next(
        current_state=_FakeState(state_name=NoopDoneState.NAME, state_status="DONE"),
        messages_history=_messages(("user", "continue")),
    ) == NextDecision(state_name=NoopDoneState.NAME)
    assert llm.calls == []


def test_decide_next_supports_property_style_state_access() -> None:
    router = LLMAssistedRouterRouter(llm=_FakeLLM())
    legacy_state = _LegacyLikeState(
        state_name=ProtocolDiscussionState.NAME,
        state_status="DONE",
    )

    decision = router.decide_next(
        current_state=cast(State, legacy_state),
        messages_history=[],
    )

    assert decision == NextDecision(state_name=DatasetState.NAME)


def test_pending_routes_to_current_state_and_uses_current_fellow_prompt_shape() -> None:
    llm = _FakeLLM(
        scripted_results=[
            {
                "state_name": ModelSelectionState.NAME,
                "router_confirmation_message_for_user": None,
            }
        ]
    )
    router = LLMAssistedRouterRouter(llm=llm)

    decision = router.decide_next(
        current_state=_FakeState(state_name=ModelSelectionState.NAME, state_status="PENDING"),
        messages_history=_messages(
            ("assistant", "old"),
            ("system", "prefer validation if the question is about warnings"),
            ("user", "why was this model picked"),
        ),
    )

    assert decision == NextDecision(state_name=ModelSelectionState.NAME)
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["config"].model == "mini"
    assert call["config"].temperature == 0.4
    assert call["history"] is None
    payload = json.loads(call["user_prompt"])
    assert payload["current_state"]["state_name"] == ModelSelectionState.NAME
    assert [item["state_name"] for item in payload["fellow_states"]] == [
        DatasetState.NAME,
        CompileAndValidateState.NAME,
    ]
    assert payload["recent_messages"] == [
        {"role": "system", "content": "prefer validation if the question is about warnings"},
        {"role": "user", "content": "why was this model picked"},
    ]


def test_pending_routes_to_fellow_state_when_llm_selects_fellow() -> None:
    llm = _FakeLLM(
        scripted_results=[
            {
                "state_name": DatasetState.NAME,
                "router_confirmation_message_for_user": None,
            }
        ]
    )
    router = LLMAssistedRouterRouter(llm=llm)

    decision = router.decide_next(
        current_state=_FakeState(state_name=CompileAndValidateState.NAME, state_status="PENDING"),
        messages_history=_messages(("user", "show me the cleaned columns")),
    )

    assert decision == NextDecision(state_name=DatasetState.NAME)


def test_pending_with_only_current_candidate_still_uses_llm_and_can_clarify() -> None:
    llm = _FakeLLM(
        scripted_results=[
            {
                "state_name": None,
                "router_confirmation_message_for_user": "What would you like to do from the done stage?",
            }
        ]
    )
    router = LLMAssistedRouterRouter(llm=llm)

    decision = router.decide_next(
        current_state=_FakeState(state_name=NoopDoneState.NAME, state_status="PENDING"),
        messages_history=_messages(("user", "change it")),
    )

    assert decision == NextDecision(
        state_name=None,
        router_confirmation_message_for_user="What would you like to do from the done stage?",
    )
    assert len(llm.calls) == 1
    payload = json.loads(llm.calls[0]["user_prompt"])
    assert payload["current_state"]["state_name"] == NoopDoneState.NAME
    assert payload["fellow_states"] == []


def test_pending_returns_confirmation_when_llm_returns_null() -> None:
    llm = _FakeLLM(
        scripted_results=[
            {
                "state_name": None,
                "router_confirmation_message_for_user": "Do you want data review or protocol discussion?",
            }
        ]
    )
    router = LLMAssistedRouterRouter(llm=llm)

    decision = router.decide_next(
        current_state=_FakeState(state_name=DatasetState.NAME, state_status="PENDING"),
        messages_history=_messages(("user", "change it")),
    )

    assert decision == NextDecision(
        state_name=None,
        router_confirmation_message_for_user="Do you want data review or protocol discussion?",
    )


def test_pending_rejects_state_outside_allowed_candidates() -> None:
    llm = _FakeLLM(
        scripted_results=[
            {
                "state_name": ModelTrainState.NAME,
                "router_confirmation_message_for_user": None,
            }
        ]
    )
    router = LLMAssistedRouterRouter(llm=llm)

    decision = router.decide_next(
        current_state=_FakeState(state_name=ModelSelectionState.NAME, state_status="PENDING"),
        messages_history=_messages(("user", "train it now")),
    )

    assert decision == NextDecision(
        state_name=None,
        router_confirmation_message_for_user=(
            "I could not map that request to an allowed stage from the current state. "
            "Please clarify the intended stage."
        ),
    )


def test_pending_returns_confirmation_when_llm_raises() -> None:
    llm = _FakeLLM(scripted_results=[RuntimeError("router llm failed")])
    router = LLMAssistedRouterRouter(llm=llm)

    decision = router.decide_next(
        current_state=_FakeState(state_name=DatasetState.NAME, state_status="PENDING"),
        messages_history=_messages(("user", "anything")),
    )

    assert decision == NextDecision(
        state_name=None,
        router_confirmation_message_for_user=(
            "Sorry I couldn't understand the intended route. Please clarify what you want to do next."
        ),
    )


def test_aborted_single_candidate_skips_llm() -> None:
    llm = _FakeLLM()
    router = LLMAssistedRouterRouter(llm=llm)

    decision = router.decide_next(
        current_state=_FakeState(
            state_name=DatasetState.NAME,
            state_status="ABORTED",
            state_error=NodeExecutionError(state_name=DatasetState.NAME, error="load failed"),
        ),
        messages_history=_messages(("user", "retry")),
    )

    assert decision == NextDecision(state_name=DatasetState.NAME)
    assert llm.calls == []


def test_aborted_routes_to_recoverable_state_using_error_context() -> None:
    llm = _FakeLLM(
        scripted_results=[
            {
                "state_name": ModelSelectionState.NAME,
                "router_confirmation_message_for_user": None,
            }
        ]
    )
    router = LLMAssistedRouterRouter(llm=llm)
    current_state = _FakeState(
        state_name=ModelTrainState.NAME,
        state_status="ABORTED",
        state_messages=_messages(
            ("assistant", "training failed"),
            ("system", "recover by revisiting model choice if fit assumptions are broken"),
        ),
        state_error=NodeExecutionError(state_name=ModelTrainState.NAME, error="fit failed"),
    )

    decision = router.decide_next(
        current_state=current_state,
        messages_history=_messages(
            ("assistant", "old"),
            ("system", "latest system route"),
            ("user", "fix it"),
        ),
    )

    assert decision == NextDecision(state_name=ModelSelectionState.NAME)
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["config"].model == "basic"
    assert call["history"] is None
    payload = json.loads(call["user_prompt"])
    assert payload["current_state"] == ModelTrainState.NAME
    assert payload["current_error"] == "fit failed"
    assert payload["current_system_message"] == "recover by revisiting model choice if fit assumptions are broken"
    assert [item["state_name"] for item in payload["recoverable_candidates"]] == [
        ProtocolDiscussionState.NAME,
        ModelSelectionState.NAME,
    ]
    assert payload["recent_messages"] == [
        {"role": "system", "content": "latest system route"},
        {"role": "user", "content": "fix it"},
    ]


def test_aborted_returns_confirmation_when_llm_returns_null() -> None:
    llm = _FakeLLM(
        scripted_results=[
            {
                "state_name": None,
                "router_confirmation_message_for_user": "Do you want to revisit protocol or model selection?",
            }
        ]
    )
    router = LLMAssistedRouterRouter(llm=llm)

    decision = router.decide_next(
        current_state=_FakeState(
            state_name=ModelTrainState.NAME,
            state_status="ABORTED",
            state_error=NodeExecutionError(state_name=ModelTrainState.NAME, error="fit failed"),
        ),
        messages_history=_messages(("user", "fix it")),
    )

    assert decision == NextDecision(
        state_name=None,
        router_confirmation_message_for_user="Do you want to revisit protocol or model selection?",
    )


def test_aborted_rejects_state_outside_recoverable_candidates() -> None:
    llm = _FakeLLM(
        scripted_results=[
            {
                "state_name": DatasetState.NAME,
                "router_confirmation_message_for_user": None,
            }
        ]
    )
    router = LLMAssistedRouterRouter(llm=llm)

    decision = router.decide_next(
        current_state=_FakeState(
            state_name=ModelTrainState.NAME,
            state_status="ABORTED",
            state_error=NodeExecutionError(state_name=ModelTrainState.NAME, error="fit failed"),
        ),
        messages_history=_messages(("user", "fix it")),
    )

    assert decision == NextDecision(
        state_name=None,
        router_confirmation_message_for_user=(
            "I could not select a valid recoverable stage. Please clarify where to recover."
        ),
    )


def test_aborted_returns_confirmation_when_llm_raises() -> None:
    llm = _FakeLLM(scripted_results=[RuntimeError("recovery llm failed")])
    router = LLMAssistedRouterRouter(llm=llm)

    decision = router.decide_next(
        current_state=_FakeState(
            state_name=ModelTrainState.NAME,
            state_status="ABORTED",
            state_error=NodeExecutionError(state_name=ModelTrainState.NAME, error="fit failed"),
        ),
        messages_history=_messages(("user", "fix it")),
    )

    assert decision == NextDecision(
        state_name=None,
        router_confirmation_message_for_user=(
            "I could not determine the best recovery stage. Please clarify where you want to recover."
        ),
    )


def test_decide_next_raises_on_unexpected_status() -> None:
    router = LLMAssistedRouterRouter(llm=_FakeLLM())

    with pytest.raises(ValueError, match="Unexpected state status"):
        router.decide_next(
            current_state=_FakeState(state_name=DatasetState.NAME, state_status="UNKNOWN"),
            messages_history=[],
        )
