from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

from python.domain.models.errors import StateNotFoundError
from python.domain.models.models import ChatMessage
from python.domain.workflows.node import Node
from python.domain.workflows.route import NextDecision, Router
from python.domain.workflows.state import Action, State, Status
from python.implementation.workflows.workflow_app import WorkflowApp


class _BaseTestState(State):
    NAME = "BASE_STATE"
    ACTION: Action = "NEEDS_INPUT"

    def __init__(
        self,
        *,
        status: Status = "PENDING",
        assistant_text: str = "assistant",
        messages: Sequence[ChatMessage] | None = None,
        action: Action | None = None,
        required: Sequence[str] = (),
        error_text: str | None = None,
        name_override: str | None = None,
    ) -> None:
        self._name = name_override or self.NAME
        self._status = status
        self._messages = list(messages or [ChatMessage(role="assistant", content=assistant_text)])
        self._action = action or self.ACTION
        self._required = tuple(required)
        self._error_text = error_text

    def name(self) -> str:
        return self._name

    def status(self) -> Status:
        return self._status

    def set_status_freez(self) -> None:
        self._status = "FREEZED"

    def action(self) -> Action:
        return self._action

    def set_status_pending(self) -> None:
        self._status = "PENDING"

    def messages(self) -> Sequence[ChatMessage]:
        return list(self._messages)

    def error(self):  # type: ignore[override]
        return None

    def pre_required_states_names(self) -> Sequence[str]:
        return self._required

    def to_json_dict(self) -> dict[str, Any]:
        return {"name": self._name, "status": self._status}

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> _BaseTestState:
        return cls(
            status=payload.get("status", "PENDING"),
            name_override=payload.get("name"),
        )

    @classmethod
    def init_empty(cls) -> _BaseTestState:
        return cls(status="PENDING")


class _StateA(_BaseTestState):
    NAME = "STATE_A"


class _StateB(_BaseTestState):
    NAME = "STATE_B"


class _DepState(_BaseTestState):
    NAME = "DEP_STATE"


@dataclass
class _FakeWorkflowStateRepo:
    conversations: dict[UUID, set[UUID]] = field(default_factory=dict)
    active_by_conversation: dict[tuple[UUID, UUID], str] = field(default_factory=dict)
    states: dict[tuple[UUID, UUID, str], State] = field(default_factory=dict)
    messages: dict[tuple[UUID, UUID], list[ChatMessage]] = field(default_factory=dict)
    stored_active_calls: list[tuple[UUID, UUID, str]] = field(default_factory=list)
    stored_state_calls: list[tuple[UUID, UUID, str]] = field(default_factory=list)
    deleted_state_calls: list[tuple[UUID, UUID, str]] = field(default_factory=list)

    def save_conversation_id(self, *, user_id: UUID, conversation_id: UUID) -> None:
        self.conversations.setdefault(user_id, set()).add(conversation_id)

    def get_conversation_ids_for_user(self, *, user_id: UUID) -> Sequence[UUID]:
        return sorted(self.conversations.get(user_id, set()), key=str)

    def is_conversation_id_for_user_id_exists(
        self, *, user_id: UUID, conversation_id: UUID
    ) -> bool:
        return conversation_id in self.conversations.get(user_id, set())

    def load_active_state_name(self, *, user_id: UUID, conversation_id: UUID) -> str | None:
        return self.active_by_conversation.get((user_id, conversation_id))

    def store_active_state_name(
        self, *, user_id: UUID, conversation_id: UUID, state_name: str
    ) -> None:
        self.active_by_conversation[(user_id, conversation_id)] = state_name
        self.stored_active_calls.append((user_id, conversation_id, state_name))

    def load_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> State | None:
        return self.states.get((user_id, conversation_id, state_name))

    def store_state(self, *, user_id: UUID, conversation_id: UUID, state: State) -> None:
        self.states[(user_id, conversation_id, state.name())] = state
        self.stored_state_calls.append((user_id, conversation_id, state.name()))

    def delete_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> None:
        self.deleted_state_calls.append((user_id, conversation_id, state_name))
        self.states.pop((user_id, conversation_id, state_name), None)

    def append_message(self, *, user_id: UUID, conversation_id: UUID, message: ChatMessage) -> None:
        self.append_messages(user_id=user_id, conversation_id=conversation_id, messages=[message])

    def append_messages(
        self, *, user_id: UUID, conversation_id: UUID, messages: Sequence[ChatMessage]
    ) -> None:
        key = (user_id, conversation_id)
        self.messages.setdefault(key, []).extend(messages)

    def load_message_history(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        limit: int = 20,
    ) -> Sequence[ChatMessage]:
        return list(self.messages.get((user_id, conversation_id), []))[-limit:]

    def clear_message_history(self, *, user_id: UUID, conversation_id: UUID) -> None:
        self.messages[(user_id, conversation_id)] = []


@dataclass
class _FakeRouter(Router):
    initial_state_name: str = _StateA.NAME
    done_state_name: str = "DONE"
    decisions: list[NextDecision] = field(default_factory=list)
    next_map: dict[str, list[str]] = field(default_factory=dict)
    decide_calls: list[dict[str, Any]] = field(default_factory=list)

    def get_initial_state_name(self) -> str:
        return self.initial_state_name

    def get_done_state_name(self) -> str:
        return self.done_state_name

    def get_next_state_names(self, current_state_name: str) -> Sequence[str]:
        return list(self.next_map.get(current_state_name, []))

    def decide_next(
        self,
        *,
        current_state: State | None,
        messages_history: Sequence[ChatMessage],
    ) -> NextDecision:
        self.decide_calls.append(
            {
                "current_state": current_state,
                "messages_history": list(messages_history),
            }
        )
        if not self.decisions:
            raise AssertionError("No router decision configured")
        return self.decisions.pop(0)


@dataclass
class _FakeNode(Node):
    _name: str
    returned_state: State
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def get_info(cls) -> str:
        return "fake-node"

    def run(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: State,
        previous_state_dependencies: Mapping[str, State],
        messages_history: Sequence[ChatMessage] | None,
    ) -> State:
        self.calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "state": state,
                "previous_state_dependencies": dict(previous_state_dependencies),
                "messages_history": list(messages_history or ()),
            }
        )
        return self.returned_state


def _build_app(
    *,
    repo: _FakeWorkflowStateRepo,
    router: _FakeRouter,
    nodes: Mapping[str, Node] | None = None,
) -> WorkflowApp:
    return WorkflowApp(
        repo=repo,
        router=router,
        nodes_by_state_name=nodes
        or {
            _StateA.NAME: _FakeNode(_StateA.NAME, _StateA()),
            _StateB.NAME: _FakeNode(_StateB.NAME, _StateB()),
        },
        state_classes_by_name={
            _StateA.NAME: _StateA,
            _StateB.NAME: _StateB,
            _DepState.NAME: _DepState,
        },
    )


def test_create_conversation_and_list_conversations() -> None:
    repo = _FakeWorkflowStateRepo()
    app = _build_app(repo=repo, router=_FakeRouter())
    user_id = uuid4()

    conversation_id = app.create_conversation(user_id)

    assert conversation_id in repo.conversations[user_id]
    assert app.list_conversations(user_id) == [conversation_id]


def test_get_last_conversation_state_returns_active_state_response() -> None:
    repo = _FakeWorkflowStateRepo()
    user_id = uuid4()
    conversation_id = uuid4()
    repo.save_conversation_id(user_id=user_id, conversation_id=conversation_id)
    repo.store_active_state_name(
        user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME
    )
    repo.store_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state=_StateA(
            messages=[
                ChatMessage(role="assistant", content="visible"),
                ChatMessage(role="system", content="hidden"),
            ]
        ),
    )
    app = _build_app(repo=repo, router=_FakeRouter())

    response = app.get_last_conversation_state(user_id=user_id, conversation_id=conversation_id)

    assert response is not None
    assert [message.content for message in response.messages] == ["visible"]
    assert response.current_stage_name == _StateA.NAME


def test_handle_initializes_initial_state_without_router() -> None:
    repo = _FakeWorkflowStateRepo()
    router = _FakeRouter(initial_state_name=_StateA.NAME)
    returned_state = _StateA(messages=[ChatMessage(role="assistant", content="hello from a")])
    node = _FakeNode(_StateA.NAME, returned_state)
    app = _build_app(repo=repo, router=router, nodes={_StateA.NAME: node})
    user_id = uuid4()
    conversation_id = uuid4()
    repo.save_conversation_id(user_id=user_id, conversation_id=conversation_id)

    response = app.handle(user_id=user_id, conversation_id=conversation_id, user_message="hi")

    assert router.decide_calls == []
    assert len(node.calls) == 1
    assert (
        repo.load_active_state_name(user_id=user_id, conversation_id=conversation_id)
        == _StateA.NAME
    )
    assert repo.stored_state_calls == [(user_id, conversation_id, _StateA.NAME)]
    assert [message.content for message in response.messages] == ["hello from a"]
    assert response.current_stage_name == _StateA.NAME


def test_handle_routes_to_next_state_and_loads_dependencies() -> None:
    repo = _FakeWorkflowStateRepo()
    router = _FakeRouter(decisions=[NextDecision(state_name=_StateB.NAME)])
    current_state = _StateA(messages=[ChatMessage(role="assistant", content="current")])
    dep_state = _DepState(messages=[ChatMessage(role="assistant", content="dep")])
    state_to_run = _StateB(
        required=[_DepState.NAME],
        messages=[ChatMessage(role="assistant", content="pending b")],
    )
    next_state = _StateB(
        messages=[ChatMessage(role="assistant", content="from b")],
    )
    node = _FakeNode(_StateB.NAME, next_state)
    app = _build_app(repo=repo, router=router, nodes={_StateB.NAME: node})
    user_id = uuid4()
    conversation_id = uuid4()
    repo.save_conversation_id(user_id=user_id, conversation_id=conversation_id)
    repo.store_active_state_name(
        user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME
    )
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=current_state)
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=dep_state)
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=state_to_run)

    response = app.handle(user_id=user_id, conversation_id=conversation_id, user_message="move")

    assert len(router.decide_calls) == 1
    assert router.decide_calls[0]["current_state"] is current_state
    assert len(node.calls) == 1
    assert node.calls[0]["previous_state_dependencies"] == {_DepState.NAME: dep_state}
    assert (
        repo.load_active_state_name(user_id=user_id, conversation_id=conversation_id)
        == _StateB.NAME
    )
    assert response.current_stage_name == _StateB.NAME
    assert [message.content for message in response.messages] == ["from b"]


def test_handle_returns_router_clarification_without_running_node() -> None:
    repo = _FakeWorkflowStateRepo()
    router = _FakeRouter(
        decisions=[NextDecision(router_confirmation_message_for_user="Need clarification")]
    )
    current_state = _StateA(messages=[ChatMessage(role="assistant", content="current")])
    node = _FakeNode(_StateA.NAME, _StateA())
    app = _build_app(repo=repo, router=router, nodes={_StateA.NAME: node})
    user_id = uuid4()
    conversation_id = uuid4()
    repo.save_conversation_id(user_id=user_id, conversation_id=conversation_id)
    repo.store_active_state_name(
        user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME
    )
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=current_state)

    response = app.handle(
        user_id=user_id, conversation_id=conversation_id, user_message="ambiguous"
    )

    assert node.calls == []
    assert [message.content for message in response.messages] == ["Need clarification"]
    history = repo.load_message_history(user_id=user_id, conversation_id=conversation_id, limit=10)
    assert [message.role for message in history] == ["user", "assistant"]


def test_handle_freezes_done_states_before_persisting() -> None:
    repo = _FakeWorkflowStateRepo()
    router = _FakeRouter(initial_state_name=_StateA.NAME)
    returned_state = _StateA(status="DONE")
    node = _FakeNode(_StateA.NAME, returned_state)
    app = _build_app(repo=repo, router=router, nodes={_StateA.NAME: node})
    user_id = uuid4()
    conversation_id = uuid4()
    repo.save_conversation_id(user_id=user_id, conversation_id=conversation_id)

    response = app.handle(user_id=user_id, conversation_id=conversation_id, user_message=None)

    assert returned_state.status() == "FREEZED"
    assert response.current_stage_status == "FREEZED"


def test_revert_to_state_deletes_downstream_states_and_appends_system_message() -> None:
    repo = _FakeWorkflowStateRepo()
    router = _FakeRouter(next_map={_StateA.NAME: [_StateB.NAME, _DepState.NAME]})
    app = _build_app(repo=repo, router=router)
    user_id = uuid4()
    conversation_id = uuid4()
    repo.save_conversation_id(user_id=user_id, conversation_id=conversation_id)
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=_StateA())
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=_StateB())
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=_DepState())

    app.revert_to_state(user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME)

    assert repo.deleted_state_calls == [
        (user_id, conversation_id, _StateA.NAME),
        (user_id, conversation_id, _StateB.NAME),
        (user_id, conversation_id, _DepState.NAME),
    ]
    assert (
        repo.load_active_state_name(user_id=user_id, conversation_id=conversation_id)
        == _StateA.NAME
    )
    last_message = repo.load_message_history(
        user_id=user_id, conversation_id=conversation_id, limit=1
    )[0]
    assert last_message.role == "system"
    assert "User reverted to state STATE_A" in last_message.content


def test_revert_to_state_raises_for_missing_state() -> None:
    app = _build_app(repo=_FakeWorkflowStateRepo(), router=_FakeRouter())

    with pytest.raises(StateNotFoundError):
        app.revert_to_state(
            user_id=uuid4(),
            conversation_id=uuid4(),
            state_name=_StateA.NAME,
        )
