from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest

from python.domain.models.errors import ConversationNotFoundError, NodeExecutionError, StateNotFoundError
from python.domain.repo.data_repo import ImageMime
from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.node import Node
from python.domain.workflows.route import NextDecision, Router
from python.domain.workflows.state import State, StateMessage
from python.domain.workflows.tool import Tool
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.workflow_app import (
    ArtifactResponse,
    WorkflowApp,
    WorkflowRequest,
)


class _BaseTestState(State):
    NAME = "BASE_STATE"
    INIT_TEXT = "init"
    REQUIRED: Sequence[str] = ()

    def __init__(
        self,
        *,
        status: str = "PENDING",
        txt_message: str | None = None,
        action: str = "NONE",
        artifact_ids: Sequence[str] | None = None,
        error_text: str | None = None,
        required: Sequence[str] | None = None,
        name_override: str | None = None,
    ) -> None:
        self._status = status
        self._txt_message = txt_message if txt_message is not None else self.INIT_TEXT
        self._action = action
        self._artifact_ids = list(artifact_ids) if artifact_ids else None
        self._error_text = error_text
        self._required = tuple(required if required is not None else self.REQUIRED)
        self._name = name_override or self.NAME

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> str:
        return self._status

    @property
    def message(self) -> StateMessage:
        return StateMessage(
            txt_message=self._txt_message,
            action=self._action,  # type: ignore[arg-type]
            artifact_ids=self._artifact_ids,
        )

    @property
    def error(self) -> NodeExecutionError | None:
        if self._error_text is None:
            return None
        return NodeExecutionError(state_name=self.name, error=self._error_text)

    def pre_required_states_names(self) -> Sequence[str]:
        return self._required

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "txt_message": self._txt_message,
            "action": self._action,
            "artifact_ids": self._artifact_ids,
            "error_text": self._error_text,
            "required": list(self._required),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> _BaseTestState:
        required_raw = payload.get("required")
        required = tuple(required_raw) if isinstance(required_raw, list) else ()
        return cls(
            status=str(payload.get("status", "PENDING")),
            txt_message=str(payload.get("txt_message", cls.INIT_TEXT)),
            action=str(payload.get("action", "NONE")),
            artifact_ids=payload.get("artifact_ids"),
            error_text=payload.get("error_text"),
            required=required,
            name_override=str(payload.get("name", cls.NAME)),
        )

    @classmethod
    def init_empty(cls) -> _BaseTestState:
        return cls(
            status="PENDING",
            txt_message=cls.INIT_TEXT,
            action="NONE",
            required=cls.REQUIRED,
            name_override=cls.NAME,
        )


class _StateA(_BaseTestState):
    NAME = "STATE_A"
    INIT_TEXT = "init-a"


class _StateB(_BaseTestState):
    NAME = "STATE_B"
    INIT_TEXT = "init-b"


class _DepState(_BaseTestState):
    NAME = "DEP_STATE"
    INIT_TEXT = "init-dep"


class _UnknownState(_BaseTestState):
    NAME = "UNKNOWN"


class _BadInitState(_BaseTestState):
    NAME = "BAD_STATE"

    @classmethod
    def init_empty(cls) -> _BaseTestState:
        return _BaseTestState(name_override="WRONG_NAME")


@dataclass
class _FakeWorkflowStateRepo:
    conversations: dict[UUID, set[UUID]] = field(default_factory=dict)
    active_by_conversation: dict[tuple[UUID, UUID], str] = field(default_factory=dict)
    states: dict[tuple[UUID, UUID, str], State] = field(default_factory=dict)
    messages: dict[tuple[UUID, UUID], list[ChatMessage]] = field(default_factory=dict)
    saved_conversations: list[tuple[UUID, UUID]] = field(default_factory=list)
    stored_active_calls: list[tuple[UUID, UUID, str]] = field(default_factory=list)
    stored_state_calls: list[tuple[UUID, UUID, str]] = field(default_factory=list)
    deleted_state_calls: list[tuple[UUID, UUID, str]] = field(default_factory=list)
    append_calls: list[tuple[UUID, UUID, ChatMessage]] = field(default_factory=list)

    def save_conversation_id(self, *, user_id: UUID, conversation_id: UUID) -> None:
        self.conversations.setdefault(user_id, set()).add(conversation_id)
        self.saved_conversations.append((user_id, conversation_id))

    def get_conversation_ids_for_user(self, *, user_id: UUID) -> Sequence[UUID]:
        return sorted(self.conversations.get(user_id, set()), key=lambda x: str(x))

    def is_conversation_id_for_user_id_exists(self, *, user_id: UUID, conversation_id: UUID) -> bool:
        return conversation_id in self.conversations.get(user_id, set())

    def load_active_state_name(self, *, user_id: UUID, conversation_id: UUID) -> str | None:
        return self.active_by_conversation.get((user_id, conversation_id))

    def store_active_state_name(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> None:
        self.active_by_conversation[(user_id, conversation_id)] = state_name
        self.stored_active_calls.append((user_id, conversation_id, state_name))

    def load_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> State | None:
        return self.states.get((user_id, conversation_id, state_name))

    def store_state(self, *, user_id: UUID, conversation_id: UUID, state: State) -> None:
        self.states[(user_id, conversation_id, state.name)] = state
        self.stored_state_calls.append((user_id, conversation_id, state.name))

    def delete_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> None:
        self.deleted_state_calls.append((user_id, conversation_id, state_name))
        self.states.pop((user_id, conversation_id, state_name), None)

    def append_message(self, *, user_id: UUID, conversation_id: UUID, message: ChatMessage) -> None:
        self.append_messages(user_id=user_id, conversation_id=conversation_id, messages=[message])

    def append_messages(self, *, user_id: UUID, conversation_id: UUID, messages: Sequence[ChatMessage]) -> None:
        key = (user_id, conversation_id)
        target = self.messages.setdefault(key, [])
        for message in messages:
            target.append(message)
            self.append_calls.append((user_id, conversation_id, message))

    def load_message_history(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        limit: int = 20,
    ) -> Sequence[ChatMessage]:
        if limit <= 0:
            return []
        key = (user_id, conversation_id)
        return list(self.messages.get(key, []))[-limit:]

    def clear_message_history(self, *, user_id: UUID, conversation_id: UUID) -> None:
        self.messages[(user_id, conversation_id)] = []


@dataclass
class _FakeDataRepo:
    artifact_mime: ImageMime = "image/png"
    artifact_bytes: bytes = b"artifact"
    csv_save_calls: list[dict[str, Any]] = field(default_factory=list)
    artifact_mime_calls: list[tuple[UUID, UUID, UUID]] = field(default_factory=list)
    artifact_bytes_calls: list[tuple[UUID, UUID, UUID, ImageMime | None]] = field(default_factory=list)

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        del user_id, conversation_id, dataset_id, limit
        return pd.DataFrame([{"x": 1}])

    def save_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        df: pd.DataFrame,
        *,
        overwrite: bool = True,
        include_index: bool = False,
    ) -> None:
        self.csv_save_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "dataset_id": dataset_id,
                "df": df.copy(),
                "overwrite": overwrite,
                "include_index": include_index,
            }
        )

    def save_artifact(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        content: bytes,
        *,
        mime: ImageMime,
        overwrite: bool = True,
    ) -> None:
        del user_id, conversation_id, artifact_id, content, mime, overwrite

    def get_artifact_mime(self, user_id: UUID, conversation_id: UUID, artifact_id: UUID) -> ImageMime:
        self.artifact_mime_calls.append((user_id, conversation_id, artifact_id))
        return self.artifact_mime

    def get_artifact_bytes(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        *,
        expected_mime: ImageMime | None = None,
    ) -> bytes:
        self.artifact_bytes_calls.append((user_id, conversation_id, artifact_id, expected_mime))
        return self.artifact_bytes


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


class _FakeTool(Tool):
    def get_tool_name(self) -> str:
        return "fake-tool"

    def get_tool_info(self) -> str:
        return "fake-tool-info"


class _FakeToolFactory(ToolFactory):
    def get_tool_names(self) -> list[str]:
        return ["fake-tool"]

    def get_tool_info(self, name: str) -> str:
        if name != "fake-tool":
            raise KeyError(name)
        return "fake-tool-info"

    def get_tools_info(self) -> dict[str, str]:
        return {"fake-tool": "fake-tool-info"}

    def has_tool(self, name: str) -> bool:
        return name == "fake-tool"

    def get_tool(self, name: str) -> Tool:
        if name != "fake-tool":
            raise KeyError(name)
        return _FakeTool()


class _FakeNode(Node):
    def __init__(self, *, name: str, output_state: State) -> None:
        self._name = name
        self._output_state = output_state
        self.calls: list[dict[str, Any]] = []

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
        tool_factory: ToolFactory,
        previous_state_dependencies: Mapping[str, State],
        messages_history: Sequence[ChatMessage] | None,
    ) -> State:
        self.calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "state": state,
                "tool_factory": tool_factory,
                "previous_state_dependencies": dict(previous_state_dependencies),
                "messages_history": list(messages_history or []),
            }
        )
        return self._output_state


def _build_app(
    *,
    repo: _FakeWorkflowStateRepo | None = None,
    data_repo: _FakeDataRepo | None = None,
    router: _FakeRouter | None = None,
    nodes: Mapping[str, Node] | None = None,
    state_classes: Mapping[str, type[State]] | None = None,
) -> tuple[WorkflowApp, _FakeWorkflowStateRepo, _FakeDataRepo, _FakeRouter]:
    fake_repo = repo or _FakeWorkflowStateRepo()
    fake_data_repo = data_repo or _FakeDataRepo()
    fake_router = router or _FakeRouter()
    app = WorkflowApp(
        repo=fake_repo,
        data_repo=fake_data_repo,
        router=fake_router,
        nodes_by_state_name=nodes or {},
        state_classes_by_name=state_classes or {},
        tool_factory=_FakeToolFactory(),
        history_limit=5,
        max_steps_per_call=1,
    )
    return app, fake_repo, fake_data_repo, fake_router


def _ids() -> tuple[UUID, UUID, UUID]:
    return uuid4(), uuid4(), uuid4()


def test_constructor_rejects_non_positive_max_steps() -> None:
    with pytest.raises(ValueError, match=r"max_steps_per_call"):
        WorkflowApp(
            repo=_FakeWorkflowStateRepo(),
            data_repo=_FakeDataRepo(),
            router=_FakeRouter(),
            nodes_by_state_name={},
            state_classes_by_name={},
            tool_factory=_FakeToolFactory(),
            max_steps_per_call=0,
        )


def test_raise_if_user_conversation_relation_exists_and_missing() -> None:
    app, repo, _, _ = _build_app()
    user_id, conversation_id, _ = _ids()
    repo.save_conversation_id(user_id=user_id, conversation_id=conversation_id)

    app.raise_if_userid_not_relates_to_conversation_id(user_id=user_id, conversation_id=conversation_id)

    with pytest.raises(ConversationNotFoundError):
        app.raise_if_userid_not_relates_to_conversation_id(user_id=user_id, conversation_id=uuid4())


def test_create_and_list_conversations_roundtrip() -> None:
    app, repo, _, _ = _build_app()
    user_id, _, _ = _ids()

    created = app.create_conversation(user_id)

    assert (user_id, created) in repo.saved_conversations
    listed = app.list_conversations(user_id)
    assert created in listed


def test_get_last_conversation_state_none_when_active_or_state_missing() -> None:
    app, repo, _, _ = _build_app()
    user_id, conversation_id, _ = _ids()

    assert app.get_last_conversation_state(user_id=user_id, conversation_id=conversation_id) is None

    repo.store_active_state_name(user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME)
    assert app.get_last_conversation_state(user_id=user_id, conversation_id=conversation_id) is None


def test_get_last_conversation_state_maps_message_flags() -> None:
    app, repo, _, _ = _build_app()
    user_id, conversation_id, _ = _ids()

    repo.store_active_state_name(user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME)
    repo.store_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state=_StateA(status="DONE", txt_message="ready", action="NEEDS_DATA", artifact_ids=["a-1"]),
    )

    resp = app.get_last_conversation_state(user_id=user_id, conversation_id=conversation_id)

    assert resp is not None
    assert resp.node_message == "ready"
    assert resp.needs_data is True
    assert resp.needs_input is False
    assert resp.artifact_ids == ["a-1"]


def test_upload_csv_data_rejects_invalid_csv_bytes() -> None:
    app, repo, _, _ = _build_app()
    user_id, conversation_id, _ = _ids()
    repo.store_active_state_name(user_id=user_id, conversation_id=conversation_id, state_name=LoadDatasetState.NAME)

    with pytest.raises(ValueError, match=r"not a valid CSV"):
        app.upload_csv_data(user_id=user_id, conversation_id=conversation_id, csv_bytes=b"")


def test_upload_csv_data_requires_load_dataset_active_state() -> None:
    app, repo, _, _ = _build_app()
    user_id, conversation_id, _ = _ids()
    repo.store_active_state_name(user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME)

    with pytest.raises(ValueError, match=r"state is not at load data set"):
        app.upload_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            csv_bytes=b"a,b\n1,2\n",
        )


def test_upload_csv_data_saves_to_data_repo_with_init_dataset_id() -> None:
    app, repo, data_repo, _ = _build_app()
    user_id, conversation_id, _ = _ids()
    repo.store_active_state_name(user_id=user_id, conversation_id=conversation_id, state_name=LoadDatasetState.NAME)

    dataset_id = app.upload_csv_data(
        user_id=user_id,
        conversation_id=conversation_id,
        csv_bytes=b"a,b\n1,2\n",
    )

    assert dataset_id == LoadDatasetState.INIT_DATA_ID
    assert len(data_repo.csv_save_calls) == 1
    call = data_repo.csv_save_calls[0]
    assert call["dataset_id"] == LoadDatasetState.INIT_DATA_ID
    assert call["overwrite"] is True
    assert call["df"].to_dict(orient="records") == [{"a": 1, "b": 2}]


def test_get_artifact_reads_mime_then_bytes_with_expected_mime() -> None:
    app, _, data_repo, _ = _build_app(data_repo=_FakeDataRepo(artifact_mime="image/jpeg", artifact_bytes=b"img"))
    user_id, conversation_id, artifact_id = _ids()

    artifact = app.get_artifact(user_id=user_id, conversation_id=conversation_id, artifact_id=artifact_id)

    assert artifact == ArtifactResponse(mime="image/jpeg", content=b"img")
    assert data_repo.artifact_mime_calls == [(user_id, conversation_id, artifact_id)]
    assert data_repo.artifact_bytes_calls == [(user_id, conversation_id, artifact_id, "image/jpeg")]


def test_revert_to_state_raises_when_state_not_found() -> None:
    app, _, _, _ = _build_app()
    user_id, conversation_id, _ = _ids()

    with pytest.raises(StateNotFoundError):
        app.revert_to_state(user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME)


def test_revert_to_state_deletes_forward_states_and_appends_system_message() -> None:
    router = _FakeRouter(next_map={_StateA.NAME: [_StateB.NAME, _DepState.NAME]})
    app, repo, _, _ = _build_app(router=router)
    user_id, conversation_id, _ = _ids()

    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=_StateA(status="DONE"))

    app.revert_to_state(user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME)

    assert repo.deleted_state_calls == [
        (user_id, conversation_id, _StateA.NAME),
        (user_id, conversation_id, _StateB.NAME),
        (user_id, conversation_id, _DepState.NAME),
    ]
    assert repo.active_by_conversation[(user_id, conversation_id)] == _StateA.NAME
    assert any("User reverted to state STATE_A" in call[2].content for call in repo.append_calls)


def test_handle_bootstraps_initial_state_and_persists_node_output() -> None:
    node = _FakeNode(
        name=_StateA.NAME,
        output_state=_StateA(status="DONE", txt_message="node-ok", action="NEEDS_INPUT", artifact_ids=["id-1"]),
    )
    router = _FakeRouter(decisions=[NextDecision(state_name=_StateA.NAME)])
    app, repo, _, _ = _build_app(
        router=router,
        nodes={_StateA.NAME: node},
        state_classes={_StateA.NAME: _StateA},
    )
    user_id, conversation_id, _ = _ids()

    resp = app.handle(WorkflowRequest(user_id=user_id, conversation_id=conversation_id, user_message=" hello "))

    assert resp.node_message == "node-ok"
    assert resp.needs_input is True
    assert resp.current_stage == _StateA.NAME
    assert repo.stored_active_calls[0] == (user_id, conversation_id, _StateA.NAME)
    assert repo.stored_state_calls == [
        (user_id, conversation_id, _StateA.NAME),
        (user_id, conversation_id, _StateA.NAME),
    ]
    assert [entry[2].role for entry in repo.append_calls] == ["user", "assistant"]
    assert len(node.calls) == 1


def test_handle_does_not_append_blank_user_message() -> None:
    node = _FakeNode(name=_StateA.NAME, output_state=_StateA(status="DONE", txt_message="ok"))
    router = _FakeRouter(decisions=[NextDecision(state_name=_StateA.NAME)])
    repo = _FakeWorkflowStateRepo()
    user_id, conversation_id, _ = _ids()
    repo.store_active_state_name(user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME)
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=_StateA(status="PENDING"))
    app, repo, _, _ = _build_app(
        repo=repo,
        router=router,
        nodes={_StateA.NAME: node},
        state_classes={_StateA.NAME: _StateA},
    )

    app.handle(WorkflowRequest(user_id=user_id, conversation_id=conversation_id, user_message="   "))

    assert [entry[2].role for entry in repo.append_calls] == ["assistant"]


def test_handle_applies_router_message_and_delete_list() -> None:
    node = _FakeNode(name=_StateB.NAME, output_state=_StateB(status="DONE", txt_message="b-done"))
    router = _FakeRouter(
        decisions=[
            NextDecision(
                state_name=_StateB.NAME,
                router_message_for_node="router says retry",
                delete_next_states_names=["S1", "S2"],
            )
        ]
    )
    repo = _FakeWorkflowStateRepo()
    user_id, conversation_id, _ = _ids()
    repo.store_active_state_name(user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME)
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=_StateA(status="DONE"))
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=_StateB(status="PENDING"))

    app, repo, _, _ = _build_app(
        repo=repo,
        router=router,
        nodes={_StateB.NAME: node},
        state_classes={_StateA.NAME: _StateA, _StateB.NAME: _StateB},
    )

    app.handle(WorkflowRequest(user_id=user_id, conversation_id=conversation_id, user_message=None))

    assert repo.deleted_state_calls == [
        (user_id, conversation_id, "S1"),
        (user_id, conversation_id, "S2"),
    ]
    assert [entry[2].role for entry in repo.append_calls] == ["system", "assistant"]
    assert node.calls[0]["messages_history"][-1].content == "router says retry"


def test_handle_uses_init_state_when_current_state_aborted() -> None:
    node = _FakeNode(name=_StateB.NAME, output_state=_StateB(status="DONE", txt_message="done-b"))
    router = _FakeRouter(decisions=[NextDecision(state_name=_StateB.NAME)])
    repo = _FakeWorkflowStateRepo()
    user_id, conversation_id, _ = _ids()
    repo.store_active_state_name(user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME)
    repo.store_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state=_StateA(status="ABORTED", error_text="broken"),
    )
    repo.store_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state=_StateB(status="DONE", txt_message="existing-b"),
    )

    app, _, _, _ = _build_app(
        repo=repo,
        router=router,
        nodes={_StateB.NAME: node},
        state_classes={_StateA.NAME: _StateA, _StateB.NAME: _StateB},
    )

    app.handle(WorkflowRequest(user_id=user_id, conversation_id=conversation_id, user_message=None))

    incoming_state = node.calls[0]["state"]
    assert incoming_state.name == _StateB.NAME
    assert incoming_state.message.txt_message == _StateB.INIT_TEXT


def test_handle_loads_required_dependencies_for_node() -> None:
    state_to_run = _StateB(status="PENDING", required=[_DepState.NAME])
    node = _FakeNode(name=_StateB.NAME, output_state=_StateB(status="DONE", txt_message="ok"))
    router = _FakeRouter(decisions=[NextDecision(state_name=_StateB.NAME)])
    repo = _FakeWorkflowStateRepo()
    user_id, conversation_id, _ = _ids()
    repo.store_active_state_name(user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME)
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=_StateA(status="DONE"))
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=state_to_run)
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=_DepState(status="DONE", txt_message="dep"))

    app, _, _, _ = _build_app(
        repo=repo,
        router=router,
        nodes={_StateB.NAME: node},
        state_classes={_StateA.NAME: _StateA, _StateB.NAME: _StateB, _DepState.NAME: _DepState},
    )

    app.handle(WorkflowRequest(user_id=user_id, conversation_id=conversation_id, user_message=None))

    deps = node.calls[0]["previous_state_dependencies"]
    assert set(deps.keys()) == {_DepState.NAME}
    assert deps[_DepState.NAME].name == _DepState.NAME


def test_handle_raises_when_node_for_decision_state_is_missing() -> None:
    router = _FakeRouter(decisions=[NextDecision(state_name=_UnknownState.NAME)])
    repo = _FakeWorkflowStateRepo()
    user_id, conversation_id, _ = _ids()
    repo.store_active_state_name(user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME)
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=_StateA(status="DONE"))
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=_UnknownState(status="PENDING"))

    app, _, _, _ = _build_app(
        repo=repo,
        router=router,
        nodes={},
        state_classes={_StateA.NAME: _StateA, _UnknownState.NAME: _UnknownState},
    )

    with pytest.raises(KeyError, match=r"no node registered"):
        app.handle(WorkflowRequest(user_id=user_id, conversation_id=conversation_id, user_message=None))


def test_handle_appends_system_error_message_when_node_returns_error() -> None:
    node = _FakeNode(
        name=_StateA.NAME,
        output_state=_StateA(status="ABORTED", txt_message="failed", error_text="boom"),
    )
    router = _FakeRouter(decisions=[NextDecision(state_name=_StateA.NAME)])
    repo = _FakeWorkflowStateRepo()
    user_id, conversation_id, _ = _ids()
    repo.store_active_state_name(user_id=user_id, conversation_id=conversation_id, state_name=_StateA.NAME)
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=_StateA(status="PENDING"))

    app, repo, _, _ = _build_app(
        repo=repo,
        router=router,
        nodes={_StateA.NAME: node},
        state_classes={_StateA.NAME: _StateA},
    )

    app.handle(WorkflowRequest(user_id=user_id, conversation_id=conversation_id, user_message=None))

    assert [entry[2].role for entry in repo.append_calls] == ["assistant", "system"]
    assert "Error returned from node boom" in repo.append_calls[-1][2].content


def test_require_state_class_and_init_empty_mismatch_validation() -> None:
    app, _, _, _ = _build_app(state_classes={_StateA.NAME: _StateA, _BadInitState.NAME: _BadInitState})

    app._require_state_class(_StateA.NAME)  # noqa: SLF001 - direct branch coverage

    with pytest.raises(KeyError, match=r"missing State class"):
        app._require_state_class("MISSING")  # noqa: SLF001

    with pytest.raises(ValueError, match=r"init_empty\(\) returned name"):
        app._init_empty_state(_BadInitState.NAME)  # noqa: SLF001
