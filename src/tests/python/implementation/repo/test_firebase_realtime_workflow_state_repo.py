from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

import python.implementation.repo.firebase_realtime_workflow_state_repo as repo_module
from python.domain.models.errors import NodeExecutionError
from python.domain.models.models import ChatMessage
from python.domain.workflows.state import Action, State
from python.implementation.repo.firebase_realtime_workflow_state_repo import (
    FirebaseRealtimeWorkflowStateRepo,
)


@dataclass
class _FakeRTDB:
    tree: dict[str, Any] = field(default_factory=dict)
    push_counter: int = 0

    def reference(self, path: str, app: object | None = None) -> _FakeRTDBRef:
        del app
        return _FakeRTDBRef(db=self, parts=_split_path(path), limit_last=None)


@dataclass
class _FakeRTDBRef:
    db: _FakeRTDB
    parts: tuple[str, ...]
    limit_last: int | None

    def child(self, name: str) -> _FakeRTDBRef:
        return _FakeRTDBRef(db=self.db, parts=(*self.parts, name), limit_last=None)

    def order_by_key(self) -> _FakeRTDBRef:
        return _FakeRTDBRef(db=self.db, parts=self.parts, limit_last=None)

    def limit_to_last(self, limit: int) -> _FakeRTDBRef:
        return _FakeRTDBRef(db=self.db, parts=self.parts, limit_last=limit)

    def get(self) -> Any:
        value = _get_value(self.db.tree, self.parts)
        if self.limit_last is not None and isinstance(value, dict):
            keys = sorted(value.keys())[-self.limit_last :]
            return {key: copy.deepcopy(value[key]) for key in keys}
        return copy.deepcopy(value)

    def set(self, value: Any) -> None:
        _set_value(self.db.tree, self.parts, value)

    def update(self, updates: dict[str, Any]) -> None:
        for raw_path, value in updates.items():
            update_parts = _split_path(raw_path)
            if value is None:
                _delete_value(self.db.tree, update_parts)
            else:
                _set_value(self.db.tree, update_parts, value)

    def delete(self) -> None:
        _delete_value(self.db.tree, self.parts)

    def push(self, value: Any) -> _FakeRTDBRef:
        existing = _get_value(self.db.tree, self.parts)
        if existing is None:
            _set_value(self.db.tree, self.parts, {})
            existing = _get_value(self.db.tree, self.parts)

        if not isinstance(existing, dict):
            raise ValueError("push target must be an object")

        self.db.push_counter += 1
        key = f"p{self.db.push_counter:06d}"
        existing[key] = copy.deepcopy(value)
        return self.child(key)


def _split_path(raw_path: str) -> tuple[str, ...]:
    return tuple(part for part in raw_path.split("/") if part)


def _get_value(tree: dict[str, Any], parts: tuple[str, ...]) -> Any:
    node: Any = tree
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set_value(tree: dict[str, Any], parts: tuple[str, ...], value: Any) -> None:
    if not parts:
        if not isinstance(value, dict):
            raise ValueError("root value must be dict")
        tree.clear()
        tree.update(copy.deepcopy(value))
        return

    node: Any = tree
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = copy.deepcopy(value)


def _delete_value(tree: dict[str, Any], parts: tuple[str, ...]) -> None:
    if not parts:
        tree.clear()
        return

    node: Any = tree
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(parts[-1], None)


@dataclass
class _DemoState(State):
    state_name: str = "DEMO_STATE"
    text: str = "hello"
    current_status: str = "PENDING"
    error_text: str | None = None

    def name(self) -> str:
        return self.state_name

    def status(self) -> str:
        return self.current_status

    def action(self) -> Action:
        return "NONE"

    def set_status_freez(self) -> None:
        self.current_status = "FREEZED"

    def set_status_pending(self) -> None:
        self.current_status = "PENDING"

    def messages(self) -> list[ChatMessage]:
        return [ChatMessage(role="assistant", content=self.text)]

    def error(self) -> NodeExecutionError | None:
        if self.error_text is None:
            return None
        return NodeExecutionError(state_name=self.state_name, error=self.error_text)

    def pre_required_states_names(self) -> list[str]:
        return []

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.state_name,
            "text": self.text,
            "status": self.current_status,
            "error_text": self.error_text,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> _DemoState:
        if payload.get("raise"):
            raise RuntimeError("deserialize failed")

        name = payload.get("name")
        if not isinstance(name, str):
            raise ValueError("name must be string")

        status = payload.get("status", "PENDING")
        if not isinstance(status, str):
            raise ValueError("status must be string")

        error_text = payload.get("error_text")
        if error_text is not None and not isinstance(error_text, str):
            raise ValueError("error_text must be string or None")

        return cls(
            state_name=name,
            text=str(payload.get("text", "")),
            current_status=status,
            error_text=error_text,
        )

    @classmethod
    def init_empty(cls) -> _DemoState:
        return cls()


@dataclass
class _MismatchState(State):
    def name(self) -> str:
        return "OTHER_STATE"

    def status(self) -> str:
        return "PENDING"

    def action(self) -> Action:
        return "NONE"

    def set_status_freez(self) -> None:
        return None

    def set_status_pending(self) -> None:
        return None

    def messages(self) -> list[ChatMessage]:
        return [ChatMessage(role="assistant", content="x")]

    def error(self) -> NodeExecutionError | None:
        return None

    def pre_required_states_names(self) -> list[str]:
        return []

    def to_json_dict(self) -> dict[str, Any]:
        return {"name": "OTHER_STATE"}

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> _MismatchState:
        del payload
        return cls()

    @classmethod
    def init_empty(cls) -> _MismatchState:
        return cls()


@dataclass
class _CredFactory:
    called: bool = False

    def __call__(self) -> object:
        self.called = True
        return object()


def _make_repo(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state_classes_by_name: dict[str, type[State]] | None = None,
) -> tuple[FirebaseRealtimeWorkflowStateRepo, _FakeRTDB]:
    fake_db = _FakeRTDB()
    monkeypatch.setattr(repo_module.db, "reference", fake_db.reference)
    repo = FirebaseRealtimeWorkflowStateRepo(
        app=object(),
        state_classes_by_name=state_classes_by_name or {"DEMO_STATE": _DemoState},
    )
    return repo, fake_db


def _ids() -> tuple[UUID, UUID]:
    return uuid4(), uuid4()


def test_constructor_validates_inputs() -> None:
    with pytest.raises(ValueError, match=r"app must not be None"):
        FirebaseRealtimeWorkflowStateRepo(app=None, state_classes_by_name={})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"must be a mapping"):
        FirebaseRealtimeWorkflowStateRepo(app=object(), state_classes_by_name=[])  # type: ignore[arg-type]


def test_save_delete_and_query_conversation_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    repo.save_conversation_id(user_id=user_id, conversation_id=conversation_id)

    index_path = ("workflow_conversation_index", str(user_id), str(conversation_id))
    meta_path = ("workflows", str(user_id), str(conversation_id), "_meta")
    assert _get_value(fake_db.tree, index_path) is True
    assert _get_value(fake_db.tree, meta_path) == {"created": True}

    second_id = uuid4()
    fake_db.reference(f"/workflow_conversation_index/{user_id}").set(
        {
            str(conversation_id): True,
            "not-a-uuid": True,
            str(second_id): True,
        }
    )

    ids = repo.get_conversation_ids_for_user(user_id=user_id)
    assert ids == sorted([conversation_id, second_id], key=lambda item: str(item))
    assert (
        repo.is_conversation_id_for_user_id_exists(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        is True
    )

    repo.delete_conversation(user_id=user_id, conversation_id=conversation_id)
    assert _get_value(fake_db.tree, index_path) is None
    assert _get_value(fake_db.tree, ("workflows", str(user_id), str(conversation_id))) is None


def test_active_state_name_roundtrip_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    assert repo.load_active_state_name(user_id=user_id, conversation_id=conversation_id) is None

    with pytest.raises(ValueError, match=r"non-empty string"):
        repo.store_active_state_name(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="  ",
        )

    repo.store_active_state_name(
        user_id=user_id,
        conversation_id=conversation_id,
        state_name="DEMO_STATE",
    )
    assert (
        repo.load_active_state_name(user_id=user_id, conversation_id=conversation_id)
        == "DEMO_STATE"
    )


def test_store_and_load_state_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    state = _DemoState(state_name="DEMO_STATE", text="roundtrip", current_status="FREEZED")
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=state)

    loaded = repo.load_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state_name="DEMO_STATE",
    )
    assert isinstance(loaded, _DemoState)
    assert loaded.name() == "DEMO_STATE"
    assert loaded.status() == "FREEZED"
    assert [msg.content for msg in loaded.messages()] == ["roundtrip"]


def test_load_state_validates_payload_and_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    with pytest.raises(ValueError, match=r"state_name must be a non-empty string"):
        repo.load_state(user_id=user_id, conversation_id=conversation_id, state_name="")

    assert (
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )
        is None
    )

    fake_db.reference(f"/workflows/{user_id}/{conversation_id}/states/DEMO_STATE").set(
        {"bad": "shape"}
    )
    with pytest.raises(ValueError, match=r"must be a JSON string blob"):
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )

    fake_db.reference(f"/workflows/{user_id}/{conversation_id}/states/DEMO_STATE").set("not-json")
    with pytest.raises(ValueError, match=r"not valid JSON"):
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )

    fake_db.reference(f"/workflows/{user_id}/{conversation_id}/states/DEMO_STATE").set(
        json.dumps([1, 2])
    )
    with pytest.raises(ValueError, match=r"must be a dict"):
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )

    fake_db.reference(f"/workflows/{user_id}/{conversation_id}/states/UNKNOWN").set(
        json.dumps({"name": "UNKNOWN"})
    )
    with pytest.raises(KeyError, match=r"No State class registered"):
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="UNKNOWN",
        )


def test_load_state_wraps_deserialization_error_and_name_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    fake_db.reference(f"/workflows/{user_id}/{conversation_id}/states/DEMO_STATE").set(
        json.dumps({"name": "DEMO_STATE", "raise": True})
    )
    with pytest.raises(ValueError, match=r"Error deserializing state"):
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )

    mismatch_repo, mismatch_db = _make_repo(
        monkeypatch,
        state_classes_by_name={"DEMO_STATE": _MismatchState},
    )
    mismatch_db.reference(f"/workflows/{user_id}/{conversation_id}/states/DEMO_STATE").set(
        json.dumps({"name": "DEMO_STATE"})
    )
    with pytest.raises(ValueError, match=r"Loaded State.name mismatch"):
        mismatch_repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )


def test_delete_state_and_message_history_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    repo.store_active_state_name(
        user_id=user_id,
        conversation_id=conversation_id,
        state_name="DEMO_STATE",
    )
    repo.store_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state=_DemoState(state_name="DEMO_STATE", text="x"),
    )
    repo.delete_state(user_id=user_id, conversation_id=conversation_id, state_name="DEMO_STATE")

    assert repo.load_active_state_name(user_id=user_id, conversation_id=conversation_id) is None
    assert (
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )
        is None
    )

    repo.append_messages(
        user_id=user_id,
        conversation_id=conversation_id,
        messages=[
            ChatMessage(role="system", content="s1"),
            ChatMessage(role="user", content="u2"),
            ChatMessage(role="assistant", content="a3"),
        ],
    )
    repo.append_message(
        user_id=user_id,
        conversation_id=conversation_id,
        message=ChatMessage(role="user", content="u4"),
    )

    history = repo.load_message_history(user_id=user_id, conversation_id=conversation_id, limit=2)
    assert [msg.content for msg in history] == ["a3", "u4"]
    assert (
        repo.load_message_history(user_id=user_id, conversation_id=conversation_id, limit=0) == []
    )

    repo.clear_message_history(user_id=user_id, conversation_id=conversation_id)
    assert (
        repo.load_message_history(user_id=user_id, conversation_id=conversation_id, limit=10) == []
    )


def test_append_and_load_message_history_roundtrips_structured_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()
    csv_id = uuid4()
    json_id = uuid4()

    repo.append_message(
        user_id=user_id,
        conversation_id=conversation_id,
        message=ChatMessage(
            role="assistant",
            content="dataset ready",
            artifact_refs=(
                {
                    "id": csv_id,
                    "kind": "data",
                    "format": "csv",
                    "artifact_meta": {"source": "summary"},
                },
                {
                    "id": json_id,
                    "kind": "data",
                    "format": "json",
                    "artifact_meta": {"vega": "true"},
                },
            ),
            artifacts=(
                {
                    "id": json_id,
                    "content": {"rows": 3},
                    "kind": "data",
                    "format": "json",
                    "artifact_meta": {"rows": "3"},
                },
            ),
            id="msg-1",
        ),
    )

    history = repo.load_message_history(user_id=user_id, conversation_id=conversation_id, limit=10)

    assert len(history) == 1
    assert history[0].content == "dataset ready"
    assert history[0].artifact_refs == [
        {
            "id": csv_id,
            "kind": "data",
            "format": "csv",
            "artifact_meta": {"source": "summary"},
        },
        {
            "id": json_id,
            "kind": "data",
            "format": "json",
            "artifact_meta": {"vega": "true"},
        },
    ]
    assert history[0].artifacts == [
        {
            "id": json_id,
            "content": {"rows": 3},
            "kind": "data",
            "format": "json",
            "artifact_meta": {"rows": "3"},
        }
    ]
    assert history[0].id == "msg-1"


def test_load_message_history_accepts_legacy_artifacts_ids_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()
    csv_id = uuid4()
    json_id = uuid4()

    fake_db.reference(f"/workflows/{user_id}/{conversation_id}/messages").set(
        {
            "p000001": {
                "role": "assistant",
                "message": "x",
                "artifacts_ids": [
                    {"id": str(csv_id), "type": "csv"},
                    {"id": str(json_id), "type": "json"},
                ],
                "id": "legacy-1",
            }
        }
    )

    history = repo.load_message_history(user_id=user_id, conversation_id=conversation_id, limit=10)

    assert len(history) == 1
    assert history[0].artifact_refs == [
        {"id": csv_id, "kind": "data", "format": "csv", "artifact_meta": None},
        {"id": json_id, "kind": "data", "format": "json", "artifact_meta": None},
    ]
    assert history[0].id == "legacy-1"


def test_load_message_history_ignores_malformed_artifact_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()
    csv_id = uuid4()
    json_id = uuid4()

    fake_db.reference(f"/workflows/{user_id}/{conversation_id}/messages").set(
        {
            "p000001": {
                "role": "assistant",
                "message": "x",
                "artifact_refs": [
                    {
                        "id": str(csv_id),
                        "kind": "data",
                        "format": "csv",
                        "artifact_meta": {"source": "summary", "count": 1},
                    },
                    {"id": "", "kind": "data", "format": "csv"},
                    {"id": str(json_id), "type": "json"},
                    {"id": "bad-1", "kind": "graph", "format": "png"},
                    {"kind": "data", "format": "csv"},
                    "bad",
                    1,
                ],
            }
        }
    )

    history = repo.load_message_history(user_id=user_id, conversation_id=conversation_id, limit=10)

    assert len(history) == 1
    assert history[0].artifact_refs == [
        {
            "id": csv_id,
            "kind": "data",
            "format": "csv",
            "artifact_meta": {"source": "summary", "count": "1"},
        },
        {
            "id": json_id,
            "kind": "data",
            "format": "json",
            "artifact_meta": None,
        },
    ]


def test_get_default_firebase_database_app_returns_existing_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_app = object()
    monkeypatch.setattr(repo_module.firebase_admin, "get_app", lambda: existing_app)

    assert FirebaseRealtimeWorkflowStateRepo.get_default_firebase_database_app() is existing_app


def test_get_default_firebase_database_app_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        repo_module.firebase_admin,
        "get_app",
        lambda: (_ for _ in ()).throw(ValueError("none")),
    )
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT_ID", raising=False)
    monkeypatch.setenv("FIREBASE_DATABASE_URL", "https://db.example")

    with pytest.raises(ValueError, match=r"GOOGLE_CLOUD_PROJECT_ID"):
        FirebaseRealtimeWorkflowStateRepo.get_default_firebase_database_app()

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_ID", "proj-1")
    monkeypatch.delenv("FIREBASE_DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match=r"FIREBASE_DATABASE_URL"):
        FirebaseRealtimeWorkflowStateRepo.get_default_firebase_database_app()


def test_get_default_firebase_database_app_initializes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        repo_module.firebase_admin,
        "get_app",
        lambda: (_ for _ in ()).throw(ValueError("none")),
    )
    cred_factory = _CredFactory()
    captured: dict[str, Any] = {}
    expected_app = object()

    def fake_initialize_app(cred: object, options: dict[str, str]) -> object:
        captured["cred"] = cred
        captured["options"] = options
        return expected_app

    monkeypatch.setattr(repo_module.credentials, "ApplicationDefault", cred_factory)
    monkeypatch.setattr(repo_module.firebase_admin, "initialize_app", fake_initialize_app)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_ID", "proj-1")
    monkeypatch.setenv("FIREBASE_DATABASE_URL", "https://db.example")

    app = FirebaseRealtimeWorkflowStateRepo.get_default_firebase_database_app()

    assert app is expected_app
    assert cred_factory.called is True
    assert captured["options"] == {
        "projectId": "proj-1",
        "databaseURL": "https://db.example",
    }
