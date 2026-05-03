from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

import python.implementation.repo.firebase_realtime_workflow_state_repo as repo_module
from python.domain.models.errors import StateConflictError
from python.domain.models.models import ChatMessage
from python.domain.workflows.node_state import NodeState
from python.domain.workflows.ochestrator_state import OchestratorState
from python.implementation.repo.firebase_realtime_workflow_state_repo import (
    FirebaseRealtimeWorkflowStateRepo,
)


# -----------------------------------------------------------------------
# Fake Firebase RTDB (in-memory)
# -----------------------------------------------------------------------


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

    def transaction(self, update_fn: Any) -> Any:
        next_value = update_fn(self.get())
        self.set(next_value)
        return next_value

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


# -----------------------------------------------------------------------
# Fake domain objects
# -----------------------------------------------------------------------


@dataclass
class _DemoNodeState(NodeState):
    state_name: str = "DEMO_STATE"
    text: str = "hello"

    def name(self) -> str:
        return self.state_name

    def clear_state(self) -> None:
        self.text = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {"name": self.state_name, "text": self.text}

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> _DemoNodeState:
        if payload.get("raise"):
            raise RuntimeError("deserialize failed")
        name = payload.get("name")
        if not isinstance(name, str):
            raise ValueError("name must be string")
        return cls(state_name=name, text=str(payload.get("text", "")))

    @classmethod
    def init_empty(cls) -> _DemoNodeState:
        return cls()


@dataclass
class _MismatchNodeState(NodeState):
    """Returns a different name than what it's registered under."""

    def name(self) -> str:
        return "OTHER_STATE"

    def clear_state(self) -> None:
        pass

    def to_json_dict(self) -> dict[str, Any]:
        return {"name": "OTHER_STATE"}

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> _MismatchNodeState:
        del payload
        return cls()

    @classmethod
    def init_empty(cls) -> _MismatchNodeState:
        return cls()


@dataclass
class _DemoOchestratorState(OchestratorState):
    _data: dict[str, Any] = field(default_factory=dict)

    def name(self) -> str:
        return "DEMO_OCHESTRATOR"

    def get_update_counter(self) -> int:
        value = self._data.get("update_counter", 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("update_counter must be a non-negative integer")
        if value < 0:
            raise ValueError("update_counter must be a non-negative integer")
        return value

    def set_update_counter(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("update_counter must be a non-negative integer")
        if value < 0:
            raise ValueError("update_counter must be a non-negative integer")
        self._data["update_counter"] = value

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._data[key] = value

    def get_current_node_name(self) -> str:
        return "DEMO_NODE"

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

    def get_working_dataset_id_and_frozen_status(self) -> tuple[UUID | None, bool]:
        return None, False

    def get_ochestration_prompt(self) -> str:
        return ""

    def to_json_dict(self) -> dict[str, Any]:
        return dict(self._data)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> _DemoOchestratorState:
        if payload.get("raise"):
            raise RuntimeError("deserialize failed")
        return cls(_data=dict(payload))

    @classmethod
    def init_empty(cls) -> _DemoOchestratorState:
        return cls()


@dataclass
class _MismatchOchestratorState(OchestratorState):
    def name(self) -> str:
        return "WRONG_NAME"

    def get_update_counter(self) -> int:
        return 0

    def set_update_counter(self, value: int) -> None:
        del value

    def get(self, key: str) -> Any:
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        pass

    def get_current_node_name(self) -> str:
        return "DEMO_NODE"

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

    def get_working_dataset_id_and_frozen_status(self) -> tuple[UUID | None, bool]:
        return None, False

    def get_ochestration_prompt(self) -> str:
        return ""

    def to_json_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> _MismatchOchestratorState:
        del payload
        return cls()

    @classmethod
    def init_empty(cls) -> _MismatchOchestratorState:
        return cls()


@dataclass
class _CredFactory:
    called: bool = False

    def __call__(self) -> object:
        self.called = True
        return object()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _make_repo(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state_classes_by_name: dict[str, type[NodeState]] | None = None,
    ochestrator_state_classes_by_name: dict[str, type[OchestratorState]] | None = None,
) -> tuple[FirebaseRealtimeWorkflowStateRepo, _FakeRTDB]:
    fake_db = _FakeRTDB()
    monkeypatch.setattr(repo_module.db, "reference", fake_db.reference)
    repo = FirebaseRealtimeWorkflowStateRepo(
        app=object(),
        state_classes_by_name=state_classes_by_name or {"DEMO_STATE": _DemoNodeState},
        ochestrator_state_classes_by_name=ochestrator_state_classes_by_name
        or {"DEMO_OCHESTRATOR": _DemoOchestratorState},
    )
    return repo, fake_db


def _ids() -> tuple[UUID, UUID]:
    return uuid4(), uuid4()


def _stored_ochestrator_payload(
    fake_db: _FakeRTDB,
    *,
    user_id: UUID,
    conversation_id: UUID,
) -> dict[str, Any]:
    raw = fake_db.reference(
        f"/workflows/{user_id}/{conversation_id}/ochestrator_state"
    ).get()
    assert isinstance(raw, str)
    envelope = json.loads(raw)
    assert isinstance(envelope, dict)
    payload = envelope["payload"]
    assert isinstance(payload, dict)
    return payload


# -----------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------


def test_constructor_validates_inputs() -> None:
    with pytest.raises(ValueError, match=r"app must not be None"):
        FirebaseRealtimeWorkflowStateRepo(
            app=None,  # type: ignore[arg-type]
            state_classes_by_name={},
            ochestrator_state_classes_by_name={},
        )

    with pytest.raises(ValueError, match=r"state_classes_by_name must be a mapping"):
        FirebaseRealtimeWorkflowStateRepo(
            app=object(),
            state_classes_by_name=[],  # type: ignore[arg-type]
            ochestrator_state_classes_by_name={},
        )

    with pytest.raises(ValueError, match=r"ochestrator_state_classes_by_name must be a mapping"):
        FirebaseRealtimeWorkflowStateRepo(
            app=object(),
            state_classes_by_name={},
            ochestrator_state_classes_by_name="bad",  # type: ignore[arg-type]
        )


# -----------------------------------------------------------------------
# Conversation persistence
# -----------------------------------------------------------------------


def test_save_and_query_conversation_ids(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_get_conversation_ids_returns_empty_for_unknown_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _make_repo(monkeypatch)
    assert repo.get_conversation_ids_for_user(user_id=uuid4()) == []


def test_is_conversation_id_exists_returns_false_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _make_repo(monkeypatch)
    assert (
        repo.is_conversation_id_for_user_id_exists(
            user_id=uuid4(), conversation_id=uuid4()
        )
        is False
    )


# -----------------------------------------------------------------------
# Orchestrator state
# -----------------------------------------------------------------------


def test_store_and_load_ochestrator_state_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    state = _DemoOchestratorState(_data={"key": "value", "count": 42})
    repo.store_ochestrator_state(
        user_id=user_id, conversation_id=conversation_id, state=state
    )
    assert state.get_update_counter() == 1
    assert (
        _stored_ochestrator_payload(
            fake_db,
            user_id=user_id,
            conversation_id=conversation_id,
        )["update_counter"]
        == 1
    )

    loaded = repo.load_ochestrator_state(
        user_id=user_id, conversation_id=conversation_id
    )
    assert isinstance(loaded, _DemoOchestratorState)
    assert loaded.name() == "DEMO_OCHESTRATOR"
    assert loaded.get("key") == "value"
    assert loaded.get("count") == 42
    assert loaded.get_update_counter() == 1


def test_store_ochestrator_state_second_save_increments_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    state = _DemoOchestratorState(_data={"key": "value"})
    repo.store_ochestrator_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state=state,
    )
    loaded = repo.load_ochestrator_state(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert isinstance(loaded, _DemoOchestratorState)

    loaded.set("key", {"next": True})
    repo.store_ochestrator_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state=loaded,
    )

    assert loaded.get_update_counter() == 2
    assert (
        _stored_ochestrator_payload(
            fake_db,
            user_id=user_id,
            conversation_id=conversation_id,
        )["update_counter"]
        == 2
    )


def test_store_ochestrator_state_stale_counter_raises_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    first = _DemoOchestratorState(_data={"key": "first"})
    repo.store_ochestrator_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state=first,
    )

    stale = _DemoOchestratorState(_data={"key": "stale", "update_counter": 0})
    with pytest.raises(StateConflictError) as exc_info:
        repo.store_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state=stale,
        )

    assert exc_info.value.code == "state_conflict"
    assert stale.get_update_counter() == 0


def test_store_ochestrator_state_legacy_payload_defaults_counter_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()
    fake_db.reference(
        f"/workflows/{user_id}/{conversation_id}/ochestrator_state"
    ).set(json.dumps({"name": "DEMO_OCHESTRATOR", "payload": {"key": "legacy"}}))

    loaded = repo.load_ochestrator_state(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert isinstance(loaded, _DemoOchestratorState)
    assert loaded.get_update_counter() == 0

    repo.store_ochestrator_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state=loaded,
    )

    assert loaded.get_update_counter() == 1
    assert (
        _stored_ochestrator_payload(
            fake_db,
            user_id=user_id,
            conversation_id=conversation_id,
        )["update_counter"]
        == 1
    )


def test_store_ochestrator_state_rejects_corrupt_stored_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()
    fake_db.reference(
        f"/workflows/{user_id}/{conversation_id}/ochestrator_state"
    ).set(
        json.dumps(
            {
                "name": "DEMO_OCHESTRATOR",
                "payload": {"key": "bad", "update_counter": "1"},
            }
        )
    )

    with pytest.raises(ValueError, match=r"payload.update_counter"):
        repo.store_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state=_DemoOchestratorState(),
        )


def test_load_ochestrator_state_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    assert (
        repo.load_ochestrator_state(user_id=user_id, conversation_id=conversation_id)
        is None
    )


def test_load_ochestrator_state_rejects_non_string_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    fake_db.reference(
        f"/workflows/{user_id}/{conversation_id}/ochestrator_state"
    ).set({"not": "a string"})

    with pytest.raises(ValueError, match=r"must be a JSON string blob"):
        repo.load_ochestrator_state(user_id=user_id, conversation_id=conversation_id)


def test_load_ochestrator_state_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    fake_db.reference(
        f"/workflows/{user_id}/{conversation_id}/ochestrator_state"
    ).set("not-json{{{")

    with pytest.raises(ValueError, match=r"is not valid JSON"):
        repo.load_ochestrator_state(user_id=user_id, conversation_id=conversation_id)


def test_load_ochestrator_state_rejects_non_dict_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    fake_db.reference(
        f"/workflows/{user_id}/{conversation_id}/ochestrator_state"
    ).set(json.dumps([1, 2]))

    with pytest.raises(ValueError, match=r"must be a dict"):
        repo.load_ochestrator_state(user_id=user_id, conversation_id=conversation_id)


def test_load_ochestrator_state_returns_none_for_empty_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    fake_db.reference(
        f"/workflows/{user_id}/{conversation_id}/ochestrator_state"
    ).set(json.dumps({"name": "", "payload": {}}))

    assert (
        repo.load_ochestrator_state(user_id=user_id, conversation_id=conversation_id)
        is None
    )


def test_load_ochestrator_state_rejects_non_dict_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    fake_db.reference(
        f"/workflows/{user_id}/{conversation_id}/ochestrator_state"
    ).set(json.dumps({"name": "DEMO_OCHESTRATOR", "payload": "bad"}))

    with pytest.raises(ValueError, match=r"must contain a dict 'payload'"):
        repo.load_ochestrator_state(user_id=user_id, conversation_id=conversation_id)


def test_load_ochestrator_state_rejects_unregistered_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    fake_db.reference(
        f"/workflows/{user_id}/{conversation_id}/ochestrator_state"
    ).set(json.dumps({"name": "UNKNOWN", "payload": {}}))

    with pytest.raises(KeyError, match=r"No OchestratorState class registered"):
        repo.load_ochestrator_state(user_id=user_id, conversation_id=conversation_id)


def test_load_ochestrator_state_wraps_deserialization_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    fake_db.reference(
        f"/workflows/{user_id}/{conversation_id}/ochestrator_state"
    ).set(json.dumps({"name": "DEMO_OCHESTRATOR", "payload": {"raise": True}}))

    with pytest.raises(ValueError, match=r"Error deserializing ochestrator_state"):
        repo.load_ochestrator_state(user_id=user_id, conversation_id=conversation_id)


def test_load_ochestrator_state_detects_name_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, fake_db = _make_repo(
        monkeypatch,
        ochestrator_state_classes_by_name={
            "DEMO_OCHESTRATOR": _MismatchOchestratorState,
        },
    )
    user_id, conversation_id = _ids()

    fake_db.reference(
        f"/workflows/{user_id}/{conversation_id}/ochestrator_state"
    ).set(json.dumps({"name": "DEMO_OCHESTRATOR", "payload": {}}))

    with pytest.raises(ValueError, match=r"OchestratorState.name mismatch"):
        repo.load_ochestrator_state(user_id=user_id, conversation_id=conversation_id)


# -----------------------------------------------------------------------
# Per-state persistence (NodeState)
# -----------------------------------------------------------------------


def test_store_and_load_state_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    state = _DemoNodeState(state_name="DEMO_STATE", text="roundtrip")
    repo.store_state(user_id=user_id, conversation_id=conversation_id, state=state)

    loaded = repo.load_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state_name="DEMO_STATE",
    )
    assert isinstance(loaded, _DemoNodeState)
    assert loaded.name() == "DEMO_STATE"
    assert loaded.text == "roundtrip"


def test_load_state_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    assert (
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )
        is None
    )


def test_load_state_validates_payload_and_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    with pytest.raises(ValueError, match=r"state_name must be a non-empty string"):
        repo.load_state(user_id=user_id, conversation_id=conversation_id, state_name="")

    # Non-string blob
    fake_db.reference(f"/workflows/{user_id}/{conversation_id}/states/DEMO_STATE").set(
        {"bad": "shape"}
    )
    with pytest.raises(ValueError, match=r"must be a JSON string blob"):
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )

    # Invalid JSON
    fake_db.reference(f"/workflows/{user_id}/{conversation_id}/states/DEMO_STATE").set("not-json")
    with pytest.raises(ValueError, match=r"not valid JSON"):
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )

    # Not a dict
    fake_db.reference(f"/workflows/{user_id}/{conversation_id}/states/DEMO_STATE").set(
        json.dumps([1, 2])
    )
    with pytest.raises(ValueError, match=r"must be a dict"):
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )

    # Unregistered state
    fake_db.reference(f"/workflows/{user_id}/{conversation_id}/states/UNKNOWN").set(
        json.dumps({"name": "UNKNOWN"})
    )
    with pytest.raises(KeyError, match=r"No State class registered"):
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="UNKNOWN",
        )


def test_load_state_wraps_deserialization_error(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_load_state_detects_name_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    mismatch_repo, mismatch_db = _make_repo(
        monkeypatch,
        state_classes_by_name={"DEMO_STATE": _MismatchNodeState},
    )
    user_id, conversation_id = _ids()

    mismatch_db.reference(f"/workflows/{user_id}/{conversation_id}/states/DEMO_STATE").set(
        json.dumps({"name": "DEMO_STATE"})
    )
    with pytest.raises(ValueError, match=r"Loaded State.name mismatch"):
        mismatch_repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )


def test_delete_state_removes_data(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    repo.store_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state=_DemoNodeState(state_name="DEMO_STATE", text="x"),
    )
    repo.delete_state(
        user_id=user_id, conversation_id=conversation_id, state_name="DEMO_STATE"
    )

    assert (
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )
        is None
    )


def test_delete_state_validates_name(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    with pytest.raises(ValueError, match=r"state_name must be a non-empty string"):
        repo.delete_state(
            user_id=user_id, conversation_id=conversation_id, state_name="  "
        )


# -----------------------------------------------------------------------
# Message history
# -----------------------------------------------------------------------


def test_append_and_load_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

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

    history = repo.load_message_history(
        user_id=user_id, conversation_id=conversation_id, limit=2
    )
    assert [msg.content for msg in history] == ["a3", "u4"]

    all_history = repo.load_message_history(
        user_id=user_id, conversation_id=conversation_id, limit=100
    )
    assert [msg.content for msg in all_history] == ["s1", "u2", "a3", "u4"]


def test_load_message_history_limit_zero_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    assert (
        repo.load_message_history(
            user_id=user_id, conversation_id=conversation_id, limit=0
        )
        == []
    )


def test_load_message_history_returns_empty_when_no_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    assert (
        repo.load_message_history(
            user_id=user_id, conversation_id=conversation_id, limit=10
        )
        == []
    )


def test_append_empty_messages_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, fake_db = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    repo.append_messages(
        user_id=user_id, conversation_id=conversation_id, messages=[]
    )
    # No messages node should exist
    assert (
        _get_value(
            fake_db.tree,
            ("workflows", str(user_id), str(conversation_id), "messages"),
        )
        is None
    )


def test_clear_message_history(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    repo.append_message(
        user_id=user_id,
        conversation_id=conversation_id,
        message=ChatMessage(role="user", content="hi"),
    )
    repo.clear_message_history(user_id=user_id, conversation_id=conversation_id)

    assert (
        repo.load_message_history(
            user_id=user_id, conversation_id=conversation_id, limit=10
        )
        == []
    )


def test_message_roundtrip_with_structured_artifacts(
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

    history = repo.load_message_history(
        user_id=user_id, conversation_id=conversation_id, limit=10
    )

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


def test_message_history_accepts_legacy_artifacts_ids(
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

    history = repo.load_message_history(
        user_id=user_id, conversation_id=conversation_id, limit=10
    )

    assert len(history) == 1
    assert history[0].artifact_refs == [
        {"id": csv_id, "kind": "data", "format": "csv", "artifact_meta": None},
        {"id": json_id, "kind": "data", "format": "json", "artifact_meta": None},
    ]
    assert history[0].id == "legacy-1"


def test_message_history_ignores_malformed_artifact_refs(
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

    history = repo.load_message_history(
        user_id=user_id, conversation_id=conversation_id, limit=10
    )

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


def test_message_without_artifacts_roundtrips_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _make_repo(monkeypatch)
    user_id, conversation_id = _ids()

    repo.append_message(
        user_id=user_id,
        conversation_id=conversation_id,
        message=ChatMessage(role="user", content="hello"),
    )

    history = repo.load_message_history(
        user_id=user_id, conversation_id=conversation_id, limit=10
    )
    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].content == "hello"
    assert history[0].artifact_refs is None
    assert history[0].artifacts is None
    assert history[0].id is None


# -----------------------------------------------------------------------
# get_default_firebase_database_app
# -----------------------------------------------------------------------


def test_get_default_firebase_database_app_returns_existing_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_app = object()
    monkeypatch.setattr(repo_module.firebase_admin, "get_app", lambda: existing_app)

    assert (
        FirebaseRealtimeWorkflowStateRepo.get_default_firebase_database_app()
        is existing_app
    )


def test_get_default_firebase_database_app_requires_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_get_default_firebase_database_app_initializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(
        repo_module.firebase_admin, "initialize_app", fake_initialize_app
    )
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_ID", "proj-1")
    monkeypatch.setenv("FIREBASE_DATABASE_URL", "https://db.example")

    app = FirebaseRealtimeWorkflowStateRepo.get_default_firebase_database_app()

    assert app is expected_app
    assert cred_factory.called is True
    assert captured["options"] == {
        "projectId": "proj-1",
        "databaseURL": "https://db.example",
    }
