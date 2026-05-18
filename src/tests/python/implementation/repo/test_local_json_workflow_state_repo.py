from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

from python.domain.models.errors import StateConflictError
from python.domain.models.models import ChatMessage
from python.domain.repo.workflow_state_repo import Conversation
from python.domain.workflows.node_state import NodeState
from python.domain.workflows.ochestrator_state import OchestratorState
from python.implementation.repo.local_json_workflow_state_repo import (
    LocalJsonWorkflowStateRepo,
)


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
        name = payload.get("name")
        if not isinstance(name, str):
            raise ValueError("name must be string")
        return cls(state_name=name, text=str(payload.get("text", "")))

    @classmethod
    def init_empty(cls) -> _DemoNodeState:
        return cls()


@dataclass
class _DemoOchestratorState(OchestratorState):
    _data: dict[str, Any] = field(default_factory=dict)

    def name(self) -> str:
        return "DEMO_OCHESTRATOR"

    def get_update_counter(self) -> int:
        value = self._data.get("update_counter", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("update_counter must be a non-negative integer")
        return value

    def set_update_counter(self, value: int) -> None:
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
        return cls(_data=dict(payload))

    @classmethod
    def init_empty(cls) -> _DemoOchestratorState:
        return cls()


def _make_repo(tmp_path) -> LocalJsonWorkflowStateRepo:
    return LocalJsonWorkflowStateRepo(
        root_path=tmp_path / "workflow_state.json",
        state_classes_by_name={"DEMO_STATE": _DemoNodeState},
        ochestrator_state_classes_by_name={"DEMO_OCHESTRATOR": _DemoOchestratorState},
    )


def _ids() -> tuple[UUID, UUID]:
    return uuid4(), uuid4()


def test_save_and_list_conversations_preserves_existing_name(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    user_id, conversation_id = _ids()

    repo.save_conversation(
        user_id=user_id,
        conversation=Conversation(
            name="First",
            conversation_id=conversation_id,
            conversation_type="causal",
            last_updated_at_utc=1.0,
        ),
    )
    repo.save_conversation(
        user_id=user_id,
        conversation=Conversation(
            name=None,
            conversation_id=conversation_id,
            conversation_type="causal",
            last_updated_at_utc=2.0,
        ),
    )

    conversations = repo.get_conversations(user_id=user_id)
    assert len(conversations) == 1
    assert conversations[0].name == "First"
    assert conversations[0].last_updated_at_utc == 2.0


def test_store_load_and_delete_state(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    user_id, conversation_id = _ids()

    repo.store_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state=_DemoNodeState(text="roundtrip"),
    )

    loaded = repo.load_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state_name="DEMO_STATE",
    )
    assert isinstance(loaded, _DemoNodeState)
    assert loaded.text == "roundtrip"

    repo.delete_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state_name="DEMO_STATE",
    )
    assert (
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )
        is None
    )


def test_store_load_ochestrator_state_and_detect_conflicts(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    user_id, conversation_id = _ids()

    first = _DemoOchestratorState(_data={"key": "first"})
    repo.store_ochestrator_state(
        user_id=user_id,
        conversation_id=conversation_id,
        state=first,
    )
    assert first.get_update_counter() == 1

    loaded = repo.load_ochestrator_state(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert isinstance(loaded, _DemoOchestratorState)
    assert loaded.get("key") == "first"
    assert loaded.get_update_counter() == 1

    stale = _DemoOchestratorState(_data={"key": "stale", "update_counter": 0})
    with pytest.raises(StateConflictError):
        repo.store_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state=stale,
        )


def test_append_load_and_clear_message_history(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    user_id, conversation_id = _ids()

    repo.append_messages(
        user_id=user_id,
        conversation_id=conversation_id,
        messages=[
            ChatMessage(role="user", content="one", created_at_utc=1.0),
            ChatMessage(role="assistant", content="two", created_at_utc=2.0),
            ChatMessage(role="user", content="three", created_at_utc=3.0),
        ],
    )

    limited = repo.load_message_history(
        user_id=user_id,
        conversation_id=conversation_id,
        limit=2,
    )
    assert [message.content for message in limited] == ["two", "three"]

    repo.clear_message_history(user_id=user_id, conversation_id=conversation_id)
    assert repo.load_message_history(user_id=user_id, conversation_id=conversation_id) == []


def test_invalid_stored_state_json_raises_value_error(tmp_path) -> None:
    repo = _make_repo(tmp_path)
    user_id, conversation_id = _ids()
    db_path = tmp_path / "workflow_state.json"
    db_path.write_text(
        json.dumps(
            {
                "workflows": {
                    str(user_id): {
                        str(conversation_id): {
                            "states": {
                                "DEMO_STATE": "not-json",
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"is not valid JSON"):
        repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name="DEMO_STATE",
        )
