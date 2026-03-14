from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Optional, Sequence, Type
from uuid import UUID

from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.state import State


class FirebaseRealtimeWorkflowStateRepo(WorkflowStateRepo):
    def __init__(
        self,
        *,
        app: Any,
        root_path: str = "/workflows",
        state_classes_by_name: Mapping[str, Type[State]],
    ) -> None:
        if not root_path.strip():
            raise ValueError("root_path must be a non-empty string")

        from firebase_admin import db

        self._root_ref = db.reference(root_path, app=app)
        self._state_classes_by_name = dict(state_classes_by_name)

    # -----------------------
    # Conversation persistence
    # -----------------------

    def save_conversation_id(self, *, user_id: UUID, conversation_id: UUID) -> None:
        conversation_ref = self._conversation_ref(user_id=user_id, conversation_id=conversation_id)
        if conversation_ref.get() is None:
            conversation_ref.set({})

    # -----------------------
    # Active stage pointer
    # -----------------------

    def get_conversation_ids_for_user(self, *, user_id: UUID) -> Sequence[UUID]:
        data = self._user_ref(user_id=user_id).get()
        if not isinstance(data, dict):
            return []

        conversation_ids: list[UUID] = []
        for key in sorted(data.keys()):
            try:
                conversation_ids.append(UUID(key))
            except ValueError:
                continue
        return conversation_ids

    def load_active_state_name(self, *, user_id: UUID, conversation_id: UUID) -> Optional[str]:
        value = self._conversation_ref(user_id=user_id, conversation_id=conversation_id).child(
            "active_state_name"
        ).get()
        return value if isinstance(value, str) and value.strip() else None

    def store_active_state_name(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> None:
        if not state_name.strip():
            raise ValueError("state_name must be a non-empty string")

        self._conversation_ref(user_id=user_id, conversation_id=conversation_id).child(
            "active_state_name"
        ).set(state_name)

    # -----------------------
    # Per-state persistence
    # -----------------------

    def load_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> Optional[State]:
        if not state_name.strip():
            raise ValueError("state_name must be a non-empty string")

        payload = self._conversation_ref(user_id=user_id, conversation_id=conversation_id).child("states").child(
            state_name
        ).get()
        if not isinstance(payload, dict):
            return None

        cls = self._state_classes_by_name.get(state_name)
        if cls is None:
            raise KeyError(f"No State class registered for state_name={state_name!r}")

        try:
            state = cls.from_json_dict(payload)
        except Exception as exc:
            raise ValueError(f"Error deserializing state '{state_name}': {exc}") from exc

        if getattr(state, "name", None) != state_name:
            raise ValueError(f"Loaded State.name mismatch: got {state.name!r}, expected {state_name!r}")

        return state

    def store_state(self, *, user_id: UUID, conversation_id: UUID, state: State) -> None:
        if not state.name.strip():
            raise ValueError("state.name must be a non-empty string")

        self._conversation_ref(user_id=user_id, conversation_id=conversation_id).child("states").child(
            state.name
        ).set(state.to_json_dict())

    def delete_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> None:
        if not state_name.strip():
            raise ValueError("state_name must be a non-empty string")

        conversation_ref = self._conversation_ref(user_id=user_id, conversation_id=conversation_id)
        active_state_name = conversation_ref.child("active_state_name").get()
        conversation_ref.child("states").child(state_name).delete()
        if active_state_name == state_name:
            conversation_ref.child("active_state_name").delete()

    # -----------------------
    # Message history
    # -----------------------

    def append_message(self, *, user_id: UUID, conversation_id: UUID, message: ChatMessage) -> None:
        self.append_messages(user_id=user_id, conversation_id=conversation_id, messages=[message])

    def append_messages(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        messages: Sequence[ChatMessage],
    ) -> None:
        if not messages:
            return

        messages_ref = self._conversation_ref(user_id=user_id, conversation_id=conversation_id).child("messages")
        for message in messages:
            messages_ref.push(self._chat_message_to_dict(message))

    def load_message_history(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        limit: int = 20,
    ) -> Sequence[ChatMessage]:
        if limit <= 0:
            return []

        data = (
            self._conversation_ref(user_id=user_id, conversation_id=conversation_id)
            .child("messages")
            .order_by_key()
            .limit_to_last(limit)
            .get()
        )
        if not isinstance(data, dict):
            return []

        messages: list[ChatMessage] = []
        for key in sorted(data.keys()):
            item = data[key]
            if isinstance(item, dict):
                messages.append(self._chat_message_from_dict(item))
        return messages

    def clear_message_history(self, *, user_id: UUID, conversation_id: UUID) -> None:
        self._conversation_ref(user_id=user_id, conversation_id=conversation_id).child("messages").delete()

    # -----------------------
    # Internals
    # -----------------------

    def _user_ref(self, *, user_id: UUID) -> Any:
        return self._root_ref.child(str(user_id))

    def _conversation_ref(self, *, user_id: UUID, conversation_id: UUID) -> Any:
        return self._user_ref(user_id=user_id).child(str(conversation_id))

    def _chat_message_to_dict(self, message: ChatMessage) -> dict[str, Any]:
        if is_dataclass(message):
            return asdict(message)
        return {"role": message.role, "content": message.content}

    def _chat_message_from_dict(self, payload: dict[str, Any]) -> ChatMessage:
        return ChatMessage(**payload)
