from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
import os
from typing import Any, Mapping, Optional, Sequence, Type
from uuid import UUID

from firebase_admin import credentials, db
import firebase_admin

from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.state import State


class FirebaseRealtimeWorkflowStateRepo(WorkflowStateRepo):
    
    @staticmethod
    def get_default_firebase_database_app() -> FirebaseRealtimeWorkflowStateRepo:
        try:
            return firebase_admin.get_app()
        except ValueError:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "").strip()
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT_ID environment variable must be set for FirebaseRealtimeWorkflowStateRepo")
            database_url = os.getenv("FIREBASE_DATABASE_URL", "").strip()
            if not database_url:
                raise ValueError("FIREBASE_DATABASE_URL environment variable must be set for FirebaseRealtimeWorkflowStateRepo")

            options: dict[str, str] = {}
            options["projectId"] = project_id
            options["databaseURL"] = database_url

            return firebase_admin.initialize_app(
                credentials.ApplicationDefault(),
                options or None,
            )
        
        
    def __init__(
        self,
        *,
        app: Any,
        state_classes_by_name: Mapping[str, Type[State]],
    ) -> None:
        root_path: str = "/workflows"
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
        value = (
            self._conversation_ref(user_id=user_id, conversation_id=conversation_id)
            .child("active_state_name")
            .get()
        )
        return value if isinstance(value, str) and value.strip() else None

    def store_active_state_name(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> None:
        if not isinstance(state_name, str) or not state_name.strip():
            raise ValueError("state_name must be a non-empty string")

        (
            self._conversation_ref(user_id=user_id, conversation_id=conversation_id)
            .child("active_state_name")
            .set(state_name)
        )

    # -----------------------
    # Per-state persistence
    # -----------------------

    def load_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> Optional[State]:
        if not isinstance(state_name, str) or not state_name.strip():
            raise ValueError("state_name must be a non-empty string")

        payload = (
            self._conversation_ref(user_id=user_id, conversation_id=conversation_id)
            .child("states")
            .child(state_name)
            .get()
        )
        if payload is None:
            return None
        if not isinstance(payload, str):
            raise ValueError(
                f"Stored state payload for state_name={state_name!r} must be a JSON string blob, "
                f"got {type(payload).__name__}"
            )

        cls = self._state_classes_by_name.get(state_name)
        if cls is None:
            raise KeyError(f"No State class registered for state_name={state_name!r}")

        try:
            state_dict = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Stored state blob for state_name={state_name!r} is not valid JSON: {exc}") from exc

        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Decoded state payload for state_name={state_name!r} must be a dict, "
                f"got {type(state_dict).__name__}"
            )

        try:
            state = cls.from_json_dict(state_dict)
        except Exception as exc:
            raise ValueError(f"Error deserializing state '{state_name}': {exc}") from exc

        if getattr(state, "name", None) != state_name:
            raise ValueError(f"Loaded State.name mismatch: got {state.name!r}, expected {state_name!r}")

        return state

    def store_state(self, *, user_id: UUID, conversation_id: UUID, state: State) -> None:
        if not isinstance(state.name, str) or not state.name.strip():
            raise ValueError("state.name must be a non-empty string")

        try:
            payload_json = json.dumps(
                state.to_json_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"State '{state.name}' is not JSON-serializable: {exc}") from exc

        (
            self._conversation_ref(user_id=user_id, conversation_id=conversation_id)
            .child("states")
            .child(state.name)
            .set(payload_json)
        )

    def delete_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> None:
        if not isinstance(state_name, str) or not state_name.strip():
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



    
        