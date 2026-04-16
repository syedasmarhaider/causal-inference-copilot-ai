from __future__ import annotations

import itertools
import json
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import firebase_admin
from firebase_admin import credentials, db

from python.domain.models.models import ChatMessage
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.workflows.node_state import NodeState
from python.domain.workflows.ochestrator_state import OchestratorState

# Monotonically increasing counter for push IDs — ensures same-millisecond messages
# are always stored and returned in insertion order.
_PUSH_ID_COUNTER: itertools.count[int] = itertools.count()


class FirebaseRealtimeWorkflowStateRepo(WorkflowStateRepo):
    """
    Firebase RTDB-backed workflow state repository.

    Storage layout::

        /workflow_conversation_index/{user_id}/{conversation_id}: true

        /workflows/{user_id}/{conversation_id}/_meta:
            created: true

        /workflows/{user_id}/{conversation_id}/ochestrator_state: json-string
        /workflows/{user_id}/{conversation_id}/states/{state_name}: json-string
        /workflows/{user_id}/{conversation_id}/messages/{push_id}: ChatMessage dict
    """

    _WORKFLOWS_ROOT = "/workflows"
    _CONVERSATION_INDEX_ROOT = "/workflow_conversation_index"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def get_default_firebase_database_app() -> Any:
        try:
            return firebase_admin.get_app()
        except ValueError:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "").strip()
            if not project_id:
                raise ValueError(
                    "GOOGLE_CLOUD_PROJECT_ID environment variable must be set"
                ) from None

            database_url = os.getenv("FIREBASE_DATABASE_URL", "").strip()
            if not database_url:
                raise ValueError(
                    "FIREBASE_DATABASE_URL environment variable must be set"
                ) from None

            return firebase_admin.initialize_app(
                credentials.ApplicationDefault(),
                {
                    "projectId": project_id,
                    "databaseURL": database_url,
                },
            )

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        app: Any,
        state_classes_by_name: Mapping[str, type[NodeState]],
        ochestrator_state_classes_by_name: Mapping[str, type[OchestratorState]],
    ) -> None:
        if app is None:
            raise ValueError("app must not be None")
        if not isinstance(state_classes_by_name, Mapping):
            raise ValueError("state_classes_by_name must be a mapping")
        if not isinstance(ochestrator_state_classes_by_name, Mapping):
            raise ValueError("ochestrator_state_classes_by_name must be a mapping")

        self._root_ref = db.reference("/", app=app)
        self._state_classes_by_name = dict(state_classes_by_name)
        self._ochestrator_state_classes_by_name = dict(
            ochestrator_state_classes_by_name
        )

    # ------------------------------------------------------------------
    # Conversation persistence
    # ------------------------------------------------------------------

    def save_conversation_id(self, *, user_id: UUID, conversation_id: UUID) -> None:
        updates = {
            self._conversation_index_path(
                user_id=user_id,
                conversation_id=conversation_id,
            ): True,
            self._conversation_meta_path(
                user_id=user_id,
                conversation_id=conversation_id,
            ): {"created": True},
        }
        self._root_ref.update(updates)

    def get_conversation_ids_for_user(self, *, user_id: UUID) -> Sequence[UUID]:
        data = self._conversation_index_user_ref(user_id=user_id).get()
        if not isinstance(data, dict):
            return []

        conversation_ids: list[UUID] = []
        for key in sorted(data.keys()):
            try:
                conversation_ids.append(UUID(key))
            except (TypeError, ValueError):
                continue
        return conversation_ids

    def is_conversation_id_for_user_id_exists(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> bool:
        value = self._conversation_index_ref(
            user_id=user_id,
            conversation_id=conversation_id,
        ).get()
        return value is not None

    # ------------------------------------------------------------------
    # Orchestrator state
    # ------------------------------------------------------------------

    def load_ochestrator_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> OchestratorState | None:
        raw = (
            self._conversation_ref(user_id=user_id, conversation_id=conversation_id)
            .child("ochestrator_state")
            .get()
        )
        if raw is None:
            return None

        if not isinstance(raw, str):
            raise ValueError(
                f"Stored ochestrator_state for conversation_id={conversation_id!r} "
                f"must be a JSON string blob, got {type(raw).__name__}"
            )

        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Stored ochestrator_state for conversation_id={conversation_id!r} "
                f"is not valid JSON: {exc}"
            ) from exc

        if not isinstance(envelope, dict):
            raise ValueError(
                f"Decoded ochestrator_state for conversation_id={conversation_id!r} "
                f"must be a dict, got {type(envelope).__name__}"
            )

        state_name = envelope.get("name")
        state_payload = envelope.get("payload")

        if not isinstance(state_name, str) or not state_name.strip():
            return None

        if not isinstance(state_payload, dict):
            raise ValueError(
                f"Decoded ochestrator_state for conversation_id={conversation_id!r} "
                f"must contain a dict 'payload', got {type(state_payload).__name__}"
            )

        cls = self._ochestrator_state_classes_by_name.get(state_name)
        if cls is None:
            raise KeyError(
                f"No OchestratorState class registered for name={state_name!r}"
            )

        try:
            state = cls.from_json_dict(state_payload)
        except Exception as exc:
            raise ValueError(
                f"Error deserializing ochestrator_state '{state_name}': {exc}"
            ) from exc

        loaded_name = self._state_name_of(state)
        if loaded_name != state_name:
            raise ValueError(
                f"Loaded OchestratorState.name mismatch: got {loaded_name!r}, "
                f"expected {state_name!r}"
            )

        return state

    def store_ochestrator_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: OchestratorState,
    ) -> None:
        state_name = self._state_name_of(state)
        if not isinstance(state_name, str) or not state_name.strip():
            raise ValueError("ochestrator state name must be a non-empty string")

        envelope = {
            "name": state_name,
            "payload": state.to_json_dict(),
        }

        try:
            payload_json = json.dumps(
                envelope,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"OchestratorState '{state_name}' is not JSON-serializable: {exc}"
            ) from exc

        path = self._conversation_path(user_id=user_id, conversation_id=conversation_id)
        self._root_ref.update({f"{path}/ochestrator_state": payload_json})

    # ------------------------------------------------------------------
    # Per-state persistence
    # ------------------------------------------------------------------

    def load_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state_name: str,
    ) -> NodeState | None:
        if not isinstance(state_name, str) or not state_name.strip():
            raise ValueError("state_name must be a non-empty string")

        raw = (
            self._conversation_ref(user_id=user_id, conversation_id=conversation_id)
            .child("states")
            .child(state_name)
            .get()
        )

        if raw is None:
            return None

        if not isinstance(raw, str):
            raise ValueError(
                f"Stored state payload for state_name={state_name!r} must be "
                f"a JSON string blob, got {type(raw).__name__}"
            )

        try:
            state_dict = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Stored state blob for state_name={state_name!r} "
                f"is not valid JSON: {exc}"
            ) from exc

        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Decoded state payload for state_name={state_name!r} "
                f"must be a dict, got {type(state_dict).__name__}"
            )

        cls = self._state_classes_by_name.get(state_name)
        if cls is None:
            raise KeyError(
                f"No State class registered for state_name={state_name!r}"
            )

        try:
            state = cls.from_json_dict(state_dict)
        except Exception as exc:
            raise ValueError(
                f"Error deserializing state '{state_name}': {exc}"
            ) from exc

        loaded_name = self._state_name_of(state)
        if loaded_name != state_name:
            raise ValueError(
                f"Loaded State.name mismatch: got {loaded_name!r}, "
                f"expected {state_name!r}"
            )

        return state

    def store_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state: NodeState,
    ) -> None:
        state_name = self._state_name_of(state)
        if not isinstance(state_name, str) or not state_name.strip():
            raise ValueError("state.name must be a non-empty string")

        try:
            payload_json = json.dumps(
                state.to_json_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"State '{state_name}' is not JSON-serializable: {exc}"
            ) from exc

        path = self._conversation_path(user_id=user_id, conversation_id=conversation_id)
        self._root_ref.update({f"{path}/states/{state_name}": payload_json})

    def delete_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state_name: str,
    ) -> None:
        if not isinstance(state_name, str) or not state_name.strip():
            raise ValueError("state_name must be a non-empty string")

        path = self._conversation_path(user_id=user_id, conversation_id=conversation_id)
        self._root_ref.update({f"{path}/states/{state_name}": None})

    # ------------------------------------------------------------------
    # Message history
    # ------------------------------------------------------------------

    def append_message(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        message: ChatMessage,
    ) -> None:
        self.append_messages(
            user_id=user_id,
            conversation_id=conversation_id,
            messages=[message],
        )

    def append_messages(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        messages: Sequence[ChatMessage],
    ) -> None:
        if not messages:
            return

        base_path = self._conversation_path(user_id=user_id, conversation_id=conversation_id)
        updates: dict[str, Any] = {}
        for message in messages:
            key = self._push_id()
            updates[f"{base_path}/messages/{key}"] = self._chat_message_to_dict(message)
        self._root_ref.update(updates)

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
        path = self._conversation_path(user_id=user_id, conversation_id=conversation_id)
        self._root_ref.update({f"{path}/messages": None})

    # ------------------------------------------------------------------
    # Internal — Firebase references & paths
    # ------------------------------------------------------------------

    def _conversation_index_user_ref(self, *, user_id: UUID) -> Any:
        return self._root_ref.child("workflow_conversation_index").child(str(user_id))

    def _conversation_index_ref(
        self, *, user_id: UUID, conversation_id: UUID
    ) -> Any:
        return self._conversation_index_user_ref(user_id=user_id).child(
            str(conversation_id)
        )

    def _user_ref(self, *, user_id: UUID) -> Any:
        return self._root_ref.child("workflows").child(str(user_id))

    def _conversation_ref(
        self, *, user_id: UUID, conversation_id: UUID
    ) -> Any:
        return self._user_ref(user_id=user_id).child(str(conversation_id))

    def _conversation_index_path(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> str:
        root = self._CONVERSATION_INDEX_ROOT.strip("/")
        return f"{root}/{user_id}/{conversation_id}"

    def _conversation_path(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> str:
        root = self._WORKFLOWS_ROOT.strip("/")
        return f"{root}/{user_id}/{conversation_id}"

    def _conversation_meta_path(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> str:
        base = self._conversation_path(
            user_id=user_id, conversation_id=conversation_id
        )
        return f"{base}/_meta"

    # ------------------------------------------------------------------
    # Internal — ChatMessage serialisation
    # ------------------------------------------------------------------

    def _chat_message_to_dict(self, message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": message.role,
            "message": message.content,
        }

        artifact_refs = self._serialize_artifact_refs(
            getattr(message, "artifact_refs", None)
        )
        artifacts = self._serialize_artifacts(
            getattr(message, "artifacts", None)
        )
        message_id = getattr(message, "id", None)

        if artifact_refs is not None:
            payload["artifact_refs"] = artifact_refs
        if artifacts is not None:
            payload["artifacts"] = artifacts
        if message_id is not None:
            payload["id"] = str(message_id)

        return payload

    def _chat_message_from_dict(self, payload: dict[str, Any]) -> ChatMessage:
        role = payload.get("role")
        content = payload.get("message", payload.get("content"))
        artifact_refs = self._normalize_artifact_refs(
            payload.get("artifact_refs", payload.get("artifacts_ids"))
        )
        artifacts = self._normalize_artifacts(payload.get("artifacts"))
        message_id = payload.get("id")

        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(
                "Invalid chat message payload: role/message must be strings"
            )

        return ChatMessage(
            role=role,  # type: ignore[arg-type]
            content=content,
            artifact_refs=artifact_refs,
            artifacts=artifacts,
            id=message_id if isinstance(message_id, str) else None,
        )

    # ------------------------------------------------------------------
    # Internal — artifact helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_artifact_refs(value: Any) -> list[dict[str, Any]] | None:
        normalized = FirebaseRealtimeWorkflowStateRepo._normalize_artifact_refs(
            value
        )
        if normalized is None:
            return None

        serialized: list[dict[str, Any]] = []
        for item in normalized:
            entry: dict[str, Any] = {
                "id": str(item["id"]),
                "kind": item["kind"],
                "format": item["format"],
            }
            meta = item.get("artifact_meta")
            if meta is not None:
                entry["artifact_meta"] = dict(meta)
            serialized.append(entry)

        return serialized

    @staticmethod
    def _serialize_artifacts(value: Any) -> list[dict[str, Any]] | None:
        normalized = FirebaseRealtimeWorkflowStateRepo._normalize_artifacts(value)
        if normalized is None:
            return None
        return [
            FirebaseRealtimeWorkflowStateRepo._jsonify_nested(item)
            for item in normalized
        ]

    @staticmethod
    def _normalize_artifact_refs(value: Any) -> list[dict[str, Any]] | None:
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            return None

        normalized: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue

            artifact_id = item.get("id")
            artifact_kind = item.get("kind")
            artifact_format = item.get("format")

            # Legacy format: {"id": ..., "type": "csv"} → kind="data", format="csv"
            if (
                artifact_kind is None
                and artifact_format is None
                and item.get("type") in {"csv", "json"}
            ):
                artifact_kind = "data"
                artifact_format = item.get("type")

            if artifact_kind not in {"graph", "data"}:
                continue
            if artifact_format not in {"csv", "json"}:
                continue

            try:
                parsed_id = (
                    artifact_id
                    if isinstance(artifact_id, UUID)
                    else UUID(str(artifact_id).strip())
                )
            except (TypeError, ValueError, AttributeError):
                continue

            normalized.append(
                {
                    "id": parsed_id,
                    "kind": artifact_kind,
                    "format": artifact_format,
                    "artifact_meta": FirebaseRealtimeWorkflowStateRepo._normalize_artifact_meta(
                        item.get("artifact_meta")
                    ),
                }
            )

        return normalized or None

    @staticmethod
    def _normalize_artifacts(value: Any) -> list[dict[str, Any]] | None:
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            return None

        normalized: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue

            entry = dict(item)

            if "id" in entry:
                try:
                    entry["id"] = (
                        entry["id"]
                        if isinstance(entry["id"], UUID)
                        else UUID(str(entry["id"]).strip())
                    )
                except (TypeError, ValueError, AttributeError):
                    entry.pop("id", None)

            kind = entry.get("kind")
            if kind is not None and kind not in {"graph", "data"}:
                entry.pop("kind", None)

            fmt = entry.get("format")
            if fmt is not None and fmt not in {"csv", "json"}:
                entry.pop("format", None)

            entry["artifact_meta"] = (
                FirebaseRealtimeWorkflowStateRepo._normalize_artifact_meta(
                    entry.get("artifact_meta")
                )
            )

            normalized.append(entry)

        return normalized or None

    @staticmethod
    def _normalize_artifact_meta(value: Any) -> dict[str, str] | None:
        if not isinstance(value, dict):
            return None

        result: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            if item is None:
                continue
            result[key] = str(item)

        return result or None

    # ------------------------------------------------------------------
    # Internal — misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _push_id() -> str:
        """Generate a time-sortable key for RTDB message entries."""
        ts_ms = int(time.time() * 1000)
        seq = next(_PUSH_ID_COUNTER)
        return f"{ts_ms:016d}_{seq:020d}_{uuid.uuid4().hex}"

    @staticmethod
    def _state_name_of(state: Any) -> str | None:
        candidate = getattr(state, "name", None)
        if callable(candidate):
            candidate = candidate()
        return candidate if isinstance(candidate, str) else None

    @staticmethod
    def _jsonify_nested(value: Any) -> Any:
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, dict):
            return {
                str(k): FirebaseRealtimeWorkflowStateRepo._jsonify_nested(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                FirebaseRealtimeWorkflowStateRepo._jsonify_nested(v) for v in value
            ]
        return value