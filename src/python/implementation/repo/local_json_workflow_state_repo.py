from __future__ import annotations

import itertools
import json
import os
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast
from uuid import UUID

from python.domain.models.errors import StateConflictError
from python.domain.models.models import ChatMessage
from python.domain.repo.workflow_state_repo import Conversation, WorkflowStateRepo
from python.domain.workflows.node_state import NodeState
from python.domain.workflows.ochestrator_state import OchestratorState

_PUSH_ID_COUNTER: itertools.count[int] = itertools.count()


class LocalJsonWorkflowStateRepo(WorkflowStateRepo):
    """
    JSON-file-backed workflow state repository for local development.

    The persisted JSON keeps a stable workflow-state shape for local runs.
    """

    _WORKFLOWS_ROOT = "workflows"
    _CONVERSATION_INDEX_ROOT = "workflow_conversation_index"

    def __init__(
        self,
        *,
        root_path: str | Path,
        state_classes_by_name: Mapping[str, type[NodeState]],
        ochestrator_state_classes_by_name: Mapping[str, type[OchestratorState]],
    ) -> None:
        self._root_path = Path(root_path).expanduser()
        self._root_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._state_classes_by_name = dict(state_classes_by_name)
        self._ochestrator_state_classes_by_name = dict(ochestrator_state_classes_by_name)

    def save_conversation(self, *, user_id: UUID, conversation: Conversation) -> None:
        with self._lock:
            data = self._load_db()
            conversation_path = self._conversation_parts(
                user_id=user_id,
                conversation_id=conversation.conversation_id,
            )
            meta_path = [*conversation_path, "_meta"]
            index_path = self._conversation_index_parts(
                user_id=user_id,
                conversation_id=conversation.conversation_id,
            )
            existing_meta = self._get_path(data, meta_path)
            existing_index = self._get_path(data, index_path)

            name = conversation.name
            if name is None and isinstance(existing_index, dict):
                existing_name = existing_index.get("name")
                if isinstance(existing_name, str):
                    name = existing_name

            self._set_path(
                data,
                index_path,
                {
                    "conversation_type": conversation.conversation_type,
                    "last_updated_at_utc": float(conversation.last_updated_at_utc),
                    "name": name,
                },
            )

            if not isinstance(existing_meta, dict):
                self._set_path(
                    data,
                    meta_path,
                    {
                        "created": True,
                        "created_at_utc": time.time(),
                    },
                )

            self._write_db(data)

    def get_conversations(self, *, user_id: UUID) -> Sequence[Conversation]:
        with self._lock:
            data = self._load_db()
            raw = self._get_path(
                data,
                [self._CONVERSATION_INDEX_ROOT, str(user_id)],
            )

        if not isinstance(raw, dict):
            return []

        conversations: list[Conversation] = []
        for key, value in raw.items():
            try:
                conversation_id = UUID(str(key))
            except (TypeError, ValueError):
                continue

            if not isinstance(value, dict):
                continue

            raw_type = value.get("conversation_type")
            if raw_type not in {"causal", "data"}:
                continue

            raw_name = value.get("name")
            if raw_name is not None and not isinstance(raw_name, str):
                continue

            last_updated_at_utc = self._coerce_float(value.get("last_updated_at_utc"))
            if last_updated_at_utc is None:
                continue

            conversations.append(
                Conversation(
                    name=raw_name,
                    conversation_id=conversation_id,
                    last_updated_at_utc=last_updated_at_utc,
                    conversation_type=cast(Any, raw_type),
                )
            )

        conversations.sort(
            key=lambda conversation: conversation.last_updated_at_utc,
            reverse=True,
        )
        return conversations

    def is_conversation_id_for_user_id_exists(
        self,
        *,
        user_id: UUID,
        conversation: Conversation,
    ) -> bool:
        with self._lock:
            data = self._load_db()
            return (
                self._get_path(
                    data,
                    self._conversation_index_parts(
                        user_id=user_id,
                        conversation_id=conversation.conversation_id,
                    ),
                )
                is not None
            )

    def load_ochestrator_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> OchestratorState | None:
        with self._lock:
            data = self._load_db()
            raw = self._get_path(
                data,
                [
                    *self._conversation_parts(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    ),
                    "ochestrator_state",
                ],
            )

        if raw is None:
            return None
        return self._deserialize_ochestrator_state(
            raw,
            conversation_id=conversation_id,
        )

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

        expected_counter = state.get_update_counter()
        saved_counter = expected_counter + 1
        payload_json = self._serialize_ochestrator_state(
            state=state,
            state_name=state_name,
            update_counter=saved_counter,
        )

        with self._lock:
            data = self._load_db()
            path = [
                *self._conversation_parts(
                    user_id=user_id,
                    conversation_id=conversation_id,
                ),
                "ochestrator_state",
            ]
            current_raw = self._get_path(data, path)
            current_counter = self._extract_ochestrator_update_counter(
                current_raw,
                conversation_id=conversation_id,
            )
            if current_counter is None:
                if expected_counter != 0:
                    raise StateConflictError(
                        state_name=state_name,
                        expected_counter=expected_counter,
                        actual_counter=None,
                    )
            elif current_counter != expected_counter:
                raise StateConflictError(
                    state_name=state_name,
                    expected_counter=expected_counter,
                    actual_counter=current_counter,
                )

            self._set_path(data, path, payload_json)
            self._write_db(data)

        state.set_update_counter(saved_counter)

    def load_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state_name: str,
    ) -> NodeState | None:
        if not isinstance(state_name, str) or not state_name.strip():
            raise ValueError("state_name must be a non-empty string")

        with self._lock:
            data = self._load_db()
            raw = self._get_path(
                data,
                [
                    *self._conversation_parts(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    ),
                    "states",
                    state_name,
                ],
            )

        if raw is None:
            return None
        return self._deserialize_state(raw, state_name=state_name)

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
            raise ValueError(f"State '{state_name}' is not JSON-serializable: {exc}") from exc

        with self._lock:
            data = self._load_db()
            self._set_path(
                data,
                [
                    *self._conversation_parts(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    ),
                    "states",
                    state_name,
                ],
                payload_json,
            )
            self._write_db(data)

    def delete_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state_name: str,
    ) -> None:
        if not isinstance(state_name, str) or not state_name.strip():
            raise ValueError("state_name must be a non-empty string")

        with self._lock:
            data = self._load_db()
            self._delete_path(
                data,
                [
                    *self._conversation_parts(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    ),
                    "states",
                    state_name,
                ],
            )
            self._write_db(data)

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

        with self._lock:
            data = self._load_db()
            messages_path = [
                *self._conversation_parts(
                    user_id=user_id,
                    conversation_id=conversation_id,
                ),
                "messages",
            ]
            for message in messages:
                self._set_path(
                    data,
                    [*messages_path, self._push_id()],
                    self._chat_message_to_dict(message),
                )
            self._write_db(data)

    def load_message_history(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        limit: int | None = 20,
    ) -> Sequence[ChatMessage]:
        if limit is not None and limit <= 0:
            return []

        with self._lock:
            data = self._load_db()
            raw = self._get_path(
                data,
                [
                    *self._conversation_parts(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    ),
                    "messages",
                ],
            )

        if not isinstance(raw, dict):
            return []

        keys = sorted(raw.keys())
        if limit is not None:
            keys = keys[-limit:]

        messages: list[ChatMessage] = []
        for key in keys:
            item = raw[key]
            if isinstance(item, dict):
                messages.append(self._chat_message_from_dict(item, message_key=key))
        return messages

    def clear_message_history(self, *, user_id: UUID, conversation_id: UUID) -> None:
        with self._lock:
            data = self._load_db()
            self._delete_path(
                data,
                [
                    *self._conversation_parts(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    ),
                    "messages",
                ],
            )
            self._write_db(data)

    def _load_db(self) -> dict[str, Any]:
        if not self._root_path.exists():
            return {}
        try:
            data = json.loads(self._root_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Local workflow state database is not valid JSON: {self._root_path}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"Local workflow state database root must be a JSON object: {self._root_path}"
            )
        return data

    def _write_db(self, data: dict[str, Any]) -> None:
        self._root_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._root_path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temp_path, self._root_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _get_path(data: dict[str, Any], parts: Sequence[str]) -> Any:
        node: Any = data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    @staticmethod
    def _set_path(data: dict[str, Any], parts: Sequence[str], value: Any) -> None:
        if not parts:
            raise ValueError("path must not be empty")
        node = data
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value

    @staticmethod
    def _delete_path(data: dict[str, Any], parts: Sequence[str]) -> None:
        if not parts:
            return
        node: Any = data
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                return
            node = node[part]
        if isinstance(node, dict):
            node.pop(parts[-1], None)

    def _conversation_index_parts(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> list[str]:
        return [
            self._CONVERSATION_INDEX_ROOT,
            str(user_id),
            str(conversation_id),
        ]

    def _conversation_parts(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> list[str]:
        return [
            self._WORKFLOWS_ROOT,
            str(user_id),
            str(conversation_id),
        ]

    def _deserialize_ochestrator_state(
        self,
        raw: Any,
        *,
        conversation_id: UUID,
    ) -> OchestratorState | None:
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
            raise KeyError(f"No OchestratorState class registered for name={state_name!r}")

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

    def _serialize_ochestrator_state(
        self,
        *,
        state: OchestratorState,
        state_name: str,
        update_counter: int,
    ) -> str:
        payload = dict(state.to_json_dict())
        payload["update_counter"] = update_counter
        envelope = {
            "name": state_name,
            "payload": payload,
        }

        try:
            return json.dumps(
                envelope,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"OchestratorState '{state_name}' is not JSON-serializable: {exc}"
            ) from exc

    def _extract_ochestrator_update_counter(
        self,
        current_raw: Any,
        *,
        conversation_id: UUID,
    ) -> int | None:
        if current_raw is None:
            return None

        if not isinstance(current_raw, str):
            raise ValueError(
                f"Stored ochestrator_state for conversation_id={conversation_id!r} "
                f"must be a JSON string blob, got {type(current_raw).__name__}"
            )

        try:
            envelope = json.loads(current_raw)
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

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Decoded ochestrator_state for conversation_id={conversation_id!r} "
                f"must contain a dict 'payload', got {type(payload).__name__}"
            )

        counter = payload.get("update_counter", 0)
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise ValueError(
                f"Decoded ochestrator_state for conversation_id={conversation_id!r} "
                "must contain a non-negative integer 'payload.update_counter'"
            )
        return counter

    def _deserialize_state(self, raw: Any, *, state_name: str) -> NodeState:
        if not isinstance(raw, str):
            raise ValueError(
                f"Stored state payload for state_name={state_name!r} must be "
                f"a JSON string blob, got {type(raw).__name__}"
            )

        try:
            state_dict = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Stored state blob for state_name={state_name!r} " f"is not valid JSON: {exc}"
            ) from exc

        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Decoded state payload for state_name={state_name!r} "
                f"must be a dict, got {type(state_dict).__name__}"
            )

        cls = self._state_classes_by_name.get(state_name)
        if cls is None:
            raise KeyError(f"No State class registered for state_name={state_name!r}")

        try:
            state = cls.from_json_dict(state_dict)
        except Exception as exc:
            raise ValueError(f"Error deserializing state '{state_name}': {exc}") from exc

        loaded_name = self._state_name_of(state)
        if loaded_name != state_name:
            raise ValueError(
                f"Loaded State.name mismatch: got {loaded_name!r}, " f"expected {state_name!r}"
            )
        return state

    def _chat_message_to_dict(self, message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": message.role,
            "message": message.content,
            "created_at_utc": float(message.created_at_utc),
        }
        artifact_refs = self._serialize_artifact_refs(getattr(message, "artifact_refs", None))
        artifacts = self._serialize_artifacts(getattr(message, "artifacts", None))
        message_id = getattr(message, "id", None)

        if artifact_refs is not None:
            payload["artifact_refs"] = artifact_refs
        if artifacts is not None:
            payload["artifacts"] = artifacts
        if message_id is not None:
            payload["id"] = str(message_id)

        return payload

    def _chat_message_from_dict(
        self,
        payload: dict[str, Any],
        *,
        message_key: str | None = None,
    ) -> ChatMessage:
        role = payload.get("role")
        content = payload.get("message", payload.get("content"))
        artifact_refs = self._normalize_artifact_refs(
            payload.get("artifact_refs", payload.get("artifacts_ids"))
        )
        artifacts = self._normalize_artifacts(payload.get("artifacts"))
        message_id = payload.get("id")
        created_at_utc = self._coerce_float(payload.get("created_at_utc"))
        if created_at_utc is None:
            created_at_utc = self._created_at_utc_from_message_key(message_key)

        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Invalid chat message payload: role/message must be strings")

        kwargs: dict[str, Any] = {
            "role": role,
            "content": content,
            "artifact_refs": artifact_refs,
            "artifacts": artifacts,
            "id": message_id if isinstance(message_id, str) else None,
        }
        if created_at_utc is not None:
            kwargs["created_at_utc"] = created_at_utc

        return ChatMessage(**kwargs)

    @staticmethod
    def _serialize_artifact_refs(value: Any) -> list[dict[str, Any]] | None:
        normalized = LocalJsonWorkflowStateRepo._normalize_artifact_refs(value)
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
        normalized = LocalJsonWorkflowStateRepo._normalize_artifacts(value)
        if normalized is None:
            return None
        return [LocalJsonWorkflowStateRepo._jsonify_nested(item) for item in normalized]

    @staticmethod
    def _normalize_artifact_refs(value: Any) -> list[dict[str, Any]] | None:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return None

        normalized: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue

            artifact_id = item.get("id")
            artifact_kind = item.get("kind")
            artifact_format = item.get("format")

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
                    artifact_id if isinstance(artifact_id, UUID) else UUID(str(artifact_id).strip())
                )
            except (TypeError, ValueError, AttributeError):
                continue

            normalized.append(
                {
                    "id": parsed_id,
                    "kind": artifact_kind,
                    "format": artifact_format,
                    "artifact_meta": LocalJsonWorkflowStateRepo._normalize_artifact_meta(
                        item.get("artifact_meta")
                    ),
                }
            )
        return normalized or None

    @staticmethod
    def _normalize_artifacts(value: Any) -> list[dict[str, Any]] | None:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
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

            entry["artifact_meta"] = LocalJsonWorkflowStateRepo._normalize_artifact_meta(
                entry.get("artifact_meta")
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

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _created_at_utc_from_message_key(message_key: str | None) -> float | None:
        if not isinstance(message_key, str):
            return None

        timestamp_part = message_key.split("_", 1)[0]
        if len(timestamp_part) != 16 or not timestamp_part.isdigit():
            return None
        return int(timestamp_part) / 1000.0

    @staticmethod
    def _push_id() -> str:
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
            return {str(k): LocalJsonWorkflowStateRepo._jsonify_nested(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [LocalJsonWorkflowStateRepo._jsonify_nested(v) for v in value]
        return value
