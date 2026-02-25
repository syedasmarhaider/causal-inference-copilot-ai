from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Type
from uuid import UUID

from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.state import State


@dataclass(frozen=True)
class JsonFileRepoConfig:
    file_name_suffix: str = ".json"
    lock_timeout_s: float = 5.0
    lock_poll_min_s: float = 0.01
    lock_poll_max_s: float = 0.10
    lock_stale_s: float = 120.0

    keep_backup: bool = True
    fsync_writes: bool = True

    max_messages_kept: int = 300  # keep last N messages in file; prevents unbounded growth



class _FileLock:
    """
    Portable lock-file based mutual exclusion.

    Implementation:
    - Acquire by creating lock file with O_CREAT|O_EXCL (atomic on POSIX/Windows).
    - Write pid + timestamp into lock file.
    - If lock exists and is older than stale threshold -> break it.
    """
    def __init__(
        self,
        lock_path: Path,
        *,
        timeout_s: float,
        stale_s: float,
        poll_min_s: float,
        poll_max_s: float,
    ) -> None:
        self._lock_path = lock_path
        self._timeout_s = timeout_s
        self._stale_s = stale_s
        self._poll_min_s = poll_min_s
        self._poll_max_s = poll_max_s
        self._fd: Optional[int] = None

    def __enter__(self) -> "_FileLock":
        deadline = time.monotonic() + self._timeout_s
        while True:
            try:
                fd = os.open(str(self._lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                self._fd = fd
                payload = f"pid={os.getpid()} ts={time.time()}\n"
                os.write(fd, payload.encode("utf-8", errors="replace"))
                os.fsync(fd)
                return self
            except FileExistsError:
                self._maybe_break_stale_lock()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timeout acquiring lock: {self._lock_path}")
                time.sleep(random.uniform(self._poll_min_s, self._poll_max_s))

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                finally:
                    self._fd = None
        finally:
            # Release lock by deleting lock file
            try:
                self._lock_path.unlink(missing_ok=True)  # py3.8+ supports missing_ok
            except TypeError:
                # py<3.8 compatibility (likely not needed)
                if self._lock_path.exists():
                    self._lock_path.unlink()

    def _maybe_break_stale_lock(self) -> None:
        try:
            st = self._lock_path.stat()
        except FileNotFoundError:
            return

        age = time.time() - st.st_mtime
        if age < self._stale_s:
            return

        # Stale lock -> break it
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            # Another process might be racing to delete; ignore
            return


class JsonFileWorkflowStateRepo(WorkflowStateRepo):
    """
    Robust JSON-file implementation (single file per conversation).

    Layout:
      base_dir/
        <user_id>/
          <conversation_id>.json
          <conversation_id>.json.lock

    File schema (v1):
    {
      "schema_version": 1,
      "active_state_name": "LOAD_DATASET" | null,
      "states": {
        "<STATE_NAME>": {
          "class": "ConcreteStateClassName",
          "payload": { ...state.to_json_dict()... }
        }
      },
      "messages": [ { ...ChatMessage... }, ... ],
      "updated_at": 1730000000.0
    }
    """

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        base_dir: str | Path,
        state_classes_by_name: Mapping[str, Type[State]],
        config: Optional[JsonFileRepoConfig] = None,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._state_classes_by_name = dict(state_classes_by_name)
        self._cfg = config or JsonFileRepoConfig()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Active stage pointer
    # -----------------------

    def load_active_state_name(self, *, user_id: UUID, conversation_id: UUID) -> Optional[str]:
        with self._lock(user_id=user_id, conversation_id=conversation_id):
            data = self._read_unlocked(user_id=user_id, conversation_id=conversation_id)
            name = data.get("active_state_name")
            return name if isinstance(name, str) and name else None

    def store_active_state_name(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> None:
        if not state_name.strip():
            raise ValueError("state_name must be a non-empty string")

        with self._lock(user_id=user_id, conversation_id=conversation_id):
            data = self._read_unlocked(user_id=user_id, conversation_id=conversation_id)
            data["active_state_name"] = state_name
            data["updated_at"] = time.time()
            self._write_unlocked(user_id=user_id, conversation_id=conversation_id, data=data)

    # -----------------------
    # Per-state persistence
    # -----------------------

    def load_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> Optional[State]:
        if not state_name.strip():
            raise ValueError("state_name must be a non-empty string")

        with self._lock(user_id=user_id, conversation_id=conversation_id):
            data = self._read_unlocked(user_id=user_id, conversation_id=conversation_id)
            states = data.get("states")
            if not isinstance(states, dict):
                return None

            rec = states.get(state_name)
            if not isinstance(rec, dict):
                return None

            payload = rec.get("payload")
            if not isinstance(payload, dict):
                return None

            cls = self._state_classes_by_name.get(state_name)
            if cls is None:
                raise KeyError(f"No State class registered for state_name={state_name!r}")

            try:
                st = cls.from_json_dict(payload)
            except Exception as e:
                raise ValueError(f"Error deserializing state '{state_name}': {e}") from e


            # sanity check
            if getattr(st, "name", None) != state_name:
                raise ValueError(f"Loaded State.name mismatch: got {st.name!r}, expected {state_name!r}")
            
            return st

    def store_state(self, *, user_id: UUID, conversation_id: UUID, state: State) -> None:
        if not state.name.strip():
            raise ValueError("state.name must be a non-empty string")

        with self._lock(user_id=user_id, conversation_id=conversation_id):
            data = self._read_unlocked(user_id=user_id, conversation_id=conversation_id)
            states = data.get("states")
            if not isinstance(states, dict):
                states = {}
                data["states"] = states

            states[state.name] = {
                "class": state.__class__.__name__,
                "payload": state.to_json_dict(),
            }

            # convenience: keep pointer aligned to last stored state
            data["active_state_name"] = state.name
            data["updated_at"] = time.time()

            self._write_unlocked(user_id=user_id, conversation_id=conversation_id, data=data)

    def delete_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> None:
        if not state_name.strip():
            raise ValueError("state_name must be a non-empty string")

        with self._lock(user_id=user_id, conversation_id=conversation_id):
            data = self._read_unlocked(user_id=user_id, conversation_id=conversation_id)
            states = data.get("states")
            if isinstance(states, dict) and state_name in states:
                del states[state_name]

            if data.get("active_state_name") == state_name:
                data["active_state_name"] = None

            data["updated_at"] = time.time()
            self._write_unlocked(user_id=user_id, conversation_id=conversation_id, data=data)

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

        with self._lock(user_id=user_id, conversation_id=conversation_id):
            data = self._read_unlocked(user_id=user_id, conversation_id=conversation_id)
            arr = data.get("messages")
            if not isinstance(arr, list):
                arr = []
                data["messages"] = arr

            for m in messages:
                arr.append(self._chat_message_to_dict(m))

            # cap growth
            if self._cfg.max_messages_kept > 0 and len(arr) > self._cfg.max_messages_kept:
                data["messages"] = arr[-self._cfg.max_messages_kept :]

            data["updated_at"] = time.time()
            self._write_unlocked(user_id=user_id, conversation_id=conversation_id, data=data)

    def load_message_history(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        limit: int = 20,
    ) -> Sequence[ChatMessage]:
        if limit <= 0:
            return []

        with self._lock(user_id=user_id, conversation_id=conversation_id):
            data = self._read_unlocked(user_id=user_id, conversation_id=conversation_id)
            arr = data.get("messages")
            if not isinstance(arr, list):
                return []

            tail = arr[-limit:]
            out: list[ChatMessage] = []
            for item in tail:
                if isinstance(item, dict):
                    msg = self._chat_message_from_dict(item)
                    if msg is not None:
                        out.append(msg)
            return out

    def clear_message_history(self, *, user_id: UUID, conversation_id: UUID) -> None:
        with self._lock(user_id=user_id, conversation_id=conversation_id):
            data = self._read_unlocked(user_id=user_id, conversation_id=conversation_id)
            data["messages"] = []
            data["updated_at"] = time.time()
            self._write_unlocked(user_id=user_id, conversation_id=conversation_id, data=data)

    # -----------------------
    # Internals
    # -----------------------

    def _conv_file(self, *, user_id: UUID, conversation_id: UUID) -> Path:
        user_dir = self._base_dir / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / f"{conversation_id}{self._cfg.file_name_suffix}"

    def _lock_file(self, *, user_id: UUID, conversation_id: UUID) -> Path:
        return self._conv_file(user_id=user_id, conversation_id=conversation_id).with_suffix(
            f"{self._cfg.file_name_suffix}.lock"
        )

    def _lock(self, *, user_id: UUID, conversation_id: UUID) -> _FileLock:
        return _FileLock(
            self._lock_file(user_id=user_id, conversation_id=conversation_id),
            timeout_s=self._cfg.lock_timeout_s,
            stale_s=self._cfg.lock_stale_s,
            poll_min_s=self._cfg.lock_poll_min_s,
            poll_max_s=self._cfg.lock_poll_max_s,
        )

    def _empty_doc(self) -> dict[str, Any]:
        return {
            "schema_version": self._SCHEMA_VERSION,
            "active_state_name": None,
            "states": {},
            "messages": [],
            "updated_at": time.time(),
        }

    def _read_unlocked(self, *, user_id: UUID, conversation_id: UUID) -> dict[str, Any]:
        path = self._conv_file(user_id=user_id, conversation_id=conversation_id)
        bak = path.with_suffix(path.suffix + ".bak")

        if not path.exists():
            return self._empty_doc()

        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
            if not isinstance(data, dict):
                raise ValueError("Root JSON is not an object")
            self._validate_doc_shape(data)
            return data
        except Exception:
            # Try backup
            if self._cfg.keep_backup and bak.exists():
                try:
                    raw_bak = bak.read_text(encoding="utf-8")
                    data_bak = json.loads(raw_bak) if raw_bak.strip() else {}
                    if isinstance(data_bak, dict):
                        self._validate_doc_shape(data_bak)
                        return data_bak
                except Exception:
                    pass

            # Keep corrupt copy
            try:
                ts = int(time.time())
                corrupt = path.with_suffix(path.suffix + f".corrupt.{ts}")
                path.rename(corrupt)
            except Exception:
                raise

            return self._empty_doc()

    def _write_unlocked(self, *, user_id: UUID, conversation_id: UUID, data: dict[str, Any]) -> None:
        self._validate_doc_shape(data, allow_missing=True)

        path = self._conv_file(user_id=user_id, conversation_id=conversation_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        bak = path.with_suffix(path.suffix + ".bak")

        payload = json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=self._json_default,
        )

        # Write temp
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            if self._cfg.fsync_writes:
                os.fsync(f.fileno())

        # Backup current (best-effort)
        if self._cfg.keep_backup and path.exists():
            try:
                # os.replace is atomic; keep last good
                os.replace(path, bak)
            except Exception:
                pass

        # Atomic replace
        os.replace(tmp, path)

        # Fsync directory for durability on POSIX
        if self._cfg.fsync_writes and os.name == "posix":
            try:
                dir_fd = os.open(str(path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                pass

    def _validate_doc_shape(self, data: dict[str, Any], *, allow_missing: bool = False) -> None:
        """
        Validates minimal invariants. If allow_missing=True, it will only enforce types where present.
        """
        def _chk(key: str, typ: type) -> None:
            if key not in data:
                if not allow_missing:
                    raise ValueError(f"Missing key: {key}")
                return
            if not isinstance(data[key], typ):
                raise ValueError(f"Key {key} must be {typ.__name__}")

        _chk("schema_version", int)
        _chk("states", dict)
        _chk("messages", list)
        # active_state_name may be None or str
        if "active_state_name" in data and data["active_state_name"] is not None and not isinstance(
            data["active_state_name"], str
        ):
            raise ValueError("active_state_name must be str or None")

        if "schema_version" in data and data["schema_version"] not in (self._SCHEMA_VERSION,):
            raise ValueError(f"Unsupported schema_version={data['schema_version']}")

    def _chat_message_to_dict(self, m: ChatMessage) -> dict[str, Any]:
        # pydantic v2 model instance
        if hasattr(m, "model_dump"):
            d = getattr(m, "model_dump")()
            return d

        # dataclass instance
        if is_dataclass(m):
            return asdict(m)

        # object with attrs
        d: dict[str, Any] = {}
        for k in ("role", "content"):
            if hasattr(m, k):
                d[k] = getattr(m, k)
        if not d:
            d["content"] = str(m)
        return d

    def _chat_message_from_dict(self, d: dict[str, Any]) -> Optional[ChatMessage]:
        # pydantic v2 classmethod
        if hasattr(ChatMessage, "model_validate"):
            try:
                return ChatMessage.model_validate(d)  # type: ignore[attr-defined]
            except Exception:
                raise


        # dataclass / normal constructor
        try:
            return ChatMessage(**d)  # type: ignore[misc]
        except Exception:
            raise
        
        
    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)  # type: ignore[return-value]
        if hasattr(obj, "to_json_dict") and callable(getattr(obj, "to_json_dict")):
            return obj.to_json_dict()
        return str(obj)