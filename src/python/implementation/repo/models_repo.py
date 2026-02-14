from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Final, Mapping, Optional, Protocol, cast
from uuid import UUID, uuid4

import joblib  # pyright: ignore[reportMissingTypeStubs]

from python.domain.repo.models_repo import ModelRecord, ModelsRepo

DEFAULT_DATASET_PATH: Final[Path] = Path(
    "./models/"
).resolve()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_dir(dir_path: Path) -> None:
    # fsync directory to persist rename metadata on POSIX
    try:
        fd = os.open(str(dir_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        # best-effort; on some FS/platforms this may fail
        pass


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid4()}")
    text = json.dumps(obj, sort_keys=True, indent=2, default=str)

    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


class _JoblibLike(Protocol):
    def dump(self, value: Any, filename: Any, compress: int = ..., protocol: int = ...) -> Any: ...
    def load(self, filename: Any, mmap_mode: Optional[str] = ...) -> Any: ...


_joblib: _JoblibLike = cast(_JoblibLike, joblib)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class FileSystemModelsRepo(ModelsRepo):
    root_dir: Path = DEFAULT_DATASET_PATH

    def __post_init__(self) -> None:
        # ensure root_dir exists and is a directory
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if not self.root_dir.is_dir():
            raise ValueError(f"root_dir is not a directory: {self.root_dir}")

    def _models_dir(self, *, user_id: UUID, conversation_id: UUID) -> Path:
        return self.root_dir / "users" / str(user_id) / "conversations" / str(conversation_id) / "models"

    def _artifact_path(self, *, user_id: UUID, conversation_id: UUID, model_id: UUID) -> Path:
        return self._models_dir(user_id=user_id, conversation_id=conversation_id) / f"{model_id}.joblib"

    def _meta_path(self, *, user_id: UUID, conversation_id: UUID, model_id: UUID) -> Path:
        return self._models_dir(user_id=user_id, conversation_id=conversation_id) / f"{model_id}.meta.json"

    def save_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
        model: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        models_dir = self._models_dir(user_id=user_id, conversation_id=conversation_id)
        models_dir.mkdir(parents=True, exist_ok=True)

        artifact = self._artifact_path(user_id=user_id, conversation_id=conversation_id, model_id=model_id)
        meta_path = self._meta_path(user_id=user_id, conversation_id=conversation_id, model_id=model_id)

        tmp = artifact.with_suffix(artifact.suffix + f".tmp.{uuid4()}")

        try:
            _joblib.dump(model, tmp)

            # fsync file contents (best-effort)
            try:
                with open(tmp, "rb") as f:
                    os.fsync(f.fileno())
            except Exception:
                pass

            os.replace(tmp, artifact)
            _fsync_dir(artifact.parent)
        finally:
            _safe_unlink(tmp)

        # write metadata after artifact exists
        meta: Dict[str, Any] = dict(metadata or {})
        meta["model_id"] = str(model_id)
        meta["user_id"] = str(user_id)
        meta["conversation_id"] = str(conversation_id)
        meta["format"] = "joblib"
        meta["artifact_name"] = artifact.name
        meta["saved_at_utc"] = meta.get("saved_at_utc") or _utc_now_iso()
        meta["artifact_bytes"] = artifact.stat().st_size
        meta["artifact_sha256"] = _sha256_file(artifact)

        _atomic_write_json(meta_path, meta)

    def load_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> ModelRecord | None:
        artifact = self._artifact_path(user_id=user_id, conversation_id=conversation_id, model_id=model_id)
        if not artifact.exists():
            return None

        meta_path = self._meta_path(user_id=user_id, conversation_id=conversation_id, model_id=model_id)
        metadata: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}

        model = _joblib.load(artifact)

        metadata.setdefault("model_id", str(model_id))
        metadata.setdefault("user_id", str(user_id))
        metadata.setdefault("conversation_id", str(conversation_id))
        metadata.setdefault("format", "joblib")
        metadata.setdefault("artifact_name", artifact.name)

        return ModelRecord(model_id=model_id, model=model, metadata=metadata)

    def model_exists(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> bool:
        return self._artifact_path(user_id=user_id, conversation_id=conversation_id, model_id=model_id).exists()

    def delete_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> None:
        _safe_unlink(self._artifact_path(user_id=user_id, conversation_id=conversation_id, model_id=model_id))
        _safe_unlink(self._meta_path(user_id=user_id, conversation_id=conversation_id, model_id=model_id))
