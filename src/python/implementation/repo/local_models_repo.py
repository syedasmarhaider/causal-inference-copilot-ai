from __future__ import annotations

import os
import pickle
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Final
from uuid import UUID

from python.domain.repo.models_repo import ModelRecord, ModelsRepo


@dataclass(frozen=True)
class _StoredModelPayload:
    model: Any
    metadata: dict[str, Any]


class LocalFileModelsRepo(ModelsRepo):
    """
    Local filesystem implementation of ModelsRepo.

    Layout:
        <root>/
          users/
            <user_id>/
              conversations/
                <conversation_id>/
                  models/
                    <model_id>/
                      record.pkl
    """

    _RECORD_FILE_NAME: Final[str] = "record.pkl"

    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir).expanduser()
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
        model: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_metadata = self._normalize_metadata(metadata)

        payload = _StoredModelPayload(
            model=model,
            metadata=normalized_metadata,
        )

        record_path = self._record_path(
            user_id=user_id,
            conversation_id=conversation_id,
            model_id=model_id,
        )
        record_path.parent.mkdir(parents=True, exist_ok=True)
        self._dump_pickle_atomic(record_path, payload)

    def load_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> ModelRecord | None:
        record_path = self._record_path(
            user_id=user_id,
            conversation_id=conversation_id,
            model_id=model_id,
        )

        if not record_path.is_file():
            return None

        payload = self._load_payload(record_path)

        return ModelRecord(
            model_id=model_id,
            model=payload.model,
            metadata=dict(payload.metadata),
        )

    def model_exists(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> bool:  # pyright: ignore[reportReturnType]
        record_path = self._record_path(
            user_id=user_id,
            conversation_id=conversation_id,
            model_id=model_id,
        )
        return record_path.is_file()

    def delete_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> None:
        model_dir = self._model_dir(
            user_id=user_id,
            conversation_id=conversation_id,
            model_id=model_id,
        )
        shutil.rmtree(model_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Path helpers
    # -------------------------------------------------------------------------

    def _conversation_dir(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> Path:
        return self._root_dir / "users" / str(user_id) / "conversations" / str(conversation_id)

    def _models_dir(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> Path:
        return (
            self._conversation_dir(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            / "models"
        )

    def _model_dir(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> Path:
        return self._models_dir(
            user_id=user_id,
            conversation_id=conversation_id,
        ) / str(model_id)

    def _record_path(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> Path:
        return (
            self._model_dir(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
            )
            / self._RECORD_FILE_NAME
        )

    # -------------------------------------------------------------------------
    # Validation / normalization
    # -------------------------------------------------------------------------

    def _normalize_metadata(
        self,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if metadata is None:
            return {}

        normalized = dict(metadata)

        for key in normalized:
            if not isinstance(key, str):
                raise TypeError("metadata keys must be strings")

        return normalized

    def _load_payload(self, path: Path) -> _StoredModelPayload:
        raw = self._load_pickle(path)

        if isinstance(raw, _StoredModelPayload):
            return raw

        # Backward/defensive compatibility if something dict-like got stored.
        if isinstance(raw, dict):
            if "model" not in raw or "metadata" not in raw:
                raise ValueError(f"stored model record at {path} is missing required keys")

            metadata = raw["metadata"]
            if not isinstance(metadata, dict):
                raise ValueError(f"stored metadata at {path} must deserialize to dict")
            if not all(isinstance(key, str) for key in metadata):
                raise ValueError(f"stored metadata at {path} must have string keys")

            return _StoredModelPayload(
                model=raw["model"],
                metadata=dict(metadata),
            )

        raise ValueError(
            f"stored model record at {path} has unsupported type: " f"{type(raw).__name__}"
        )

    # -------------------------------------------------------------------------
    # Pickle IO helpers
    # -------------------------------------------------------------------------

    def _dump_pickle_atomic(self, path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                suffix=".tmp",
                delete=False,
            ) as tmp:
                pickle.dump(obj, tmp, protocol=pickle.HIGHEST_PROTOCOL)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)

            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _load_pickle(self, path: Path) -> Any:
        with path.open("rb") as handle:
            return pickle.load(handle)
