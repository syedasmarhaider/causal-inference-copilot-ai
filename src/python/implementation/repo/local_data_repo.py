from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final
from uuid import UUID

import pandas as pd

from python.domain.repo.data_repo import DataRepo, ImageMime


class LocalFileDataRepo(DataRepo):
    _ARTIFACT_CONTENT_FILE: Final[str] = "content.bin"
    _ARTIFACT_META_FILE: Final[str] = "metadata.json"
    _ALLOWED_MIMES: Final[set[str]] = {
        "image/png",
        "image/jpeg",
        "image/webp",
    }

    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir).expanduser()
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        start: int = 0,
        limit: int | None = None,
    ) -> pd.DataFrame:
        if start < 0:
            raise ValueError("start must be >= 0")
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")

        path = self._csv_path(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
        )
        self._require_file(path, "csv dataset")

        skiprows = range(1, start + 1) if start > 0 else None
        return pd.read_csv(path, skiprows=skiprows, nrows=limit)

    def save_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        df: pd.DataFrame,
        *,
        overwrite: bool = True,
        include_index: bool = False,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas.DataFrame")

        path = self._csv_path(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite:
            self._write_csv_atomic(path=path, df=df, include_index=include_index)
            return

        if path.exists():
            raise FileExistsError(f"csv dataset already exists: {path}")

        with path.open("x", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=include_index)
            handle.flush()
            os.fsync(handle.fileno())

    def get_json_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
    ) -> str:
        path = self._json_path(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
        )
        self._require_file(path, "json dataset")
        return path.read_text(encoding="utf-8")

    def save_json_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        json_data: str,
        *,
        overwrite: bool = True,
    ) -> None:
        if not isinstance(json_data, str):
            raise TypeError("json_data must be a string")

        # Validate that the payload is actually JSON.
        try:
            json.loads(json_data)
        except json.JSONDecodeError as exc:
            raise ValueError("json_data must be valid JSON") from exc

        path = self._json_path(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
        )
        self._write_text(path=path, content=json_data, overwrite=overwrite)

    def save_artifact(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        content: bytes,
        *,
        mime: ImageMime,
        overwrite: bool = True,
    ) -> None:
        self._validate_mime(mime)

        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise TypeError("content must be bytes-like")

        artifact_dir = self._artifact_dir(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        content_path = artifact_dir / self._ARTIFACT_CONTENT_FILE
        meta_path = artifact_dir / self._ARTIFACT_META_FILE

        if not overwrite and (content_path.exists() or meta_path.exists()):
            raise FileExistsError(f"artifact already exists: {artifact_dir}")

        metadata = {
            "mime": mime,
            "size_bytes": len(content),
            "artifact_id": str(artifact_id),
            "conversation_id": str(conversation_id),
            "user_id": str(user_id),
        }
        metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)

        try:
            self._write_bytes(
                path=content_path,
                content=bytes(content),
                overwrite=overwrite,
            )
            self._write_text(
                path=meta_path,
                content=metadata_json,
                overwrite=overwrite,
            )
        except Exception:
            if not overwrite:
                content_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
            raise

    def get_artifact_mime(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
    ) -> ImageMime:
        metadata = self._read_artifact_metadata(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
        )
        mime = metadata.get("mime")
        self._validate_mime(mime)
        return mime

    def get_artifact_bytes(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        *,
        expected_mime: ImageMime | None = None,
    ) -> bytes:
        actual_mime = self.get_artifact_mime(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
        )

        if expected_mime is not None and actual_mime != expected_mime:
            raise ValueError(
                f"artifact mime mismatch: expected={expected_mime}, actual={actual_mime}"
            )

        content_path = (
            self._artifact_dir(
                user_id=user_id,
                conversation_id=conversation_id,
                artifact_id=artifact_id,
            )
            / self._ARTIFACT_CONTENT_FILE
        )
        self._require_file(content_path, "artifact content")
        return content_path.read_bytes()

    # -------------------------------------------------------------------------
    # Path helpers
    # -------------------------------------------------------------------------

    def _conversation_dir(self, user_id: UUID, conversation_id: UUID) -> Path:
        return self._root_dir / "users" / str(user_id) / "conversations" / str(conversation_id)

    def _datasets_dir(self, user_id: UUID, conversation_id: UUID) -> Path:
        return self._conversation_dir(user_id, conversation_id) / "datasets"

    def _artifacts_root(self, user_id: UUID, conversation_id: UUID) -> Path:
        return self._conversation_dir(user_id, conversation_id) / "artifacts"

    def _csv_path(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
    ) -> Path:
        return self._datasets_dir(user_id, conversation_id) / f"{dataset_id}.csv"

    def _json_path(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
    ) -> Path:
        return self._datasets_dir(user_id, conversation_id) / f"{dataset_id}.json"

    def _artifact_dir(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
    ) -> Path:
        return self._artifacts_root(user_id, conversation_id) / str(artifact_id)

    # -------------------------------------------------------------------------
    # Artifact metadata helpers
    # -------------------------------------------------------------------------

    def _read_artifact_metadata(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
    ) -> dict[str, object]:
        meta_path = (
            self._artifact_dir(
                user_id=user_id,
                conversation_id=conversation_id,
                artifact_id=artifact_id,
            )
            / self._ARTIFACT_META_FILE
        )
        self._require_file(meta_path, "artifact metadata")

        raw = meta_path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"artifact metadata is corrupt: {meta_path}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"artifact metadata must be a JSON object: {meta_path}")

        return data

    def _validate_mime(self, mime: str) -> None:
        if mime not in self._ALLOWED_MIMES:
            raise ValueError(
                f"unsupported mime type: {mime!r}. "
                f"Allowed values: {sorted(self._ALLOWED_MIMES)}"
            )

    # -------------------------------------------------------------------------
    # IO helpers
    # -------------------------------------------------------------------------

    def _require_file(self, path: Path, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    def _write_text(self, path: Path, content: str, *, overwrite: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite:
            self._write_text_atomic(path, content)
            return

        if path.exists():
            raise FileExistsError(f"file already exists: {path}")

        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    def _write_bytes(self, path: Path, content: bytes, *, overwrite: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite:
            self._write_bytes_atomic(path, content)
            return

        if path.exists():
            raise FileExistsError(f"file already exists: {path}")

        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    def _write_text_atomic(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=path.parent,
                delete=False,
            ) as tmp:
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)

            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _write_bytes_atomic(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                delete=False,
            ) as tmp:
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)

            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _write_csv_atomic(
        self,
        *,
        path: Path,
        df: pd.DataFrame,
        include_index: bool,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=path.parent,
                suffix=".csv.tmp",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)

            df.to_csv(tmp_path, index=include_index)

            # Re-open once so the file data is flushed to disk before replace.
            with tmp_path.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
