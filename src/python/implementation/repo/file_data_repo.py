# src/python/implementation/repo/file_data_repo.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional
from uuid import UUID, uuid4

import pandas as pd

from python.domain.repo.data_repo import DataRepo, ImageMime, ArtifactRef  # <- ensure ArtifactRef exists in domain


# Root folder for file-backed storage.
DATA_ROOT: Final[Path] = Path("./data").resolve()
CSV_FILENAME: Final[str] = "data.csv"

ARTIFACTS_DIRNAME: Final[str] = "artifacts"
ARTIFACT_BASENAME: Final[str] = "artifact"
ARTIFACT_META_FILENAME: Final[str] = "meta.json"

_MIME_TO_EXT: Final[dict[ImageMime, str]] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
_EXT_TO_MIME: Final[dict[str, ImageMime]] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _validate_image_bytes(mime: ImageMime, content: bytes) -> None:
    if not content:
        raise ValueError("artifact content is empty")

    if mime == "image/png":
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("content does not look like a PNG (bad signature)")
        return

    if mime == "image/jpeg":
        if not content.startswith(b"\xFF\xD8"):  # JPEG SOI
            raise ValueError("content does not look like a JPEG (bad signature)")
        return

    if mime == "image/webp":
        # RIFF....WEBP
        if len(content) < 12 or not (content.startswith(b"RIFF") and content[8:12] == b"WEBP"):
            raise ValueError("content does not look like a WEBP (bad signature)")
        return

    raise ValueError(f"unsupported mime: {mime!r}")


@dataclass(frozen=True)
class FileDataRepo(DataRepo):
    """
    File-backed repo.

    CSV layout:
      ./data/<user_id>/<conversation_id>/<dataset_id>/data.csv

    Artifact layout (folder per artifact_id):
      ./data/<user_id>/<conversation_id>/artifacts/<artifact_id>/artifact.<ext>
      ./data/<user_id>/<conversation_id>/artifacts/<artifact_id>/meta.json
    """

    # ---------------- CSV ----------------
    def _dataset_csv_path(self, user_id: UUID, conversation_id: UUID, dataset_id: UUID) -> Path:
        return (DATA_ROOT / str(user_id) / str(conversation_id) / str(dataset_id) / CSV_FILENAME).resolve()

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        csv_path = self._dataset_csv_path(user_id, conversation_id, dataset_id)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at path: {csv_path}")
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV path is not a file: {csv_path}")

        if limit is not None and limit <= 0:
            raise ValueError(f"limit must be a positive int or None, got: {limit!r}")

        try:
            return pd.read_csv(  # pyright: ignore[reportUnknownMemberType]
                csv_path,
                nrows=limit,
                low_memory=False,
            )
        except Exception as e:
            raise ValueError(f"Failed to read CSV at {csv_path}: {e}") from e

    def save_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        df: pd.DataFrame,
        *,
        overwrite: bool = True,
        include_index: bool = False,
    ) -> Path:
        target = self._dataset_csv_path(user_id, conversation_id, dataset_id)

        try:
            target.relative_to(DATA_ROOT)
        except ValueError as e:
            raise ValueError(f"Resolved target path escapes DATA_ROOT: {target}") from e

        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing CSV at: {target}")

        tmp = target.with_suffix(target.suffix + f".tmp-{uuid4().hex}")
        try:
            df.to_csv(tmp, index=include_index)  # pyright: ignore[reportUnknownMemberType]
            tmp.replace(target)
            return target
        except Exception as e:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise ValueError(f"Failed to write CSV at {target}: {e}") from e

    # ---------------- ARTIFACTS ----------------
    def _artifacts_root(self, user_id: UUID, conversation_id: UUID) -> Path:
        return (DATA_ROOT / str(user_id) / str(conversation_id) / ARTIFACTS_DIRNAME).resolve()

    def _artifact_dir(self, user_id: UUID, conversation_id: UUID, artifact_id: UUID) -> Path:
        return (self._artifacts_root(user_id, conversation_id) / str(artifact_id)).resolve()

    def _artifact_meta_path(self, user_id: UUID, conversation_id: UUID, artifact_id: UUID) -> Path:
        return (self._artifact_dir(user_id, conversation_id, artifact_id) / ARTIFACT_META_FILENAME).resolve()

    def _artifact_file_path(self, user_id: UUID, conversation_id: UUID, artifact_id: UUID, *, mime: ImageMime) -> Path:
        ext = _MIME_TO_EXT[mime]
        return (self._artifact_dir(user_id, conversation_id, artifact_id) / f"{ARTIFACT_BASENAME}{ext}").resolve()

    def _assert_under_data_root(self, p: Path) -> None:
        try:
            p.relative_to(DATA_ROOT)
        except ValueError as e:
            raise ValueError(f"Resolved path escapes DATA_ROOT: {p}") from e

    def save_artifact(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        content: bytes,
        *,
        mime: ImageMime,
        overwrite: bool = True,
    ) -> ArtifactRef:
        _validate_image_bytes(mime, content)

        artifact_dir = self._artifact_dir(user_id, conversation_id, artifact_id)
        self._assert_under_data_root(artifact_dir)

        if artifact_dir.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing artifact dir at: {artifact_dir}")

        artifact_dir.mkdir(parents=True, exist_ok=True)

        # If overwriting, remove old image files (prevents ambiguity if mime changes)
        if overwrite:
            for child in artifact_dir.iterdir():
                if child.is_file() and child.suffix.lower() in _EXT_TO_MIME:
                    try:
                        child.unlink()
                    except Exception:
                        pass

        img_path = self._artifact_file_path(user_id, conversation_id, artifact_id, mime=mime)
        meta_path = self._artifact_meta_path(user_id, conversation_id, artifact_id)

        self._assert_under_data_root(img_path)
        self._assert_under_data_root(meta_path)

        img_tmp = img_path.with_suffix(img_path.suffix + f".tmp-{uuid4().hex}")
        meta_tmp = meta_path.with_suffix(meta_path.suffix + f".tmp-{uuid4().hex}")

        try:
            # atomic write image
            img_tmp.write_bytes(content)
            img_tmp.replace(img_path)

            # atomic write meta
            meta = {
                "artifact_id": str(artifact_id),
                "mime": mime,
                "filename": img_path.name,
            }
            meta_tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            meta_tmp.replace(meta_path)

            size = img_path.stat().st_size
            return ArtifactRef(
                user_id=user_id,
                conversation_id=conversation_id,
                artifact_id=artifact_id,
                mime=mime,
                path=img_path,
                size_bytes=size,
            )
        except Exception as e:
            for tmp in (img_tmp, meta_tmp):
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
            raise ValueError(f"Failed to save artifact {artifact_id} under {artifact_dir}: {e}") from e

    def get_artifact_ref(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        *,
        expected_mime: Optional[ImageMime] = None,
    ) -> ArtifactRef:
        artifact_dir = self._artifact_dir(user_id, conversation_id, artifact_id)
        self._assert_under_data_root(artifact_dir)

        meta_path = self._artifact_meta_path(user_id, conversation_id, artifact_id)
        self._assert_under_data_root(meta_path)

        if not meta_path.exists() or not meta_path.is_file():
            raise FileNotFoundError("artifact not found")

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"artifact meta is unreadable: {meta_path}: {e}") from e

        mime = meta.get("mime")
        filename = meta.get("filename")

        if mime not in _MIME_TO_EXT:
            raise ValueError(f"artifact meta has unsupported mime: {mime!r}")
        if not isinstance(filename, str) or not filename:
            raise ValueError("artifact meta missing/invalid filename")

        mime_typed: ImageMime = mime  # type: ignore[assignment]

        if expected_mime is not None and mime_typed != expected_mime:
            raise ValueError(f"mime mismatch: expected {expected_mime}, got {mime_typed}")

        img_path = (artifact_dir / filename).resolve()
        self._assert_under_data_root(img_path)

        # ensure resolved file stays under artifact_dir (symlink/path-trick defense)
        try:
            img_path.relative_to(artifact_dir)
        except ValueError as e:
            raise ValueError(f"resolved artifact path escapes artifact_dir: {img_path}") from e

        if not img_path.exists() or not img_path.is_file():
            raise FileNotFoundError("artifact not found")

        size = img_path.stat().st_size
        return ArtifactRef(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
            mime=mime_typed,
            path=img_path,
            size_bytes=size,
        )