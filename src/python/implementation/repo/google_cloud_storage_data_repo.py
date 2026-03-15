from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Final, Optional
from uuid import UUID

import pandas as pd
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

from python.domain.repo.data_repo import DataRepo, ImageMime

CSV_FILENAME: Final[str] = "data.csv"
DATASETS_DIRNAME: Final[str] = "datasets"
ARTIFACTS_DIRNAME: Final[str] = "artifacts"
ARTIFACT_META_FILENAME: Final[str] = "meta.json"
ARTIFACT_BASENAME: Final[str] = "artifact"

_MIME_TO_EXT: Final[dict[ImageMime, str]] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip().strip("/")


def _validate_image_bytes(mime: ImageMime, content: bytes) -> None:
    if not content:
        raise ValueError("artifact content is empty")

    if mime == "image/png":
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("content does not look like a PNG (bad signature)")
        return

    if mime == "image/jpeg":
        if not content.startswith(b"\xFF\xD8"):
            raise ValueError("content does not look like a JPEG (bad signature)")
        return

    if mime == "image/webp":
        if len(content) < 12 or not (content.startswith(b"RIFF") and content[8:12] == b"WEBP"):
            raise ValueError("content does not look like a WEBP (bad signature)")
        return

    raise ValueError(f"unsupported mime: {mime!r}")


@dataclass(frozen=True)
class GoogleCloudStorageDataRepo(DataRepo):
    bucket_name: str
    root_prefix: str = "data"
    project_id: str | None = None
    timeout_seconds: float = 60.0
    client: storage.Client = field(init=False, repr=False)
    bucket: storage.Bucket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bucket_name = self.bucket_name.strip()
        if not bucket_name:
            raise ValueError("bucket_name must be a non-empty string")

        client = storage.Client(project=self.project_id)
        bucket = client.bucket(bucket_name)

        object.__setattr__(self, "bucket_name", bucket_name)
        object.__setattr__(self, "root_prefix", _normalize_prefix(self.root_prefix))
        object.__setattr__(self, "client", client)
        object.__setattr__(self, "bucket", bucket)

    # ------------------------------------------------------------------
    # Object naming
    # ------------------------------------------------------------------

    def _join(self, *parts: str) -> str:
        return "/".join(part.strip("/") for part in parts if part and part.strip("/"))

    def _dataset_blob_name(self, user_id: UUID, conversation_id: UUID, dataset_id: UUID) -> str:
        return self._join(
            self.root_prefix,
            "users",
            str(user_id),
            "conversations",
            str(conversation_id),
            DATASETS_DIRNAME,
            str(dataset_id),
            CSV_FILENAME,
        )

    def _artifact_dir_prefix(self, user_id: UUID, conversation_id: UUID, artifact_id: UUID) -> str:
        return self._join(
            self.root_prefix,
            "users",
            str(user_id),
            "conversations",
            str(conversation_id),
            ARTIFACTS_DIRNAME,
            str(artifact_id),
        )

    def _artifact_blob_name(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        *,
        mime: ImageMime,
    ) -> str:
        ext = _MIME_TO_EXT[mime]
        return self._join(
            self._artifact_dir_prefix(user_id, conversation_id, artifact_id),
            f"{ARTIFACT_BASENAME}{ext}",
        )

    def _artifact_meta_blob_name(self, user_id: UUID, conversation_id: UUID, artifact_id: UUID) -> str:
        return self._join(
            self._artifact_dir_prefix(user_id, conversation_id, artifact_id),
            ARTIFACT_META_FILENAME,
        )

    def _artifact_candidate_blob_names(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
    ) -> dict[ImageMime, str]:
        return {
            mime: self._artifact_blob_name(
                user_id=user_id,
                conversation_id=conversation_id,
                artifact_id=artifact_id,
                mime=mime,
            )
            for mime in _MIME_TO_EXT
        }

    def _blob(self, blob_name: str) -> storage.Blob:
        return self.bucket.blob(blob_name)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        if limit is not None and limit <= 0:
            raise ValueError(f"limit must be a positive int or None, got: {limit!r}")

        blob_name = self._dataset_blob_name(user_id, conversation_id, dataset_id)
        blob = self._blob(blob_name)

        if not blob.exists(client=self.client, timeout=self.timeout_seconds):
            raise FileNotFoundError(f"CSV not found for dataset_id={dataset_id}")

        try:
            csv_bytes = blob.download_as_bytes(timeout=self.timeout_seconds)
            return pd.read_csv(
                io.BytesIO(csv_bytes),
                nrows=limit,
                low_memory=False,
            )
        except FileNotFoundError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to read CSV for dataset_id={dataset_id}: {e}") from e

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
        blob_name = self._dataset_blob_name(user_id, conversation_id, dataset_id)
        blob = self._blob(blob_name)

        try:
            csv_text = df.to_csv(index=include_index)  # pyright: ignore[reportUnknownMemberType]
        except Exception as e:
            raise ValueError(f"Failed to serialize CSV for dataset_id={dataset_id}: {e}") from e

        upload_kwargs: dict[str, object] = {
            "data": csv_text,
            "content_type": "text/csv; charset=utf-8",
            "timeout": self.timeout_seconds,
        }
        if not overwrite:
            upload_kwargs["if_generation_match"] = 0

        try:
            blob.upload_from_string(**upload_kwargs)
        except PreconditionFailed as e:
            raise FileExistsError(f"Refusing to overwrite existing CSV for dataset_id={dataset_id}") from e
        except Exception as e:
            raise ValueError(f"Failed to write CSV for dataset_id={dataset_id}: {e}") from e

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

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
        _validate_image_bytes(mime, content)

        artifact_blob_name = self._artifact_blob_name(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
            mime=mime,
        )
        meta_blob_name = self._artifact_meta_blob_name(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
        )

        artifact_blob = self._blob(artifact_blob_name)
        meta_blob = self._blob(meta_blob_name)

        artifact_upload_kwargs: dict[str, object] = {
            "data": content,
            "content_type": mime,
            "timeout": self.timeout_seconds,
        }
        meta_upload_kwargs: dict[str, object] = {
            "data": json.dumps({"mime": mime}, sort_keys=True),
            "content_type": "application/json",
            "timeout": self.timeout_seconds,
        }

        if not overwrite:
            artifact_upload_kwargs["if_generation_match"] = 0
            meta_upload_kwargs["if_generation_match"] = 0

        try:
            artifact_blob.upload_from_string(**artifact_upload_kwargs)

            try:
                meta_blob.upload_from_string(**meta_upload_kwargs)
            except Exception:
                if not overwrite:
                    try:
                        artifact_blob.delete(timeout=self.timeout_seconds)
                    except Exception:
                        pass
                raise

            if overwrite:
                for other_mime, other_blob_name in self._artifact_candidate_blob_names(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    artifact_id=artifact_id,
                ).items():
                    if other_mime == mime:
                        continue
                    try:
                        self._blob(other_blob_name).delete(timeout=self.timeout_seconds)
                    except NotFound:
                        pass
                    except Exception:
                        # best-effort cleanup; metadata already points to the new mime
                        pass

        except PreconditionFailed as e:
            raise FileExistsError(
                f"Refusing to overwrite existing artifact for artifact_id={artifact_id}"
            ) from e
        except FileExistsError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to save artifact {artifact_id}: {e}") from e

    def _probe_artifact_mime(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
    ) -> ImageMime:
        found: list[ImageMime] = []

        for mime, blob_name in self._artifact_candidate_blob_names(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
        ).items():
            if self._blob(blob_name).exists(client=self.client, timeout=self.timeout_seconds):
                found.append(mime)

        if not found:
            raise FileNotFoundError("artifact not found")

        if len(found) > 1:
            raise ValueError(
                f"artifact state is ambiguous for artifact_id={artifact_id}: multiple image blobs exist"
            )

        return found[0]

    def get_artifact_mime(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
    ) -> ImageMime:
        meta_blob_name = self._artifact_meta_blob_name(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
        )
        meta_blob = self._blob(meta_blob_name)

        try:
            raw = meta_blob.download_as_bytes(timeout=self.timeout_seconds)
            parsed = json.loads(raw.decode("utf-8"))
            mime = parsed.get("mime")

            if mime not in _MIME_TO_EXT:
                return self._probe_artifact_mime(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    artifact_id=artifact_id,
                )

            mime_typed: ImageMime = mime
            artifact_blob_name = self._artifact_blob_name(
                user_id=user_id,
                conversation_id=conversation_id,
                artifact_id=artifact_id,
                mime=mime_typed,
            )

            if self._blob(artifact_blob_name).exists(client=self.client, timeout=self.timeout_seconds):
                return mime_typed

            return self._probe_artifact_mime(
                user_id=user_id,
                conversation_id=conversation_id,
                artifact_id=artifact_id,
            )

        except NotFound:
            return self._probe_artifact_mime(
                user_id=user_id,
                conversation_id=conversation_id,
                artifact_id=artifact_id,
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._probe_artifact_mime(
                user_id=user_id,
                conversation_id=conversation_id,
                artifact_id=artifact_id,
            )
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to resolve artifact mime for artifact_id={artifact_id}: {e}") from e

    def get_artifact_bytes(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        *,
        expected_mime: Optional[ImageMime] = None,
    ) -> bytes:
        actual_mime = self.get_artifact_mime(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
        )

        if expected_mime is not None and actual_mime != expected_mime:
            raise ValueError(f"mime mismatch: expected {expected_mime}, got {actual_mime}")

        artifact_blob_name = self._artifact_blob_name(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
            mime=actual_mime,
        )
        artifact_blob = self._blob(artifact_blob_name)

        try:
            return artifact_blob.download_as_bytes(timeout=self.timeout_seconds)
        except NotFound as e:
            raise FileNotFoundError("artifact not found") from e
        except Exception as e:
            raise ValueError(f"Failed to read artifact {artifact_id}: {e}") from e