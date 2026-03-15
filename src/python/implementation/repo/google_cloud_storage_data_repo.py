from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Final, Optional
from uuid import UUID

import pandas as pd
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

from python.domain.repo.data_repo import DataRepo, ImageMime

DEFAULT_GCS_DATA_PREFIX: Final[str] = "data"
DEFAULT_GCS_TIMEOUT_SECONDS: Final[float] = 60.0

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
    bucket: storage.Bucket

    @staticmethod
    def get_default_bucket() -> storage.Bucket:
        project_name = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or None
        client = storage.Client(project=project_name)
        bucket_name = os.getenv("GCS_DATA_BUCKET_NAME", "").strip()
        if not bucket_name:
            raise ValueError("GCS_DATA_BUCKET_NAME must be configured for GoogleCloudStorageDataRepo")
        return client.bucket(bucket_name)

    def __post_init__(self) -> None:
        bucket_name = getattr(self.bucket, "name", "").strip()
        if not bucket_name:
            raise ValueError("bucket must have a non-empty name")

    def _join(self, *parts: str) -> str:
        return "/".join(part.strip("/") for part in parts if part and part.strip("/"))

    def _dataset_blob_name(self, user_id: UUID, conversation_id: UUID, dataset_id: UUID) -> str:
        return self._join(
            DEFAULT_GCS_DATA_PREFIX,
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
            DEFAULT_GCS_DATA_PREFIX,
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
        return self._join(
            self._artifact_dir_prefix(user_id, conversation_id, artifact_id),
            f"{ARTIFACT_BASENAME}{_MIME_TO_EXT[mime]}",
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

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        if limit is not None and limit <= 0:
            raise ValueError(f"limit must be a positive int or None, got: {limit!r}")

        blob = self._blob(self._dataset_blob_name(user_id, conversation_id, dataset_id))
        if not blob.exists(timeout=DEFAULT_GCS_TIMEOUT_SECONDS):
            raise FileNotFoundError(f"CSV not found for dataset_id={dataset_id}")

        try:
            csv_bytes = blob.download_as_bytes(timeout=DEFAULT_GCS_TIMEOUT_SECONDS)
            return pd.read_csv(io.BytesIO(csv_bytes), nrows=limit, low_memory=False)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to read CSV for dataset_id={dataset_id}: {exc}") from exc

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
        blob = self._blob(self._dataset_blob_name(user_id, conversation_id, dataset_id))

        try:
            csv_text = df.to_csv(index=include_index)  # pyright: ignore[reportUnknownMemberType]
        except Exception as exc:
            raise ValueError(f"Failed to serialize CSV for dataset_id={dataset_id}: {exc}") from exc

        upload_kwargs: dict[str, object] = {
            "data": csv_text,
            "content_type": "text/csv; charset=utf-8",
            "timeout": DEFAULT_GCS_TIMEOUT_SECONDS,
        }
        if not overwrite:
            upload_kwargs["if_generation_match"] = 0

        try:
            blob.upload_from_string(**upload_kwargs)
        except PreconditionFailed as exc:
            raise FileExistsError(f"Refusing to overwrite existing CSV for dataset_id={dataset_id}") from exc
        except Exception as exc:
            raise ValueError(f"Failed to write CSV for dataset_id={dataset_id}: {exc}") from exc

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

        artifact_blob = self._blob(
            self._artifact_blob_name(
                user_id=user_id,
                conversation_id=conversation_id,
                artifact_id=artifact_id,
                mime=mime,
            )
        )
        meta_blob = self._blob(
            self._artifact_meta_blob_name(
                user_id=user_id,
                conversation_id=conversation_id,
                artifact_id=artifact_id,
            )
        )

        artifact_upload_kwargs: dict[str, object] = {
            "data": content,
            "content_type": mime,
            "timeout": DEFAULT_GCS_TIMEOUT_SECONDS,
        }
        meta_upload_kwargs: dict[str, object] = {
            "data": json.dumps({"mime": mime}, sort_keys=True),
            "content_type": "application/json",
            "timeout": DEFAULT_GCS_TIMEOUT_SECONDS,
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
                        artifact_blob.delete(timeout=DEFAULT_GCS_TIMEOUT_SECONDS)
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
                        self._blob(other_blob_name).delete(timeout=DEFAULT_GCS_TIMEOUT_SECONDS)
                    except NotFound:
                        pass
                    except Exception:
                        pass

        except PreconditionFailed as exc:
            raise FileExistsError(
                f"Refusing to overwrite existing artifact for artifact_id={artifact_id}"
            ) from exc
        except FileExistsError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to save artifact {artifact_id}: {exc}") from exc

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
            if self._blob(blob_name).exists(timeout=DEFAULT_GCS_TIMEOUT_SECONDS):
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
        meta_blob = self._blob(
            self._artifact_meta_blob_name(
                user_id=user_id,
                conversation_id=conversation_id,
                artifact_id=artifact_id,
            )
        )

        try:
            raw = meta_blob.download_as_bytes(timeout=DEFAULT_GCS_TIMEOUT_SECONDS)
            parsed = json.loads(raw.decode("utf-8"))
            mime = parsed.get("mime")
            if mime not in _MIME_TO_EXT:
                return self._probe_artifact_mime(user_id, conversation_id, artifact_id)

            mime_typed: ImageMime = mime
            artifact_blob = self._blob(
                self._artifact_blob_name(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    artifact_id=artifact_id,
                    mime=mime_typed,
                )
            )
            if artifact_blob.exists(timeout=DEFAULT_GCS_TIMEOUT_SECONDS):
                return mime_typed

            return self._probe_artifact_mime(user_id, conversation_id, artifact_id)
        except NotFound:
            return self._probe_artifact_mime(user_id, conversation_id, artifact_id)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._probe_artifact_mime(user_id, conversation_id, artifact_id)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to resolve artifact mime for artifact_id={artifact_id}: {exc}") from exc

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

        artifact_blob = self._blob(
            self._artifact_blob_name(
                user_id=user_id,
                conversation_id=conversation_id,
                artifact_id=artifact_id,
                mime=actual_mime,
            )
        )

        try:
            return artifact_blob.download_as_bytes(timeout=DEFAULT_GCS_TIMEOUT_SECONDS)
        except NotFound as exc:
            raise FileNotFoundError("artifact not found") from exc
        except Exception as exc:
            raise ValueError(f"Failed to read artifact {artifact_id}: {exc}") from exc
