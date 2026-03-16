from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Optional, Protocol, cast
from uuid import UUID

import joblib  # pyright: ignore[reportMissingTypeStubs]
from google.api_core.exceptions import NotFound
from google.cloud import storage  # pyright: ignore[reportMissingTypeStubs]
from google.cloud.storage.retry import DEFAULT_RETRY  # pyright: ignore[reportMissingTypeStubs]

from python.domain.repo.models_repo import ModelRecord, ModelsRepo

DEFAULT_GCS_MODELS_PREFIX: Final[str] = "models"
DEFAULT_GCS_TIMEOUT_SECONDS: Final[float] = 60.0
DEFAULT_GCS_UPLOAD_TIMEOUT_SECONDS: Final[float] = 300.0
DEFAULT_GCS_UPLOAD_RETRY_TIMEOUT_SECONDS: Final[float] = 900.0
DEFAULT_GCS_UPLOAD_CHUNK_SIZE_BYTES: Final[int] = 8 * 1024 * 1024
_UPLOAD_CHUNK_SIZE_GRANULARITY_BYTES: Final[int] = 256 * 1024

log = logging.getLogger(__name__)


def _env_float_seconds(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning(
            "Invalid %s value=%r. Falling back to default=%s.",
            name,
            raw,
            default,
        )
        return default
    if value <= 0:
        log.warning(
            "Non-positive %s value=%r. Falling back to default=%s.",
            name,
            raw,
            default,
        )
        return default
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning(
            "Invalid %s value=%r. Falling back to default=%s.",
            name,
            raw,
            default,
        )
        return default
    if value <= 0:
        log.warning(
            "Non-positive %s value=%r. Falling back to default=%s.",
            name,
            raw,
            default,
        )
        return default
    return value


def _gcs_io_timeout_seconds() -> float:
    return _env_float_seconds("GCS_MODELS_TIMEOUT_SECONDS", DEFAULT_GCS_TIMEOUT_SECONDS)


def _gcs_upload_timeout_seconds() -> float:
    return _env_float_seconds("GCS_MODELS_UPLOAD_TIMEOUT_SECONDS", DEFAULT_GCS_UPLOAD_TIMEOUT_SECONDS)


def _gcs_upload_retry_timeout_seconds() -> float:
    return _env_float_seconds(
        "GCS_MODELS_UPLOAD_RETRY_TIMEOUT_SECONDS",
        DEFAULT_GCS_UPLOAD_RETRY_TIMEOUT_SECONDS,
    )


def _gcs_upload_chunk_size_bytes() -> int:
    chunk_size = _env_int("GCS_MODELS_UPLOAD_CHUNK_SIZE_BYTES", DEFAULT_GCS_UPLOAD_CHUNK_SIZE_BYTES)
    if chunk_size % _UPLOAD_CHUNK_SIZE_GRANULARITY_BYTES != 0:
        log.warning(
            "%s=%s is not a multiple of %s bytes. Falling back to default=%s.",
            "GCS_MODELS_UPLOAD_CHUNK_SIZE_BYTES",
            chunk_size,
            _UPLOAD_CHUNK_SIZE_GRANULARITY_BYTES,
            DEFAULT_GCS_UPLOAD_CHUNK_SIZE_BYTES,
        )
        return DEFAULT_GCS_UPLOAD_CHUNK_SIZE_BYTES
    return chunk_size


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, default=str)


class _JoblibLike(Protocol):
    def dump(
        self,
        value: Any,
        filename: Any,
        compress: int = ...,
        protocol: int = ...,
    ) -> Any: ...

    def load(self, filename: Any, mmap_mode: Optional[str] = ...) -> Any: ...


_joblib: _JoblibLike = cast(_JoblibLike, joblib)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


@dataclass(frozen=True)
class GoogleCloudStorageModelsRepo(ModelsRepo):
    bucket: storage.Bucket
    
    @staticmethod
    def get_default_bucket() -> storage.Bucket:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "").strip() or None
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT_ID environment variable must be set for GoogleCloudStorageModelsRepo")
        client = storage.Client(project=project_id)
        bucket_name = os.getenv("GCS_MODELS_BUCKET_NAME", "").strip()
        if not bucket_name:
            raise ValueError("GCS_MODELS_BUCKET_NAME must be configured for GoogleCloudStorageModelsRepo")
        return client.bucket(bucket_name)

    def __post_init__(self) -> None:
        bucket_name = getattr(self.bucket, "name", "").strip()
        if not bucket_name:
            raise ValueError("bucket must have a non-empty name")

    def _models_prefix(self, *, user_id: UUID, conversation_id: UUID) -> str:
        parts = (
            DEFAULT_GCS_MODELS_PREFIX,
            "users",
            str(user_id),
            "conversations",
            str(conversation_id),
            "models",
        )
        return "/".join(part for part in parts if part)

    def _artifact_blob_name(self, *, user_id: UUID, conversation_id: UUID, model_id: UUID) -> str:
        return f"{self._models_prefix(user_id=user_id, conversation_id=conversation_id)}/{model_id}.joblib"

    def _meta_blob_name(self, *, user_id: UUID, conversation_id: UUID, model_id: UUID) -> str:
        return f"{self._models_prefix(user_id=user_id, conversation_id=conversation_id)}/{model_id}.meta.json"

    def _artifact_blob(self, *, user_id: UUID, conversation_id: UUID, model_id: UUID) -> storage.Blob:
        return self.bucket.blob(
            self._artifact_blob_name(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
            )
        )

    def _meta_blob(self, *, user_id: UUID, conversation_id: UUID, model_id: UUID) -> storage.Blob:
        return self.bucket.blob(
            self._meta_blob_name(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
            )
        )

    def _artifact_gcs_uri(self, *, user_id: UUID, conversation_id: UUID, model_id: UUID) -> str:
        return f"gs://{self.bucket.name}/{self._artifact_blob_name(user_id=user_id, conversation_id=conversation_id, model_id=model_id)}"

    def _make_temp_path(self, suffix: str) -> Path:
        fd, path = tempfile.mkstemp(prefix="models_repo_", suffix=suffix)
        os.close(fd)
        return Path(path)

    def _build_saved_metadata(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
        artifact_bytes: int,
        artifact_sha256: str,
        app_metadata: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "model_id": str(model_id),
            "user_id": str(user_id),
            "conversation_id": str(conversation_id),
            "format": "joblib",
            "artifact_name": f"{model_id}.joblib",
            "artifact_blob_name": self._artifact_blob_name(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
            ),
            "artifact_gcs_uri": self._artifact_gcs_uri(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
            ),
            "artifact_bytes": artifact_bytes,
            "artifact_sha256": artifact_sha256,
            "saved_at_utc": _utc_now_iso(),
            "repo_backend": "gcs",
            "bucket": self.bucket.name,
            "app_metadata": dict(app_metadata or {}),
        }

    def _load_metadata(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> dict[str, Any]:
        io_timeout = _gcs_io_timeout_seconds()
        try:
            payload = self._meta_blob(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
            ).download_as_text(timeout=io_timeout)
        except NotFound:
            return {}
        except Exception:
            log.warning(
                "Failed to load metadata for model_id=%s from bucket=%s.",
                model_id,
                self.bucket.name,
                exc_info=True,
            )
            return {}

        try:
            decoded = json.loads(payload)
        except Exception:
            return {}

        return decoded if isinstance(decoded, dict) else {}

    def _metadata_defaults(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> dict[str, Any]:
        return {
            "model_id": str(model_id),
            "user_id": str(user_id),
            "conversation_id": str(conversation_id),
            "format": "joblib",
            "artifact_name": f"{model_id}.joblib",
            "artifact_blob_name": self._artifact_blob_name(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
            ),
            "artifact_gcs_uri": self._artifact_gcs_uri(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
            ),
            "repo_backend": "gcs",
            "bucket": self.bucket.name,
            "app_metadata": {},
        }

    def save_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
        model: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        artifact_blob = self._artifact_blob(
            user_id=user_id,
            conversation_id=conversation_id,
            model_id=model_id,
        )
        meta_blob = self._meta_blob(
            user_id=user_id,
            conversation_id=conversation_id,
            model_id=model_id,
        )

        tmp_artifact = self._make_temp_path(".joblib")
        try:
            _joblib.dump(model, tmp_artifact)
            artifact_size_bytes = tmp_artifact.stat().st_size
            upload_timeout = _gcs_upload_timeout_seconds()
            upload_retry_timeout = _gcs_upload_retry_timeout_seconds()
            io_timeout = _gcs_io_timeout_seconds()
            upload_chunk_size = _gcs_upload_chunk_size_bytes()
            upload_retry = DEFAULT_RETRY.with_timeout(upload_retry_timeout)
            artifact_blob.chunk_size = upload_chunk_size

            artifact_blob.metadata = {
                "model_id": str(model_id),
                "user_id": str(user_id),
                "conversation_id": str(conversation_id),
                "format": "joblib",
            }
            log.warning(
                "Uploading model artifact to GCS. model_id=%s bucket=%s blob=%s size_bytes=%s timeout_seconds=%s retry_timeout_seconds=%s chunk_size_bytes=%s",
                model_id,
                self.bucket.name,
                artifact_blob.name,
                artifact_size_bytes,
                upload_timeout,
                upload_retry_timeout,
                upload_chunk_size,
            )
            artifact_blob.upload_from_filename(
                filename=str(tmp_artifact),
                content_type="application/octet-stream",
                timeout=upload_timeout,
                retry=upload_retry,
            )
            log.warning(
                "Model artifact upload completed. model_id=%s bucket=%s blob=%s size_bytes=%s",
                model_id,
                self.bucket.name,
                artifact_blob.name,
                artifact_size_bytes,
            )

            repo_metadata = self._build_saved_metadata(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
                artifact_bytes=artifact_size_bytes,
                artifact_sha256=_sha256_file(tmp_artifact),
                app_metadata=metadata,
            )
            meta_blob.upload_from_string(
                data=_json_dumps(repo_metadata),
                content_type="application/json",
                timeout=io_timeout,
                retry=upload_retry,
            )
            log.warning(
                "Model metadata upload completed. model_id=%s bucket=%s blob=%s",
                model_id,
                self.bucket.name,
                meta_blob.name,
            )
        except Exception:
            log.warning(
                "Model save failed in GoogleCloudStorageModelsRepo. model_id=%s bucket=%s artifact_blob=%s",
                model_id,
                self.bucket.name,
                artifact_blob.name,
                exc_info=True,
            )
            raise
        finally:
            _safe_unlink(tmp_artifact)

    def load_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> ModelRecord | None:
        artifact_blob = self._artifact_blob(
            user_id=user_id,
            conversation_id=conversation_id,
            model_id=model_id,
        )
        io_timeout = _gcs_io_timeout_seconds()
        if not artifact_blob.exists(timeout=io_timeout):
            return None

        metadata = self._load_metadata(
            user_id=user_id,
            conversation_id=conversation_id,
            model_id=model_id,
        )

        tmp_artifact = self._make_temp_path(".joblib")
        try:
            artifact_blob.download_to_filename(
                filename=str(tmp_artifact),
                timeout=io_timeout,
            )
            model = _joblib.load(tmp_artifact)
        finally:
            _safe_unlink(tmp_artifact)

        for key, value in self._metadata_defaults(
            user_id=user_id,
            conversation_id=conversation_id,
            model_id=model_id,
        ).items():
            metadata.setdefault(key, value)

        return ModelRecord(model_id=model_id, model=model, metadata=metadata)

    def model_exists(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> bool:
        return bool(
            self._artifact_blob(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
            ).exists(timeout=_gcs_io_timeout_seconds())
        )

    def delete_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> None:
        for blob in (
            self._artifact_blob(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
            ),
            self._meta_blob(
                user_id=user_id,
                conversation_id=conversation_id,
                model_id=model_id,
            ),
        ):
            try:
                blob.delete(timeout=_gcs_io_timeout_seconds())
            except NotFound:
                pass
            except Exception:
                log.warning(
                    "Failed to delete model blob from GCS. model_id=%s bucket=%s blob=%s",
                    model_id,
                    self.bucket.name,
                    blob.name,
                    exc_info=True,
                )
                pass
