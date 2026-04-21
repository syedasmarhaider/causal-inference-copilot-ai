from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from google.api_core.exceptions import NotFound

import python.implementation.repo.google_cloud_storage_model_repo as model_repo_module
from python.implementation.repo.google_cloud_storage_model_repo import (
    DEFAULT_GCS_UPLOAD_CHUNK_SIZE_BYTES,
    GoogleCloudStorageModelsRepo,
)


@dataclass
class _FakeRetry:
    timeout: float | None = None

    def with_timeout(self, timeout: float) -> _FakeRetry:
        return _FakeRetry(timeout=timeout)


@dataclass
class _FakeBlob:
    name: str
    present: bool = False
    bytes_data: bytes = b""
    chunk_size: int | None = None
    metadata: dict[str, str] | None = None
    upload_filename_calls: list[dict[str, Any]] = field(default_factory=list)
    upload_string_calls: list[dict[str, Any]] = field(default_factory=list)
    download_filename_calls: list[dict[str, Any]] = field(default_factory=list)
    download_text_calls: list[dict[str, Any]] = field(default_factory=list)
    exists_calls: list[float | None] = field(default_factory=list)
    delete_calls: list[float | None] = field(default_factory=list)
    next_upload_filename_errors: list[Exception] = field(default_factory=list)
    next_upload_string_errors: list[Exception] = field(default_factory=list)
    next_download_filename_errors: list[Exception] = field(default_factory=list)
    next_download_text_errors: list[Exception] = field(default_factory=list)
    next_delete_errors: list[Exception] = field(default_factory=list)

    def exists(self, *, timeout: float | None = None) -> bool:
        self.exists_calls.append(timeout)
        return self.present

    def upload_from_filename(
        self,
        *,
        filename: str,
        content_type: str,
        timeout: float,
        retry: Any,
    ) -> None:
        self.upload_filename_calls.append(
            {
                "filename": filename,
                "content_type": content_type,
                "timeout": timeout,
                "retry": retry,
            }
        )
        if self.next_upload_filename_errors:
            raise self.next_upload_filename_errors.pop(0)
        self.bytes_data = Path(filename).read_bytes()
        self.present = True

    def upload_from_string(
        self,
        *,
        data: str,
        content_type: str,
        timeout: float,
        retry: Any,
    ) -> None:
        self.upload_string_calls.append(
            {
                "data": data,
                "content_type": content_type,
                "timeout": timeout,
                "retry": retry,
            }
        )
        if self.next_upload_string_errors:
            raise self.next_upload_string_errors.pop(0)
        self.bytes_data = data.encode("utf-8")
        self.present = True

    def download_to_filename(self, *, filename: str, timeout: float) -> None:
        self.download_filename_calls.append({"filename": filename, "timeout": timeout})
        if self.next_download_filename_errors:
            raise self.next_download_filename_errors.pop(0)
        if not self.present:
            raise NotFound("missing")
        Path(filename).write_bytes(self.bytes_data)

    def download_as_text(self, *, timeout: float) -> str:
        self.download_text_calls.append({"timeout": timeout})
        if self.next_download_text_errors:
            raise self.next_download_text_errors.pop(0)
        if not self.present:
            raise NotFound("missing")
        return self.bytes_data.decode("utf-8")

    def delete(self, *, timeout: float) -> None:
        self.delete_calls.append(timeout)
        if self.next_delete_errors:
            raise self.next_delete_errors.pop(0)
        if not self.present:
            raise NotFound("missing")
        self.present = False
        self.bytes_data = b""


@dataclass
class _FakeBucket:
    name: str
    blobs: dict[str, _FakeBlob] = field(default_factory=dict)

    def blob(self, blob_name: str) -> _FakeBlob:
        if blob_name not in self.blobs:
            self.blobs[blob_name] = _FakeBlob(name=blob_name)
        return self.blobs[blob_name]


@dataclass
class _ClientRecorder:
    project_seen: str | None = None
    bucket_seen: str | None = None

    def __call__(self, *, project: str | None = None) -> _ClientRecorder:
        self.project_seen = project
        return self

    def bucket(self, bucket_name: str) -> _FakeBucket:
        self.bucket_seen = bucket_name
        return _FakeBucket(name=bucket_name)


@dataclass
class _FakeJoblib:
    loaded_value: Any = None

    def dump(self, value: Any, filename: Path, compress: int = 0, protocol: int = 0) -> None:
        payload = json.dumps({"model": value}, default=str).encode("utf-8")
        Path(filename).write_bytes(payload)

    def load(self, filename: Path, mmap_mode: str | None = None) -> Any:
        if self.loaded_value is not None:
            return self.loaded_value
        return json.loads(Path(filename).read_text(encoding="utf-8"))


def _ids() -> tuple[UUID, UUID, UUID]:
    return uuid4(), uuid4(), uuid4()


def test_get_default_bucket_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _ClientRecorder()
    monkeypatch.setattr(model_repo_module.storage, "Client", recorder)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_ID", "project-x")
    monkeypatch.setenv("GCS_MODELS_BUCKET_NAME", "models-bucket")

    bucket = GoogleCloudStorageModelsRepo.get_default_bucket()

    assert isinstance(bucket, _FakeBucket)
    assert recorder.project_seen == "project-x"
    assert recorder.bucket_seen == "models-bucket"


def test_get_default_bucket_requires_project_and_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _ClientRecorder()
    monkeypatch.setattr(model_repo_module.storage, "Client", recorder)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT_ID", raising=False)
    monkeypatch.setenv("GCS_MODELS_BUCKET_NAME", "models-bucket")

    with pytest.raises(ValueError, match=r"GOOGLE_CLOUD_PROJECT_ID"):
        GoogleCloudStorageModelsRepo.get_default_bucket()

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_ID", "project-x")
    monkeypatch.delenv("GCS_MODELS_BUCKET_NAME", raising=False)
    with pytest.raises(ValueError, match=r"GCS_MODELS_BUCKET_NAME"):
        GoogleCloudStorageModelsRepo.get_default_bucket()


def test_constructor_rejects_empty_bucket_name() -> None:
    with pytest.raises(ValueError, match=r"non-empty name"):
        GoogleCloudStorageModelsRepo(bucket=_FakeBucket(name=""))


def test_save_model_uploads_artifact_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    user_id, conversation_id, model_id = _ids()
    bucket = _FakeBucket(name="models")
    repo = GoogleCloudStorageModelsRepo(bucket=bucket)

    fake_joblib = _FakeJoblib()
    monkeypatch.setattr(model_repo_module, "_joblib", fake_joblib)
    monkeypatch.setattr(model_repo_module, "DEFAULT_RETRY", _FakeRetry())
    monkeypatch.setattr(model_repo_module, "_utc_now_iso", lambda: "2026-03-30T00:00:00+00:00")
    monkeypatch.setenv("GCS_MODELS_UPLOAD_TIMEOUT_SECONDS", "22")
    monkeypatch.setenv("GCS_MODELS_UPLOAD_RETRY_TIMEOUT_SECONDS", "33")
    monkeypatch.setenv("GCS_MODELS_TIMEOUT_SECONDS", "11")
    monkeypatch.setenv("GCS_MODELS_UPLOAD_CHUNK_SIZE_BYTES", str(512 * 1024))

    repo.save_model(
        user_id=user_id,
        conversation_id=conversation_id,
        model_id=model_id,
        model={"coef": [1.0, 2.0]},
        metadata={"source": "unit-test"},
    )

    artifact_name = repo._artifact_blob_name(
        user_id=user_id, conversation_id=conversation_id, model_id=model_id
    )  # noqa: SLF001
    meta_name = repo._meta_blob_name(
        user_id=user_id, conversation_id=conversation_id, model_id=model_id
    )  # noqa: SLF001

    artifact_blob = bucket.blobs[artifact_name]
    meta_blob = bucket.blobs[meta_name]

    assert artifact_blob.present is True
    assert artifact_blob.chunk_size == 512 * 1024
    assert artifact_blob.metadata == {
        "model_id": str(model_id),
        "user_id": str(user_id),
        "conversation_id": str(conversation_id),
        "format": "joblib",
    }
    assert artifact_blob.upload_filename_calls[0]["timeout"] == 22.0
    assert isinstance(artifact_blob.upload_filename_calls[0]["retry"], _FakeRetry)
    assert artifact_blob.upload_filename_calls[0]["retry"].timeout == 33.0

    metadata_payload = json.loads(meta_blob.upload_string_calls[0]["data"])
    assert metadata_payload["model_id"] == str(model_id)
    assert metadata_payload["bucket"] == "models"
    assert metadata_payload["app_metadata"] == {"source": "unit-test"}
    assert metadata_payload["saved_at_utc"] == "2026-03-30T00:00:00+00:00"


def test_save_model_falls_back_for_invalid_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    user_id, conversation_id, model_id = _ids()
    bucket = _FakeBucket(name="models")
    repo = GoogleCloudStorageModelsRepo(bucket=bucket)

    monkeypatch.setattr(model_repo_module, "_joblib", _FakeJoblib())
    monkeypatch.setattr(model_repo_module, "DEFAULT_RETRY", _FakeRetry())
    monkeypatch.setenv("GCS_MODELS_UPLOAD_CHUNK_SIZE_BYTES", "123")

    repo.save_model(
        user_id=user_id,
        conversation_id=conversation_id,
        model_id=model_id,
        model={"x": 1},
    )

    artifact_name = repo._artifact_blob_name(
        user_id=user_id, conversation_id=conversation_id, model_id=model_id
    )  # noqa: SLF001
    assert bucket.blobs[artifact_name].chunk_size == DEFAULT_GCS_UPLOAD_CHUNK_SIZE_BYTES


def test_load_model_returns_none_when_artifact_missing() -> None:
    user_id, conversation_id, model_id = _ids()
    repo = GoogleCloudStorageModelsRepo(bucket=_FakeBucket(name="models"))

    assert (
        repo.load_model(user_id=user_id, conversation_id=conversation_id, model_id=model_id) is None
    )


def test_load_model_downloads_and_applies_metadata_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id, conversation_id, model_id = _ids()
    bucket = _FakeBucket(name="models")
    repo = GoogleCloudStorageModelsRepo(bucket=bucket)

    artifact_name = repo._artifact_blob_name(
        user_id=user_id, conversation_id=conversation_id, model_id=model_id
    )  # noqa: SLF001
    artifact_blob = bucket.blob(artifact_name)
    artifact_blob.present = True
    artifact_blob.bytes_data = b"serialized"

    fake_joblib = _FakeJoblib(loaded_value={"loaded": True})
    monkeypatch.setattr(model_repo_module, "_joblib", fake_joblib)

    record = repo.load_model(user_id=user_id, conversation_id=conversation_id, model_id=model_id)

    assert record is not None
    assert record.model_id == model_id
    assert record.model == {"loaded": True}
    assert record.metadata["model_id"] == str(model_id)
    assert record.metadata["artifact_name"] == f"{model_id}.joblib"
    assert record.metadata["app_metadata"] == {}


def test_model_exists_and_delete_model_are_best_effort() -> None:
    user_id, conversation_id, model_id = _ids()
    bucket = _FakeBucket(name="models")
    repo = GoogleCloudStorageModelsRepo(bucket=bucket)

    artifact_name = repo._artifact_blob_name(
        user_id=user_id, conversation_id=conversation_id, model_id=model_id
    )  # noqa: SLF001
    meta_name = repo._meta_blob_name(
        user_id=user_id, conversation_id=conversation_id, model_id=model_id
    )  # noqa: SLF001

    artifact_blob = bucket.blob(artifact_name)
    meta_blob = bucket.blob(meta_name)

    artifact_blob.present = True
    meta_blob.present = True

    assert (
        repo.model_exists(user_id=user_id, conversation_id=conversation_id, model_id=model_id)
        is True
    )

    meta_blob.next_delete_errors.append(RuntimeError("transient"))
    repo.delete_model(user_id=user_id, conversation_id=conversation_id, model_id=model_id)

    assert artifact_blob.present is False


def test_load_metadata_returns_empty_on_not_found_or_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id, conversation_id, model_id = _ids()
    bucket = _FakeBucket(name="models")
    repo = GoogleCloudStorageModelsRepo(bucket=bucket)

    meta_name = repo._meta_blob_name(
        user_id=user_id, conversation_id=conversation_id, model_id=model_id
    )  # noqa: SLF001
    meta_blob = bucket.blob(meta_name)
    meta_blob.next_download_text_errors.append(NotFound("missing"))

    assert (
        repo._load_metadata(user_id=user_id, conversation_id=conversation_id, model_id=model_id)
        == {}
    )  # noqa: SLF001

    meta_blob.present = True
    meta_blob.bytes_data = b"not-json"
    assert (
        repo._load_metadata(user_id=user_id, conversation_id=conversation_id, model_id=model_id)
        == {}
    )  # noqa: SLF001
