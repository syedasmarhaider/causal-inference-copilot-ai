from __future__ import annotations

import json
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pandas as pd
import pytest
from google.api_core.exceptions import NotFound, PreconditionFailed

from python.implementation.repo.google_cloud_storage_data_repo import (
    DEFAULT_GCS_TIMEOUT_SECONDS,
    GoogleCloudStorageDataRepo,
    _validate_image_bytes,
)


@dataclass
class _FakeBlob:
    name: str
    present: bool = False
    data: bytes = b""
    upload_calls: list[dict[str, object]] = field(default_factory=list)
    delete_calls: list[dict[str, object]] = field(default_factory=list)
    exists_calls: list[float | None] = field(default_factory=list)
    next_upload_errors: list[Exception] = field(default_factory=list)
    next_download_errors: list[Exception] = field(default_factory=list)
    next_delete_errors: list[Exception] = field(default_factory=list)
    next_exists_errors: list[Exception] = field(default_factory=list)

    def exists(self, *, timeout: float | None = None) -> bool:
        self.exists_calls.append(timeout)
        if self.next_exists_errors:
            raise self.next_exists_errors.pop(0)
        return self.present

    def download_as_bytes(self, *, timeout: float | None = None) -> bytes:
        if self.next_download_errors:
            raise self.next_download_errors.pop(0)
        if not self.present:
            raise NotFound("missing")
        return self.data

    def upload_from_string(self, **kwargs: object) -> None:
        self.upload_calls.append(dict(kwargs))
        if self.next_upload_errors:
            raise self.next_upload_errors.pop(0)

        raw_data = kwargs.get("data", b"")
        if isinstance(raw_data, str):
            self.data = raw_data.encode("utf-8")
        elif isinstance(raw_data, bytes):
            self.data = raw_data
        else:
            self.data = str(raw_data).encode("utf-8")

        self.present = True

    def delete(self, *, timeout: float | None = None) -> None:
        self.delete_calls.append({"timeout": timeout})
        if self.next_delete_errors:
            raise self.next_delete_errors.pop(0)
        if not self.present:
            raise NotFound("missing")
        self.present = False
        self.data = b""


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
    requested_project: str | None = None
    requested_bucket: str | None = None

    def __call__(self, *, project: str | None = None) -> _ClientRecorder:
        self.requested_project = project
        return self

    def bucket(self, bucket_name: str) -> _FakeBucket:
        self.requested_bucket = bucket_name
        return _FakeBucket(name=bucket_name)


def _ids() -> tuple[UUID, UUID, UUID]:
    return uuid4(), uuid4(), uuid4()


def _png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"payload"


def _jpeg_bytes() -> bytes:
    return b"\xFF\xD8" + b"payload"


def _webp_bytes() -> bytes:
    return b"RIFF" + b"xxxx" + b"WEBP" + b"payload"


def test_get_default_bucket_uses_env_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _ClientRecorder()
    monkeypatch.setattr(
        "python.implementation.repo.google_cloud_storage_data_repo.storage.Client",
        recorder,
    )
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_ID", "my-project")
    monkeypatch.setenv("GCS_DATA_BUCKET_NAME", "my-data-bucket")

    bucket = GoogleCloudStorageDataRepo.get_default_bucket()

    assert isinstance(bucket, _FakeBucket)
    assert bucket.name == "my-data-bucket"
    assert recorder.requested_project == "my-project"
    assert recorder.requested_bucket == "my-data-bucket"


def test_get_default_bucket_requires_bucket_name(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _ClientRecorder()
    monkeypatch.setattr(
        "python.implementation.repo.google_cloud_storage_data_repo.storage.Client",
        recorder,
    )
    monkeypatch.delenv("GCS_DATA_BUCKET_NAME", raising=False)

    with pytest.raises(ValueError, match=r"GCS_DATA_BUCKET_NAME"):
        GoogleCloudStorageDataRepo.get_default_bucket()


def test_constructor_rejects_bucket_with_empty_name() -> None:
    with pytest.raises(ValueError, match=r"non-empty name"):
        GoogleCloudStorageDataRepo(bucket=_FakeBucket(name=""))


def test_get_csv_data_reads_dataframe_and_applies_limit() -> None:
    user_id, conversation_id, dataset_id = _ids()
    bucket = _FakeBucket(name="bucket")
    repo = GoogleCloudStorageDataRepo(bucket=bucket)

    blob_name = repo._dataset_blob_name(user_id, conversation_id, dataset_id)  # noqa: SLF001
    blob = bucket.blob(blob_name)
    blob.data = b"a,b\n1,2\n3,4\n"
    blob.present = True

    frame = repo.get_csv_data(user_id, conversation_id, dataset_id, limit=1)

    assert frame.to_dict(orient="records") == [{"a": 1, "b": 2}]
    assert blob.exists_calls == [DEFAULT_GCS_TIMEOUT_SECONDS]


def test_get_csv_data_validates_limit_and_missing_blob() -> None:
    user_id, conversation_id, dataset_id = _ids()
    repo = GoogleCloudStorageDataRepo(bucket=_FakeBucket(name="bucket"))

    with pytest.raises(ValueError, match=r"limit must be a positive int"):
        repo.get_csv_data(user_id, conversation_id, dataset_id, limit=0)

    with pytest.raises(FileNotFoundError, match=r"CSV not found"):
        repo.get_csv_data(user_id, conversation_id, dataset_id)


def test_save_csv_data_serializes_and_respects_overwrite_flag() -> None:
    user_id, conversation_id, dataset_id = _ids()
    bucket = _FakeBucket(name="bucket")
    repo = GoogleCloudStorageDataRepo(bucket=bucket)

    repo.save_csv_data(
        user_id,
        conversation_id,
        dataset_id,
        pd.DataFrame([{"x": 7}, {"x": 8}]),
        overwrite=False,
    )

    blob_name = repo._dataset_blob_name(user_id, conversation_id, dataset_id)  # noqa: SLF001
    call = bucket.blobs[blob_name].upload_calls[0]
    assert call["content_type"] == "text/csv; charset=utf-8"
    assert call["if_generation_match"] == 0
    assert "x" in str(call["data"])


def test_save_csv_data_maps_precondition_failed_to_file_exists() -> None:
    user_id, conversation_id, dataset_id = _ids()
    bucket = _FakeBucket(name="bucket")
    repo = GoogleCloudStorageDataRepo(bucket=bucket)

    blob_name = repo._dataset_blob_name(user_id, conversation_id, dataset_id)  # noqa: SLF001
    bucket.blob(blob_name).next_upload_errors.append(PreconditionFailed("exists"))

    with pytest.raises(FileExistsError, match=r"Refusing to overwrite existing CSV"):
        repo.save_csv_data(user_id, conversation_id, dataset_id, pd.DataFrame([{"x": 1}]))


def test_get_and_save_json_data_roundtrip_and_respect_overwrite_flag() -> None:
    user_id, conversation_id, dataset_id = _ids()
    bucket = _FakeBucket(name="bucket")
    repo = GoogleCloudStorageDataRepo(bucket=bucket)

    repo.save_json_data(
        user_id,
        conversation_id,
        dataset_id,
        json.dumps({"chart": "ok"}),
        overwrite=False,
    )

    blob_name = repo._json_blob_name(user_id, conversation_id, dataset_id)  # noqa: SLF001
    call = bucket.blobs[blob_name].upload_calls[0]
    assert call["content_type"] == "application/json; charset=utf-8"
    assert call["if_generation_match"] == 0
    assert repo.get_json_data(user_id, conversation_id, dataset_id) == '{"chart": "ok"}'


def test_get_and_save_json_data_map_missing_and_precondition_errors() -> None:
    user_id, conversation_id, dataset_id = _ids()
    bucket = _FakeBucket(name="bucket")
    repo = GoogleCloudStorageDataRepo(bucket=bucket)

    with pytest.raises(FileNotFoundError, match=r"JSON not found"):
        repo.get_json_data(user_id, conversation_id, dataset_id)

    blob_name = repo._json_blob_name(user_id, conversation_id, dataset_id)  # noqa: SLF001
    bucket.blob(blob_name).next_upload_errors.append(PreconditionFailed("exists"))

    with pytest.raises(FileExistsError, match=r"Refusing to overwrite existing JSON"):
        repo.save_json_data(user_id, conversation_id, dataset_id, "{}")


@pytest.mark.parametrize(
    ("mime", "content", "error_pattern"),
    [
        ("image/png", b"bad", r"PNG"),
        ("image/jpeg", b"bad", r"JPEG"),
        ("image/webp", b"bad", r"WEBP"),
    ],
)
def test_validate_image_bytes_rejects_invalid_signatures(
    mime: str,
    content: bytes,
    error_pattern: str,
) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        _validate_image_bytes(mime, content)  # type: ignore[arg-type]


def test_save_artifact_writes_meta_and_prunes_other_mimes_on_overwrite() -> None:
    user_id, conversation_id, artifact_id = _ids()
    bucket = _FakeBucket(name="bucket")
    repo = GoogleCloudStorageDataRepo(bucket=bucket)

    png_blob_name = repo._artifact_blob_name(  # noqa: SLF001
        user_id,
        conversation_id,
        artifact_id,
        mime="image/png",
    )
    jpeg_blob_name = repo._artifact_blob_name(  # noqa: SLF001
        user_id,
        conversation_id,
        artifact_id,
        mime="image/jpeg",
    )
    bucket.blob(jpeg_blob_name).present = True
    bucket.blob(jpeg_blob_name).data = _jpeg_bytes()

    repo.save_artifact(
        user_id,
        conversation_id,
        artifact_id,
        _png_bytes(),
        mime="image/png",
        overwrite=True,
    )

    assert bucket.blobs[png_blob_name].present is True
    meta_name = repo._artifact_meta_blob_name(user_id, conversation_id, artifact_id)  # noqa: SLF001
    meta = json.loads(bucket.blobs[meta_name].data.decode("utf-8"))
    assert meta == {"mime": "image/png"}
    assert bucket.blobs[jpeg_blob_name].present is False


def test_save_artifact_rolls_back_binary_when_meta_upload_fails_with_no_overwrite() -> None:
    user_id, conversation_id, artifact_id = _ids()
    bucket = _FakeBucket(name="bucket")
    repo = GoogleCloudStorageDataRepo(bucket=bucket)

    artifact_blob_name = repo._artifact_blob_name(  # noqa: SLF001
        user_id,
        conversation_id,
        artifact_id,
        mime="image/png",
    )
    meta_blob_name = repo._artifact_meta_blob_name(user_id, conversation_id, artifact_id)  # noqa: SLF001

    bucket.blob(meta_blob_name).next_upload_errors.append(RuntimeError("meta write failed"))

    with pytest.raises(ValueError, match=r"Failed to save artifact"):
        repo.save_artifact(
            user_id,
            conversation_id,
            artifact_id,
            _png_bytes(),
            mime="image/png",
            overwrite=False,
        )

    artifact_blob = bucket.blobs[artifact_blob_name]
    assert artifact_blob.present is False
    assert artifact_blob.delete_calls, "artifact blob should be deleted on rollback"


def test_save_artifact_maps_precondition_failed_to_file_exists() -> None:
    user_id, conversation_id, artifact_id = _ids()
    bucket = _FakeBucket(name="bucket")
    repo = GoogleCloudStorageDataRepo(bucket=bucket)

    artifact_blob_name = repo._artifact_blob_name(  # noqa: SLF001
        user_id,
        conversation_id,
        artifact_id,
        mime="image/png",
    )
    bucket.blob(artifact_blob_name).next_upload_errors.append(PreconditionFailed("exists"))

    with pytest.raises(FileExistsError, match=r"Refusing to overwrite existing artifact"):
        repo.save_artifact(
            user_id,
            conversation_id,
            artifact_id,
            _png_bytes(),
            mime="image/png",
            overwrite=False,
        )


def test_get_artifact_mime_uses_metadata_when_valid() -> None:
    user_id, conversation_id, artifact_id = _ids()
    bucket = _FakeBucket(name="bucket")
    repo = GoogleCloudStorageDataRepo(bucket=bucket)

    meta_blob = bucket.blob(repo._artifact_meta_blob_name(user_id, conversation_id, artifact_id))  # noqa: SLF001
    meta_blob.present = True
    meta_blob.data = b'{"mime":"image/webp"}'

    webp_blob = bucket.blob(
        repo._artifact_blob_name(user_id, conversation_id, artifact_id, mime="image/webp")  # noqa: SLF001
    )
    webp_blob.present = True

    assert repo.get_artifact_mime(user_id, conversation_id, artifact_id) == "image/webp"


def test_get_artifact_mime_falls_back_to_probe_on_bad_meta() -> None:
    user_id, conversation_id, artifact_id = _ids()
    bucket = _FakeBucket(name="bucket")
    repo = GoogleCloudStorageDataRepo(bucket=bucket)

    meta_blob = bucket.blob(repo._artifact_meta_blob_name(user_id, conversation_id, artifact_id))  # noqa: SLF001
    meta_blob.present = True
    meta_blob.data = b"not-json"

    png_blob = bucket.blob(
        repo._artifact_blob_name(user_id, conversation_id, artifact_id, mime="image/png")  # noqa: SLF001
    )
    png_blob.present = True

    assert repo.get_artifact_mime(user_id, conversation_id, artifact_id) == "image/png"


def test_get_artifact_mime_rejects_ambiguous_probe() -> None:
    user_id, conversation_id, artifact_id = _ids()
    bucket = _FakeBucket(name="bucket")
    repo = GoogleCloudStorageDataRepo(bucket=bucket)

    for mime, content in (
        ("image/png", _png_bytes()),
        ("image/jpeg", _jpeg_bytes()),
    ):
        blob = bucket.blob(repo._artifact_blob_name(user_id, conversation_id, artifact_id, mime=mime))  # noqa: SLF001
        blob.present = True
        blob.data = content

    with pytest.raises(ValueError, match=r"ambiguous"):
        repo.get_artifact_mime(user_id, conversation_id, artifact_id)


def test_get_artifact_bytes_checks_expected_mime_and_maps_not_found() -> None:
    user_id, conversation_id, artifact_id = _ids()
    bucket = _FakeBucket(name="bucket")
    repo = GoogleCloudStorageDataRepo(bucket=bucket)

    meta_blob = bucket.blob(repo._artifact_meta_blob_name(user_id, conversation_id, artifact_id))  # noqa: SLF001
    meta_blob.present = True
    meta_blob.data = b'{"mime":"image/png"}'

    png_blob = bucket.blob(
        repo._artifact_blob_name(user_id, conversation_id, artifact_id, mime="image/png")  # noqa: SLF001
    )
    png_blob.present = True
    png_blob.data = _png_bytes()

    with pytest.raises(ValueError, match=r"mime mismatch"):
        repo.get_artifact_bytes(
            user_id,
            conversation_id,
            artifact_id,
            expected_mime="image/jpeg",
        )

    png_blob.next_download_errors.append(NotFound("gone"))
    with pytest.raises(FileNotFoundError, match=r"artifact not found"):
        repo.get_artifact_bytes(user_id, conversation_id, artifact_id)
