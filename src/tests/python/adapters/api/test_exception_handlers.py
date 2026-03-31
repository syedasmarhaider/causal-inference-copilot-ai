from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from python.adapters.api.exception_handlers import (
    map_workflow_error_to_http_exception,
    register_exception_handlers,
)
from python.domain.models.errors import (
    ArtifactNotFoundError,
    AuthenticationError,
    ConversationNotFoundError,
    DataUploadError,
    InvalidStateError,
    StateNotFoundError,
    ValidationError,
    WorkflowError,
)


@pytest.mark.parametrize(
    ("exc", "status_code", "detail"),
    [
        (
            ConversationNotFoundError(user_id=uuid4(), conversation_id=uuid4()),
            404,
            "conversation not found",
        ),
        (StateNotFoundError(state_name="S1"), 404, "state not found"),
        (ArtifactNotFoundError(artifact_id=uuid4()), 404, "artifact not found"),
        (AuthenticationError("missing auth"), 401, "authentication failed"),
        (ValidationError(field="field", reason="invalid"), 422, "validation failed"),
        (DataUploadError(reason="bad csv"), 400, "data upload failed"),
        (InvalidStateError(state_name="S1", reason="done"), 409, "invalid workflow state"),
        (FileNotFoundError("missing file"), 404, "resource not found"),
        (FileExistsError("already exists"), 409, "resource already exists"),
    ],
)
def test_map_workflow_error_to_http_exception_maps_known_errors(
    exc: Exception,
    status_code: int,
    detail: str,
) -> None:
    http_exc = map_workflow_error_to_http_exception(exc)
    assert http_exc is not None
    assert http_exc.status_code == status_code
    assert http_exc.detail == detail


def test_map_workflow_error_to_http_exception_returns_none_for_unmapped_base_error() -> None:
    http_exc = map_workflow_error_to_http_exception(WorkflowError("unknown workflow issue"))
    assert http_exc is None


def test_exception_handlers_return_mapped_and_safe_fallback_errors() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/mapped")
    async def mapped() -> dict[str, bool]:
        raise ConversationNotFoundError(user_id=uuid4(), conversation_id=uuid4())

    @app.get("/unmapped-workflow")
    async def unmapped_workflow() -> dict[str, bool]:
        raise WorkflowError("unexpected workflow failure")

    @app.get("/unhandled")
    async def unhandled() -> dict[str, bool]:
        raise RuntimeError("unexpected runtime failure")

    @app.get("/http-exc")
    async def http_exc() -> dict[str, bool]:
        raise HTTPException(status_code=418, detail="teapot")

    @app.get("/file-not-found")
    async def file_not_found() -> dict[str, bool]:
        raise FileNotFoundError("missing file")

    @app.get("/file-exists")
    async def file_exists() -> dict[str, bool]:
        raise FileExistsError("already exists")

    client = TestClient(app, raise_server_exceptions=False)

    mapped_response = client.get("/mapped")
    assert mapped_response.status_code == 404
    assert mapped_response.json() == {"detail": "conversation not found"}

    unmapped_workflow_response = client.get("/unmapped-workflow")
    assert unmapped_workflow_response.status_code == 500
    assert unmapped_workflow_response.json() == {"detail": "Internal server error"}

    unhandled_response = client.get("/unhandled")
    assert unhandled_response.status_code == 500
    assert unhandled_response.json() == {"detail": "Internal server error"}

    http_exc_response = client.get("/http-exc")
    assert http_exc_response.status_code == 418
    assert http_exc_response.json() == {"detail": "teapot"}

    file_not_found_response = client.get("/file-not-found")
    assert file_not_found_response.status_code == 404
    assert file_not_found_response.json() == {"detail": "resource not found"}

    file_exists_response = client.get("/file-exists")
    assert file_exists_response.status_code == 409
    assert file_exists_response.json() == {"detail": "resource already exists"}
