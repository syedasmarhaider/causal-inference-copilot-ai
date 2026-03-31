from __future__ import annotations

from http import HTTPStatus
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from python.adapters.api.exception_handlers import (
    UNKNOWN_ERROR_CODE,
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


class UnknownWorkflowError(WorkflowError):
    code = "workflow_error"


@pytest.mark.parametrize(
    ("exc", "status_code"),
    [
        (
            ConversationNotFoundError(user_id=uuid4(), conversation_id=uuid4()),
            404,
        ),
        (StateNotFoundError(state_name="S1"), 404),
        (ArtifactNotFoundError(artifact_id=uuid4()), 404),
        (AuthenticationError("missing auth"), 401),
        (ValidationError(field="field", reason="invalid"), 422),
        (DataUploadError(reason="bad csv"), 400),
        (InvalidStateError(state_name="S1", reason="done"), 409),
        (FileNotFoundError("missing file"), 404),
        (FileExistsError("already exists"), 409),
    ],
)
def test_map_workflow_error_to_http_exception_maps_known_errors(
    exc: Exception,
    status_code: int,
) -> None:
    http_exc = map_workflow_error_to_http_exception(exc)
    assert http_exc is not None
    assert http_exc.status_code == status_code
    assert http_exc.detail == str(exc)


def test_map_workflow_error_to_http_exception_returns_none_for_unmapped_base_error() -> None:
    http_exc = map_workflow_error_to_http_exception(UnknownWorkflowError("unknown workflow issue"))
    assert http_exc is None


def test_exception_handlers_return_mapped_and_safe_fallback_errors() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    mapped_user_id = uuid4()
    mapped_conversation_id = uuid4()

    @app.get("/mapped")
    async def mapped() -> dict[str, bool]:
        raise ConversationNotFoundError(
            user_id=mapped_user_id,
            conversation_id=mapped_conversation_id,
        )

    @app.get("/unmapped-workflow")
    async def unmapped_workflow() -> dict[str, bool]:
        raise UnknownWorkflowError("unexpected workflow failure")

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

    @app.get("/invalid-state")
    async def invalid_state() -> dict[str, bool]:
        raise InvalidStateError(state_name="S1", reason="done")

    @app.get("/http-500")
    async def http_500() -> dict[str, bool]:
        raise HTTPException(status_code=500, detail="should not leak")

    class BrokenValidationError(ValidationError):
        def __str__(self) -> str:
            raise RuntimeError("str failed")

    @app.get("/broken-mapped")
    async def broken_mapped() -> dict[str, bool]:
        raise BrokenValidationError(field="field", reason="invalid")

    class BrokenWorkflowError(WorkflowError):
        code = "workflow_error"

        def __str__(self) -> str:
            raise RuntimeError("str failed")

    @app.get("/broken-unmapped")
    async def broken_unmapped() -> dict[str, bool]:
        raise BrokenWorkflowError("unknown")

    client = TestClient(app, raise_server_exceptions=False)

    mapped_response = client.get("/mapped")
    assert mapped_response.status_code == 404
    assert mapped_response.json() == {
        "code": "conversation_not_found",
        "message": "Conversation not found",
        "detail": str(
            ConversationNotFoundError(
                user_id=mapped_user_id,
                conversation_id=mapped_conversation_id,
            )
        ),
    }

    unmapped_workflow_response = client.get("/unmapped-workflow")
    assert unmapped_workflow_response.status_code == 500
    assert unmapped_workflow_response.json() == {
        "code": "workflow_error",
        "message": "Internal server error",
        "detail": "Internal server error",
    }

    unhandled_response = client.get("/unhandled")
    assert unhandled_response.status_code == 500
    assert unhandled_response.json() == {
        "code": UNKNOWN_ERROR_CODE,
        "message": "Internal server error",
        "detail": "Internal server error",
    }

    http_exc_response = client.get("/http-exc")
    assert http_exc_response.status_code == 418
    assert http_exc_response.json() == {
        "code": UNKNOWN_ERROR_CODE,
        "message": HTTPStatus(418).phrase,
        "detail": "teapot",
    }

    http_500_response = client.get("/http-500")
    assert http_500_response.status_code == 500
    assert http_500_response.json() == {
        "code": UNKNOWN_ERROR_CODE,
        "message": "Internal server error",
        "detail": "Internal server error",
    }

    file_not_found_response = client.get("/file-not-found")
    assert file_not_found_response.status_code == 404
    assert file_not_found_response.json() == {
        "code": UNKNOWN_ERROR_CODE,
        "message": "Resource not found",
        "detail": "missing file",
    }

    file_exists_response = client.get("/file-exists")
    assert file_exists_response.status_code == 409
    assert file_exists_response.json() == {
        "code": UNKNOWN_ERROR_CODE,
        "message": "Resource already exists",
        "detail": "already exists",
    }

    invalid_state_response = client.get("/invalid-state")
    assert invalid_state_response.status_code == 409
    assert invalid_state_response.json() == {
        "code": "invalid_state",
        "message": "Invalid workflow state",
        "detail": "Invalid state 'S1': done",
    }

    broken_mapped_response = client.get("/broken-mapped")
    assert broken_mapped_response.status_code == 422
    assert broken_mapped_response.json() == {
        "code": "validation_failed",
        "message": "Validation failed",
        "detail": "BrokenValidationError",
    }

    broken_unmapped_response = client.get("/broken-unmapped")
    assert broken_unmapped_response.status_code == 500
    assert broken_unmapped_response.json() == {
        "code": "workflow_error",
        "message": "Internal server error",
        "detail": "Internal server error",
    }
