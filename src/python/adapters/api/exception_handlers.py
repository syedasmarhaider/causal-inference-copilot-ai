from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

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
from python.implementation.service.logging.default_logging import get_logger

log = get_logger(__name__)

MappedErrorType: TypeAlias = type[Exception]


@dataclass(frozen=True)
class ErrorMapping:
    status_code: int
    detail: str


_WORKFLOW_ERROR_MAPPING: tuple[tuple[MappedErrorType, ErrorMapping], ...] = (
    (ConversationNotFoundError, ErrorMapping(status_code=404, detail="conversation not found")),
    (StateNotFoundError, ErrorMapping(status_code=404, detail="state not found")),
    (ArtifactNotFoundError, ErrorMapping(status_code=404, detail="artifact not found")),
    (AuthenticationError, ErrorMapping(status_code=401, detail="authentication failed")),
    (ValidationError, ErrorMapping(status_code=422, detail="validation failed")),
    (DataUploadError, ErrorMapping(status_code=400, detail="data upload failed")),
    (InvalidStateError, ErrorMapping(status_code=409, detail="invalid workflow state")),
    (FileNotFoundError, ErrorMapping(status_code=404, detail="resource not found")),
    (FileExistsError, ErrorMapping(status_code=409, detail="resource already exists")),
)


def map_workflow_error_to_http_exception(exc: Exception) -> HTTPException | None:
    for error_type, mapped in _WORKFLOW_ERROR_MAPPING:
        if isinstance(exc, error_type):
            return HTTPException(status_code=mapped.status_code, detail=mapped.detail)
    return None


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(WorkflowError)
    async def _workflow_error_handler(_: Request, exc: WorkflowError) -> JSONResponse:
        http_exc = map_workflow_error_to_http_exception(exc)
        if http_exc is None:
            log.exception(
                "unmapped workflow error",
                error=exc,
                error_type=type(exc).__name__,
            )
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})

        if http_exc.status_code >= 500:
            log.exception(
                "workflow error mapped to server error",
                error=exc,
                error_type=type(exc).__name__,
                status_code=http_exc.status_code,
            )

        return JSONResponse(status_code=http_exc.status_code, content={"detail": http_exc.detail})

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        http_exc = map_workflow_error_to_http_exception(exc)
        if http_exc is not None:
            if http_exc.status_code >= 500:
                log.exception(
                    "mapped non-workflow exception to server error",
                    error=exc,
                    error_type=type(exc).__name__,
                    status_code=http_exc.status_code,
                )
            return JSONResponse(status_code=http_exc.status_code, content={"detail": http_exc.detail})

        log.exception(
            "unhandled exception in API adapter",
            error=exc,
            error_type=type(exc).__name__,
        )
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


__all__ = [
    "map_workflow_error_to_http_exception",
    "register_exception_handlers",
]
