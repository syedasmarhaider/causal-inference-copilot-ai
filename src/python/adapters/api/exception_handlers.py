from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, TypeAlias, cast

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
UNKNOWN_ERROR_CODE = "unknown code"


@dataclass(frozen=True)
class ErrorSpec:
    status_code: int
    message: str


_INTERNAL_SERVER_ERROR = ErrorSpec(
    status_code=500,
    message="Internal server error",
)

_WORKFLOW_ERROR_SPECS: tuple[tuple[MappedErrorType, ErrorSpec], ...] = (
    (ConversationNotFoundError, ErrorSpec(status_code=404, message="Conversation not found")),
    (StateNotFoundError, ErrorSpec(status_code=404, message="State not found")),
    (ArtifactNotFoundError, ErrorSpec(status_code=404, message="Artifact not found")),
    (AuthenticationError, ErrorSpec(status_code=401, message="Authentication failed")),
    (ValidationError, ErrorSpec(status_code=422, message="Validation failed")),
    (DataUploadError, ErrorSpec(status_code=400, message="Data upload failed")),
    (InvalidStateError, ErrorSpec(status_code=409, message="Invalid workflow state")),
    (FileNotFoundError, ErrorSpec(status_code=404, message="Resource not found")),
    (FileExistsError, ErrorSpec(status_code=409, message="Resource already exists")),
)


def _safe_str(value: object) -> str:
    try:
        text = str(value)
    except Exception:
        return type(value).__name__

    normalized = text.strip()
    return normalized or type(value).__name__


def _resolve_error_code(exc: Exception) -> str:
    try:
        value = getattr(exc, "code", None)
    except Exception:
        return UNKNOWN_ERROR_CODE

    if isinstance(value, str):
        return value
    return UNKNOWN_ERROR_CODE


def _get_error_spec(exc: Exception) -> ErrorSpec | None:
    for error_type, spec in _WORKFLOW_ERROR_SPECS:
        if isinstance(exc, error_type):
            return spec
    return None


def _is_server_error(status_code: int) -> bool:
    return status_code >= 500


def _error_response(*, spec: ErrorSpec, code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=spec.status_code,
        content={
            "code": code,
            "message": spec.message,
            "detail": detail,
        },
    )


def _internal_server_error_response(exc: Exception) -> JSONResponse:
    return _error_response(
        spec=_INTERNAL_SERVER_ERROR,
        code=_resolve_error_code(exc),
        detail="Internal server error",
    )


def map_workflow_error_to_http_exception(exc: Exception) -> HTTPException | None:
    spec = _get_error_spec(exc)
    if spec is None:
        return None

    detail = "Internal server error" if _is_server_error(spec.status_code) else _safe_str(exc)
    return HTTPException(status_code=spec.status_code, detail=detail)


def _build_http_exception_response(exc: HTTPException) -> JSONResponse:
    if _is_server_error(exc.status_code):
        log.exception(
            "http exception mapped to server error",
            error_type=type(exc).__name__,
            error_detail=_safe_str(exc),
            status_code=exc.status_code,
        )
        return _internal_server_error_response(exc)

    phrase = (
        HTTPStatus(exc.status_code).phrase
        if exc.status_code in HTTPStatus._value2member_map_
        else "Request failed"
    )
    spec = ErrorSpec(status_code=exc.status_code, message=phrase)

    detail_payload = exc.detail
    if isinstance(detail_payload, dict):
        d = cast(dict[str, Any], detail_payload)
        detail = _safe_str(d.get("detail", detail_payload))
    else:
        detail = _safe_str(detail_payload)

    return _error_response(
        spec=spec,
        code=_resolve_error_code(exc),
        detail=detail,
    )


def _build_mapped_exception_response(exc: Exception, spec: ErrorSpec) -> JSONResponse:
    if _is_server_error(spec.status_code):
        log.exception(
            "mapped exception resolved to server error",
            error_type=type(exc).__name__,
            error_detail=_safe_str(exc),
            status_code=spec.status_code,
        )
        return _internal_server_error_response(exc)

    return _error_response(
        spec=spec,
        code=_resolve_error_code(exc),
        detail=_safe_str(exc),
    )


def _build_unhandled_exception_response(exc: Exception) -> JSONResponse:
    log.exception(
        "unhandled exception in API adapter",
        error_type=type(exc).__name__,
        error_detail=_safe_str(exc),
    )
    return _internal_server_error_response(exc)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return _build_http_exception_response(exc)

    @app.exception_handler(WorkflowError)
    async def _workflow_error_handler(_: Request, exc: WorkflowError) -> JSONResponse:
        spec = _get_error_spec(exc)
        if spec is None:
            log.exception(
                "unmapped workflow error so server error",
                error_type=type(exc).__name__,
                error_detail=_safe_str(exc),
            )
            return _internal_server_error_response(exc)
        return _build_mapped_exception_response(exc, spec)

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        spec = _get_error_spec(exc)
        if spec is not None:
            return _build_mapped_exception_response(exc, spec)
        return _build_unhandled_exception_response(exc)


__all__ = [
    "map_workflow_error_to_http_exception",
    "register_exception_handlers",
]
