from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

_DEFAULT_SERVICE_NAME = "causal-inference-copilot-ai"
_LOG_LEVEL_ENV = "LOG_LEVEL"
_SERVICE_NAME_ENV = "LOG_SERVICE_NAME"
_logger_factory: "LoggerFactory | None" = None
_request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
_trace_id_ctx: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id_ctx: ContextVar[str | None] = ContextVar("span_id", default=None)
_trace_sampled_ctx: ContextVar[bool | None] = ContextVar("trace_sampled", default=None)


class AppLogger(Protocol):
    def bind(self, **tags: Any) -> "AppLogger": ...

    def debug(self, message: str, **fields: Any) -> None: ...

    def info(self, message: str, **fields: Any) -> None: ...

    def warning(self, message: str, **fields: Any) -> None: ...

    def error(self, message: str, **fields: Any) -> None: ...

    def exception(self, message: str, **fields: Any) -> None: ...


LoggerFactory = Callable[[str, dict[str, Any]], AppLogger]


@dataclass(frozen=True)
class LogContextTokens:
    request_id: Token[str | None]
    trace_id: Token[str | None]
    span_id: Token[str | None]
    trace_sampled: Token[bool | None]


class JSONLogFormatter(logging.Formatter):
    def __init__(self, *, service_name: str) -> None:
        super().__init__()
        self._service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self._service_name,
            "logger": record.name,
            "message": record.getMessage(),
        }

        tags = getattr(record, "tags", None)
        if isinstance(tags, dict) and tags:
            payload["tags"] = _normalize_fields(tags)

        fields = getattr(record, "fields", None)
        if isinstance(fields, dict) and fields:
            payload["fields"] = _normalize_fields(fields)

        context = getattr(record, "context", None)
        if isinstance(context, dict) and context:
            payload.update(_normalize_fields(context))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


class DefaultAppLogger(AppLogger):
    def __init__(self, logger: logging.Logger, *, tags: dict[str, Any] | None = None) -> None:
        self._logger = logger
        self._tags = _normalize_fields(tags or {})

    def bind(self, **tags: Any) -> AppLogger:
        merged = dict(self._tags)
        merged.update(_normalize_fields(tags))
        return DefaultAppLogger(self._logger, tags=merged)

    def debug(self, message: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, message, fields)

    def info(self, message: str, **fields: Any) -> None:
        self._emit(logging.INFO, message, fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._emit(logging.WARNING, message, fields)

    def error(self, message: str, **fields: Any) -> None:
        self._emit(logging.ERROR, message, fields)

    def exception(self, message: str, **fields: Any) -> None:
        self._emit(logging.ERROR, message, fields, exc_info=True)

    def _emit(
        self,
        level: int,
        message: str,
        fields: dict[str, Any],
        *,
        exc_info: bool = False,
    ) -> None:
        kwargs: dict[str, Any] = {
            "stacklevel": 3,
        }

        extra: dict[str, Any] = {}
        if self._tags:
            extra["tags"] = dict(self._tags)

        normalized_fields = _normalize_fields(fields)
        if normalized_fields:
            extra["fields"] = normalized_fields

        context = get_log_context()
        if context:
            extra["context"] = context

        if extra:
            kwargs["extra"] = extra

        if exc_info:
            kwargs["exc_info"] = True

        self._logger.log(level, message, **kwargs)


def set_app_logger_factory(factory: LoggerFactory | None) -> None:
    global _logger_factory
    _logger_factory = factory


def set_log_context(
    *,
    request_id: str | None,
    trace_id: str | None,
    span_id: str | None = None,
    trace_sampled: bool | None = None,
) -> LogContextTokens:
    return LogContextTokens(
        request_id=_request_id_ctx.set(request_id),
        trace_id=_trace_id_ctx.set(trace_id),
        span_id=_span_id_ctx.set(span_id),
        trace_sampled=_trace_sampled_ctx.set(trace_sampled),
    )


def reset_log_context(tokens: LogContextTokens) -> None:
    _request_id_ctx.reset(tokens.request_id)
    _trace_id_ctx.reset(tokens.trace_id)
    _span_id_ctx.reset(tokens.span_id)
    _trace_sampled_ctx.reset(tokens.trace_sampled)


def get_log_context() -> dict[str, Any]:
    context: dict[str, Any] = {}
    request_id = _request_id_ctx.get()
    trace_id = _trace_id_ctx.get()
    span_id = _span_id_ctx.get()
    trace_sampled = _trace_sampled_ctx.get()

    if request_id:
        context["request_id"] = request_id
    if trace_id:
        context["trace_id"] = trace_id
    if span_id:
        context["span_id"] = span_id
    if trace_sampled is not None:
        context["trace_sampled"] = trace_sampled

    return context


def get_logger(
    logger_name: str,
    *,
    component: str | None = None,
    log_type: str | None = None,
) -> AppLogger:
    tags: dict[str, Any] = {}
    if component:
        tags["component"] = component
    if log_type:
        tags["type"] = log_type

    if _logger_factory is not None:
        return _logger_factory(logger_name, tags)

    return DefaultAppLogger(logging.getLogger(logger_name), tags=tags)


def get_app_logger(
    logger_name: str,
    *,
    component: str | None = None,
    log_type: str | None = None,
) -> AppLogger:
    return get_logger(logger_name, component=component, log_type=log_type)


def configure_default_logging(
    *,
    service_name: str | None = None,
    level: str | int | None = None,
    force: bool = False,
) -> None:
    resolved_level = _resolve_log_level(level)
    resolved_service_name = service_name or os.getenv(_SERVICE_NAME_ENV, _DEFAULT_SERVICE_NAME)
    formatter = JSONLogFormatter(service_name=resolved_service_name)

    root_logger = logging.getLogger()

    if root_logger.handlers and not force:
        root_logger.setLevel(resolved_level)
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
        return

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logging.basicConfig(level=resolved_level, handlers=[stream_handler], force=force)


def _resolve_log_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level

    if isinstance(level, str) and level.strip():
        normalized_level = level.strip().upper()
    else:
        normalized_level = os.getenv(_LOG_LEVEL_ENV, "INFO").strip().upper()

    level_mapping = logging.getLevelNamesMapping()
    return level_mapping.get(normalized_level, logging.INFO)


def _normalize_fields(values: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


__all__ = [
    "JSONLogFormatter",
    "AppLogger",
    "DefaultAppLogger",
    "LogContextTokens",
    "configure_default_logging",
    "get_log_context",
    "get_logger",
    "get_app_logger",
    "reset_log_context",
    "set_app_logger_factory",
    "set_log_context",
]
