from __future__ import annotations

import io
import json
import logging
from typing import Any

import python.implementation.service.logging.default_logging as logging_module
from python.implementation.service.logging.default_logging import DefaultAppLogger, JSONLogFormatter


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _isolated_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger


def test_default_app_logger_emits_tags_and_fields() -> None:
    logger = _isolated_logger("tests.logging.capture")
    handler = _CaptureHandler()
    logger.addHandler(handler)

    app_logger = DefaultAppLogger(
        logger,
        tags={"component": "WorkflowApp", "type": "workflow_service"},
    )
    app_logger.info("workflow started", user_id="u1", step=2)

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.getMessage() == "workflow started"
    assert record.tags == {"component": "WorkflowApp", "type": "workflow_service"}
    assert record.fields == {"user_id": "u1", "step": 2}


def test_default_app_logger_bind_merges_tags() -> None:
    logger = _isolated_logger("tests.logging.bind")
    handler = _CaptureHandler()
    logger.addHandler(handler)

    parent = DefaultAppLogger(logger, tags={"component": "WorkflowApp", "type": "workflow_service"})
    child = parent.bind(state_name="LOAD_DATASET")

    child.warning("state entered")

    record = handler.records[0]
    assert record.tags == {
        "component": "WorkflowApp",
        "type": "workflow_service",
        "state_name": "LOAD_DATASET",
    }


def test_json_log_formatter_includes_tags_fields_and_service() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONLogFormatter(service_name="svc-name"))

    logger = _isolated_logger("tests.logging.formatter")
    logger.addHandler(handler)

    app_logger = DefaultAppLogger(logger, tags={"component": "Router", "type": "workflow_router"})
    app_logger.error("decision failed", reason="bad-choice", retry=False)

    payload = json.loads(stream.getvalue().strip())
    assert payload["service"] == "svc-name"
    assert payload["logger"] == "tests.logging.formatter"
    assert payload["message"] == "decision failed"
    assert payload["tags"] == {"component": "Router", "type": "workflow_router"}
    assert payload["fields"] == {"reason": "bad-choice", "retry": False}


def test_default_app_logger_includes_request_context_fields() -> None:
    logger = _isolated_logger("tests.logging.request_context")
    handler = _CaptureHandler()
    logger.addHandler(handler)

    tokens = logging_module.set_log_context(
        request_id="req-1",
        trace_id="a" * 32,
        span_id="b" * 16,
        trace_sampled=True,
    )
    try:
        app_logger = DefaultAppLogger(logger, tags={"component": "WorkflowApp"})
        app_logger.info("contextual log")
    finally:
        logging_module.reset_log_context(tokens)

    record = handler.records[0]
    assert record.context == {
        "request_id": "req-1",
        "trace_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "span_id": "bbbbbbbbbbbbbbbb",
        "trace_sampled": True,
    }


def test_json_log_formatter_renders_request_context_as_top_level_fields() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONLogFormatter(service_name="svc-name"))

    logger = _isolated_logger("tests.logging.context_formatter")
    logger.addHandler(handler)

    tokens = logging_module.set_log_context(
        request_id="req-2",
        trace_id="c" * 32,
        span_id=None,
        trace_sampled=False,
    )
    try:
        app_logger = DefaultAppLogger(logger, tags={"component": "WorkflowApp"})
        app_logger.info("context in json")
    finally:
        logging_module.reset_log_context(tokens)

    payload = json.loads(stream.getvalue().strip())
    assert payload["request_id"] == "req-2"
    assert payload["trace_id"] == "cccccccccccccccccccccccccccccccc"
    assert payload["trace_sampled"] is False
    assert "span_id" not in payload


def test_reset_log_context_restores_previous_values() -> None:
    outer_tokens = logging_module.set_log_context(
        request_id="outer-req",
        trace_id="d" * 32,
        span_id=None,
        trace_sampled=None,
    )
    nested_tokens = logging_module.set_log_context(
        request_id="inner-req",
        trace_id="e" * 32,
        span_id="f" * 16,
        trace_sampled=True,
    )
    logging_module.reset_log_context(nested_tokens)

    current = logging_module.get_log_context()
    assert current["request_id"] == "outer-req"
    assert current["trace_id"] == "dddddddddddddddddddddddddddddddd"
    assert "span_id" not in current

    logging_module.reset_log_context(outer_tokens)
    assert logging_module.get_log_context() == {}


def test_get_logger_uses_custom_factory() -> None:
    captured: dict[str, Any] = {}

    class _CustomLogger:
        def bind(self, **tags: Any) -> _CustomLogger:
            return self

        def debug(self, message: str, **fields: Any) -> None:
            captured["debug"] = (message, fields)

        def info(self, message: str, **fields: Any) -> None:
            captured["info"] = (message, fields)

        def warning(self, message: str, **fields: Any) -> None:
            captured["warning"] = (message, fields)

        def error(self, message: str, **fields: Any) -> None:
            captured["error"] = (message, fields)

        def exception(self, message: str, **fields: Any) -> None:
            captured["exception"] = (message, fields)

    def _factory(logger_name: str, tags: dict[str, Any]) -> _CustomLogger:
        captured["logger_name"] = logger_name
        captured["tags"] = tags
        return _CustomLogger()

    logging_module.set_app_logger_factory(_factory)
    try:
        logger = logging_module.get_logger(
            "tests.logging.factory",
            component="FactoryComponent",
            log_type="factory_type",
        )
        logger.info("hello", request_id="r-1")
    finally:
        logging_module.set_app_logger_factory(None)

    assert captured["logger_name"] == "tests.logging.factory"
    assert captured["tags"] == {
        "component": "FactoryComponent",
        "type": "factory_type",
    }
    assert captured["info"] == ("hello", {"request_id": "r-1"})
