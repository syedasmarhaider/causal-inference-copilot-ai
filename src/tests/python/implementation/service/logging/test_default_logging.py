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
