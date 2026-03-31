from __future__ import annotations

import logging
import re

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from python.adapters.api.request_context_middleware import (
    RequestContextMiddleware,
    extract_trace_context_from_headers,
)
from python.implementation.service.logging.default_logging import get_logger


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _isolated_logger(name: str) -> tuple[logging.Logger, _CaptureHandler]:
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = _CaptureHandler()
    logger.addHandler(handler)
    return logger, handler


def test_extract_trace_context_prefers_traceparent_over_platform_headers() -> None:
    traceparent_trace_id = "a" * 32
    traceparent_span_id = "b" * 16
    traceparent = f"00-{traceparent_trace_id}-{traceparent_span_id}-01"

    headers = {
        "traceparent": traceparent,
        "x-cloud-trace-context": f"{'c' * 32}/123;o=1",
        "x-amzn-trace-id": "Root=1-5f84c7a1-4d3c2b1a0f9e8d7c6b5a4f3e;Parent=53995c3f42cd8ad8;Sampled=1",
    }

    context = extract_trace_context_from_headers(headers)

    assert context.trace_id == traceparent_trace_id
    assert context.span_id == traceparent_span_id
    assert context.trace_sampled is True
    assert context.source == "traceparent"


def test_extract_trace_context_parses_gcp_header() -> None:
    trace_id = "c" * 32
    context = extract_trace_context_from_headers(
        {
            "x-cloud-trace-context": f"{trace_id}/123456;o=1",
        }
    )

    assert context.trace_id == trace_id
    assert context.span_id == "000000000001e240"
    assert context.trace_sampled is True
    assert context.source == "x-cloud-trace-context"


def test_extract_trace_context_parses_aws_header() -> None:
    header = "Root=1-5f84c7a1-4d3c2b1a0f9e8d7c6b5a4f3e;Parent=53995c3f42cd8ad8;Sampled=0"
    context = extract_trace_context_from_headers({"x-amzn-trace-id": header})

    assert context.trace_id == "5f84c7a14d3c2b1a0f9e8d7c6b5a4f3e"
    assert context.span_id == "53995c3f42cd8ad8"
    assert context.trace_sampled is False
    assert context.source == "x-amzn-trace-id"


def test_request_context_middleware_populates_headers_state_and_logs() -> None:
    logger_name = "tests.request_context_middleware"
    _, capture = _isolated_logger(logger_name)
    app_logger = get_logger(logger_name, component="TestEndpoint", log_type="test")

    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ping")
    async def ping(request: Request) -> dict[str, str | None]:
        app_logger.info("inside endpoint")
        return {
            "request_id": request.state.request_id,
            "trace_id": request.state.trace_id,
            "span_id": request.state.span_id,
        }

    client = TestClient(app)
    trace_id = "1" * 32
    span_id = "2" * 16
    response = client.get(
        "/ping",
        headers={
            "X-Request-ID": "req-123",
            "traceparent": f"00-{trace_id}-{span_id}-01",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "req-123"
    assert payload["trace_id"] == trace_id
    assert payload["span_id"] == span_id

    assert response.headers["x-request-id"] == "req-123"
    assert response.headers["x-trace-id"] == trace_id
    assert response.headers["x-span-id"] == span_id

    assert len(capture.records) == 1
    assert capture.records[0].context == {
        "request_id": "req-123",
        "trace_id": trace_id,
        "span_id": span_id,
        "trace_sampled": True,
    }


def test_request_context_middleware_generates_ids_when_missing() -> None:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ping")
    async def ping(request: Request) -> dict[str, str | None]:
        return {
            "request_id": request.state.request_id,
            "trace_id": request.state.trace_id,
            "span_id": request.state.span_id,
        }

    client = TestClient(app)
    response = client.get("/ping")

    assert response.status_code == 200
    payload = response.json()

    assert re.fullmatch(r"[0-9a-f]{32}", payload["request_id"] or "")
    assert re.fullmatch(r"[0-9a-f]{32}", payload["trace_id"] or "")
    assert payload["span_id"] is None

    assert response.headers["x-request-id"] == payload["request_id"]
    assert response.headers["x-trace-id"] == payload["trace_id"]
