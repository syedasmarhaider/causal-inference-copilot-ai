from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from python.implementation.service.logging.default_logging import (
    reset_log_context,
    set_log_context,
)

_TRACEPARENT_PATTERN = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
_GCP_TRACE_PATTERN = re.compile(r"^([0-9a-f]{32})(?:/([0-9]+|[0-9a-f]{16}))?(?:;o=([01]))?$")
_AWS_ROOT_PATTERN = re.compile(r"^1-([0-9a-f]{8})-([0-9a-f]{24})$")
_HEX16_PATTERN = re.compile(r"^[0-9a-f]{16}$")


@dataclass(frozen=True)
class TraceContext:
    trace_id: str | None
    span_id: str | None = None
    trace_sampled: bool | None = None
    source: str | None = None


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = _normalize_request_id(request.headers.get("x-request-id"))
        trace_context = extract_trace_context_from_headers(request.headers)

        trace_id = trace_context.trace_id or uuid4().hex

        request.state.request_id = request_id
        request.state.trace_id = trace_id
        request.state.span_id = trace_context.span_id
        request.state.trace_sampled = trace_context.trace_sampled

        tokens = set_log_context(
            request_id=request_id,
            trace_id=trace_id,
            span_id=trace_context.span_id,
            trace_sampled=trace_context.trace_sampled,
        )
        try:
            response = await call_next(request)
        finally:
            reset_log_context(tokens)

        response.headers.setdefault("X-Request-ID", request_id)
        response.headers.setdefault("X-Trace-ID", trace_id)
        if trace_context.span_id:
            response.headers.setdefault("X-Span-ID", trace_context.span_id)
        return response


def extract_trace_context_from_headers(headers: Mapping[str, str]) -> TraceContext:
    traceparent = _parse_traceparent(headers.get("traceparent"))
    if traceparent is not None:
        return traceparent

    gcp_trace = _parse_x_cloud_trace_context(headers.get("x-cloud-trace-context"))
    if gcp_trace is not None:
        return gcp_trace

    aws_trace = _parse_x_amzn_trace_id(headers.get("x-amzn-trace-id"))
    if aws_trace is not None:
        return aws_trace

    return TraceContext(trace_id=None)


def _normalize_request_id(raw_request_id: str | None) -> str:
    if raw_request_id is None:
        return uuid4().hex
    trimmed = raw_request_id.strip()
    return trimmed if trimmed else uuid4().hex


def _parse_traceparent(raw_header: str | None) -> TraceContext | None:
    if raw_header is None:
        return None

    header = raw_header.strip().lower()
    match = _TRACEPARENT_PATTERN.fullmatch(header)
    if match is None:
        return None

    trace_id, span_id, flags_hex = match.groups()
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return None

    flags = int(flags_hex, 16)
    sampled = bool(flags & 0x01)

    return TraceContext(
        trace_id=trace_id,
        span_id=span_id,
        trace_sampled=sampled,
        source="traceparent",
    )


def _parse_x_cloud_trace_context(raw_header: str | None) -> TraceContext | None:
    if raw_header is None:
        return None

    header = raw_header.strip().lower()
    match = _GCP_TRACE_PATTERN.fullmatch(header)
    if match is None:
        return None

    trace_id, raw_span_id, raw_sampled = match.groups()
    if trace_id == "0" * 32:
        return None

    span_id = _normalize_gcp_span_id(raw_span_id)
    sampled = _parse_sampled_flag(raw_sampled)

    return TraceContext(
        trace_id=trace_id,
        span_id=span_id,
        trace_sampled=sampled,
        source="x-cloud-trace-context",
    )


def _parse_x_amzn_trace_id(raw_header: str | None) -> TraceContext | None:
    if raw_header is None:
        return None

    pairs: dict[str, str] = {}
    for part in raw_header.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip().lower()
        if key and value:
            pairs[key] = value

    raw_root = pairs.get("root")
    if raw_root is None:
        return None

    root_match = _AWS_ROOT_PATTERN.fullmatch(raw_root)
    if root_match is None:
        return None

    trace_id = "".join(root_match.groups())
    if trace_id == "0" * 32:
        return None

    raw_parent = pairs.get("parent")
    span_id = raw_parent if raw_parent and _HEX16_PATTERN.fullmatch(raw_parent) else None
    sampled = _parse_sampled_flag(pairs.get("sampled"))

    return TraceContext(
        trace_id=trace_id,
        span_id=span_id,
        trace_sampled=sampled,
        source="x-amzn-trace-id",
    )


def _normalize_gcp_span_id(raw_span_id: str | None) -> str | None:
    if not raw_span_id:
        return None

    if _HEX16_PATTERN.fullmatch(raw_span_id):
        return raw_span_id

    try:
        span_value = int(raw_span_id, 10)
    except ValueError:
        return None

    if span_value < 0 or span_value > (2**64 - 1):
        return None

    return f"{span_value:016x}"


def _parse_sampled_flag(raw_sampled: str | None) -> bool | None:
    if raw_sampled is None:
        return None

    val = raw_sampled.strip().lower()
    if val in {"1", "true"}:
        return True
    if val in {"0", "false"}:
        return False
    return None


__all__ = [
    "RequestContextMiddleware",
    "TraceContext",
    "extract_trace_context_from_headers",
]
