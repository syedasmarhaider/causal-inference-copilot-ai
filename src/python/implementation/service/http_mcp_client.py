from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx

from python.domain.service.mcp_client import McpClient, McpError, McpToolResult, JSONDict


def _cast_json_dict(x: Any) -> JSONDict:
    if not isinstance(x, dict):
        raise TypeError(f"Expected dict JSON, got {type(x)}")
    return x


def _try_parse_json_text(s: str) -> Any:
    s2 = (s or "").strip()
    if not s2:
        return s
    try:
        return json.loads(s2)
    except Exception:
        return s


def _unwrap_tool_result(result_obj: Any) -> Any:
    """
    FastMCP (and many MCP servers) return tool outputs in a content envelope like:
      { "content": [ { "type": "json", "json": {...} } ] }
    or:
      { "content": [ { "type": "text", "text": "{...}" } ] }

    This function extracts a useful payload for app logic.
    If it cannot unwrap, it returns the input as-is.
    """
    if not isinstance(result_obj, dict):
        return result_obj

    content = result_obj.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            t = str(first.get("type") or "").lower()
            if t == "json" and "json" in first:
                return first.get("json")
            if t == "text" and "text" in first:
                return _try_parse_json_text(str(first.get("text")))
    # Some servers might directly return the JSON in `result` without envelope.
    return result_obj


@dataclass
class HttpMcpClient(McpClient):
    """
    MCP Streamable HTTP client (JSON-RPC over POST).

    - Handles initialize + notifications/initialized handshake.
    - Persists session via `mcp-session-id` header.
    - On session invalidation (common 404 case), re-initializes once and retries.
    """

    endpoint: str
    protocol_version: str = "2025-06-18"
    client_name: str = "causal-copilot"
    client_version: str = "0.1.0"
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        self._endpoint = self.endpoint.rstrip("/")
        self._session_id: Optional[str] = None
        self._next_id: int = 1
        self._http = httpx.Client(timeout=self.timeout_s)
        self._initialized: bool = False

    def close(self) -> None:
        try:
            self._http.close()
        finally:
            self._initialized = False
            self._session_id = None

    # -----------------------------
    # internals
    # -----------------------------
    def _headers(self) -> Dict[str, str]:
        h = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            "mcp-protocol-version": self.protocol_version,
        }
        if self._session_id:
            h["mcp-session-id"] = self._session_id
        return h

    def _new_id(self) -> int:
        v = self._next_id
        self._next_id += 1
        return v

    def _post(self, payload: JSONDict, *, expect_response: bool) -> Tuple[Optional[JSONDict], httpx.Response]:
        try:
            r = self._http.post(self._endpoint, headers=self._headers(), content=json.dumps(payload))
        except httpx.TimeoutException as e:
            raise McpError(f"MCP request timed out: {e}") from e
        except httpx.RequestError as e:
            raise McpError(f"MCP request error: {e}") from e

        if not expect_response:
            return None, r

        ctype = (r.headers.get("content-type") or "").lower()

        if "application/json" in ctype:
            try:
                return _cast_json_dict(r.json()), r
            except Exception as e:
                raise McpError(f"Invalid JSON response from MCP server: {e}") from e

        # If your server returns SSE at some point, add parsing here.
        raise McpError(f"Unsupported MCP response content-type: {ctype}")

    def _reset_session(self) -> None:
        self._initialized = False
        self._session_id = None

    # -----------------------------
    # MCP interface
    # -----------------------------
    def ensure_initialized(self) -> None:
        if self._initialized:
            return

        init_id = self._new_id()
        init_req: JSONDict = {
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": self.client_name, "version": self.client_version},
            },
        }

        obj, resp = self._post(init_req, expect_response=True)
        if obj is None:
            raise McpError("Missing initialize response.")
        if "error" in obj:
            raise McpError(f"MCP initialize error: {obj['error']}")

        sid = resp.headers.get("mcp-session-id")
        if sid:
            self._session_id = sid

        # notifications/initialized (no response expected)
        notif: JSONDict = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        _none, _r2 = self._post(notif, expect_response=False)

        self._initialized = True

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> McpToolResult:
        """
        Calls tools/call and returns a normalized `McpToolResult.data`:

        - If server wraps result in content envelope -> returns unwrapped JSON/text.
        - Else returns raw result as-is.

        Also retries once if the server likely rejected the session (common with streamable-http servers).
        """
        # Ensure handshake
        self.ensure_initialized()

        def _do_call() -> McpToolResult:
            req_id = self._new_id()
            req: JSONDict = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments or {}},
            }

            obj, resp = self._post(req, expect_response=True)
            if obj is None:
                raise McpError("Missing tools/call response.")
            if "error" in obj:
                raise McpError(f"MCP tools/call error: {obj['error']}")
            if "result" not in obj:
                raise McpError(f"MCP tools/call missing result: {obj}")

            raw_result = obj["result"]
            data = _unwrap_tool_result(raw_result)
            return McpToolResult(data=data, raw_response=obj)

        try:
            return _do_call()
        except McpError as e:
            # Heuristic retry: if session got invalidated server-side,
            # clear session and re-init once.
            # (Many servers respond 404 without JSON; if that happens, _post will raise McpError too.)
            self._reset_session()
            self.ensure_initialized()
            return _do_call()
