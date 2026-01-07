from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

JSONDict = Dict[str, Any]


class McpError(RuntimeError):
    """Base exception for MCP client failures."""


@dataclass(frozen=True)
class McpToolResult:
    """
    Normalized tool output.

    - `data`: best-effort extracted JSON payload (dict/list/primitive) suitable for app logic.
    - `raw_response`: full JSON-RPC response dict for debugging / logging.
    """
    data: Any
    raw_response: JSONDict


@runtime_checkable
class McpClient(Protocol):
    """
    Domain interface for MCP clients.

    Implementations may be HTTP, stdio, in-proc, etc.
    """

    def ensure_initialized(self) -> None:
        """Perform MCP handshake if needed. Must be idempotent."""
        ...

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> McpToolResult:
        """Call a tool and return normalized output."""
        ...

    def close(self) -> None:
        """Release any underlying resources (HTTP client, streams, etc.)."""
        ...
