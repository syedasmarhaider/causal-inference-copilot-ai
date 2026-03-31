from __future__ import annotations

from typing import Any, TypeAlias
from uuid import UUID

JSONValue: TypeAlias = Any
JSONDict: TypeAlias = dict[str, JSONValue]


def uuid_from_any(v: Any) -> UUID | None:
    if v is None:
        return None
    if isinstance(v, UUID):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        return UUID(s)
    raise ValueError(f"Invalid UUID value: {v!r}")


def uuid_to_str(v: UUID | None) -> str | None:
    return str(v) if v is not None else None

def safe_err(e: Exception, limit: int = 500) -> str:
    s = str(e).strip()
    return s[:limit] if s else e.__class__.__name__


BOOL_TRUE = {"true", "1", "yes"}
BOOL_FALSE = {"false", "0", "no"}
# ---------- core ----------
ScalarValue = str | int | float | bool  # or Strict* variants if you want strictness
