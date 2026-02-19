from __future__ import annotations

import math
from typing import Any, Dict, Optional, TypeAlias
from uuid import UUID

DEFAULT_MODEL_GEMNI: str = "gemini-2.5-flash"

JSONValue: TypeAlias = Any
JSONDict: TypeAlias = Dict[str, JSONValue]

def json_sanitize(x: Any) -> Any:
    """
    Convert arbitrary nested structures into JSON-serializable primitives.
    - float NaN/Inf -> None
    - numpy scalars -> .item()
    - unknown objects -> str(x)
    """
    if x is None or isinstance(x, (str, bool, int)):
        return x

    if isinstance(x, float):
        return x if math.isfinite(x) else None

    # numpy / pandas scalars
    if hasattr(x, "item"):
        try:
            return _json_sanitize(x.item())  # type: ignore[attr-defined]
        except Exception:
            pass

    if isinstance(x, dict):
        out: Dict[str, Any] = {}
        for k, v in x.items(): # pyright: ignore[reportUnknownVariableType]
            k_str: str = str(k) # pyright: ignore[reportUnknownArgumentType]
            out[k_str] = json_sanitize(v)
        return out

    if isinstance(x, (list, tuple)):
        return [json_sanitize(v) for v in x] # pyright: ignore[reportUnknownVariableType]

    return str(x)


def uuid_from_any(v: Any) -> Optional[UUID]:
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


def uuid_to_str(v: Optional[UUID]) -> Optional[str]:
    return str(v) if v is not None else None
