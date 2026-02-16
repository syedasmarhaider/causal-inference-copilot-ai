from __future__ import annotations

import math
from typing import Any, Dict, TypeAlias

DEFAULT_MODEL_GEMNI: str = "gemini-2.5-flash"

JSONValue: TypeAlias = Any
JSONDict: TypeAlias = Dict[str, JSONValue]

# {
#   "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
#   "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"


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
