
from __future__ import annotations
from datetime import datetime, timezone

import inspect
from typing import Any, Dict, Mapping, Optional, Set, Type


# a hack
_EMPTY = inspect._empty # pyright: ignore[reportPrivateUsage]

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def jsonish_default(v: Any) -> Any:
    """Return JSON-friendly default if possible; otherwise repr()."""
    if v is _EMPTY:
        return None
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)) and all(
        x is None or isinstance(x, (bool, int, float, str)) for x in v # pyright: ignore[reportUnknownVariableType]
    ):
        return list(v)  # type: ignore[return-value]
    if isinstance(v, dict):
        # keep dict only if it is JSON-ish
        ok = True
        out: Dict[str, Any] = {}
        for k, vv in v.items(): # pyright: ignore[reportUnknownVariableType]
            if not isinstance(k, str):
                ok = False
                break
            if vv is None or isinstance(vv, (bool, int, float, str)):
                out[k] = vv
            else:
                ok = False
                break
        return out if ok else repr(v) # pyright: ignore[reportUnknownArgumentType]
    return repr(v) # pyright: ignore[reportUnknownArgumentType]


def _param_meta(p: inspect.Parameter) -> Dict[str, Any]:
    ann = None if p.annotation is _EMPTY else repr(p.annotation)
    return {
        "required": (p.default is _EMPTY),
        "default": jsonish_default(p.default),
        "kind": str(p.kind),        # KEYWORD_ONLY, POSITIONAL_OR_KEYWORD, etc.
        "annotation": ann,
    }


def build_init_fit_param_maps(
    cls: Type[Any],
    *,
    fit_exclude_names: Optional[Set[str]] = None,
    fit_include_names: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Returns:
      {
        "init": { param_name: {required, default, kind, annotation}, ... },
        "fit":  { param_name: {required, default, kind, annotation}, ... },
      }

    Notes:
      - init excludes "self"
      - fit excludes data args by default ("self","Y","T","X","W","Z")
      - You can whitelist fit params via fit_include_names if you want (e.g. {"cache_values","inference"}).
    """
    if fit_exclude_names is None:
        fit_exclude_names = {"self", "Y", "T", "X", "W", "Z"}

    # ---- __init__ map ----
    init_sig = inspect.signature(cls.__init__)
    init_map: Dict[str, Dict[str, Any]] = {}
    for name, p in init_sig.parameters.items():
        if name == "self":
            continue
        init_map[name] = _param_meta(p)

    # ---- fit map ----
    fit_map: Dict[str, Dict[str, Any]] = {}
    if hasattr(cls, "fit"):
        fit_sig = inspect.signature(cls.fit)  # type: ignore[attr-defined]
        for name, p in fit_sig.parameters.items():
            if name in fit_exclude_names:
                continue
            if fit_include_names is not None and name not in fit_include_names:
                continue
            fit_map[name] = _param_meta(p)

    return {"init": init_map, "fit": fit_map}


def validate_flat_options(
    options: Mapping[str, Any],
    *,
    init_map: Mapping[str, Any],
    fit_map: Mapping[str, Any],
) -> None:
    """
    Validate that flat options keys are known to either __init__ or fit.
    Raise ValueError on unknown keys.
    """
    allowed = set(init_map.keys()) | set(fit_map.keys())
    unknown = [k for k in options.keys() if k not in allowed]
    if unknown:
        raise ValueError(f"Unknown option keys: {unknown}")


def split_flat_options(
    options: Mapping[str, Any],
    *,
    init_map: Mapping[str, Any],
    fit_map: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split a flat options dict into:
      - init_kwargs (for cls(**init_kwargs))
      - fit_kwargs  (for est.fit(..., **fit_kwargs))
    """
    validate_flat_options(options, init_map=init_map, fit_map=fit_map)
    init_kwargs: Dict[str, Any] = {}
    fit_kwargs: Dict[str, Any] = {}

    for k, v in options.items():
        if k in init_map:
            init_kwargs[k] = v
        else:
            fit_kwargs[k] = v

    return init_kwargs, fit_kwargs