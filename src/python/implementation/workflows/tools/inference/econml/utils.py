# from __future__ import annotations

# from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast
# from uuid import UUID

# import numpy as np
# import pandas as pd

# from python.workflows.tools.inference.models.causal_command import Issue


# # -------------------------
# # Issue helpers
# # -------------------------
# def need(path: str, msg: str, *, fix: Optional[str] = None, required: Optional[List[str]] = None) -> Issue:
#     return Issue(code="MISSING", message=msg, path=path, fix_hint=fix, required=required)


# def invalid(path: str, msg: str, *, fix: Optional[str] = None) -> Issue:
#     return Issue(code="INVALID", message=msg, path=path, fix_hint=fix)


# def unsupported(path: str, msg: str, *, fix: Optional[str] = None) -> Issue:
#     return Issue(code="UNSUPPORTED", message=msg, path=path, fix_hint=fix)


# # -------------------------
# # kwargs validation
# # -------------------------
# def validate_kwargs(*, provided: Mapping[str, Any], allowed: Sequence[str], path: str) -> Tuple[Dict[str, Any], List[Issue]]:
#     allowed_set = set(allowed)
#     out: Dict[str, Any] = {}
#     issues: List[Issue] = []

#     for k, v in provided.items():
#         if k not in allowed_set:
#             issues.append(
#                 invalid(
#                     f"{path}.{k}",
#                     f"Unknown option '{k}'.",
#                     fix=f"Allowed keys: {sorted(allowed_set)}",
#                 )
#             )
#             continue
#         out[k] = v

#     return out, issues


# # -------------------------
# # dataframe column resolvers
# # -------------------------
# def resolve_col(df: pd.DataFrame, col: str, *, role: str) -> Any:
#     if col not in df.columns:
#         raise KeyError(f"Missing column '{col}' for role={role}")
#     return df[col]


# def resolve_cols(df: pd.DataFrame, cols: Sequence[str], *, role: str) -> pd.DataFrame:
#     missing = [c for c in cols if c not in df.columns]
#     if missing:
#         raise KeyError(f"Missing columns {missing} for role={role}")
#     return df[list(cols)]


# def covariates_or_none(df: pd.DataFrame, cols: Sequence[str]) -> Optional[pd.DataFrame]:
#     if not cols:
#         return None
#     return resolve_cols(df, cols, role="covariates")


# # -------------------------
# # registry-based estimator builder
# # -------------------------
# def build_from_registry(
#     registry: Mapping[str, type],
#     spec: Any,
#     *,
#     role_path: str,
#     allow_auto: bool,
#     default: Any,
#     allow_names: Optional[set[str]] = None,
# ) -> Tuple[Any, Optional[Issue]]:
#     """
#     spec allowed:
#       - None -> default
#       - "auto" -> "auto" (iff allow_auto)
#       - "registry.key" -> registry.key()
#       - {"name": "registry.key", "kwargs": {...}} -> registry.key(**kwargs)
#       - already-constructed estimator instance -> returned as-is
#     allow_names: optional restriction subset (adapter-specific constraint, but generic mechanism).
#     """
#     if spec is None:
#         spec = default

#     if isinstance(spec, str):
#         if spec == "auto":
#             if not allow_auto:
#                 return None, unsupported(role_path, "'auto' is not allowed here.")
#             return "auto", None

#         if allow_names is not None and spec not in allow_names:
#             return None, unsupported(
#                 role_path,
#                 f"Estimator '{spec}' is not allowed for this role.",
#                 fix=f"Allowed: {sorted(allow_names)}",
#             )

#         ctor = registry.get(spec)
#         if ctor is None:
#             return None, unsupported(
#                 role_path,
#                 f"Estimator '{spec}' is not supported by this adapter.",
#                 fix=f"Supported: {sorted(registry.keys())}",
#             )
#         try:
#             return ctor(), None
#         except Exception as e:
#             return None, invalid(role_path, f"Failed to construct '{spec}': {e}")

#     if isinstance(spec, Mapping):
#         name = cast(Optional[str], spec.get("name")) # pyright: ignore[reportUnknownMemberType]
#         kwargs = cast(Dict[str, Any], spec.get("kwargs") or {}) # pyright: ignore[reportUnknownMemberType]

#         if not name:
#             return None, invalid(role_path, "Missing 'name' in estimator spec.", fix="Use {'name': ..., 'kwargs': {...}}")

#         if allow_names is not None and name not in allow_names:
#             return None, unsupported(
#                 role_path,
#                 f"Estimator '{name}' is not allowed for this role.",
#                 fix=f"Allowed: {sorted(allow_names)}",
#             )

#         ctor = registry.get(name)
#         if ctor is None:
#             return None, unsupported(
#                 role_path,
#                 f"Estimator '{name}' is not supported by this adapter.",
#                 fix=f"Supported: {sorted(registry.keys())}",
#             )
#         try:
#             return ctor(**kwargs), None
#         except Exception as e:
#             return None, invalid(role_path, f"Failed to construct '{name}': {e}")

#     # already-constructed estimator instance (python-only)
#     return spec, None


# # -------------------------
# # json-safe serialization for outputs/meta
# # -------------------------
# def json_safe(x: Any) -> Any:
#     if x is None:
#         return None
#     if isinstance(x, (str, int, float, bool)):
#         return x
#     if isinstance(x, UUID):
#         return str(x)

#     if isinstance(x, (np.integer, np.floating)):
#         return x.item()
#     if isinstance(x, np.ndarray):
#         return x.tolist()

#     if isinstance(x, pd.DataFrame):
#         return {"__type__": "DataFrame", "shape": [int(x.shape[0]), int(x.shape[1])], "columns": [str(c) for c in x.columns]}
#     if isinstance(x, pd.Series):
#         return {"__type__": "Series", "shape": [int(x.shape[0])], "name": str(x.name)}

#     if isinstance(x, Mapping):
#         return {str(k): json_safe(v) for k, v in x.items()} # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
#     if isinstance(x, (list, tuple)):
#         return [json_safe(v) for v in x] # pyright: ignore[reportUnknownVariableType]

#     return str(x)
