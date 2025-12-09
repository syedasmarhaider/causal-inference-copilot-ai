from __future__ import annotations

from typing import Any, Dict, List, cast
from uuid import UUID

from domain.service.econml_backdoor_service import EconmlBackdoorService, JSONDict



class EconmlBackdoorMcpService(EconmlBackdoorService):
    """
    Implementation of EconmlBackdoorService that talks to your separate
    EconML MCP server (the one exposing backdoor_* and metadata_* tools).

    Transport details (HTTP URL, etc.) are hidden behind FastMCP's Client.
    """

    def __init__(self, client: Any) -> None: # pyright: ignore[reportUnknownParameterType]
        self._client = client

    async def _call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Internal helper: call an MCP tool and return its structured .data payload.

        FastMCP wraps tool results in CallToolResult.
        - .data           -> hydrated Python objects (dict/list/etc.)
        - .structured_content -> raw JSON if you need it
        """
        async with self._client:
            result = await self._client.call_tool(name, args)
        # For your server, return types are dict / list[dict], so .data is fine.
        return result.data

    # ---------- MetaData ----------

    async def metadata_get(self, dataset_id: UUID) -> JSONDict:
        data = await self._call_tool(
            "metadata_get",
            {"dataset_id": str(dataset_id)},
        )
        if not isinstance(data, dict):
            raise TypeError(f"metadata_get expected dict, got {type(data)!r}")
        return cast(JSONDict, data)

    async def metadata_create_or_update(
        self,
        dataset_id: UUID,
        *,
        treatment_type: str,
        outcome_type: str,
        treatment_cols: List[str],
        outcome_col: str,
        role_overrides: Dict[str, str] | None = None,
        notes: str | None = None,
    ) -> JSONDict:
        payload: Dict[str, Any] = {
            "dataset_id": str(dataset_id),
            "treatment_type": treatment_type,
            "outcome_type": outcome_type,
            "treatment_cols": treatment_cols,
            "outcome_col": outcome_col,
        }
        if role_overrides is not None:
            payload["role_overrides"] = role_overrides
        if notes is not None:
            payload["notes"] = notes

        data = await self._call_tool("metadata_create_or_update", payload)
        if not isinstance(data, dict):
            raise TypeError(
                f"metadata_create_or_update expected dict, got {type(data)!r}"
            )
        return cast(JSONDict, data)

    # ---------- Estimators ----------

    async def backdoor_list_estimators(self) -> List[JSONDict]:
        data = await self._call_tool("backdoor_list_estimators", {})
        if not isinstance(data, list):
            raise TypeError(
                f"backdoor_list_estimators expected list, got {type(data)!r}"
            )
        # Mild runtime check to keep the contract sane
        return cast(List[JSONDict], data)

    # ---------- Fit ----------

    async def backdoor_fit(
        self,
        dataset_id: UUID,
        estimator_id: str,
        *,
        run_tag: str | None = None,
        init_params: JSONDict | None = None,
        fit_params: JSONDict | None = None,
    ) -> JSONDict:
        payload: Dict[str, Any] = {
            "dataset_id": str(dataset_id),
            "estimator_id": estimator_id,
            "run_tag": run_tag,
            "init_params": init_params or {},
            "fit_params": fit_params or {},
        }
        data = await self._call_tool("backdoor_fit", payload)
        if not isinstance(data, dict):
            raise TypeError(f"backdoor_fit expected dict, got {type(data)!r}")
        return cast(JSONDict, data)

    # ---------- Effects ----------

    async def backdoor_effects(
        self,
        dataset_id: UUID,
        model_id: UUID,
        estimator_id: str,
        *,
        target: str,
        t0: Any | None = None,
        t1: Any | None = None,
        base_t: Any | None = None,
        row_indices: List[int] | None = None,
        unit_ids: List[Any] | None = None,
        x_new: Dict[str, List[Any]] | None = None,
        return_ci: bool = True,
        alpha: float = 0.05,
        options: JSONDict | None = None,
    ) -> JSONDict:
        payload: Dict[str, Any] = {
            "dataset_id": str(dataset_id),
            "model_id": str(model_id),
            "estimator_id": estimator_id,
            "target": target,
            "return_ci": return_ci,
            "alpha": alpha,
            "options": options or {},
        }
        if t0 is not None:
            payload["t0"] = t0
        if t1 is not None:
            payload["t1"] = t1
        if base_t is not None:
            payload["base_t"] = base_t
        if row_indices is not None:
            payload["row_indices"] = row_indices
        if unit_ids is not None:
            payload["unit_ids"] = unit_ids
        if x_new is not None:
            payload["x_new"] = x_new

        data = await self._call_tool("backdoor_effects", payload)
        if not isinstance(data, dict):
            raise TypeError(f"backdoor_effects expected dict, got {type(data)!r}")
        return cast(JSONDict, data)
