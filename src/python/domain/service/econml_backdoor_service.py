from __future__ import annotations

from typing import Any, Dict, List, Protocol
from uuid import UUID

JSONDict = Dict[str, Any]


class EconmlBackdoorService(Protocol):
    """
    Abstract contract for talking to the EconML backdoor engine.
    Implementations live in implementation/service/.
    """

    # ---------- MetaData ----------

    async def metadata_get(self, dataset_id: UUID) -> JSONDict:
        """Fetch MetaData JSON for a dataset_id."""
        ...

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
        """Create or update MetaData and return the canonical MetaData JSON."""
        ...

    # ---------- Estimators ----------

    async def backdoor_list_estimators(self) -> List[JSONDict]:
        """Return catalog of estimators (id, capabilities, notes, ...)."""
        ...

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
        """Fit model and return BackdoorFitResult JSON (includes model_id)."""
        ...

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
        """Query ATE / CATE / marginal effects and return BackdoorEffectResult."""
        ...
