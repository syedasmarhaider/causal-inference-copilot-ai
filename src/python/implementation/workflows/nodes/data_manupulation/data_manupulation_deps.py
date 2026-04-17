from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)


@dataclass(frozen=True)
class DataManupulationDeps:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel | None

    @classmethod
    def from_request(cls, request: NodeRequest) -> DataManupulationDeps:
        dataset_id_raw: Any = request.orchestrator_state.get("working_dataset_id")
        dataset_summary_raw: Any = request.orchestrator_state.get("latest_dataset_summary")
        if not isinstance(dataset_id_raw, UUID):
            raise TypeError("dataset_id must be a UUID")
        if dataset_summary_raw is not None and not isinstance(
            dataset_summary_raw, DatasetSummaryModel
        ):
            raise TypeError("dataset_summary must be of type DatasetSummaryModel")

        return cls(
            dataset_id=dataset_id_raw,
            dataset_summary=dataset_summary_raw,
        )

__all__ = ["DataManupulationDeps"]
