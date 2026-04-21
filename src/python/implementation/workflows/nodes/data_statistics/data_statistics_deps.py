from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)


@dataclass(frozen=True)
class DataStatisticsDeps:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel

    @classmethod
    def from_request(cls, request: NodeRequest) -> DataStatisticsDeps:
        dataset_id_raw: Any = request.orchestrator_state.get("working_dataset_id")
        dataset_summary_raw: Any = request.orchestrator_state.get("latest_dataset_summary")

        if dataset_id_raw is None:
            raise ValueError("dataset_id is required but was not found in orchestrator state")
        if dataset_summary_raw is None:
            raise ValueError("dataset_summary is required but was not found in orchestrator state")

        if not isinstance(dataset_id_raw, UUID):
            raise TypeError("dataset_id must be a UUID")
        if not isinstance(dataset_summary_raw, DatasetSummaryModel):
            raise TypeError("dataset_summary must be of type DatasetSummaryModel")

        return cls(
            dataset_id=dataset_id_raw,
            dataset_summary=dataset_summary_raw,
        )
    