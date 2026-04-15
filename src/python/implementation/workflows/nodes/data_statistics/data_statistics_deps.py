from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from python.domain.models.errors import StateDependencyError
from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.nodes.data_statistics.data_statistics_state import (
    DataStatisticsState,
)
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)


@dataclass(frozen=True)
class DataStatisticsDeps:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel

    @classmethod
    def from_request(cls, request: NodeRequest) -> DataStatisticsDeps:
        raw_context = request.orchestrator_state.get(request.node_state.name())
        if raw_context is None:
            raise _missing_dependency_error("dataset_id", "dataset_summary")

        dataset_id_raw: Any
        dataset_summary_raw: Any

        if isinstance(raw_context, Mapping):
            dataset_id_raw = raw_context.get("dataset_id")
            dataset_summary_raw = raw_context.get("dataset_summary")
        else:
            raise ValueError(
                "DATA_STATISTICS dependency payload must be a dict with dataset_id and "
                "dataset_summary"
            )

        if dataset_id_raw is None or dataset_summary_raw is None:
            raise _missing_dependency_error("dataset_id", "dataset_summary")
        
        if not isinstance(dataset_summary_raw, DatasetSummaryModel):
            raise TypeError("dataset_summary must be of type DatasetSummaryModel")
        
        if not isinstance(dataset_id_raw,  UUID):
            raise TypeError("dataset_id must be a UUID")

        return cls(dataset_id=dataset_id_raw, dataset_summary=dataset_summary_raw)
    

def _missing_dependency_error(*missing: str) -> StateDependencyError:
    return StateDependencyError(
        DataStatisticsState.NAME,
        DataStatisticsState.NAME,
        list(missing),
    )