from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from python.domain.models.errors import StateDependencyError
from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_state import (
    DataManupulationState,
)
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

        if dataset_id_raw is None:
            raise _missing_dependency_error("dataset_id")

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


def _missing_dependency_error(*missing: str) -> StateDependencyError:
    return StateDependencyError(
        DataManupulationState.NAME,
        DataManupulationState.NAME,
        list(missing),
    )


__all__ = ["DataManupulationDeps"]
