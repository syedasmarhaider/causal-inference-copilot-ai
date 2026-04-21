from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


@dataclass(frozen=True)
class ProtocolDiscussionDeps:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel

    @classmethod
    def from_request(cls, request: NodeRequest) -> ProtocolDiscussionDeps:
        dataset_id_raw = request.orchestrator_state.get("working_dataset_id")
        dataset_summary_raw = request.orchestrator_state.get("latest_dataset_summary")

        if dataset_id_raw is None:
            raise ValueError("ProtocolDiscussionDeps: dataset_id is required but was not found in compilation state")
        if dataset_summary_raw is None:
            raise ValueError("ProtocolDiscussionDeps: dataset_summary is required but was not found in compilation state")
        
        if not isinstance(dataset_id_raw, UUID):
            raise TypeError("ProtocolDiscussionDeps: dataset_id must be a UUID")
        if not isinstance(dataset_summary_raw, DatasetSummaryModel):
            raise TypeError("ProtocolDiscussionDeps: dataset_summary must be of type DatasetSummaryModel")

        return cls(
            dataset_id=dataset_id_raw,
            dataset_summary=dataset_summary_raw,
        )
