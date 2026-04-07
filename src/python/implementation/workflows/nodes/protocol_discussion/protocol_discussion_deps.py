from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from python.domain.models.errors import StateDependencyError
from python.domain.workflows.ochestrator_state import ReadOnlyOchestratorState
from python.implementation.workflows.nodes.dataset.dataset_state import DatasetState
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


@dataclass(frozen=True)
class ProtocolDiscussionDeps:
    dataset_summary: DatasetSummaryModel
    dataset_id: UUID
    
    @classmethod
    def from_loaded(cls, readonly_orchestrator_state: ReadOnlyOchestratorState) -> ProtocolDiscussionDeps:
        dataset_id = readonly_orchestrator_state.get("working_dataset_id")
        summary    = readonly_orchestrator_state.get("working_dataset_summary")
        if dataset_id is None or summary is None:
            raise StateDependencyError(
                "PROTOCOL_DISCUSSION",
                "PROTOCOL_DISCUSSION",
                [DatasetState.NAME],
            )
            
        return cls(dataset_summary=summary, dataset_id=dataset_id)
