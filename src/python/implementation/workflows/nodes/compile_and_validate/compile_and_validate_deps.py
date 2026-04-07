from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from python.domain.models.errors import  StateDependencyError
from python.domain.workflows.ochestrator_state import ReadOnlyOchestratorState
from python.implementation.workflows.nodes.dataset.dataset_state import DatasetState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


@dataclass(frozen=True)
class CompileAndValidateDeps:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel
    protocol_discussion: str

    @classmethod
    def from_loaded(cls, readonly_orchestrator_state: ReadOnlyOchestratorState) -> CompileAndValidateDeps:
        dataset_id = readonly_orchestrator_state.get("working_dataset_id")
        summary    = readonly_orchestrator_state.get("working_dataset_summary")
        protocol_discussion = readonly_orchestrator_state.get("protocol_discussion")
        if dataset_id is None or summary is None or protocol_discussion is None:
            raise StateDependencyError(
                "COMPILE_AND_VALIDATE",
                "COMPILE_AND_VALIDATE",
                [DatasetState.NAME, ProtocolDiscussionState.NAME],
            )

        return cls(
            dataset_id=dataset_id,
            dataset_summary=summary,
            protocol_discussion=protocol_discussion,
        )
