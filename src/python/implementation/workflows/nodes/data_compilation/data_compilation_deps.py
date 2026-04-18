from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    CausalSpecDraft,
)

@dataclass(frozen=True)
class DataCompilationDeps:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel
    protocol_discussion: str
    protocol_cleaning_instructions: str | None
    causal_spec_draft: CausalSpecDraft

    @classmethod
    def from_request(cls, request: NodeRequest) -> DataCompilationDeps:
        dataset_id_raw = request.orchestrator_state.get("working_dataset_id")
        dataset_summary_raw = request.orchestrator_state.get("latest_dataset_summary")
        protocol_discussion_raw = request.orchestrator_state.get("protocol_discussion")
        protocol_cleaning_instructions_raw = request.orchestrator_state.get(
            "protocol_cleaning_instructions"
        )
        causal_spec_draft_raw = request.orchestrator_state.get("causal_spec_draft")

        if dataset_id_raw is None:
            raise ValueError("DataCompilationDeps: dataset_id is required but was not found in compilation state")
        if dataset_summary_raw is None:
            raise ValueError("DataCompilationDeps: dataset_summary is required but was not found in compilation state")
        if protocol_discussion_raw is None:
            raise ValueError("DataCompilationDeps: protocol_discussion is required but was not found in compilation state")
        if causal_spec_draft_raw is None:
            raise ValueError("DataCompilationDeps: causal_spec_draft is required but was not found in compilation state")
        
        if not isinstance(dataset_id_raw, UUID):
            raise TypeError("DataCompilationDeps: dataset_id must be a UUID")
        if not isinstance(dataset_summary_raw, DatasetSummaryModel):
            raise TypeError("DataCompilationDeps: dataset_summary must be of type DatasetSummaryModel")
        if not isinstance(protocol_discussion_raw, str):
            raise TypeError("DataCompilationDeps: protocol_discussion must be a string")
        if (
            protocol_cleaning_instructions_raw is not None
            and not isinstance(protocol_cleaning_instructions_raw, str)
        ):
            raise TypeError(
                "DataCompilationDeps: protocol_cleaning_instructions must be a string or None"
            )
        if not isinstance(causal_spec_draft_raw, CausalSpecDraft):
            raise TypeError("DataCompilationDeps: causal_spec_draft must be of type CausalSpecDraft")

        return cls(
            dataset_id=dataset_id_raw,
            dataset_summary=dataset_summary_raw,
            protocol_discussion=protocol_discussion_raw,
            protocol_cleaning_instructions=protocol_cleaning_instructions_raw,
            causal_spec_draft=causal_spec_draft_raw,
        )
