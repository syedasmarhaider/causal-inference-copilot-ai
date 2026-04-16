from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.utils.utils import uuid_from_any


@dataclass(frozen=True)
class ProtocolDiscussionDeps:
    dataset_id: UUID | None
    dataset_summary: DatasetSummaryModel | None

    @classmethod
    def from_request(cls, request: NodeRequest) -> ProtocolDiscussionDeps:
        dataset_id_raw = request.orchestrator_state.get("working_dataset_id")
        summary_raw = request.orchestrator_state.get("latest_dataset_summary")

        dataset_id = uuid_from_any(dataset_id_raw)

        dataset_summary: DatasetSummaryModel | None
        if summary_raw is None:
            dataset_summary = None
        elif isinstance(summary_raw, DatasetSummaryModel):
            dataset_summary = summary_raw
        elif isinstance(summary_raw, str):
            dataset_summary = DatasetSummaryModel.model_validate_json(summary_raw)
        else:
            dataset_summary = DatasetSummaryModel.model_validate(summary_raw)

        return cls(dataset_id=dataset_id, dataset_summary=dataset_summary)
