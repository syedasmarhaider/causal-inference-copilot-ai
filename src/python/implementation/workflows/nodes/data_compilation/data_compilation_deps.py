from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)
from python.implementation.workflows.utils.utils import uuid_from_any


@dataclass(frozen=True)
class DataCompilationDeps:
    dataset_id: UUID | None
    dataset_summary: DatasetSummaryModel | None
    protocol_discussion: str | None

    @classmethod
    def from_request(cls, request: NodeRequest) -> DataCompilationDeps:
        dataset_id_raw = request.orchestrator_state.get("working_dataset_id")
        dataset_summary_raw = request.orchestrator_state.get("latest_dataset_summary")
        protocol_discussion_raw = request.orchestrator_state.get("protocol_discussion")

        dataset_id = uuid_from_any(dataset_id_raw)

        dataset_summary: DatasetSummaryModel | None
        if dataset_summary_raw is None:
            dataset_summary = None
        elif isinstance(dataset_summary_raw, DatasetSummaryModel):
            dataset_summary = dataset_summary_raw
        elif isinstance(dataset_summary_raw, str):
            dataset_summary = DatasetSummaryModel.model_validate_json(dataset_summary_raw)
        else:
            dataset_summary = DatasetSummaryModel.model_validate(dataset_summary_raw)

        if protocol_discussion_raw is None:
            protocol_discussion = None
        elif isinstance(protocol_discussion_raw, str):
            protocol_discussion = protocol_discussion_raw.strip() or None
        else:
            raise TypeError("protocol_discussion must be str|null")

        return cls(
            dataset_id=dataset_id,
            dataset_summary=dataset_summary,
            protocol_discussion=protocol_discussion,
        )
