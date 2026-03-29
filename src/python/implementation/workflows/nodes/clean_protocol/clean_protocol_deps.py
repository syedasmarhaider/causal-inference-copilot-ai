from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Sequence
from uuid import UUID

from python.domain.workflows.state import State
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


@dataclass(frozen=True)
class CleanProtocolDeps:
    id: UUID
    summary: DatasetSummaryModel
    protocol_discsussion: str

    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (
            LoadDatasetState.NAME,
            ProtocolDiscussionState.NAME,
        )

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "CleanProtocolDeps":
        # ---- LoadDatasetState ----
        ld = loaded.get(LoadDatasetState.NAME)
        if ld is None:
            raise ValueError(f"CleanProtocolDeps: missing {LoadDatasetState.NAME}")
        if not isinstance(ld, LoadDatasetState):
            raise ValueError(
                f"CleanProtocolDeps: invalid {LoadDatasetState.NAME} "
                f"(expected LoadDatasetState, got {type(ld).__name__})"
            )
        
        # ---- ProtocolDiscussionState ----
        pd = loaded.get(ProtocolDiscussionState.NAME)
        if pd is None:
            raise ValueError(f"CleanProtocolDeps: missing {ProtocolDiscussionState.NAME}")
        if not isinstance(pd, ProtocolDiscussionState):
            raise ValueError(
                f"CleanProtocolDeps: invalid {ProtocolDiscussionState.NAME} "
                f"(expected ProtocolDiscussionState, got {type(pd).__name__})"
            )
        if ld.payload.id is None:
            raise ValueError(f"CleanProtocolDeps: {LoadDatasetState.NAME} is not DONE yet (missing dataset id)")
        if ld.payload.summary is None:
            raise ValueError(f"CleanProtocolDeps: {LoadDatasetState.NAME} is not DONE yet (missing dataset summary)")
        if pd.payload.discussion == "":
            raise ValueError(f"CleanProtocolDeps: {ProtocolDiscussionState.NAME} is not DONE yet (missing discussion summary)")
        return cls(id=ld.payload.id, summary=ld.payload.summary, protocol_discsussion=pd.payload.discussion)
