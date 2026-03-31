from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from python.domain.models.errors import StateDependencyError
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
    def from_loaded(cls, loaded: Mapping[str, State]) -> CleanProtocolDeps:
        # ---- LoadDatasetState ----
        ld = loaded.get(LoadDatasetState.NAME)
        if ld is None:
            raise StateDependencyError(f"CleanProtocolDeps: missing {LoadDatasetState.NAME}", to_state="CleanProtocolDeps", missing_dependencies=[LoadDatasetState.NAME])
        if not isinstance(ld, LoadDatasetState):
            raise StateDependencyError(f"CleanProtocolDeps: invalid {LoadDatasetState.NAME} "
                                       f"(expected LoadDatasetState, got {type(ld).__name__})", to_state="CleanProtocolDeps", missing_dependencies=[LoadDatasetState.NAME])
        
        # ---- ProtocolDiscussionState ----
        pd = loaded.get(ProtocolDiscussionState.NAME)
        if pd is None:
            raise StateDependencyError(f"CleanProtocolDeps: missing {ProtocolDiscussionState.NAME}", to_state="CleanProtocolDeps", missing_dependencies=[ProtocolDiscussionState.NAME])
        if not isinstance(pd, ProtocolDiscussionState):
            raise StateDependencyError(f"CleanProtocolDeps: invalid {ProtocolDiscussionState.NAME} "
                                       f"(expected ProtocolDiscussionState, got {type(pd).__name__})", to_state="CleanProtocolDeps", missing_dependencies=[ProtocolDiscussionState.NAME])
        if ld.payload.id is None:
            raise StateDependencyError(f"CleanProtocolDeps: {LoadDatasetState.NAME} is not DONE yet (missing dataset id)", to_state="CleanProtocolDeps", missing_dependencies=[LoadDatasetState.NAME])
        if ld.payload.summary is None:
            raise StateDependencyError(f"CleanProtocolDeps: {LoadDatasetState.NAME} is not DONE yet (missing dataset summary)", to_state="CleanProtocolDeps", missing_dependencies=[LoadDatasetState.NAME])
        if pd.payload.discussion == "":
            raise StateDependencyError(f"CleanProtocolDeps: {ProtocolDiscussionState.NAME} is not DONE yet (missing discussion summary)", to_state="CleanProtocolDeps", missing_dependencies=[ProtocolDiscussionState.NAME])
        return cls(id=ld.payload.id, summary=ld.payload.summary, protocol_discsussion=pd.payload.discussion)
