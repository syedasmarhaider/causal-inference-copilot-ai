from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from python.domain.models.errors import StateDependencyError
from python.domain.workflows.state import State
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


@dataclass(frozen=True)
class ProtocolDiscussionDeps:
    dataset_summary : DatasetSummaryModel
    
    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return [LoadDatasetState.NAME]

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> ProtocolDiscussionDeps:
        # ---- LoadDatasetState ----
        ld = loaded.get(LoadDatasetState.NAME)
        if ld is None:
            raise StateDependencyError(f"ProtocolDiscussionDeps: missing {LoadDatasetState.NAME}", to_state="ProtocolDiscussionDeps", missing_dependencies=[LoadDatasetState.NAME])
        if not isinstance(ld, LoadDatasetState):
            raise StateDependencyError(f"ProtocolDiscussionDeps: invalid {LoadDatasetState.NAME} "
                                       f"(expected LoadDatasetState, got {type(ld).__name__})", to_state="ProtocolDiscussionDeps", missing_dependencies=[LoadDatasetState.NAME])
        if ld.payload.summary is None:
            raise StateDependencyError(f"ProtocolDiscussionDeps: {LoadDatasetState.NAME} is not DONE yet (missing dataset summary)", to_state="ProtocolDiscussionDeps", missing_dependencies=[LoadDatasetState.NAME])
        return cls(dataset_summary=ld.payload.summary)