from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from python.domain.models.errors import StateDependencyError
from python.domain.workflows.state import State
from python.implementation.workflows.nodes.dataset.dataset_state import DatasetState
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


@dataclass(frozen=True)
class ProtocolDiscussionDeps:
    dataset_summary: DatasetSummaryModel
    dataset_id: UUID

    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return [DatasetState.NAME]

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> ProtocolDiscussionDeps:
        # ---- DatasetState ----
        ds = loaded.get(DatasetState.NAME)
        if ds is None:
            raise StateDependencyError(
                f"ProtocolDiscussionDeps: missing {DatasetState.NAME}",
                to_state="ProtocolDiscussionDeps",
                missing_dependencies=[DatasetState.NAME],
            )
        if not isinstance(ds, DatasetState):
            raise StateDependencyError(
                f"ProtocolDiscussionDeps: invalid {DatasetState.NAME} "
                f"(expected DatasetState, got {type(ds).__name__})",
                to_state="ProtocolDiscussionDeps",
                missing_dependencies=[DatasetState.NAME],
            )
        if len(ds.payload.dataset_iterations) == 0:
            raise StateDependencyError(
                f"ProtocolDiscussionDeps: {DatasetState.NAME} is not DONE yet (missing dataset iterations)",
                to_state="ProtocolDiscussionDeps",
                missing_dependencies=[DatasetState.NAME],
            )
        latest_iteration = ds.payload.dataset_iterations[-1]
        latest_summary = ds.payload.latest_summary
        if latest_summary is None:
            raise StateDependencyError(
                f"ProtocolDiscussionDeps: {DatasetState.NAME} is not DONE yet (missing dataset summary)",
                to_state="ProtocolDiscussionDeps",
                missing_dependencies=[DatasetState.NAME],
            )
        return cls(dataset_summary=latest_summary, dataset_id=latest_iteration.dataset_id)
