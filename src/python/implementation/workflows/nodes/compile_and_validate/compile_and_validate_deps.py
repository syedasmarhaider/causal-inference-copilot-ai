from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from python.domain.models.errors import NodeExecutionError, StateDependencyError
from python.domain.workflows.state import State
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
    def pre_required_states_names(cls) -> Sequence[str]:
        return [DatasetState.NAME, ProtocolDiscussionState.NAME]

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> CompileAndValidateDeps:
        dataset_state = loaded.get(DatasetState.NAME)
        if dataset_state is None or not isinstance(dataset_state, DatasetState):
            raise StateDependencyError(
                "COMPILE_AND_VALIDATE",
                "COMPILE_AND_VALIDATE",
                [DatasetState.NAME],
            )

        if not dataset_state.payload.dataset_iterations:
            raise StateDependencyError(
                "COMPILE_AND_VALIDATE",
                "COMPILE_AND_VALIDATE",
                [DatasetState.NAME],
            )

        latest_iteration = dataset_state.payload.dataset_iterations[-1]
        if latest_iteration.summary is None:
            raise StateDependencyError(
                "COMPILE_AND_VALIDATE",
                "COMPILE_AND_VALIDATE",
                [DatasetState.NAME],
            )

        protocol_state = loaded.get(ProtocolDiscussionState.NAME)
        if protocol_state is None or not isinstance(protocol_state, ProtocolDiscussionState):
            raise StateDependencyError(
                "COMPILE_AND_VALIDATE",
                "COMPILE_AND_VALIDATE",
                [ProtocolDiscussionState.NAME],
            )

        if protocol_state.payload.phase != "CONFIRMED":
            raise StateDependencyError(
                "COMPILE_AND_VALIDATE",
                "COMPILE_AND_VALIDATE",
                [ProtocolDiscussionState.NAME],
            )

        if not protocol_state.payload.discussion.strip():
            raise NodeExecutionError(
                state_name="COMPILE_AND_VALIDATE",
                error="Protocol discussion is empty and cannot be compiled.",
            )

        return cls(
            dataset_id=latest_iteration.dataset_id,
            dataset_summary=latest_iteration.summary,
            protocol_discussion=protocol_state.payload.discussion,
        )
