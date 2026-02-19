from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from python.domain.workflows.state import State
from python.domain.workflows.state_dep import StateDep
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import ProtocolDiscussionState


@dataclass(frozen=True)
class CompileProtocolDeps(StateDep):
    load_dataset: LoadDatasetState
    protocol_discussion: ProtocolDiscussionState

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "CompileProtocolDeps":
        ld = loaded.get(LoadDatasetState.NAME)
        if not isinstance(ld, LoadDatasetState):
            raise ValueError(
                f"CompileProtocolDeps: missing/invalid {LoadDatasetState.NAME} (got {type(ld).__name__ if ld else None})"
            )
        pd = loaded.get(ProtocolDiscussionState.NAME)
        if not isinstance(pd, ProtocolDiscussionState):
            raise ValueError(
                f"CompileProtocolDeps: missing/invalid {ProtocolDiscussionState.NAME} (got {type(pd).__name__ if pd else None})"
            )
        return cls(load_dataset=ld, protocol_discussion=pd)
