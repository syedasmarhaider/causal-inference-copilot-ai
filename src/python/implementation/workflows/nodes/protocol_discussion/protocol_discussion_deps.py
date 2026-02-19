from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from python.domain.workflows.state import State
from python.domain.workflows.state_dep import StateDep
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState


@dataclass(frozen=True)
class ProtocolDiscussionDeps(StateDep):
    load_dataset: LoadDatasetState

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "ProtocolDiscussionDeps":
        ld = loaded.get(LoadDatasetState.NAME)
        if not isinstance(ld, LoadDatasetState):
            raise ValueError(
                f"ProtocolDiscussionDeps: missing/invalid {LoadDatasetState.NAME} (got {type(ld).__name__ if ld else None})"
            )
        return cls(load_dataset=ld)
