from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Sequence

from python.domain.workflows.state import State
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState


@dataclass(frozen=True)
class ProtocolDiscussionDeps:
    load_dataset: LoadDatasetState
    
    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return [LoadDatasetState.NAME]

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "ProtocolDiscussionDeps":
        # ---- LoadDatasetState ----
        ld = loaded.get(LoadDatasetState.NAME)
        if ld is None:
            raise ValueError(f"ProtocolDiscussionDeps: missing {LoadDatasetState.NAME}")
        if not isinstance(ld, LoadDatasetState):
            raise ValueError(
                f"ProtocolDiscussionDeps: invalid {LoadDatasetState.NAME} "
                f"(expected LoadDatasetState, got {type(ld).__name__})"
            )
        return cls(load_dataset=ld)