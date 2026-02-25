from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Sequence

from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import ProtocolDiscussionState


@dataclass(frozen=True)
class CompileProtocolDeps:
    load_dataset: LoadDatasetState
    protocol_discussion: ProtocolDiscussionState
    
    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (LoadDatasetState.NAME, ProtocolDiscussionState.NAME)

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, Any]) -> "CompileProtocolDeps":
        # ---- LoadDatasetState ----
        ld = loaded.get(LoadDatasetState.NAME)
        if ld is None:
            raise ValueError(f"CompileProtocolDeps: missing {LoadDatasetState.NAME}")
        if not isinstance(ld, LoadDatasetState):
            raise ValueError(
                f"CompileProtocolDeps: invalid {LoadDatasetState.NAME} "
                f"(expected LoadDatasetState, got {type(ld).__name__})"
            )
        
        # ---- ProtocolDiscussionState ----
        pd = loaded.get(ProtocolDiscussionState.NAME)
        if pd is None:
            raise ValueError(f"CompileProtocolDeps: missing {ProtocolDiscussionState.NAME}")
        if not isinstance(pd, ProtocolDiscussionState):
            raise ValueError(
                f"CompileProtocolDeps: invalid {ProtocolDiscussionState.NAME} "
                f"(expected ProtocolDiscussionState, got {type(pd).__name__})"
            )    
        return cls(load_dataset=ld, protocol_discussion=pd)