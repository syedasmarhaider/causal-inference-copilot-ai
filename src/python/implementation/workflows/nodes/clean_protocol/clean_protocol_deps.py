from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Sequence

from python.domain.workflows.state import State
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)


@dataclass(frozen=True)
class CleanProtocolDeps:
    load_dataset: LoadDatasetState
    compile_protocol: CompileProtocolState
    protocol_discussion: ProtocolDiscussionState

    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (
            LoadDatasetState.NAME,
            ProtocolDiscussionState.NAME,
            CompileProtocolState.NAME,
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
        
        # ---- CompileProtocolState ----
        cp = loaded.get(CompileProtocolState.NAME)
        if cp is None:
            raise ValueError(f"CleanProtocolDeps: missing {CompileProtocolState.NAME}")
        if not isinstance(cp, CompileProtocolState):
            raise ValueError(
                f"CleanProtocolDeps: invalid {CompileProtocolState.NAME} "
                f"(expected CompileProtocolState, got {type(cp).__name__})"
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

        return cls(load_dataset=ld, compile_protocol=cp, protocol_discussion=pd)
