from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from python.domain.workflows.state import State
from python.domain.workflows.state_dep import StateDep
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState


@dataclass(frozen=True)
class CompileInferenceDeps(StateDep):
    load_dataset: LoadDatasetState
    compile_protocol: CompileProtocolState

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "CompileInferenceDeps":
        ld = loaded.get(LoadDatasetState.NAME)
        if not isinstance(ld, LoadDatasetState):
            raise ValueError(
                f"CompileInferenceDeps: missing/invalid {LoadDatasetState.NAME} (got {type(ld).__name__ if ld else None})"
            )
        pd = loaded.get(CompileProtocolState.NAME)
        if not isinstance(pd, CompileProtocolState):
            raise ValueError(
                f"CompileInferenceDeps: missing/invalid {CompileProtocolState.NAME} (got {type(pd).__name__ if pd else None})"
            )
        return cls(load_dataset=ld, compile_protocol=pd)
