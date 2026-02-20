from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from python.domain.workflows.state import State
from python.domain.workflows.state_dep import StateDep
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState


@dataclass(frozen=True)
class ValidateCleanProtocolDeps(StateDep):
    compile_protocol: CompileProtocolState
    clean_protocol: CleanProtocolState

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "ValidateCleanProtocolDeps":
        pd = loaded.get(CompileProtocolState.NAME)
        if not isinstance(pd, CompileProtocolState):
            raise ValueError(
                f"ValidateCleanProtocolDeps: missing/invalid {CompileProtocolState.NAME} (got {type(pd).__name__ if pd else None})"
            )
        ci = loaded.get(CleanProtocolState.NAME)
        if not isinstance(ci, CleanProtocolState):
            raise ValueError(
                f"ValidateCleanProtocolDeps: missing/invalid {CleanProtocolState.NAME} (got {type(ci).__name__ if ci else None})"
            )
        return cls(compile_protocol=pd, clean_protocol=ci)
