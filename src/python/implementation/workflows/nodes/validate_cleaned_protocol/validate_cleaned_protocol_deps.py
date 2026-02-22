from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Sequence

from python.domain.workflows.state import State
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState


@dataclass(frozen=True)
class ValidateCleanProtocolDeps:
    compile_protocol: CompileProtocolState
    clean_protocol: CleanProtocolState
    
    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (CompileProtocolState.NAME, CleanProtocolState.NAME)

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "ValidateCleanProtocolDeps":
        # ---- CompileProtocolState ----
        cp = loaded.get(CompileProtocolState.NAME)
        if cp is None:
            raise ValueError(f"ValidateCleanProtocolDeps: missing {CompileProtocolState.NAME}")
        if not isinstance(cp, CompileProtocolState):
            raise ValueError(
                f"ValidateCleanProtocolDeps: invalid {CompileProtocolState.NAME} "
                f"(expected CompileProtocolState, got {type(cp).__name__})"
            )
        
        # ---- CleanProtocolState ----
        clp = loaded.get(CleanProtocolState.NAME)
        if clp is None:
            raise ValueError(f"ValidateCleanProtocolDeps: missing {CleanProtocolState.NAME}")
        if not isinstance(clp, CleanProtocolState):
            raise ValueError(
                f"ValidateCleanProtocolDeps: invalid {CleanProtocolState.NAME} "
                f"(expected CleanProtocolState, got {type(clp).__name__})"
            )
        
        return cls(compile_protocol=cp, clean_protocol=clp)