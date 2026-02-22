from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Sequence

from python.domain.workflows.state import State
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_state import ValidateCleanProtocolState


@dataclass(frozen=True)
class TransformProtocolDeps:
    compile_protocol: CompileProtocolState
    clean_protocol: CleanProtocolState 
    validate_cleaned_protocol: ValidateCleanProtocolState
    
    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (
            CompileProtocolState.NAME,
            CleanProtocolState.NAME,
            ValidateCleanProtocolState.NAME,
        )

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "TransformProtocolDeps":
        # ---- CompileProtocolState ----
        cp = loaded.get(CompileProtocolState.NAME)
        if cp is None:
            raise ValueError(f"TransformProtocolDeps: missing {CompileProtocolState.NAME}")
        if not isinstance(cp, CompileProtocolState):
            raise ValueError(
                f"TransformProtocolDeps: invalid {CompileProtocolState.NAME} "
                f"(expected CompileProtocolState, got {type(cp).__name__})"
            )
        
        # ---- CleanProtocolState ----
        clp = loaded.get(CleanProtocolState.NAME)
        if clp is None:
            raise ValueError(f"TransformProtocolDeps: missing {CleanProtocolState.NAME}")
        if not isinstance(clp, CleanProtocolState):
            raise ValueError(
                f"TransformProtocolDeps: invalid {CleanProtocolState.NAME} "
                f"(expected CleanProtocolState, got {type(clp).__name__})"
            )
        
        # ---- ValidateCleanedProtocolState ----
        vc = loaded.get(ValidateCleanProtocolState.NAME)
        if vc is None:
            raise ValueError(f"TransformProtocolDeps: missing {ValidateCleanProtocolState.NAME}")
        if not isinstance(vc, ValidateCleanProtocolState):
            raise ValueError(
                f"TransformProtocolDeps: invalid {ValidateCleanProtocolState.NAME} "
                f"(expected ValidateCleanProtocolState, got {type(vc).__name__})"
            )
        
        return cls(compile_protocol=cp, clean_protocol=clp, validate_cleaned_protocol=vc)