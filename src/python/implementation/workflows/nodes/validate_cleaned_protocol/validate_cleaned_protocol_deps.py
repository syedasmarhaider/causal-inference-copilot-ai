from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Sequence

from python.domain.workflows.state import State
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState


@dataclass(frozen=True)
class ValidateCleanProtocolDeps:
    clean_protocol: CleanProtocolState
    
    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (CleanProtocolState.NAME,)

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "ValidateCleanProtocolDeps":
        
        # ---- CleanProtocolState ----
        clp = loaded.get(CleanProtocolState.NAME)
        if clp is None:
            raise ValueError(f"ValidateCleanProtocolDeps: missing {CleanProtocolState.NAME}")
        if not isinstance(clp, CleanProtocolState):
            raise ValueError(
                f"ValidateCleanProtocolDeps: invalid {CleanProtocolState.NAME} "
                f"(expected CleanProtocolState, got {type(clp).__name__})"
            )
        
        return cls( clean_protocol=clp)