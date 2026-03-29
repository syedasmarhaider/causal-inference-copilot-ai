from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from python.domain.models.errors import StateDependencyError
from python.domain.workflows.state import State
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import (
    CleanProtocolState,
)
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec


@dataclass(frozen=True)
class ValidateCleanProtocolDeps:
    dataset_id: UUID
    causal_spec: CausalSpec
    
    
    
    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (CleanProtocolState.NAME,)

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> ValidateCleanProtocolDeps:
        
        # ---- CleanProtocolState ----
        clp = loaded.get(CleanProtocolState.NAME)
        if clp is None:
            raise StateDependencyError(f"ValidateCleanProtocolDeps: missing {CleanProtocolState.NAME}", to_state="ValidateCleanProtocolDeps", missing_dependencies=[CleanProtocolState.NAME])
        if not isinstance(clp, CleanProtocolState):
            raise StateDependencyError(f"ValidateCleanProtocolDeps: invalid {CleanProtocolState.NAME} "
                                       f"(expected CleanProtocolState, got {type(clp).__name__})", to_state="ValidateCleanProtocolDeps", missing_dependencies=[CleanProtocolState.NAME])
        
        if clp.payload.clean_dataset_id is None:
            raise StateDependencyError(f"ValidateCleanProtocolDeps: {CleanProtocolState.NAME} is not DONE yet (missing dataset id)", to_state="ValidateCleanProtocolDeps", missing_dependencies=[CleanProtocolState.NAME])
        if clp.payload.compiled_causal_spec is None:
            raise StateDependencyError(f"ValidateCleanProtocolDeps: {CleanProtocolState.NAME} is not DONE yet (missing compiled causal spec)", to_state="ValidateCleanProtocolDeps", missing_dependencies=[CleanProtocolState.NAME])
        return cls(dataset_id=clp.payload.clean_dataset_id, causal_spec=clp.payload.compiled_causal_spec)