from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Sequence

from python.domain.workflows.state import State
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_state import TransformProtocolState


@dataclass(frozen=True)
class ConfirmTransformedProtocolDeps:
    transform_protocol: TransformProtocolState
    
    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (TransformProtocolState.NAME,)

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "ConfirmTransformedProtocolDeps":
        # ---- TransformProtocolState ----
        tp = loaded.get(TransformProtocolState.NAME)
        if tp is None:
            raise ValueError(f"ConfirmTransformedProtocolDeps: missing {TransformProtocolState.NAME}")
        if not isinstance(tp, TransformProtocolState):
            raise ValueError(
                f"ConfirmTransformedProtocolDeps: invalid {TransformProtocolState.NAME} "
                f"(expected TransformProtocolState, got {type(tp).__name__})"
            )
        return cls(transform_protocol=tp)
        