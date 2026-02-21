from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from python.domain.workflows.state_dep import StateDep
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_state import TransformProtocolState


@dataclass(frozen=True)
class ConfirmTransformedProtocolDeps(StateDep):
    transform_protocol: TransformProtocolState

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, Any]) -> "ConfirmTransformedProtocolDeps":
        raw_tp = loaded.get(TransformProtocolState.NAME)
        if raw_tp is None:
            raise ValueError(f"ConfirmTransformedProtocolDeps: missing {TransformProtocolState.NAME}")

        if isinstance(raw_tp, Mapping):
            tp = TransformProtocolState.from_json_dict(dict(raw_tp))  # pyright: ignore[reportUnknownArgumentType]
        elif isinstance(raw_tp, TransformProtocolState):
            tp = raw_tp
        else:
            raise ValueError(
                f"ConfirmTransformedProtocolDeps: invalid {TransformProtocolState.NAME} "
                f"(expected payload mapping or TransformProtocolState, got {type(raw_tp).__name__})"
            )

        return cls(transform_protocol=tp)