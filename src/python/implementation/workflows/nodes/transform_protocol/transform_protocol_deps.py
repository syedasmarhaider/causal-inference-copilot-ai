from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from python.domain.workflows.state_dep import StateDep
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_state import ValidateCleanProtocolState


@dataclass(frozen=True)
class TransformProtocolDeps(StateDep):
    compile_protocol: CompileProtocolState
    clean_protocol: CleanProtocolState 
    validate_cleaned_protocol: ValidateCleanProtocolState

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, Any]) -> "TransformProtocolDeps":
        # ---- CompileProtocolState ----
        raw_cp = loaded.get(CompileProtocolState.NAME)
        if raw_cp is None:
            raise ValueError(f"TransformProtocolDeps: missing {CompileProtocolState.NAME}")

        if isinstance(raw_cp, Mapping):
            cp = CompileProtocolState.from_json_dict(dict(raw_cp))  # pyright: ignore[reportUnknownArgumentType]
        elif isinstance(raw_cp, CompileProtocolState):
            cp = raw_cp
        else:
            raise ValueError(
                f"TransformProtocolDeps: invalid {CompileProtocolState.NAME} "
                f"(expected payload mapping or CompileProtocolState, got {type(raw_cp).__name__})"
            )

        # ---- CleanProtocolState ----
        raw_cl = loaded.get(CleanProtocolState.NAME)
        if raw_cl is None:
            raise ValueError(f"TransformProtocolDeps: missing {CleanProtocolState.NAME}")

        if isinstance(raw_cl, Mapping):
            cl = CleanProtocolState.from_json_dict(dict(raw_cl))  # pyright: ignore[reportUnknownArgumentType]
        elif isinstance(raw_cl, CleanProtocolState):
            cl = raw_cl
        else:
            raise ValueError(
                f"TransformProtocolDeps: invalid {CleanProtocolState.NAME} "
                f"(expected payload mapping or CleanProtocolState, got {type(raw_cl).__name__})"
            )
            
        raw_vcp = loaded.get(ValidateCleanProtocolState.NAME)
        if raw_vcp is None:
            raise ValueError(f"TransformProtocolDeps: missing {ValidateCleanProtocolState.NAME}")
        if isinstance(raw_vcp, Mapping):
            vcp = ValidateCleanProtocolState.from_json_dict(dict(raw_vcp))  # pyright: ignore[reportUnknownArgumentType]
        elif isinstance(raw_vcp, ValidateCleanProtocolState):
            vcp = raw_vcp
        else:
            raise ValueError(
                f"TransformProtocolDeps: invalid {ValidateCleanProtocolState.NAME} "
                f"(expected payload mapping or ValidateCleanProtocolState, got {type(raw_vcp).__name__})"
            )
                 

        return cls(compile_protocol=cp, clean_protocol=cl, validate_cleaned_protocol=vcp)