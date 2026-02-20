from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from python.domain.workflows.state_dep import StateDep
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState


@dataclass(frozen=True)
class ValidateCleanProtocolDeps(StateDep):
    compile_protocol: CompileProtocolState
    clean_protocol: CleanProtocolState

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, Any]) -> "ValidateCleanProtocolDeps":
        # ---- CompileProtocolState ----
        raw_cp = loaded.get(CompileProtocolState.NAME)
        if raw_cp is None:
            raise ValueError(f"ValidateCleanProtocolDeps: missing {CompileProtocolState.NAME}")

        if isinstance(raw_cp, Mapping):
            cp = CompileProtocolState.from_json_dict(dict(raw_cp))  # pyright: ignore[reportUnknownArgumentType]
        elif isinstance(raw_cp, CompileProtocolState):
            cp = raw_cp
        else:
            raise ValueError(
                f"ValidateCleanProtocolDeps: invalid {CompileProtocolState.NAME} "
                f"(expected payload mapping or CompileProtocolState, got {type(raw_cp).__name__})"
            )

        # ---- CleanProtocolState ----
        raw_cl = loaded.get(CleanProtocolState.NAME)
        if raw_cl is None:
            raise ValueError(f"ValidateCleanProtocolDeps: missing {CleanProtocolState.NAME}")

        if isinstance(raw_cl, Mapping):
            cl = CleanProtocolState.from_json_dict(dict(raw_cl))  # pyright: ignore[reportUnknownArgumentType]
        elif isinstance(raw_cl, CleanProtocolState):
            cl = raw_cl
        else:
            raise ValueError(
                f"ValidateCleanProtocolDeps: invalid {CleanProtocolState.NAME} "
                f"(expected payload mapping or CleanProtocolState, got {type(raw_cl).__name__})"
            )

        return cls(compile_protocol=cp, clean_protocol=cl)