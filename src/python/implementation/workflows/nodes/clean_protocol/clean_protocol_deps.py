from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from python.domain.workflows.state_dep import StateDep
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState


@dataclass(frozen=True)
class CleanProtocolDeps(StateDep):
    load_dataset: LoadDatasetState
    compile_protocol: CompileProtocolState

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, Any]) -> "CleanProtocolDeps":
        # ---- LoadDatasetState ----
        raw_ld = loaded.get(LoadDatasetState.NAME)
        if raw_ld is None:
            raise ValueError(f"CleanProtocolDeps: missing {LoadDatasetState.NAME}")

        if isinstance(raw_ld, Mapping):
            ld = LoadDatasetState.from_json_dict(dict(raw_ld))  # pyright: ignore[reportUnknownArgumentType] (payload-only dict)
        elif isinstance(raw_ld, LoadDatasetState):
            ld = raw_ld
        else:
            raise ValueError(
                f"CleanProtocolDeps: invalid {LoadDatasetState.NAME} "
                f"(expected payload mapping or LoadDatasetState, got {type(raw_ld).__name__})"
            )

        # ---- CompileProtocolState ----
        raw_cp = loaded.get(CompileProtocolState.NAME)
        if raw_cp is None:
            raise ValueError(f"CleanProtocolDeps: missing {CompileProtocolState.NAME}")

        if isinstance(raw_cp, Mapping):
            cp = CompileProtocolState.from_json_dict(dict(raw_cp))  # pyright: ignore[reportUnknownArgumentType] (payload-only dict)
        elif isinstance(raw_cp, CompileProtocolState):
            cp = raw_cp
        else:
            raise ValueError(
                f"CleanProtocolDeps: invalid {CompileProtocolState.NAME} "
                f"(expected payload mapping or CompileProtocolState, got {type(raw_cp).__name__})"
            )

        return cls(load_dataset=ld, compile_protocol=cp)