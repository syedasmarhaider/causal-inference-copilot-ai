from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from python.domain.workflows.state_dep import StateDep
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import ProtocolDiscussionState


@dataclass(frozen=True)
class CompileProtocolDeps(StateDep):
    load_dataset: LoadDatasetState
    protocol_discussion: ProtocolDiscussionState

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, Any]) -> "CompileProtocolDeps":
        raw_ld = loaded.get(LoadDatasetState.NAME)
        if raw_ld is None:
            raise ValueError(f"CompileProtocolDeps: missing {LoadDatasetState.NAME}")

        if isinstance(raw_ld, Mapping):
            ld = LoadDatasetState.from_json_dict(dict(raw_ld)) # pyright: ignore[reportUnknownArgumentType]
        elif isinstance(raw_ld, LoadDatasetState):
            # allow already-instantiated state; keep as-is
            ld = raw_ld
        else:
            raise ValueError(
                f"CompileProtocolDeps: invalid {LoadDatasetState.NAME} "
                f"(expected payload mapping or LoadDatasetState, got {type(raw_ld).__name__})"
            )

        # ---- ProtocolDiscussionState ----
        raw_pd = loaded.get(ProtocolDiscussionState.NAME)
        if raw_pd is None:
            raise ValueError(f"CompileProtocolDeps: missing {ProtocolDiscussionState.NAME}")

        if isinstance(raw_pd, Mapping):
            pd = ProtocolDiscussionState.from_json_dict(dict(raw_pd))  # pyright: ignore[reportUnknownArgumentType]
        elif isinstance(raw_pd, ProtocolDiscussionState):
            pd = raw_pd
        else:
            raise ValueError(
                f"CompileProtocolDeps: invalid {ProtocolDiscussionState.NAME} "
                f"(expected payload mapping or ProtocolDiscussionState, got {type(raw_pd).__name__})"
            )

        return cls(load_dataset=ld, protocol_discussion=pd)