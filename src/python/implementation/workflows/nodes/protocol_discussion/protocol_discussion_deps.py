from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from python.domain.workflows.state_dep import StateDep
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState


@dataclass(frozen=True)
class ProtocolDiscussionDeps(StateDep):
    load_dataset: LoadDatasetState

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, Any]) -> "ProtocolDiscussionDeps":
        raw = loaded.get(LoadDatasetState.NAME)
        if raw is None:
            raise ValueError(f"ProtocolDiscussionDeps: missing {LoadDatasetState.NAME}")
        
        if isinstance(raw, Mapping):
            ld = LoadDatasetState.from_json_dict(dict(raw)) # pyright: ignore[reportUnknownArgumentType]
            return cls(load_dataset=ld)

        raise ValueError(
            f"ProtocolDiscussionDeps: invalid {LoadDatasetState.NAME} "
            f"(expected payload mapping, got {type(raw).__name__})"
        )