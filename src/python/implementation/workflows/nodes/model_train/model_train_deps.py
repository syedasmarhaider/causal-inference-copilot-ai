from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Sequence, TypeVar

from python.domain.workflows.state import State

from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState
from python.implementation.workflows.nodes.model_selection.mode_selection_state import ModelSelectionState


T = TypeVar("T", bound=State)


@dataclass(frozen=True, slots=True)
class ModelTrainDeps:
    clean_protocol: CleanProtocolState
    model_selection: ModelSelectionState

    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (
            CleanProtocolState.NAME,
            ModelSelectionState.NAME,
        )

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "ModelTrainDeps":
        def _get(name: str, expected_type: type[T]) -> T:
            st = loaded.get(name)
            if st is None:
                raise ValueError(f"ModelTrainDeps: missing {name}")
            if not isinstance(st, expected_type):
                raise ValueError(
                    f"ModelTrainDeps: invalid {name} "
                    f"(expected {expected_type.__name__}, got {type(st).__name__})"
                )
            return st
        cl = _get(CleanProtocolState.NAME, CleanProtocolState)
        ms = _get(ModelSelectionState.NAME, ModelSelectionState)

        return cls(
            clean_protocol=cl,
            model_selection=ms,
        )