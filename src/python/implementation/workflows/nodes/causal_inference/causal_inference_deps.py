from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Sequence, TypeVar

from python.domain.workflows.state import State

from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState
from python.implementation.workflows.nodes.model_selection.mode_selection_state import ModelSelectionState
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainState


T = TypeVar("T", bound=State)


@dataclass(frozen=True, slots=True)
class CausalInferenceDeps:
    clean_protocol: CleanProtocolState
    model_selection: ModelSelectionState
    model_train: ModelTrainState

    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (
            CompileProtocolState.NAME,
            CleanProtocolState.NAME,
            ModelSelectionState.NAME,
            ModelTrainState.NAME,
        )

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "CausalInferenceDeps":
        def _get(name: str, expected_type: type[T]) -> T:
            st = loaded.get(name)
            if st is None:
                raise ValueError(f"CausalInferenceDeps: missing {name}")
            if not isinstance(st, expected_type):
                raise ValueError(
                    f"CausalInferenceDeps: invalid {name} "
                    f"(expected {expected_type.__name__}, got {type(st).__name__})"
                )
            return st
        cl = _get(CleanProtocolState.NAME, CleanProtocolState)
        ms = _get(ModelSelectionState.NAME, ModelSelectionState)
        mt = _get(ModelTrainState.NAME, ModelTrainState)

        return cls(
            clean_protocol=cl,
            model_selection=ms,
            model_train=mt,
        )