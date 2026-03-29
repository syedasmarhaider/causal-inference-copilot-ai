from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar
from uuid import UUID

from python.domain.models.errors import StateDependencyError
from python.domain.workflows.state import State
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import (
    CleanProtocolState,
)
from python.implementation.workflows.nodes.model_selection.mode_selection_state import (
    ModelSelectionState,
)
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel

T = TypeVar("T", bound=State)


@dataclass(frozen=True, slots=True)
class ModelTrainDeps:
    causal_specs: CausalSpec
    dataset_summary: DatasetSummaryModel
    dataset_id: UUID

    selected_model: str

    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (
            CleanProtocolState.NAME,
            ModelSelectionState.NAME,
        )

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> ModelTrainDeps:
        def _get(name: str, expected_type: type[T]) -> T:
            st = loaded.get(name)
            if st is None:
                raise StateDependencyError(f"ModelTrainDeps: missing {name}", to_state="ModelTrainDeps", missing_dependencies=[
                    name,
                ])
            if not isinstance(st, expected_type):
                raise StateDependencyError(f"ModelTrainDeps: invalid {name} (expected {expected_type.__name__}, got {type(st).__name__})", to_state="ModelTrainDeps", missing_dependencies=[name])
            return st
        cl = _get(CleanProtocolState.NAME, CleanProtocolState)
        ms = _get(ModelSelectionState.NAME, ModelSelectionState)
        
        if cl.payload.summary is None:
            raise StateDependencyError(f"ModelTrainDeps: {CleanProtocolState.NAME} is not DONE yet (missing clean dataset summary)", to_state="ModelTrainDeps", missing_dependencies=[CleanProtocolState.NAME])
        if cl.payload.compiled_causal_spec is None:
            raise StateDependencyError(f"ModelTrainDeps: {CleanProtocolState.NAME} is not DONE yet (missing compiled causal spec)", to_state="ModelTrainDeps", missing_dependencies=[CleanProtocolState.NAME])
        if ms.payload.confirmed_model_selection is None or ms.payload.confirmed_model_selection.selected_model is None:
            raise StateDependencyError(f"ModelTrainDeps: {ModelSelectionState.NAME} is not DONE yet (missing confirmed model selection)", to_state="ModelTrainDeps", missing_dependencies=[ModelSelectionState.NAME])
        if cl.payload.clean_dataset_id is None:
            raise StateDependencyError(f"ModelTrainDeps: {CleanProtocolState.NAME} is not DONE yet (missing dataset id)", to_state="ModelTrainDeps", missing_dependencies=[CleanProtocolState.NAME])
        
        return cls(
            causal_specs=cl.payload.compiled_causal_spec,
            dataset_summary=cl.payload.summary,
            dataset_id=cl.payload.clean_dataset_id,
            selected_model=ms.payload.confirmed_model_selection.selected_model,
        )