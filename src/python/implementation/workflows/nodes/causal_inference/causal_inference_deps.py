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
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainState
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel

T = TypeVar("T", bound=State)


@dataclass(frozen=True, slots=True)
class CausalInferenceDeps:
    causal_specs: CausalSpec
    dataset_summary: DatasetSummaryModel
    dataset_id: UUID
    trained_model_id: UUID
    selected_model: str
    column_transformation_plan: TransformPlan | None
    order_effect_modifiers: Sequence[str] | None
    order_covariates: Sequence[str] | None

    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (
            CleanProtocolState.NAME,
            ModelSelectionState.NAME,
            ModelTrainState.NAME,
        )

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> CausalInferenceDeps:
        def _get(name: str, expected_type: type[T]) -> T:
            st = loaded.get(name)
            if st is None:
                raise StateDependencyError(f"CausalInferenceDeps: missing {name}", to_state="CausalInferenceDeps", missing_dependencies=[
                    name,
                ])
            if not isinstance(st, expected_type):
                raise StateDependencyError(f"CausalInferenceDeps: invalid {name} (expected {expected_type.__name__}, got {type(st).__name__})", to_state="CausalInferenceDeps", missing_dependencies=[name])
            return st
        cl = _get(CleanProtocolState.NAME, CleanProtocolState)
        ms = _get(ModelSelectionState.NAME, ModelSelectionState)
        mt = _get(ModelTrainState.NAME, ModelTrainState)

        if cl.payload.summary is None:
            raise StateDependencyError(f"CausalInferenceDeps: {CleanProtocolState.NAME} is not DONE yet (missing clean dataset summary)", to_state="CausalInferenceDeps", missing_dependencies=[CleanProtocolState.NAME])
        if cl.payload.compiled_causal_spec is None:
            raise StateDependencyError(f"CausalInferenceDeps: {CleanProtocolState.NAME} is not DONE yet (missing compiled causal spec)", to_state="CausalInferenceDeps", missing_dependencies=[CleanProtocolState.NAME])
        if cl.payload.clean_dataset_id is None:
            raise StateDependencyError(f"CausalInferenceDeps: {CleanProtocolState.NAME} is not DONE yet (missing dataset id)", to_state="CausalInferenceDeps", missing_dependencies=[CleanProtocolState.NAME])
        if ms.payload.confirmed_model_selection is None or ms.payload.confirmed_model_selection.selected_model is None:
            raise StateDependencyError(f"CausalInferenceDeps: {ModelSelectionState.NAME} is not DONE yet (missing confirmed model selection)", to_state="CausalInferenceDeps", missing_dependencies=[ModelSelectionState.NAME])
        if mt.payload.trained_model_id is None:
            raise StateDependencyError(f"CausalInferenceDeps: {ModelTrainState.NAME} is not DONE yet (missing trained model id)", to_state="CausalInferenceDeps", missing_dependencies=[ModelTrainState.NAME])
        
        
        return cls(
            causal_specs=cl.payload.compiled_causal_spec,
            dataset_summary=cl.payload.summary,
            dataset_id=cl.payload.clean_dataset_id,
            trained_model_id=mt.payload.trained_model_id,
            selected_model=ms.payload.confirmed_model_selection.selected_model,
            column_transformation_plan=mt.payload.column_transformation_plan,
            order_effect_modifiers=mt.payload.order_effect_modifiers,
            order_covariates=mt.payload.order_covariates,
        )