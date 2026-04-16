from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.nodes.data_compilation.data_compilation_state import (
    DataCompilationState,
)
from python.implementation.workflows.nodes.model_selection.mode_selection_state import (
    ModelSelectionState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)


@dataclass(frozen=True)
class ModelTrainDeps:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel
    causal_spec: CausalSpec
    transformation_plan: TransformPlan
    selected_model: str

    @classmethod
    def from_request(cls, request: NodeRequest) -> ModelTrainDeps:
        compilation_raw = request.orchestrator_state.get(DataCompilationState.NAME)
        selection_raw = request.orchestrator_state.get(ModelSelectionState.NAME)

        if compilation_raw is not None and not isinstance(compilation_raw, Mapping):
            raise TypeError(
                "DATA_COMPILATION payload must be a dict with compiled dataset and setup values"
            )
        if selection_raw is not None and not isinstance(selection_raw, Mapping):
            raise TypeError(
                "MODEL_SELECTION payload must be a dict with selected model values"
            )

        compilation = cast(Mapping[str, Any] | None, compilation_raw)
        selection = cast(Mapping[str, Any] | None, selection_raw)

        dataset_id_raw = None if compilation is None else compilation.get("working_dataset_id")
        if dataset_id_raw is None:
            dataset_id = None
        elif isinstance(dataset_id_raw, UUID):
            dataset_id = dataset_id_raw
        else:
            raise TypeError("new_dataset_id must be a UUID")

        dataset_summary_raw = (
            None if compilation is None else compilation.get("latest_dataset_summary")
        )
        if dataset_summary_raw is None:
            dataset_summary = None
        elif isinstance(dataset_summary_raw, DatasetSummaryModel):
            dataset_summary = dataset_summary_raw
        elif isinstance(dataset_summary_raw, str):
            dataset_summary = DatasetSummaryModel.model_validate_json(dataset_summary_raw)
        else:
            dataset_summary = DatasetSummaryModel.model_validate(dataset_summary_raw)

        causal_spec_raw = None if compilation is None else compilation.get("causal_spec")
        if causal_spec_raw is None:
            causal_spec = None
        elif isinstance(causal_spec_raw, CausalSpec):
            causal_spec = causal_spec_raw
        else:
            causal_spec = CausalSpec.model_validate(causal_spec_raw)

        transformation_plan_raw = (
            None
            if compilation is None
            else compilation.get(
                "transformation_plan",
                compilation.get("data_transformation_plan"),
            )
        )
        if transformation_plan_raw is None:
            transformation_plan = None
        elif isinstance(transformation_plan_raw, TransformPlan):
            transformation_plan = transformation_plan_raw
        else:
            transformation_plan = TransformPlan.model_validate(transformation_plan_raw)

        selected_model_raw = None if selection is None else selection.get("selected_model")
        if selected_model_raw is None:
            selected_model = None
        elif isinstance(selected_model_raw, str):
            selected_model = selected_model_raw.strip() or None
        else:
            raise TypeError("selected_model must be a non-empty string or null")

        if dataset_id is None:
            raise ValueError("ModelTrainDeps: dataset_id is required but was not found in compilation state")
        if dataset_summary is None:
            raise ValueError("ModelTrainDeps: dataset_summary is required but was not found in compilation state")
        if causal_spec is None:
            raise ValueError("ModelTrainDeps: causal_spec is required but was not found in compilation state")
        if transformation_plan is None:
            raise ValueError("ModelTrainDeps: transformation_plan is required but was not found in compilation state")
        if selected_model is None:
            raise ValueError("ModelTrainDeps: selected_model is required but was not found in selection state")

        return cls(
            dataset_id=dataset_id,
            dataset_summary=dataset_summary,
            causal_spec=causal_spec,
            transformation_plan=transformation_plan,
            selected_model=selected_model,
        )


__all__ = ["ModelTrainDeps"]
