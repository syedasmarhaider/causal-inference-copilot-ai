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
from python.implementation.workflows.nodes.model_train.model_train_state import (
    ModelTrainState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)
from python.implementation.workflows.utils.utils import uuid_from_any


@dataclass(frozen=True)
class CausalInferenceDeps:
    dataset_id: UUID | None
    dataset_summary: DatasetSummaryModel | None
    causal_spec: CausalSpec | None
    transformation_plan: TransformPlan | None
    selected_model: str | None
    trained_model_id: UUID | None

    @classmethod
    def from_request(cls, request: NodeRequest) -> CausalInferenceDeps:
        compilation_raw = request.orchestrator_state.get(DataCompilationState.NAME)
        selection_raw = request.orchestrator_state.get(ModelSelectionState.NAME)
        training_raw = request.orchestrator_state.get(ModelTrainState.NAME)

        if compilation_raw is not None and not isinstance(compilation_raw, Mapping):
            raise TypeError(
                "DATA_COMPILATION payload must be a dict with compiled dataset and setup values"
            )
        if selection_raw is not None and not isinstance(selection_raw, Mapping):
            raise TypeError(
                "MODEL_SELECTION payload must be a dict with selected model values"
            )
        if training_raw is not None and not isinstance(training_raw, Mapping):
            raise TypeError(
                "MODEL_TRAIN payload must be a dict with trained model values"
            )

        compilation = cast(Mapping[str, Any] | None, compilation_raw)
        selection = cast(Mapping[str, Any] | None, selection_raw)
        training = cast(Mapping[str, Any] | None, training_raw)

        dataset_id = uuid_from_any(
            None if compilation is None else compilation.get("new_dataset_id")
        )

        dataset_summary_raw = (
            None if compilation is None else compilation.get("new_dataset_summary")
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

        trained_model_id = uuid_from_any(
            None if training is None else training.get("trained_model_id")
        )

        return cls(
            dataset_id=dataset_id,
            dataset_summary=dataset_summary,
            causal_spec=causal_spec,
            transformation_plan=transformation_plan,
            selected_model=selected_model,
            trained_model_id=trained_model_id,
        )


__all__ = ["CausalInferenceDeps"]
