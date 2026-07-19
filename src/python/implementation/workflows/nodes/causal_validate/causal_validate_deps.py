from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)


@dataclass(frozen=True)
class CausalValidateDeps:
    """The same confirmed upstream context required by causal inference.

    ``trained_model_id`` is a dependency marker: outer-CV validation fits temporary models
    for its folds and does not call the already trained model.  Requiring it guarantees that
    validation is
    only available after normal training, and lets the node invalidate cached
    validation after a retrain.
    """

    dataset_id: UUID
    dataset_summary: DatasetSummaryModel
    causal_spec: CausalSpec
    transformation_plan: TransformPlan
    selected_model: str
    trained_model_id: UUID

    @classmethod
    def from_request(cls, request: NodeRequest) -> CausalValidateDeps:
        dataset_id_raw = request.orchestrator_state.get("working_dataset_id")
        dataset_summary_raw = request.orchestrator_state.get("latest_dataset_summary")
        causal_spec_raw = request.orchestrator_state.get("causal_spec")
        transformation_plan_raw = request.orchestrator_state.get("data_transformation_plan")
        selected_model_raw = request.orchestrator_state.get("selected_model")
        trained_model_id_raw = request.orchestrator_state.get("trained_model_id")

        required_values = {
            "dataset_id": dataset_id_raw,
            "dataset_summary": dataset_summary_raw,
            "causal_spec": causal_spec_raw,
            "transformation_plan": transformation_plan_raw,
            "selected_model": selected_model_raw,
            "trained_model_id": trained_model_id_raw,
        }
        missing = [name for name, value in required_values.items() if value is None]
        if missing:
            raise ValueError(
                "CausalValidateDeps requires confirmed upstream values: " + ", ".join(missing)
            )

        if not isinstance(dataset_id_raw, UUID):
            raise TypeError("CausalValidateDeps: dataset_id must be a UUID")
        if not isinstance(dataset_summary_raw, DatasetSummaryModel):
            raise TypeError("CausalValidateDeps: dataset_summary must be a DatasetSummaryModel")
        if not isinstance(causal_spec_raw, CausalSpec):
            raise TypeError("CausalValidateDeps: causal_spec must be a CausalSpec")
        if not isinstance(transformation_plan_raw, TransformPlan):
            raise TypeError("CausalValidateDeps: transformation_plan must be a TransformPlan")
        if not isinstance(selected_model_raw, str):
            raise TypeError("CausalValidateDeps: selected_model must be a string")
        if not isinstance(trained_model_id_raw, UUID):
            raise TypeError("CausalValidateDeps: trained_model_id must be a UUID")

        return cls(
            dataset_id=dataset_id_raw,
            dataset_summary=dataset_summary_raw,
            causal_spec=causal_spec_raw,
            transformation_plan=transformation_plan_raw,
            selected_model=selected_model_raw,
            trained_model_id=trained_model_id_raw,
        )


__all__ = ["CausalValidateDeps"]
