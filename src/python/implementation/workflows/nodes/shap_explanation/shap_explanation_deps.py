from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)
from python.implementation.workflows.utils.utils import uuid_from_any


@dataclass(frozen=True)
class ShapExplanationDeps:
    dataset_id: UUID
    dataset_summary: DatasetSummaryModel
    causal_spec: CausalSpec
    transformation_plan: TransformPlan
    selected_model: str
    trained_model_id: UUID
    shap_values_dataset_id: UUID | None = None
    shap_values_summary: dict[str, Any] | None = None
    shap_values_source_signature: str | None = None

    @classmethod
    def from_request(cls, request: NodeRequest) -> ShapExplanationDeps:
        dataset_id_raw = request.orchestrator_state.get("working_dataset_id")
        dataset_summary_raw = request.orchestrator_state.get("latest_dataset_summary")
        causal_spec_raw = request.orchestrator_state.get("causal_spec")
        transformation_plan_raw = request.orchestrator_state.get("data_transformation_plan")
        selected_model_raw = request.orchestrator_state.get("selected_model")
        trained_model_id_raw = request.orchestrator_state.get("trained_model_id")
        shap_values_dataset_id_raw = _get_optional_state_value(
            request,
            "shap_values_dataset_id",
        )
        shap_values_summary_raw = _get_optional_state_value(
            request,
            "shap_values_summary",
        )
        shap_values_source_signature_raw = _get_optional_state_value(
            request,
            "shap_values_source_signature",
        )

        if dataset_id_raw is None:
            raise ValueError("ShapExplanationDeps: dataset_id is required")
        if dataset_summary_raw is None:
            raise ValueError("ShapExplanationDeps: dataset_summary is required")
        if causal_spec_raw is None:
            raise ValueError("ShapExplanationDeps: causal_spec is required")
        if transformation_plan_raw is None:
            raise ValueError("ShapExplanationDeps: transformation_plan is required")
        if selected_model_raw is None:
            raise ValueError("ShapExplanationDeps: selected_model is required")
        if trained_model_id_raw is None:
            raise ValueError("ShapExplanationDeps: trained_model_id is required")

        if not isinstance(dataset_id_raw, UUID):
            raise TypeError("ShapExplanationDeps: dataset_id must be a UUID")
        if not isinstance(dataset_summary_raw, DatasetSummaryModel):
            raise TypeError("ShapExplanationDeps: dataset_summary must be a DatasetSummaryModel")
        if not isinstance(causal_spec_raw, CausalSpec):
            raise TypeError("ShapExplanationDeps: causal_spec must be a CausalSpec")
        if not isinstance(transformation_plan_raw, TransformPlan):
            raise TypeError("ShapExplanationDeps: transformation_plan must be a TransformPlan")
        if not isinstance(selected_model_raw, str):
            raise TypeError("ShapExplanationDeps: selected_model must be a string")
        if not isinstance(trained_model_id_raw, UUID):
            raise TypeError("ShapExplanationDeps: trained_model_id must be a UUID")
        if shap_values_summary_raw is not None and not isinstance(
            shap_values_summary_raw,
            dict,
        ):
            raise TypeError("ShapExplanationDeps: shap_values_summary must be a dict")
        if shap_values_source_signature_raw is not None and not isinstance(
            shap_values_source_signature_raw,
            str,
        ):
            raise TypeError("ShapExplanationDeps: shap_values_source_signature must be a string")

        return cls(
            dataset_id=dataset_id_raw,
            dataset_summary=dataset_summary_raw,
            causal_spec=causal_spec_raw,
            transformation_plan=transformation_plan_raw,
            selected_model=selected_model_raw,
            trained_model_id=trained_model_id_raw,
            shap_values_dataset_id=uuid_from_any(shap_values_dataset_id_raw),
            shap_values_summary=shap_values_summary_raw,
            shap_values_source_signature=shap_values_source_signature_raw,
        )


def _get_optional_state_value(request: NodeRequest, key: str) -> Any:
    try:
        return request.orchestrator_state.get(key)
    except KeyError:
        return None


__all__ = ["ShapExplanationDeps"]
