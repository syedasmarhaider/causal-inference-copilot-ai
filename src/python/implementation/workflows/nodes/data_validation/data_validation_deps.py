from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec


@dataclass(frozen=True)
class DataValidationDeps:
    dataset_id: UUID | None
    causal_spec: CausalSpec | None
    transformation_plan: TransformPlan | None

    @classmethod
    def from_request(cls, request: NodeRequest) -> DataValidationDeps:
        dataset_id_raw: Any = request.orchestrator_state.get("working_dataset_id")
        causal_spec_raw: Any = request.orchestrator_state.get("causal_spec")
        transformation_plan_raw: Any = request.orchestrator_state.get("data_transformation_plan")

        if dataset_id_raw is not None and not isinstance(dataset_id_raw, UUID):
            raise TypeError("dataset_id must be a UUID if provided")
        if causal_spec_raw is not None and not isinstance(causal_spec_raw, CausalSpec):
            raise TypeError("causal_spec must be of type CausalSpec if provided")
        if transformation_plan_raw is not None and not isinstance(transformation_plan_raw, TransformPlan):
            raise TypeError("transformation_plan must be of type TransformPlan if provided")
    
        return cls(
            dataset_id=dataset_id_raw,
            causal_spec=causal_spec_raw,
            transformation_plan=transformation_plan_raw,
        )


__all__ = ["DataValidationDeps"]
