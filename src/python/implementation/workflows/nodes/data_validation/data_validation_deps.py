from __future__ import annotations

from collections.abc import Mapping
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
        raw_context = request.orchestrator_state.get(request.node_state.name())
        if raw_context is None:
            return cls(
                dataset_id=None,
                causal_spec=None,
                transformation_plan=None,
            )

        if not isinstance(raw_context, Mapping):
            raise TypeError(
                "DATA_VALIDATION dependency payload must be a dict with dataset_id, "
                "causal_spec, and transformation_plan"
            )

        dataset_id_raw: Any = raw_context.get("working_dataset_id")
        causal_spec_raw: Any = raw_context.get("causal_spec")
        transformation_plan_raw: Any = raw_context.get("data_transformation_plan")

        if dataset_id_raw is not None and not isinstance(dataset_id_raw, UUID):
            raise TypeError("dataset_id must be a UUID")
        if causal_spec_raw is not None and not isinstance(causal_spec_raw, CausalSpec):
            raise TypeError("causal_spec must be of type CausalSpec")
        if transformation_plan_raw is not None and not isinstance(
            transformation_plan_raw, TransformPlan
        ):
            raise TypeError("transformation_plan must be of type TransformPlan")

        return cls(
            dataset_id=dataset_id_raw,
            causal_spec=causal_spec_raw,
            transformation_plan=transformation_plan_raw,
        )


__all__ = ["DataValidationDeps"]
