from __future__ import annotations

from dataclasses import field, dataclass

from python.domain.models.errors import StateDependencyError
from python.domain.models.validation import ValidationIssueModel
from python.domain.workflows.ochestrator_state import ReadOnlyOchestratorState
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec

@dataclass(frozen=True, slots=True)
class ModelSelectionDeps:
    causal_spec: CausalSpec
    data_transformation_plan: TransformPlan
    validation_issues: list[ValidationIssueModel] = field(default_factory=list)

    @classmethod
    def from_loaded(cls, ready_only_ochestration_state: ReadOnlyOchestratorState) -> ModelSelectionDeps:
        causal_spec = ready_only_ochestration_state.get("causal_spec")
        data_transformation_plan = ready_only_ochestration_state.get("data_transformation_plan")
        validation_issues: list[ValidationIssueModel] = ready_only_ochestration_state.get("validation_issues") or []
        if causal_spec is None or data_transformation_plan is None:
            raise StateDependencyError(
                "MODEL_SELECTION",
                "MODEL_SELECTION",
                ["COMPILE_AND_VALIDATE"],
            )
        return cls(
            causal_spec=causal_spec,
            data_transformation_plan=data_transformation_plan,
            validation_issues=validation_issues,
        )
        
      
