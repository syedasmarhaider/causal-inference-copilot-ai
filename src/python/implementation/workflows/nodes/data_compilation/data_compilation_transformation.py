




from dataclasses import dataclass

from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


@dataclass(frozen=True)
class TransformationResult:
    transformation_plan: TransformPlan | None
    required_dataset_changes: str | None
    addtional_suggestions_to_user: str
    


def transform(
    *,
    transformation_instructions: str,
    causal_spec: CausalSpec,
    data_summary: DatasetSummaryModel,
) -> TransformationResult:
    