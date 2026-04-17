from __future__ import annotations

from uuid import uuid4

import pytest

from python.implementation.workflows.nodes.data_compilation.data_compilation_node import (
    DataCompilationNode,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_state import (
    DataCompilationState,
)
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_state import (
    DataManupulationState,
)
from python.implementation.workflows.nodes.model_selection.mode_selection_state import (
    ModelSelectionState,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)
from python.implementation.workflows.ochestrator.writable_ochestrator_state import (
    WritableOchestratorState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


def _summary(n_rows: int) -> DatasetSummaryModel:
    return DatasetSummaryModel(n_rows=n_rows, profiles=[])


def _causal_spec() -> CausalSpec:
    return CausalSpec.model_validate(
        {
            "treatment_spec": {
                "kind": "binary",
                "column": "treatment",
                "treated": "drug",
                "control": "control",
            },
            "outcome_spec": {
                "kind": "binary",
                "column": "outcome",
                "event": "event",
                "non_event": "non_event",
            },
            "covariates": ["age"],
            "effect_modifiers": ["isex"],
            "experiment_type": "RCT",
        }
    )


def _transform_plan() -> TransformPlan:
    return TransformPlan.model_validate(
        {
            "columns": [
                {
                    "column": "age",
                    "role": "covariate",
                    "encoding": {"preset": "num_standard"},
                }
            ]
        }
    )


def test_data_compilation_dataset_only_publish_preserves_protocol_and_invalidates_downstream() -> (
    None
):
    state = WritableOchestratorState.init_empty()
    source_dataset_id = uuid4()
    compiled_dataset_id = uuid4()
    repaired_dataset_id = uuid4()

    state.set(
        DataManupulationState.NAME,
        {
            "working_dataset_id": source_dataset_id,
            "latest_dataset_summary": _summary(100),
        },
    )
    state.set(
        ProtocolDiscussionState.NAME,
        {
            "protocol_discussion": "Confirmed protocol discussion.",
            "protocol_cleaning_instructions": "Keep treatment, outcome, age, and isex.",
        },
    )
    state.set(
        DataCompilationState.NAME,
        {
            "working_dataset_id": compiled_dataset_id,
            "latest_dataset_summary": _summary(90),
            "causal_spec": _causal_spec(),
            "data_transformation_plan": _transform_plan(),
            "validation_issues": [],
            "is_validated": True,
        },
    )
    state.set(
        ModelSelectionState.NAME,
        {
            "selected_model": "dr_learner",
            "selection_reasoning": "Reasonable baseline choice.",
        },
    )

    state.set(
        DataCompilationState.NAME,
        {
            "working_dataset_id": repaired_dataset_id,
            "latest_dataset_summary": _summary(88),
            "publish_dataset_only": True,
        },
    )

    assert state.get("working_dataset_id") == repaired_dataset_id
    assert state.get("latest_dataset_summary").model_dump(mode="json") == _summary(88).model_dump(
        mode="json"
    )
    assert state.get("protocol_discussion") == "Confirmed protocol discussion."
    assert (
        state.get("protocol_cleaning_instructions")
        == "Keep treatment, outcome, age, and isex."
    )
    assert state.get("causal_spec") is None
    assert state.get("data_transformation_plan") is None
    assert state.get("working_dataset_frozen") is False
    assert state.get("validation_issues") == []
    assert state.get("is_validated") is False
    assert state.get("selected_model") is None
    assert state.get("selection_reasoning") is None
    assert state.get_current_node_name() == DataCompilationNode.NAME


def test_data_compilation_dataset_only_publish_allows_partial_payload() -> None:
    state = WritableOchestratorState.init_empty()
    source_dataset_id = uuid4()
    repaired_dataset_id = uuid4()

    state.set(
        DataManupulationState.NAME,
        {
            "working_dataset_id": source_dataset_id,
            "latest_dataset_summary": _summary(50),
        },
    )
    state.set(
        ProtocolDiscussionState.NAME,
        {
            "protocol_discussion": "Confirmed protocol discussion.",
            "protocol_cleaning_instructions": "Normalize treatment and outcome values.",
        },
    )

    state.set(
        DataCompilationState.NAME,
        {
            "working_dataset_id": repaired_dataset_id,
            "latest_dataset_summary": _summary(48),
            "publish_dataset_only": True,
        },
    )

    assert state.get("working_dataset_id") == repaired_dataset_id
    assert state.get("causal_spec") is None
    assert state.get("data_transformation_plan") is None


def test_data_compilation_full_publish_still_requires_full_payload() -> None:
    state = WritableOchestratorState.init_empty()
    source_dataset_id = uuid4()

    state.set(
        DataManupulationState.NAME,
        {
            "working_dataset_id": source_dataset_id,
            "latest_dataset_summary": _summary(25),
        },
    )
    state.set(
        ProtocolDiscussionState.NAME,
        {
            "protocol_discussion": "Confirmed protocol discussion.",
            "protocol_cleaning_instructions": "Keep the study columns only.",
        },
    )

    with pytest.raises(KeyError, match="DATA_COMPILATION updates must include"):
        state.set(
            DataCompilationState.NAME,
            {
                "working_dataset_id": uuid4(),
                "latest_dataset_summary": _summary(24),
            },
        )
