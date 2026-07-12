from __future__ import annotations

from uuid import uuid4

import pytest

from python.implementation.workflows.nodes.data_compilation.data_compilation_node import (
    DataCompilationNode,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_state import (
    DataCompilationPayloadModel,
    DataCompilationState,
)
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_state import (
    DataManupulationState,
)
from python.implementation.workflows.nodes.model_selection.mode_selection_state import (
    ModelSelectionState,
)
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)
from python.implementation.workflows.nodes.shap_explanation.shap_explanation_state import (
    ShapExplanationState,
)
from python.implementation.workflows.ochestrator.causal_ochestrator_state import (
    CausalOchestratorState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import CausalSpecDraft
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
            "id_col": "auto_id",
        }
    )


def _causal_draft() -> CausalSpecDraft:
    return CausalSpecDraft(
        treatment_column="treatment",
        outcome_column="outcome",
        covariates=["age"],
        effect_modifiers=["isex"],
        target_population="all rows",
        study_type="RCT",
        time_zero="baseline treatment assignment",
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


def _complete_state_through_training() -> CausalOchestratorState:
    state = CausalOchestratorState.init_empty()
    state.set(
        DataManupulationState.NAME,
        {
            "working_dataset_id": uuid4(),
            "latest_dataset_summary": _summary(100),
        },
    )
    state.set(
        ProtocolDiscussionState.NAME,
        {"causal_spec_draft": _causal_draft()},
    )
    state.set(
        DataCompilationState.NAME,
        {
            "working_dataset_id": uuid4(),
            "latest_dataset_summary": _summary(90),
            "causal_spec_draft": _causal_draft(),
            "causal_spec": _causal_spec(),
            "data_transformation_plan": _transform_plan(),
            "working_dataset_frozen": True,
            "validation_issues": [],
            "is_validated": True,
        },
    )
    state.set(
        ModelSelectionState.NAME,
        {
            "selected_model": "econml.dr.ForestDRLearner",
            "selection_reasoning": "Selected for heterogeneity.",
        },
    )
    state.set(
        ModelTrainState.NAME,
        {
            "trained_model_id": uuid4(),
            "training_warnings": [],
            "training_spec": {"selected_model": "econml.dr.ForestDRLearner"},
        },
    )
    return state


def test_shap_explanation_update_is_allowed_after_training() -> None:
    state = _complete_state_through_training()
    shap_dataset_id = uuid4()

    state.set(
        ShapExplanationState.NAME,
        {
            "shap_values_dataset_id": shap_dataset_id,
            "shap_values_summary": {"status": "COMPLETED", "row_count": 10},
            "shap_values_source_signature": "signature",
        },
    )

    assert state.get("shap_values_dataset_id") == shap_dataset_id
    assert state.get("shap_values_summary") == {"status": "COMPLETED", "row_count": 10}
    assert state.get("shap_values_source_signature") == "signature"


def test_shap_explanation_update_requires_trained_model() -> None:
    state = CausalOchestratorState.init_empty()

    with pytest.raises(ValueError, match="Stage 5 incomplete"):
        state.set(
            ShapExplanationState.NAME,
            {
                "shap_values_dataset_id": uuid4(),
                "shap_values_summary": {"status": "COMPLETED"},
                "shap_values_source_signature": "signature",
            },
        )


def test_data_compilation_dataset_only_publish_preserves_draft_and_invalidates_downstream() -> None:
    state = CausalOchestratorState.init_empty()
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
        {"causal_spec_draft": _causal_draft()},
    )
    state.set(
        DataCompilationState.NAME,
        {
            "working_dataset_id": compiled_dataset_id,
            "latest_dataset_summary": _summary(90),
            "causal_spec_draft": _causal_draft(),
            "causal_spec": _causal_spec(),
            "data_transformation_plan": _transform_plan(),
            "working_dataset_frozen": True,
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
            "causal_spec_draft": _causal_draft(),
            "publish_dataset_only": True,
        },
    )

    assert state.get("working_dataset_id") == repaired_dataset_id
    assert state.get("latest_dataset_summary").model_dump(mode="json") == _summary(88).model_dump(
        mode="json"
    )
    with pytest.raises(KeyError):
        state.get("protocol_discussion")
    with pytest.raises(KeyError):
        state.get("protocol_cleaning_instructions")
    assert state.get("causal_spec_draft").model_dump(mode="json") == _causal_draft().model_dump(
        mode="json"
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
    state = CausalOchestratorState.init_empty()
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
        {"causal_spec_draft": _causal_draft()},
    )

    state.set(
        DataCompilationState.NAME,
        {
            "working_dataset_id": repaired_dataset_id,
            "latest_dataset_summary": _summary(48),
        },
    )

    assert state.get("working_dataset_id") == repaired_dataset_id
    assert state.get("causal_spec_draft").model_dump(mode="json") == _causal_draft().model_dump(
        mode="json"
    )
    assert state.get("causal_spec") is None
    assert state.get("data_transformation_plan") is None


def test_data_compilation_acceptance_still_requires_full_payload() -> None:
    state = CausalOchestratorState.init_empty()
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
        {"causal_spec_draft": _causal_draft()},
    )

    with pytest.raises(KeyError, match="DATA_COMPILATION acceptance updates must include"):
        state.set(
            DataCompilationState.NAME,
            {
                "working_dataset_id": uuid4(),
                "latest_dataset_summary": _summary(24),
                "causal_spec_draft": _causal_draft(),
                "causal_spec": _causal_spec(),
            },
        )


def test_data_compilation_confirmation_does_not_duplicate_preview_dataset_id() -> None:
    state = CausalOchestratorState.init_empty()
    source_dataset_id = uuid4()
    preview_dataset_id = uuid4()

    state.set(
        DataManupulationState.NAME,
        {
            "working_dataset_id": source_dataset_id,
            "latest_dataset_summary": _summary(100),
        },
    )
    state.set(ProtocolDiscussionState.NAME, {"causal_spec_draft": _causal_draft()})
    state.set(
        DataCompilationState.NAME,
        {
            "working_dataset_id": preview_dataset_id,
            "latest_dataset_summary": _summary(90),
            "causal_spec_draft": _causal_draft(),
        },
    )

    dataset_ids_after_preview = state.get("working_dataset_ids")
    assert dataset_ids_after_preview[-2:] == [source_dataset_id, preview_dataset_id]
    assert state.get_current_node_name() == DataCompilationNode.NAME

    state.set(
        DataCompilationState.NAME,
        {
            "working_dataset_id": preview_dataset_id,
            "latest_dataset_summary": _summary(90),
            "causal_spec_draft": _causal_draft(),
            "causal_spec": _causal_spec(),
            "data_transformation_plan": _transform_plan(),
            "working_dataset_frozen": True,
            "validation_issues": [],
            "is_validated": True,
        },
    )

    assert state.get("working_dataset_ids") == dataset_ids_after_preview
    assert state.get("working_dataset_id") == preview_dataset_id
    assert state.get("working_dataset_frozen") is True
    assert state.get("is_validated") is True


def test_data_compilation_revert_request_pops_preview_dataset_id() -> None:
    state = CausalOchestratorState.init_empty()
    source_dataset_id = uuid4()
    preview_dataset_id = uuid4()
    source_summary = _summary(100)

    state.set(
        DataManupulationState.NAME,
        {
            "working_dataset_id": source_dataset_id,
            "latest_dataset_summary": source_summary,
        },
    )
    state.set(ProtocolDiscussionState.NAME, {"causal_spec_draft": _causal_draft()})
    state.set(
        DataCompilationState.NAME,
        {
            "working_dataset_id": preview_dataset_id,
            "latest_dataset_summary": _summary(90),
            "causal_spec_draft": _causal_draft(),
        },
    )

    state.set(
        DataCompilationState.NAME,
        {
            "working_dataset_id": source_dataset_id,
            "latest_dataset_summary": source_summary,
            "causal_spec_draft": _causal_draft(),
            "revert_request": True,
        },
    )

    assert state.get("working_dataset_id") == source_dataset_id
    assert state.get("working_dataset_ids")[-1] == source_dataset_id
    assert preview_dataset_id not in state.get("working_dataset_ids")
    assert state.get("latest_dataset_summary").model_dump(mode="json") == source_summary.model_dump(
        mode="json"
    )
    assert state.get("causal_spec") is None
    assert state.get("data_transformation_plan") is None
    assert state.get("working_dataset_frozen") is False
    assert state.get_current_node_name() == DataCompilationNode.NAME


def test_data_compilation_payload_tracks_source_and_drops_stale_fields() -> None:
    source_dataset_id = uuid4()
    preview_dataset_id = uuid4()
    payload = DataCompilationPayloadModel.model_validate(
        {
            "source_dataset_id": source_dataset_id,
            "source_dataset_summary": _summary(100),
            "source_causal_spec_draft": _causal_draft().model_dump(mode="json"),
            "compiled_dataset_id": preview_dataset_id,
            "compiled_dataset_summary": _summary(90),
            "cleaning_summary": "Cleaning summary",
            "source_protocol_discussion": "old field",
            "source_protocol_cleaning_instructions": "old field",
            "missingness_decisions": {"decisions": []},
        }
    )

    assert payload.source_dataset_id == source_dataset_id
    assert payload.source_dataset_summary is not None
    assert payload.source_causal_spec_draft is not None
    assert payload.compiled_dataset_id == preview_dataset_id
    assert payload.cleaning_summary == "Cleaning summary"
