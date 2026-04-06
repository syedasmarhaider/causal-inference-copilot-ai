from __future__ import annotations

from python.implementation.workflows.nodes.causal_inference.causal_inference_node import (
    CausalInferenceNode,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_node import (
    CompileAndValidateNode,
)
from python.implementation.workflows.nodes.dataset.dataset_node import DatasetNode
from python.implementation.workflows.nodes.model_selection.model_selection_node import (
    ModelSelectionNode,
)
from python.implementation.workflows.nodes.model_train.model_train_node import ModelTrainNode
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import ProtocolDiscussionNode
from python.implementation.workflows.ochestrator.ochestrator_global_state import OchestratorReadOnlyGlobalState






class Ochestrator:
    def _needs_node_name(self, global_state: OchestratorReadOnlyGlobalState) -> str:
        working_dataset_id = global_state.get("working_dataset_id")
        working_dataset_summary = global_state.get("working_dataset_summary")
        protocol_discussed = bool(global_state.get("protocol_discussed"))
        working_dataset_froozen = bool(global_state.get("working_dataset_froozen"))
        causal_spec = global_state.get("causal_spec")
        data_transformation_plan = global_state.get("data_transformation_plan")
        validation_issues = global_state.get("validation_issues") or []
        validation_issues_accepted = bool(global_state.get("validation_issues_accepted"))
        selected_model = global_state.get("selected_model")
        model_training_id = global_state.get("model_training_id")

        if working_dataset_id is None:
            return DatasetNode.NAME

        if working_dataset_summary is None:
            return DatasetNode.NAME

        if not protocol_discussed:
            return ProtocolDiscussionNode.NAME

        if not working_dataset_froozen:
            return DatasetNode.NAME

        # Stage 4-5: compile and validate
        if causal_spec is None:
            return CompileAndValidateNode.NAME

        if data_transformation_plan is None:
            return CompileAndValidateNode.NAME

        if validation_issues and not validation_issues_accepted:
            return CompileAndValidateNode.NAME

        # Stage 6: model selection
        if selected_model is None:
            return ModelSelectionNode.NAME

        # Stage 7: model training
        if model_training_id is None:
            return ModelTrainNode.NAME

        # Final stage: inference
        return CausalInferenceNode.NAME
             
