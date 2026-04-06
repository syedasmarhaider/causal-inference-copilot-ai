from __future__ import annotations
from collections.abc import Sequence
from typing import Mapping
from uuid import UUID

from python.domain.models.models import ChatMessage
from python.domain.repo.analytics_repo import AnalyticsRepo
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.state import State
from python.implementation.workflows.nodes.causal_inference.causal_inference_node import (
    CausalInferenceNode,
)
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import CausalInferenceState
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_node import (
    CompileAndValidateNode,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_state import CompileAndValidateState
from python.implementation.workflows.nodes.dataset.dataset_node import DatasetNode
from python.implementation.workflows.nodes.dataset.dataset_state import DatasetState
from python.implementation.workflows.nodes.model_selection.mode_selection_state import ModelSelectionState
from python.implementation.workflows.nodes.model_selection.model_selection_node import (
    ModelSelectionNode,
)
from python.implementation.workflows.nodes.model_train.model_train_node import ModelTrainNode
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainState
from python.implementation.workflows.nodes.noop_done.noop_done_node import NoopDoneNode
from python.implementation.workflows.nodes.noop_done.noop_done_state import NoopDoneState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import ProtocolDiscussionNode
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import ProtocolDiscussionState
from python.implementation.workflows.ochestrator.ochestrator_global_state import OchestratorReadOnlyGlobalState, OchestratorWritableGlobalState
from python.implementation.workflows.tools.tools_factory import DefaultToolFactory






class Ochestrator:
    _llm: LLMService
    workflow_repo: WorkflowStateRepo
    def __init__(
        self,
        workflow_repo: WorkflowStateRepo,
        llm: LLMService,
        data_repo: DataRepo,
        models_repo: ModelsRepo,
        analytics_repo: AnalyticsRepo,
    ) -> None:
        self.nodes_by_name = init_all_nodoes_with_name_as_key(
            llm=llm,
            data_repo=data_repo,
            models_repo=models_repo,
            analytics_repo=analytics_repo,
        )
        self.next_state_names_by_current_state_name = init_next_state_names()
        self.recoverable_states_map = recoverable_states_map()
        self.node_name_to_description = get_node_name_with_description()
        
        
        
    def answer(self, *, conversation_id: UUID, user_id: UUID, user_message: ChatMessage) -> ChatMessage:
        # TODO: implement tnx and lock later
        self.workflow_repo.append_message(user_id=user_id,
                                          conversation_id=conversation_id,
                                          message=user_message)
        
        messages_history = self.workflow_repo.load_message_history(user_id=user_id, conversation_id=conversation_id, limit=15)
        ochestrator_state = self.workflow_repo.load_ochestrator_state(user_id=user_id, conversation_id=conversation_id)
        if ochestrator_state is None:
            ochestrator_state = OchestratorWritableGlobalState.init_empty()
            self.workflow_repo.store_ochestrator_state(user_id=user_id, conversation_id=conversation_id, state=ochestrator_state)
        
        if not isinstance(ochestrator_state, OchestratorWritableGlobalState):    
            raise ValueError("Ochestrator state should be of type OchestratorWritableGlobalState")
        
        node_name_to_run = None
        
        raise NotImplementedError("Implement node name decision logic based on ochestrator state and messages history")
        

            
        
        
    def _get_next_node_name_to_run(self, *, user_id: UUID, conversation_id: UUID, ochestrator_state: OchestratorWritableGlobalState) -> str:
        last_active_node_name = ochestrator_state.get_last_active_node_name()
        if last_active_node_name is None:
            return DatasetNode.NAME
            
        last_state = self.workflow_repo.load_state(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=last_active_node_name,
         )
            
        if last_state is None:
            raise ValueError(f"State for last active node {last_active_node_name} not found")
        
        if last_state.status() == "DONE" or last_state.status() == "FREEZED":
               next_state = self.next_state_names_by_current_state_name.get(last_active_node_name)
               return self.next_state_names_by_current_state_name.get(last_active_node_name, DatasetNode.NAME)
            
            
        
        
        
        current_state = self.workflow_repo.load_current_state(conversation_id)
        global_state = self.workflow_repo.load_global_state(conversation_id)

        needs_node_name = self._needs_node_name(global_state)

        node = self.nodes_by_name[needs_node_name]

        new_state = node.run(
            user_id=user_id,
            conversation_id=conversation_id,
            state=current_state,
            readonly_global_state=global_state,
            messages_history=None,  # TODO: pass messages history
        )

        self.workflow_repo.save_new_state(conversation_id, new_state)

        next_node_name = self.next_state_names_by_current_state_name[new_state.name()]
        next_node_description = (
            self.node_name_to_description[next_node_name]
            if next_node_name is not None
            else "No next node, workflow is done"
        )

        return f"State updated to {new_state.name()}. Next step: {next_node_description}."

    
    
    
    
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
             





def init_next_state_names() -> Mapping[str, str | None]:
    return {
        ProtocolDiscussionState.NAME: DatasetState.NAME,
        DatasetState.NAME: CompileAndValidateNode.NAME,
        CompileAndValidateState.NAME: ModelSelectionState.NAME,
        ModelSelectionState.NAME: ModelTrainState.NAME,
        ModelTrainState.NAME: CausalInferenceState.NAME,
        CausalInferenceState.NAME: NoopDoneState.NAME,
    }

def recoverable_states_map() -> Mapping[str, Sequence[str]]:
    return {
        DatasetState.NAME: (DatasetState.NAME,),
        ProtocolDiscussionState.NAME: (ProtocolDiscussionState.NAME,),
        CompileAndValidateState.NAME: (
            DatasetState.NAME,
            ProtocolDiscussionState.NAME,
        ),
        ModelSelectionState.NAME: (ProtocolDiscussionState.NAME,),
        ModelTrainState.NAME: (
            ProtocolDiscussionState.NAME,
            ModelSelectionState.NAME,
        ),
        CausalInferenceState.NAME: (
            ModelSelectionState.NAME,
            ProtocolDiscussionState.NAME,
        ),
        NoopDoneState.NAME: (NoopDoneState.NAME,),
    }


def get_node_name_with_description() -> Mapping[str, str]:
    return {
        DatasetNode.NAME: DatasetNode.get_info(),
        ProtocolDiscussionNode.NAME: ProtocolDiscussionNode.get_info(),
        CompileAndValidateNode.NAME: CompileAndValidateNode.get_info(),
        ModelSelectionNode.NAME: ModelSelectionNode.get_info(),
        ModelTrainNode.NAME: ModelTrainNode.get_info(),
        CausalInferenceNode.NAME: CausalInferenceNode.get_info(),
        NoopDoneNode.NAME: NoopDoneNode.get_info(),
    }


def build_state_classes_by_name() -> Mapping[str, type[State]]:
    return {
        DatasetState.NAME: DatasetState,
        ProtocolDiscussionState.NAME: ProtocolDiscussionState,
        CompileAndValidateState.NAME: CompileAndValidateState,
        ModelSelectionState.NAME: ModelSelectionState,
        ModelTrainState.NAME: ModelTrainState,
        CausalInferenceState.NAME: CausalInferenceState,
        NoopDoneState.NAME: NoopDoneState,
    }



def init_all_nodoes_with_name_as_key(
    llm: LLMService,
    data_repo: DataRepo,
    models_repo: ModelsRepo,
    analytics_repo: AnalyticsRepo,
) -> dict[str, Node]:
    tool_factory = DefaultToolFactory(
        data_repo=data_repo,
        models_repo=models_repo,
        analytics_repo=analytics_repo,
        llm_service=llm,
    )

    dataset_node = DatasetNode(data_repo=data_repo, llm=llm, tools_factory=tool_factory)
    protocol_discussion_node = ProtocolDiscussionNode(llm=llm)
    compile_and_validate_node = CompileAndValidateNode(
        llm=llm,
        data_repo=data_repo,
        tool_factory=tool_factory,
    )
    model_selection_node = ModelSelectionNode(llm=llm, tool_factory=tool_factory)
    model_train_node = ModelTrainNode(
        llm=llm,
        data_repo=data_repo,
        tool_factory=tool_factory,
    )
    causal_inference_node = CausalInferenceNode(
        llm=llm,
        data_repo=data_repo,
        tool_factory=tool_factory,
    )
    done_node = NoopDoneNode()

    return {
        dataset_node.name: dataset_node,
        protocol_discussion_node.name: protocol_discussion_node,
        compile_and_validate_node.name: compile_and_validate_node,
        model_selection_node.name: model_selection_node,
        model_train_node.name: model_train_node,
        causal_inference_node.name: causal_inference_node,
        done_node.name: done_node,
    }
    
