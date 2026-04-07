from __future__ import annotations
from collections.abc import Sequence
from typing import Any, Literal, Mapping
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from python.domain.models.models import ChatMessage
from python.domain.repo.analytics_repo import AnalyticsRepo
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import LLMConfig, LLMService
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
from python.implementation.workflows.ochestrator.ochestrator_prompts import OCHESTRATOR_ABORTED_SYSTEM_PROMPT, OCHESTRATOR_PENDING_ORCHESTRATOR_ANSWER_SYSTEM_PROMPT, OCHESTRATOR_PENDING_ROUTE_SYSTEM_PROMPT
from python.implementation.workflows.tools.tools_factory import DefaultToolFactory






class Ochestrator:
    _llm: LLMService
    _workflow_repo: WorkflowStateRepo
    _recoverable_candidates_map: Mapping[str, Sequence[str]]
    _node_name_to_description_map: Mapping[str, str]
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
        self._node_name_to_description_map = get_node_name_with_description()
        
        
        
    def answer(self, *, conversation_id: UUID, user_id: UUID, user_message: ChatMessage) -> ChatMessage:
        # TODO: implement tnx and lock later
        self._workflow_repo.append_message(user_id=user_id,
                                          conversation_id=conversation_id,
                                          message=user_message)
        
        messages_history = self._workflow_repo.load_message_history(user_id=user_id, conversation_id=conversation_id, limit=15)
        ochestrator_state = self._workflow_repo.load_ochestrator_state(user_id=user_id, conversation_id=conversation_id)
        if ochestrator_state is None:
            ochestrator_state = OchestratorWritableGlobalState.init_empty()
            self._workflow_repo.store_ochestrator_state(user_id=user_id, conversation_id=conversation_id, state=ochestrator_state)
        
        if not isinstance(ochestrator_state, OchestratorWritableGlobalState):    
            raise ValueError("Ochestrator state should be of type OchestratorWritableGlobalState")
        
        node_name_to_run = ochestrator_state.get_last_active_node_name()
        if node_name_to_run is None:
            node_name_to_run = DatasetNode.NAME
        
        
        
        raise NotImplementedError("Implement node name decision logic based on ochestrator state and messages history")
        

            
        
        

    def _decide_node_name_or_ochestration_on_pending(
        self,
        *,
        current_state: State,
        ochestrator_state: OchestratorReadOnlyGlobalState,
        messages_history: Sequence[ChatMessage] | None,
    ) -> str:
        last_2_messages = _last_two_messages(messages_history)
        current_state_name = current_state.name()
        latest_user_message = _latest_user_message(messages_history)
        if latest_user_message is None:
            return current_state_name

        latest_state_system_message = _latest_system_message(current_state.messages())
        current_node_description = self._node_name_to_description_map.get(
            current_state_name,
            "",
        )
        dataset_node_description = self._node_name_to_description_map.get(
            DatasetNode.NAME,
            "",
        )

        decision = self._llm.generate_json(
            schema=_PendingRouteDecision,
            system_prompt=OCHESTRATOR_PENDING_ROUTE_SYSTEM_PROMPT,
            user_prompt=(
                f"Current pending state: {current_state_name!r}\n"
                f"Current state description: {current_node_description!r}\n"
                f"Dataset state description: {dataset_node_description!r}\n"
                f"Latest user message: {latest_user_message!r}\n"
                f"Latest system message inside current state: {latest_state_system_message!r}\n"
                "Return the best route intent."
            ),
            config=LLMConfig(model="basic", temperature=0.1),
            history=last_2_messages,
            max_attempts=2,
        )

        match decision.route_intent:
            case "CURRENT_STATE":
                return current_state_name

            case "DATASET":
                return DatasetState.NAME

            case "ORCHESTRATOR_ANSWER":
                return "ORCHESTRATOR_ANSWER"

        raise ValueError(
            f"Unsupported pending route intent: {decision.route_intent!r}"
        )
    
    def _answer_orchestrator_question_on_pending(
        self,
        *,
        current_state: State,
        ochestrator_state: OchestratorReadOnlyGlobalState,
        messages_history: Sequence[ChatMessage] | None,
    ) -> ChatMessage:
        last_4_messages = messages_history[-4:] if messages_history else []
        latest_user_message = _latest_user_message(messages_history)
        if latest_user_message is None:
            raise ValueError("Latest user message is required for pending orchestrator answer")

        current_state_name = current_state.name()
        current_node_description = self._node_name_to_description_map.get(
            current_state_name,
            "",
        )
        latest_state_system_message = _latest_system_message(current_state.messages())
        orchestrator_payload = _build_pending_orchestrator_payload(ochestrator_state)

        next_state_name = self.next_state_names_by_current_state_name.get(current_state_name)
        next_state_description = (
            self._node_name_to_description_map.get(next_state_name, "")
            if next_state_name is not None
            else ""
        )

        answer_text = self._llm.generate(
            system_prompt=OCHESTRATOR_PENDING_ORCHESTRATOR_ANSWER_SYSTEM_PROMPT,
            user_prompt=(
                f"Current pending state: {current_state_name!r}\n"
                f"Current state description: {current_node_description!r}\n"
                f"Latest system message inside current state: {latest_state_system_message!r}\n"
                f"Orchestrator payload: {orchestrator_payload!r}\n"
                f"Next state after completion of current state: {next_state_name!r}\n"
                f"Next state description: {next_state_description!r}\n"
                f"User question: {latest_user_message!r}\n"
                "Answer as the orchestrator. "
                "Explain the current state and what the user can do now. "
                "If useful, mention what comes next only as future context after completion."
            ),
            history=last_4_messages,
            config=LLMConfig(model="basic", temperature=0.2),  
        ).content

        return ChatMessage(role="assistant", content=answer_text)
    
    
    def decide_node_name_on_done(self, *, current_state: State) -> str | None:
        current_state_name = current_state.name()
        if current_state_name not in self.next_state_names_by_current_state_name:
            raise ValueError(f"Unsupported state name: {current_state_name}")
        return self.next_state_names_by_current_state_name[current_state_name]
    
    def _decide_node_name_on_abort(
        self,
        *,
        current_state: State,
    ) -> str:
        current_state_name = current_state.name()
        candidates = self.recoverable_states_map.get(current_state_name, (current_state_name,))
        if len(candidates) == 1:
            return candidates[0]

        state_error = current_state.error()
        current_error = state_error.error if state_error is not None else None
        current_system_message = _latest_system_message(current_state.messages())

        candidate_context = _build_aborted_candidate_context(
            candidates=candidates,
            node_descriptions=self._node_name_to_description_map,
        )
        
        decision = self._llm.generate_json(
                schema=_AbortedRouteDecision,
                system_prompt=OCHESTRATOR_ABORTED_SYSTEM_PROMPT,
                user_prompt=(
                    f"The current node is {current_state_name} and it is in ABORTED status. "
                    f"The error is: {current_error!r}. "
                    f"The latest system message is: {current_system_message!r}. "
                    f"Choose the most appropriate node to route to from the following candidates: {candidate_context}"
                ),
                config=LLMConfig(model="basic", temperature=0.1),
                history=None,
                max_attempts=2,
            )
        if decision.state_name is None:
            raise ValueError("LLM failed to choose a state to route to on ABORTED node")
        if decision.state_name not in candidates:
            raise ValueError(f"LLM chose an invalid state to route to on ABORTED node: {decision.state_name!r}, candidates were: {candidates}")
        return decision.state_name
        
        
    def _clear_ochestration_state_on_abort(self, *,state: State, ochestrator_state: OchestratorWritableGlobalState) -> OchestratorWritableGlobalState:
        if state.status() != "ABORTED":
            return ochestrator_state
        
        node_name = ochestrator_state.get_last_active_node_name()
        if node_name is None:
            raise ValueError(
                "last_active_node_name must be set before clearing on aborted node"
            )
        
        recoverable_states = self.recoverable_states_map.get(node_name)
        if recoverable_states is None:
            raise ValueError(f"Unsupported node name: {node_name}")
        
        match state:
            case DatasetState():
                raise ValueError("Dataset node is not recoverable, cannot be aborted")
            case ProtocolDiscussionState():
                ochestrator_state.clear_protocol_discussed()
            case CompileAndValidateState():   
                ochestrator_state.clear_causal_configuration()
            case ModelSelectionState():
                ochestrator_state.clear_selected_model()
            case ModelTrainState():
                ochestrator_state.clear_model_training_id()
            case CausalInferenceState():
                raise ValueError("Causal inference node is not recoverable, cannot be aborted")
            case _:
                raise ValueError(f"unsupported node name: {node_name}")
    
        return ochestrator_state    
    
    def _update_ochestration_state_if_node_done_or_dataset(
        self,
        *,
        ochestrator_state: OchestratorWritableGlobalState,
        state: State,
    ) -> OchestratorWritableGlobalState:
        if state.status() != "DONE" or state.name() == DatasetNode.NAME:
            return ochestrator_state
        
        node_name = ochestrator_state.get_last_active_node_name()
        if node_name is None:
            raise ValueError(
                "last_active_node_name must be set before updating a completed node"
            )
            
        match state:
            case DatasetState() as dataset_state:
                    latest_iteration_dataset_id = dataset_state.payload.dataset_iterations[-1].dataset_id
                    latest_iteration_dataset_summary = dataset_state.payload.latest_summary
                    if  latest_iteration_dataset_summary is None:
                            raise ValueError("Latest iteration dataset ID and summary must be set when protocol is discussed")
                    if ochestrator_state.get("freezing_working_dataset") is True:
                        return ochestrator_state
                        
                    if ochestrator_state.get("protocol_discussed") is True:
                        ochestrator_state.set_freeze_working_dataset(
                            latest_iteration_dataset_id,
                            latest_iteration_dataset_summary,
                        )
                    else:
                        ochestrator_state.set_working_dataset(dataset_id=latest_iteration_dataset_id, summary=latest_iteration_dataset_summary)  
                    
            case ProtocolDiscussionState():
                 ochestrator_state.set_protocol_discussed()
            
            case CompileAndValidateState() as compile_and_validate_state:
                inference_ready_spec = compile_and_validate_state.payload.inference_ready_causal_spec
                if inference_ready_spec is None:  
                    raise ValueError("Inference ready causal spec must be set when compile and validate is done")
                ochestrator_state.set_causal_configuration(
                    causal_spec=inference_ready_spec.causal_spec,
                    data_transformation_plan=inference_ready_spec.transformation_plan,
                    validation_issues=compile_and_validate_state.payload.validation_issues,
                )
                
            case ModelSelectionState() as model_selection_state:
                if model_selection_state.payload.confirmed_model_selection is None or model_selection_state.payload.confirmed_model_selection.selected_model is None:
                    raise ValueError("Confirmed model selection must be set when model selection node is done")
                ochestrator_state.set_selected_model(model_selection_state.payload.confirmed_model_selection.selected_model)

            case ModelTrainState() as model_train_state:
                if model_train_state.payload.trained_model_id is None:
                    raise ValueError("Trained model ID must be set when model train node is done")
                ochestrator_state.set_model_training_id(model_train_state.payload.trained_model_id)

            case CausalInferenceNode.NAME:
                # No persistent orchestrator-global mutation required here.
                pass

            case _:
                raise ValueError(f"unsupported node name: {node_name}")

        return ochestrator_state
        
    
    
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
             



def _last_two_messages(messages_history: Sequence[ChatMessage] | None) -> list[ChatMessage]:
    if not messages_history:
        return []
    non_empty_messages = [message for message in messages_history if message.content.strip()]
    if not non_empty_messages:
        return []
    return list(non_empty_messages[-2:])


def _latest_user_message(messages_history: Sequence[ChatMessage] | None) -> str | None:
    if not messages_history:
        return None
    for message in reversed(messages_history):
        if message.role != "user":
            continue
        content = message.content.strip()
        if content:
            return content
    return None


def _latest_system_message(messages: Sequence[ChatMessage]) -> str | None:
    for message in reversed(messages):
        if message.role == "system" and message.content.strip():
            return message.content.strip()
    return None



class _PendingRouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    route_intent: Literal[
        "CURRENT_STATE",
        "DATASET",
        "ORCHESTRATOR_ANSWER",
    ]
    
def _build_pending_orchestrator_payload(
    ochestrator_state: OchestratorReadOnlyGlobalState,
) -> dict[str, Any]:
    keys = (
        "causal_spec",
        "data_transformation_plan",
        "validation_issues",
        "selected_model",
    )
    return {key: ochestrator_state.get(key) for key in keys}


def init_next_state_names() -> Mapping[str, str | None]:
    return {
        ProtocolDiscussionState.NAME: DatasetState.NAME,
        DatasetState.NAME: CompileAndValidateNode.NAME,
        CompileAndValidateState.NAME: ModelSelectionState.NAME,
        ModelSelectionState.NAME: ModelTrainState.NAME,
        ModelTrainState.NAME: CausalInferenceState.NAME,
        CausalInferenceState.NAME: NoopDoneState.NAME,
    }


class _AbortedRouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    state_name: str | None = None

def _build_aborted_candidate_context(
    *,
    candidates: Sequence[str],
    node_descriptions: Mapping[str, str],
) -> list[dict[str, str]]:
    return [
        {
            "state_name": state_name,
            "node_info": node_descriptions.get(state_name, ""),
        }
        for state_name in candidates
    ]    
    
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
    
