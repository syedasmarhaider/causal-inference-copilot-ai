from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Sequence

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.route import NextDecision, Router
from python.domain.workflows.state import State
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_node import CleanProtocolNode
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_node import CompileProtocolNode
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.confirm_transformed_protocol.confirm_transformed_protocol_node import ConfirmTransformedProtocolNode
from python.implementation.workflows.nodes.confirm_transformed_protocol.confirm_transformed_protocol_state import ConfirmTransformedProtocolState
from python.implementation.workflows.nodes.load_dataset.load_dataset_node import LoadDatasetNode
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.nodes.noop_done.noop_done_node import NoopDoneNode
from python.implementation.workflows.nodes.noop_done.noop_done_state import NoopDoneState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import ProtocolDiscussionNode
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import ProtocolDiscussionState
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_node import TransformProtocolNode
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_state import TransformProtocolState
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_node import ValidateCleanProtocolNode
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_state import ValidateCleanProtocolState
from python.implementation.workflows.utils.utils import DEFAULT_MODEL_GEMNI

class LLMAssistedRouterRouter(Router):
    def __init__(
        self,
        *,
        llm: LLMService,
        config: Optional[LLMConfig] = None,
    ) -> None:
        self._llm = llm
        self._config = config or LLMConfig(temperature=0.0)
        self._next_state_names_map: Mapping[str, Optional[str]] = init_next_state_names()
        self._node_name_to_description_map: Mapping[str, str] = get_node_name_with_description()

    def decide_next(
        self,
        *,
        current_state: Optional[State],
        user_message: Optional[str],
        messages_history: Sequence[ChatMessage],
    ) -> NextDecision:
        if current_state is None:
            return NextDecision(state_name=LoadDatasetState.NAME, router_message_for_node=None)

        status = current_state.status
        
        if status == "PENDING":
            return NextDecision(state_name=current_state.name, router_message_for_node=None)
        
        if status == "DONE":
            next_name = self._next_state_names_map.get(current_state.name)
            return NextDecision(
                    state_name=next_name,
                    router_message_for_node=None,
            )

        # Aborted -> LLM remediation later (stub for now)
        if status == "ABORTED":
            return NextDecision(
                state_name=current_state.name,
                router_message_for_node="State aborted. LLM remediation routing not implemented yet.",
            )

        raise ValueError(f"Router: unexpected status={status!r} for state={current_state.name!r}")



def _node_prompt_for_router(node_name: str) -> str:
    return f"{node_name}: {get_node_name_with_description().get(node_name, 'No description available')}"
    """You are a router that decides the next node in a workflow.
    "Current node is aborted. Given the node error message returned
     other node descriptions, and user message, 
     decide which node should run next to best recover from the error. 
     node should always be selected from the node given list and always be prev node from the current node.
     Compose a detailed message for node selected by you to understand the context and error and fix it.
     Output should be only name of the selected node and message for the node in JSON format with keys"""

        

def init_next_state_names() -> Mapping[str, Optional[str]]:
    return {
        LoadDatasetState.NAME: ProtocolDiscussionState.NAME,
        ProtocolDiscussionState.NAME: CompileProtocolState.NAME,
        CompileProtocolState.NAME: CleanProtocolState.NAME,
        CleanProtocolState.NAME: ValidateCleanProtocolState.NAME,
        ValidateCleanProtocolState.NAME: TransformProtocolState.NAME,
        TransformProtocolState.NAME: ConfirmTransformedProtocolState.NAME,
        ConfirmTransformedProtocolState.NAME: NoopDoneState.NAME,
        NoopDoneState.NAME: None,
    }

def get_node_name_with_description() -> Mapping[str, str]:
    return {
        LoadDatasetState.NAME: LoadDatasetNode.get_info(),
        ProtocolDiscussionState.NAME: ProtocolDiscussionNode.get_info(),
        CompileProtocolState.NAME: CompileProtocolNode.get_info(),
        CleanProtocolState.NAME: CleanProtocolNode.get_info(),
        ValidateCleanProtocolState.NAME: ValidateCleanProtocolNode.get_info(),
        TransformProtocolState.NAME: TransformProtocolNode.get_info(),
        ConfirmTransformedProtocolState.NAME: ConfirmTransformedProtocolNode.get_info(),
        NoopDoneState.NAME: NoopDoneNode.get_info(),
    }


def init_all_nodoes_with_name_as_key(llm: LLMService, data_repo: DataRepo, models_repo: ModelsRepo) -> dict[str, Node]:
    load_dataset_node = LoadDatasetNode(data_repo=data_repo, llm=llm)
    protocol_discussion_node = ProtocolDiscussionNode(
        llm=llm,
        model_name=DEFAULT_MODEL_GEMNI,
     )
    compiled_protocol_node = CompileProtocolNode(
        llm=llm,
        model_name=DEFAULT_MODEL_GEMNI,
     )
    clean_protocol_node = CleanProtocolNode(
         data_repo=data_repo,
     )
     
    validate_cleaned_protocol_node = ValidateCleanProtocolNode(
        data_repo=data_repo,
        llm=llm,
        model_name=DEFAULT_MODEL_GEMNI,
        )
    transform_protocol_node = TransformProtocolNode(
        data_repo=data_repo,
        llm=llm,
        model_name=DEFAULT_MODEL_GEMNI,
    )
    confirm_transformed_protocol_node = ConfirmTransformedProtocolNode(
        llm=llm,
        model_name=DEFAULT_MODEL_GEMNI,
     )
    done_node = NoopDoneNode()
    
    return {
        load_dataset_node.name: load_dataset_node,
        protocol_discussion_node.name: protocol_discussion_node,
        compiled_protocol_node.name: compiled_protocol_node,
        clean_protocol_node.name: clean_protocol_node,
        validate_cleaned_protocol_node.name: validate_cleaned_protocol_node,
        transform_protocol_node.name: transform_protocol_node,
        confirm_transformed_protocol_node.name: confirm_transformed_protocol_node,
        done_node.name: done_node,
    }
    
        
     
    
