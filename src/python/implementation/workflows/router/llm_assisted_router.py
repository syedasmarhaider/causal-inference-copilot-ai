from __future__ import annotations

from collections.abc import Mapping
import json
from string import Template
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
        model_name: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self._model_name = model_name or DEFAULT_MODEL_GEMNI
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
            
            
        if status == "ABORTED":
            return _decision_on_aborted_state(
                llm=self._llm,
                model_name=self._model_name,
                current_state=current_state,
                get_next_state_names_map=self._next_state_names_map,
                get_node_name_to_description_map=self._node_name_to_description_map,
                user_message=user_message,
                messages_history=messages_history,
            )
            
            
        raise ValueError(f"Router: unexpected status={status!r} for state={current_state.name!r}")




def _decision_on_aborted_state(
    *, 
    llm: LLMService,
    model_name: str,
    current_state: State,
    get_next_state_names_map: Mapping[str, Optional[str]],
    get_node_name_to_description_map: Mapping[str, str],
    user_message: Optional[str],
    messages_history: Sequence[ChatMessage],
) -> NextDecision:
    
        last_10_messages: list[ChatMessage] = list(messages_history[-10:]) if messages_history else []
        prompt = _node_prompt_for_router()
        prompt_filled = prompt.substitute(
            current_node_name=current_state.name,
            current_node_error=current_state.error or "",
            user_message=user_message or "",
            next_state_names_map=json.dumps(get_next_state_names_map, ensure_ascii=False),
            node_name_to_description_map=json.dumps(get_node_name_to_description_map, ensure_ascii=False),
        )
        
        return llm.generate_json(
            config=LLMConfig(model=model_name, temperature=0.3),
            system_prompt="Decide fallback node",
            user_prompt=prompt_filled,
            history=last_10_messages,
            schema=NextDecision,
            max_attempts=3,
        )
        
        


def _node_prompt_for_router() -> Template:
    return Template(
        """
You are an LLM-assisted workflow router.

Goal
- The current node is ABORTED. Choose the next node to run to best recover.

Inputs (some may be null)
- current_node_name: $current_node_name
- current_node_error: $current_node_error
- user_message: $user_message
- next_state_names_map: $next_state_names_map
- node_name_to_description_map: $node_name_to_description_map

Hard Rules
1) You MUST select exactly ONE next_node_name from the available_nodes list.
2) The selected next_node_name MUST be a previous node relative to current_node_name (i.e., earlier stage).
3) If current_node_error is null, use user_message and messages_history to infer the best recovery step.
4) If user_message and messages_history are null/empty, pick the safest closest previous node and instruct it to re-derive/validate prerequisites.

Output (STRICT JSON ONLY; no extra text)
{
  "next_node_name": "<one node name from available_nodes>",
  "router_message_for_node": "<detailed message for the selected node>"
}
""".strip()
    )

        

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
    
        
     
    
