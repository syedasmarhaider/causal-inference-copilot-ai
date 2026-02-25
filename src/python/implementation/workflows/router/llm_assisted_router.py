from __future__ import annotations

from collections.abc import Mapping
import json
from string import Template
from typing import Optional, Sequence, Type

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
    
    
    def get_initial_state_name(self) -> str:
        return LoadDatasetState.NAME
    
    def get_done_state_name(self) -> str:
        return NoopDoneState.NAME
    
    def decide_next(
        self,
        *,
        current_state: Optional[State],
        messages_history: Sequence[ChatMessage],
    ) -> NextDecision:
        if current_state is None:
            return NextDecision(state_name=LoadDatasetState.NAME, router_message_for_node=None)

        status = current_state.status
        
        if status == "PENDING":
            return NextDecision(state_name=current_state.name, router_message_for_node=None)
        
        if status == "DONE":
            if current_state.name is NoopDoneState.NAME:
                return NextDecision(state_name=NoopDoneState.NAME, router_message_for_node=None)  
            next_name = self._next_state_names_map.get(current_state.name)
            if next_name is None:
                raise ValueError(f"Router has no next state defined for current state {current_state.name!r} with DONE status.")
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
    messages_history: Optional[Sequence[ChatMessage]],
) -> NextDecision:
    last_10_messages: list[ChatMessage] = list(messages_history[-10:]) if messages_history else []
    
    prev_map: dict[str, str] = {
        nxt: cur for cur, nxt in get_next_state_names_map.items() if nxt is not None
    }

    allowed_prev: set[str] = set()
    cursor = current_state.name
    while cursor in prev_map:
        cursor = prev_map[cursor]
        allowed_prev.add(cursor)

    if not allowed_prev:
        allowed_prev = {current_state.name}
        
    prompt = _node_prompt_for_router()
    prompt_filled = prompt.substitute(
        current_node_name=current_state.name,
        current_node_error=current_state.error or "null",
        next_state_names_map=json.dumps(dict(get_next_state_names_map), ensure_ascii=False),
        node_name_to_description_map=json.dumps(dict(get_node_name_to_description_map), ensure_ascii=False),
        allowed_previous_states=json.dumps(sorted(allowed_prev), ensure_ascii=False),
    )

    decision = llm.generate_json(
        config=LLMConfig(model=model_name, temperature=0.3),
        system_prompt="Decide fallback state (must be previous). Output STRICT JSON matching schema.",
        user_prompt=prompt_filled,
        history=last_10_messages,
        schema=NextDecision,
        max_attempts=3,
    )

    chosen = decision.state_name

    # ---- validation: must not be None ----
    if not chosen:
        raise ValueError(f"LLM failed to select a state for aborted '{current_state.name}'. Got empty/null response.\n\nLLM message:\n{decision.router_message_for_node or ''}"
        )

    # ---- validation: membership ----
    if chosen not in get_node_name_to_description_map or chosen not in get_next_state_names_map:
        raise ValueError(f"LLM selected invalid state '{chosen}' for aborted '{current_state.name}'. It must be a key in next_state_names_map and node_name_to_description_map.\n\nLLM message:\n{decision.router_message_for_node or ''}"
        )

    # ---- validation: previous constraint ----
    if chosen not in allowed_prev:
        raise ValueError(f"LLM selected state '{chosen}' which is not a previous state of '{current_state.name}' for recovery. Allowed previous states are: {sorted(allowed_prev)}.\n\nLLM message:\n{decision.router_message_for_node or ''}"
        )
        
    return decision
        


def _node_prompt_for_router() -> Template:
    return Template(
        """
You are an LLM-assisted workflow router.

Goal
- The current node is ABORTED. Choose the next node to run to best recover.

Inputs (some may be null)
- current_node_name: $current_node_name
- current_node_error: $current_node_error
- next_state_names_map: $next_state_names_map
- node_name_to_description_map: $node_name_to_description_map
- allowed_previous_states: $allowed_previous_states

Hard Rules
1) You MUST select exactly ONE state_name from allowed_previous_states.
2) The selected state_name MUST be a key in node_name_to_description_map.
3) Prefer the closest previous state that can fix the error with minimal rollback.

Output (STRICT JSON ONLY; no extra text)
{
  "state_name": "<one state name from allowed_previous_states>",
  "router_message_for_node": "<detailed message for the selected node>"
}
""".strip()
    )


def build_state_classes_by_name() -> Mapping[str, Type[State]]:
    return {
        LoadDatasetState.NAME: LoadDatasetState,
        ProtocolDiscussionState.NAME: ProtocolDiscussionState,
        CompileProtocolState.NAME: CompileProtocolState,
        CleanProtocolState.NAME: CleanProtocolState,
        ValidateCleanProtocolState.NAME: ValidateCleanProtocolState,
        TransformProtocolState.NAME: TransformProtocolState,
        ConfirmTransformedProtocolState.NAME: ConfirmTransformedProtocolState,
        NoopDoneState.NAME: NoopDoneState,
    }
        

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
    
        
     
    
