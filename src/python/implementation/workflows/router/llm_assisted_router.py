from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from string import Template

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.route import NextDecision, Router
from python.domain.workflows.state import State
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.causal_inference.causal_inference_node import (
    CausalInferenceNode,
)
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import (
    CausalInferenceState,
)
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_node import (
    CleanProtocolNode,
)
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import (
    CleanProtocolState,
)
from python.implementation.workflows.nodes.load_dataset.load_dataset_node import LoadDatasetNode
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.nodes.model_selection.mode_selection_state import (
    ModelSelectionState,
)
from python.implementation.workflows.nodes.model_selection.model_selection_node import (
    ModelSelectionNode,
)
from python.implementation.workflows.nodes.model_train.model_train_node import ModelTrainNode
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainState
from python.implementation.workflows.nodes.noop_done.noop_done_node import NoopDoneNode
from python.implementation.workflows.nodes.noop_done.noop_done_state import NoopDoneState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import (
    ProtocolDiscussionNode,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_node import (
    ValidateCleanProtocolNode,
)
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_state import (
    ValidateCleanProtocolState,
)

log = get_logger(__name__, component="LLMAssistedRouterRouter", log_type="workflow_router")


class LLMAssistedRouterRouter(Router):
    def __init__(
        self,
        *,
        llm: LLMService,
    ) -> None:
        self._llm = llm
        self._next_state_names_map: Mapping[str, str | None] = init_next_state_names()
        self._node_name_to_description_map: Mapping[str, str] = get_node_name_with_description()
        log.info(
            "router initialized",
            states_count=len(self._next_state_names_map),
            described_nodes_count=len(self._node_name_to_description_map),
        )
    
    
    def get_initial_state_name(self) -> str:
        return LoadDatasetState.NAME
    
    def get_done_state_name(self) -> str:
        return NoopDoneState.NAME
    
    def get_next_state_names(
        self,
        current_state_name: str,
    ) -> Sequence[str]:
        next_states: list[str] = []
        cursor = current_state_name
        while True:
            nxt = self._next_state_names_map.get(cursor)
            if nxt is None:
                break
            next_states.append(nxt)
            cursor = nxt
        return next_states
    
    def decide_next(
        self,
        *,
        current_state: State | None,
        messages_history: Sequence[ChatMessage],
    ) -> NextDecision:
        if current_state is None:
            log.debug("router selected initial state because current state is missing")
            return NextDecision(state_name=LoadDatasetState.NAME, router_message_for_node=None)

        status = current_state.status
        
        if status == "PENDING":
            log.debug("router kept current state because status is pending", state_name=current_state.name)
            return NextDecision(state_name=current_state.name, router_message_for_node=None)
        
        if status == "DONE":
            if current_state.name == NoopDoneState.NAME:
                log.debug("router kept done state because terminal node reached")
                return NextDecision(state_name=NoopDoneState.NAME, router_message_for_node=None)  
            next_name = self._next_state_names_map.get(current_state.name)
            if next_name is None:
                log.error(
                    "router has no next state mapping for done state",
                    state_name=current_state.name,
                )
                raise ValueError(f"Router has no next state defined for current state {current_state.name!r} with DONE status.")
            log.info(
                "router advanced from done state to next state",
                current_state_name=current_state.name,
                next_state_name=next_name,
            )
            return NextDecision(
                    state_name=next_name,
                    router_message_for_node=None,
            )
            
            
        if status == "ABORTED":
            log.info("router entered aborted recovery flow", state_name=current_state.name)
            return self._decision_on_aborted_state(
                current_state=current_state,
                messages_history=messages_history,
            )
            
        log.error(
            "router received unexpected state status",
            state_name=current_state.name,
            status=status,
        )
        raise ValueError(f"Router: unexpected status={status!r} for state={current_state.name!r}")
    
    def _decision_on_aborted_state(
        self,
        *,
        current_state: State,
        messages_history: Sequence[ChatMessage] | None,
    ) -> NextDecision:
        last_10_messages: list[ChatMessage] = list(messages_history[-10:]) if messages_history else []
        
        prev_map: dict[str, str] = {
            nxt: cur for cur, nxt in self._next_state_names_map.items() if nxt is not None
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
            next_state_names_map=json.dumps(dict(self._next_state_names_map), ensure_ascii=False),
            node_name_to_description_map=json.dumps(dict(self._node_name_to_description_map), ensure_ascii=False),
            allowed_previous_states=json.dumps(sorted(allowed_prev), ensure_ascii=False),
        )

        decision = self._llm.generate_json(
            config=LLMConfig(model="basic", temperature=0.3),
            system_prompt="Decide fallback state (must be previous). Output STRICT JSON matching schema.",
            user_prompt=prompt_filled,
            history=last_10_messages,
            schema=NextDecision,
            max_attempts=3,
        )

        chosen = decision.state_name

        # ---- validation: must not be None ----
        if not chosen:
            log.error(
                "router llm returned empty state selection",
                current_state_name=current_state.name,
            )
            raise ValueError(f"LLM failed to select a state for aborted '{current_state.name}'. Got empty/null response.\n\nLLM message:\n{decision.router_message_for_node or ''}"
            )

        # ---- validation: membership ----
        if chosen not in self._node_name_to_description_map or chosen not in self._next_state_names_map:
            log.error(
                "router llm returned invalid state selection",
                current_state_name=current_state.name,
                selected_state_name=chosen,
            )
            raise ValueError(f"LLM selected invalid state '{chosen}' for aborted '{current_state.name}'. It must be a key in next_state_names_map and node_name_to_description_map.\n\nLLM message:\n{decision.router_message_for_node or ''}"
            )

        # ---- validation: previous constraint ----
        if chosen not in allowed_prev:
            log.error(
                "router llm selected non-previous state",
                current_state_name=current_state.name,
                selected_state_name=chosen,
                allowed_previous_states=sorted(allowed_prev),
            )
            raise ValueError(f"LLM selected state '{chosen}' which is not a previous state of '{current_state.name}' for recovery. Allowed previous states are: {sorted(allowed_prev)}.\n\nLLM message:\n{decision.router_message_for_node or ''}"
            )

        next_states =  self.get_next_state_names(chosen)      
        decision.delete_next_states_names = next_states if next_states else None  
        log.info(
            "router selected fallback state for aborted recovery",
            current_state_name=current_state.name,
            selected_state_name=chosen,
            delete_states_count=len(next_states),
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
3) Prefer the closest previous state that can fix the error with minimal rollback and those states which requires user input.
4) Focus on last messages in error so that you can understand which state to return to best fix the error.
5) Always try to solve error with directing to some node without distrubing user about techincal messages.

Output (STRICT JSON ONLY; no extra text)
{
  "state_name": "<one state name from allowed_previous_states>",
  "router_message_for_node": "<detailed message for the selected node>"
}
""".strip()
    )


def build_state_classes_by_name() -> Mapping[str, type[State]]:
    return {
        LoadDatasetState.NAME: LoadDatasetState,
        ProtocolDiscussionState.NAME: ProtocolDiscussionState,
        CleanProtocolState.NAME: CleanProtocolState,
        ValidateCleanProtocolState.NAME: ValidateCleanProtocolState,
        ModelSelectionState.NAME: ModelSelectionState,
        ModelTrainState.NAME: ModelTrainState,
        CausalInferenceState.NAME: CausalInferenceState,
        NoopDoneState.NAME: NoopDoneState,
    }
        

def init_next_state_names() -> Mapping[str, str | None]:
    return {
        LoadDatasetState.NAME: ProtocolDiscussionState.NAME,
        ProtocolDiscussionState.NAME: CleanProtocolState.NAME,
        CleanProtocolState.NAME: ValidateCleanProtocolState.NAME,
        ValidateCleanProtocolState.NAME: ModelSelectionState.NAME,
        ModelSelectionState.NAME: ModelTrainState.NAME,
        ModelTrainState.NAME: CausalInferenceState.NAME,
        CausalInferenceState.NAME: NoopDoneState.NAME,
        NoopDoneState.NAME: None,
    }

def get_node_name_with_description() -> Mapping[str, str]:
    return{
        LoadDatasetNode.NAME: LoadDatasetNode.get_info(),
        ProtocolDiscussionNode.NAME: ProtocolDiscussionNode.get_info(),
        CleanProtocolNode.NAME: CleanProtocolNode.get_info(),
        ValidateCleanProtocolNode.NAME: ValidateCleanProtocolNode.get_info(),
        ModelSelectionState.NAME: ModelSelectionNode.get_info(),
        ModelTrainState.NAME: ModelTrainNode.get_info(),
        CausalInferenceState.NAME: CausalInferenceNode.get_info(),
        NoopDoneState.NAME: NoopDoneNode.get_info(),
    }


def init_all_nodoes_with_name_as_key(llm: LLMService, data_repo: DataRepo, models_repo: ModelsRepo) -> dict[str, Node]:
    load_dataset_node = LoadDatasetNode(data_repo=data_repo, llm=llm)
    protocol_discussion_node = ProtocolDiscussionNode(
        llm=llm,
     )
    clean_protocol_node = CleanProtocolNode(
         data_repo=data_repo,
        llm=llm,
     )
     
    validate_cleaned_protocol_node = ValidateCleanProtocolNode(
        data_repo=data_repo,
        llm=llm,
        )
    
    model_selection_node = ModelSelectionNode(
        llm=llm,
     )
    
    model_train_node = ModelTrainNode(
        llm=llm,
        data_repo=data_repo,
     )
    
    inference_node = CausalInferenceNode(
        llm=llm,
        data_repo=data_repo,
     )
    
    
   
    done_node = NoopDoneNode()
    
    return {
        load_dataset_node.name: load_dataset_node,
        protocol_discussion_node.name: protocol_discussion_node,
        clean_protocol_node.name: clean_protocol_node,
        validate_cleaned_protocol_node.name: validate_cleaned_protocol_node,
        model_selection_node.name: model_selection_node,
        model_train_node.name: model_train_node,
        inference_node.name: inference_node,
        done_node.name: done_node,
    }
    
        
     
    
