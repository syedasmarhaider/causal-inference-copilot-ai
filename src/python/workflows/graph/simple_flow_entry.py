from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Final, Mapping
from uuid import UUID

from langchain_core.messages import  HumanMessage

from python.domain.repo.conversation_repo import ConversationRepo
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.service.llm_service import LLMService
from python.workflows.graph.simple_flow_router import WorkflowRouter
from python.workflows.nodes.compile_protocol_state import make_compile_protocol_state_node
from python.workflows.nodes.compile_protocol_state import make_compile_protocol_state_node
from python.workflows.nodes.load_dataset import make_load_dataset_node
from python.workflows.nodes.model_fit import make_model_fit_node
from python.workflows.nodes.model_params_fit_discussion_node import make_model_params_fit_discussion_node
from python.workflows.nodes.model_selection_discussion_node import make_model_selection_discussion_node
from python.workflows.nodes.model_selection_node import make_model_selection_node
from python.workflows.nodes.prepare_inference_ready_state import make_prepare_inference_ready_node
from python.workflows.nodes.protocol_discussion import make_protocol_discussion_node
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState, get_init_conversation_state
from python.workflows.state.control_state import  Stage, Status
from python.workflows.tools.inference.causal_inference_factory import CausalInferenceFactory
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI


DEFAULT_DATASET_PATH: Final[Path] = Path(
    "./data/486f4975-6cd9-4261-a122-e6b0fc46462d/data.csv"
).resolve()


@dataclass(frozen=True)
class WorkflowConfig:
    data_repo: DataRepo
    models_repo: ModelsRepo
    llm: LLMService
    model_name: str = DEFAULT_MODEL_GEMNI


@dataclass(frozen=True)
class WorkflowResponse:
    node_message: str | None
    needs_input: bool
    current_stage: Stage
    current_stage_status: Status


def _noop(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
    return state


    
def _build_nodes(cfg: WorkflowConfig) -> Mapping[Stage, CallableNodeFunc]:
    return {
        "LOAD_DATASET": make_load_dataset_node(data_repo=cfg.data_repo, llm=cfg.llm, model_name=cfg.model_name),
        "PROTOCOL_DISCUSSION": make_protocol_discussion_node(
            llm=cfg.llm,
            model_name=cfg.model_name,
        ),
        "COMPILE_PROTOCOL": make_compile_protocol_state_node(
            llm=cfg.llm,
            model_name=cfg.model_name,
        ), 
        "VALIDATE_PROTOCOL_STATIC": make_validate_protocol_static_node(
            data_repo=cfg.data_repo,
            llm=cfg.llm,
            model_name=cfg.model_name,
        ),
        "VALIDATE_PROTOCOL_STATIC_DISCUSSION": make_validate_protocol_discussion_node(
            llm=cfg.llm,
            model_name=cfg.model_name,
        ),
        "INFERENCE_READY": make_prepare_inference_ready_node(
            data_repo=cfg.data_repo,
        ),
        "MODEL_SELECTION": make_model_selection_node(
            llm=cfg.llm,
            model_name=cfg.model_name,
            
        ),
        "MODEL_SELECTION_DISCUSSION": make_model_selection_discussion_node(
            llm=cfg.llm,
            model_name=cfg.model_name,
            
        ),
        "MODEL_PARAMS_FIT_DISCUSSION": make_model_params_fit_discussion_node(
            llm=cfg.llm,
            model_name=cfg.model_name,
            causal_factory = CausalInferenceFactory.create_default(
              data_repo=cfg.data_repo,
              models_repo=cfg.models_repo,
            ),
        ),
        "MODEL_FIT": make_model_fit_node(
            causal_factory = CausalInferenceFactory.create_default(
              data_repo=cfg.data_repo,
              models_repo=cfg.models_repo,
            ),
        ),
        "DONE": _noop,
    }



class SimpleWorkflow:
    def __init__(
        self,
        *,
        repo: ConversationRepo,
        cfg: WorkflowConfig,
    ) -> None:
        self._repo = repo

        nodes: Final[Mapping[Stage, CallableNodeFunc]] = _build_nodes(cfg)
        self._router = WorkflowRouter(
            llm=cfg.llm,
            model_name=cfg.model_name,
            nodes=nodes,
        )

    def invoke(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        user_text: str | None = None,
    ) -> WorkflowResponse:
        state = self._repo.load(user_id=user_id, conversation_id=conversation_id)
        
       
     
        
        if state is None:
            state = get_init_conversation_state(UUID("486f4975-6cd9-4261-a122-e6b0fc46462d"))
        if isinstance(user_text, str):
            txt = user_text.strip()
            if txt:
                state["messages"].append(HumanMessage(content=txt))
                
                

        node_fn, routed_state = self._router.route(state)
        out_state = node_fn(user_id, conversation_id, routed_state)

        control = out_state["control"]
        node_msg = control.get("node_message", None)
        needs_input = control.get("action_required") == "NEEDS_INPUT"

        if needs_input:
            control["action_required"] = "NONE"

        self._repo.save(user_id=user_id, conversation_id=conversation_id, state=out_state)
        logging.warning("State after processing protocol:" + get_string_protocol_state(out_state.get("protocol", None)))
        inference_ready = out_state.get('inference_ready')
        inference_ready_summary = get_inference_ready_state_summary(inference_ready) if inference_ready is not None else 'None'
        logging.warning(f"State after processing protocol: inference_ready={inference_ready_summary}")
        return WorkflowResponse(
            node_message=node_msg,
            needs_input=needs_input,
            current_stage=control["current_stage"],
            current_stage_status=control["current_stage_status"],
        )
