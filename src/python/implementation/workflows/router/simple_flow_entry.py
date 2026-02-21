from __future__ import annotations

from dataclasses import dataclass
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
from python.workflows.nodes.validate_inference_ready import make_validate_inference_ready_node
from python.workflows.nodes.validate_inference_ready_discussion import make_validate_inference_ready_discussion_node
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState, get_init_conversation_state
from python.workflows.graph.router_control import  Stage, Status
from python.workflows.tools.inference.causal_inference_factory import CausalInferenceFactory
from python.workflows.utils.utils import DEFAULT_MODEL_GEMNI


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
        out_state["router_message"] = None
        needs_input = control.get("action_required") == "NEEDS_INPUT"

        if needs_input:
            control["action_required"] = "NONE"

        self._repo.save(user_id=user_id, conversation_id=conversation_id, state=out_state)       
        return WorkflowResponse(
            node_message=node_msg,
            needs_input=needs_input,
            current_stage=control["current_stage"],
            current_stage_status=control["current_stage_status"],
        )
