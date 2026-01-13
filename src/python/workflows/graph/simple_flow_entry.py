from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping
from uuid import UUID

from langchain_core.messages import BaseMessage, HumanMessage

from python.domain.repo.conversation_repo import ConversationRepo
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMService
from python.domain.service.mcp_client import McpClient
from python.workflows.graph.simple_flow_router import WorkflowRouter
from python.workflows.nodes.load_dataset import make_load_dataset_node
from python.workflows.nodes.propose_and_confirm_metadata import make_propose_and_confirm_metadata_node
from python.workflows.state.conversation_state import CallableNodeFunc, ConversationState
from python.workflows.state.control_state import ControlState, Stage, Status
from python.workflows.state.dataset_state import DatasetState
from python.workflows.state.metadata_state import MetadataState, empty_metadata
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI


DEFAULT_DATASET_PATH: Final[Path] = Path(
    "./data/486f4975-6cd9-4261-a122-e6b0fc46462d/data.csv"
).resolve()


@dataclass(frozen=True)
class WorkflowConfig:
    data_repo: DataRepo
    llm: LLMService
    mcp: McpClient
    model_name: str = DEFAULT_MODEL_GEMNI


@dataclass(frozen=True)
class WorkflowResponse:
    node_message: str | None
    needs_input: bool
    current_stage: Stage
    current_stage_status: Status


def _noop(user_id: UUID, conversation_id: UUID, state: ConversationState) -> ConversationState:
    return state


def _new_state() -> ConversationState:
    control: ControlState = {
        "current_stage": "LOAD_DATASET",
        "current_stage_status": "PENDING",
        "action_required": "NONE",
        "node_message": None,
    }

    dataset: DatasetState = {
        "id": UUID("486f4975-6cd9-4261-a122-e6b0fc46462d"),
        "load_error": None,
    }

    metadata: MetadataState = empty_metadata()
    messages: list[BaseMessage] = []

    return {
        "control": control,
        "dataset": dataset,
        "metadata": metadata,
        "messages": messages,
    }
    
    
def _build_nodes(cfg: WorkflowConfig) -> Mapping[Stage, CallableNodeFunc]:
    return {
        "LOAD_DATASET": make_load_dataset_node(cfg.data_repo, cfg.llm, model_name=cfg.model_name),
        "PROPOSE_AND_CONFIRM_METADATA": make_propose_and_confirm_metadata_node(
            llm=cfg.llm,
            model_name=cfg.model_name,
        ),
        "DONE": _noop,
    }


def _extract_node_message(control: ControlState) -> str | None:
    msg = control.get("node_message")
    if not isinstance(msg, str):
        return None
    msg = msg.strip()
    return msg or None


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
            state = _new_state()
 
        if isinstance(user_text, str):
            txt = user_text.strip()
            if txt:
                state["messages"].append(HumanMessage(content=txt))

        node_fn, routed_state = self._router.route(state)
        out_state = node_fn(user_id, conversation_id, routed_state)

        control = out_state["control"]
        node_msg = _extract_node_message(control)
        needs_input = control.get("action_required") == "NEEDS_INPUT"

        # surface needs_input in response, but do not persist NEEDS_INPUT
        if needs_input:
            control["action_required"] = "NONE"

        self._repo.save(user_id=user_id, conversation_id=conversation_id, state=out_state)

        return WorkflowResponse(
            node_message=node_msg,
            needs_input=needs_input,
            current_stage=control["current_stage"],
            current_stage_status=control["current_stage_status"],
        )
