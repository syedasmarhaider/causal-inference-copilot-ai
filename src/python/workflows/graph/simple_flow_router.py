from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final, Mapping


from python.domain.repo.conversation_repo import ConversationRepo
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMService
from python.domain.service.mcp_client import McpClient

from python.workflows.graph.simple_flow_entry import SimpleWorkflow, WorkflowConfig
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState, Stage

from python.workflows.nodes.get_file import make_get_file_node
from python.workflows.nodes.load_dataset import make_load_dataset_node
from python.workflows.nodes.propose_and_confirm_metadata import make_propose_and_confirm_metadata

from python.workflows.utils.user_message_builder import build_user_message_with_llm

NodeFn = Callable[[ConversationState], ConversationState]


# =============================================================================
# Static routes
# =============================================================================
@dataclass(frozen=True)
class RouteSpec:
    nxt: Stage
    prv: Stage


DEFAULT_ROUTES: Final[dict[Stage, RouteSpec]] = {
    "GET_FILE": RouteSpec(nxt="LOAD_DATASET", prv="GET_FILE"),
    "LOAD_DATASET": RouteSpec(nxt="PROPOSE_METADATA", prv="GET_FILE"),
    "PROPOSE_METADATA": RouteSpec(nxt="CONFIRM_METADATA", prv="LOAD_DATASET"),
    "CONFIRM_METADATA": RouteSpec(nxt="DONE", prv="PROPOSE_METADATA"),
    "DONE": RouteSpec(nxt="DONE", prv="CONFIRM_METADATA"),
}


def _done_node() -> NodeFn:
    """
    Final stage node.

    Runner behavior:
      - sees node_message/post_action,
      - emits ONE AIMessage,
      - clears node_message,
      - keeps stage/status consistent with cfg.
    """
    def _fn(state: ConversationState) -> ConversationState:
        c: ControlState = state["control"]

        node_msg = c.get("node_message") or "Done."
        c2: ControlState = {
            **c,
            "stage": "DONE",
            "status": "DONE",
            "post_action": "PRESENT",
            "node_message": node_msg,
            "pending_stage": None,
        }
        return {**state, "control": c2}

    return _fn


# =============================================================================
# Builders
# =============================================================================
def build_default_nodes(
    *,
    data_repo: DataRepo,
    llm: LLMService,
    mcp_client: McpClient,  # kept for signature stability / future nodes
) -> dict[Stage, NodeFn]:
    _ = mcp_client  # reserved (future)

    meta_node = make_propose_and_confirm_metadata(llm=llm, data_repo=data_repo)

    return {
        "GET_FILE": make_get_file_node(llm),
        "LOAD_DATASET": make_load_dataset_node(data_repo),
        "PROPOSE_METADATA": meta_node,
        "CONFIRM_METADATA": meta_node,
        "DONE": _done_node(),
    }


def build_workflow_config(
    *,
    routes: Mapping[Stage, RouteSpec] = DEFAULT_ROUTES,
    max_internal_steps: int = 32,
) -> WorkflowConfig:
    next_stage: dict[Stage, Stage] = {}
    prev_stage: dict[Stage, Stage] = {}
    valid_stages: set[Stage] = set()

    for stg, spec in routes.items():
        next_stage[stg] = spec.nxt
        prev_stage[stg] = spec.prv
        valid_stages.add(stg)

    return WorkflowConfig(
        next_stage=next_stage,
        prev_stage=prev_stage,
        valid_stages=valid_stages,
        max_internal_steps=max_internal_steps,
    )


def build_simple_workflow(
    *,
    repo: ConversationRepo,
    data_repo: DataRepo,
    llm: LLMService,
    mcp_client: McpClient,
) -> SimpleWorkflow:
    nodes = build_default_nodes(data_repo=data_repo, llm=llm, mcp_client=mcp_client)
    cfg = build_workflow_config()

    def enhance(state: ConversationState) -> str:
            return build_user_message_with_llm(llm=llm, state=state)

    return SimpleWorkflow(repo=repo, nodes=nodes, cfg=cfg, enhance=enhance)


def build_simple_copilot_app(
    *,
    repo: ConversationRepo,
    data_repo: DataRepo,
    llm: LLMService,
    mcp_client: McpClient,
) -> SimpleWorkflow:
    return build_simple_workflow(
        repo=repo,
        data_repo=data_repo,
        llm=llm,
        mcp_client=mcp_client,
    )