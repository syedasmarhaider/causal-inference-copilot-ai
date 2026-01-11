from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, cast

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

NodeFn = Callable[[ConversationState], ConversationState]


# =============================================================================
# Static, explicit routing table (SimpleWorkflow uses cfg.next_stage / cfg.prev_stage)
# =============================================================================
@dataclass(frozen=True)
class RouteSpec:
    nxt: Stage
    prv: Stage


DEFAULT_ROUTES: dict[Stage, RouteSpec] = {
    "GET_FILE": RouteSpec(nxt="LOAD_DATASET", prv="GET_FILE"),
    "LOAD_DATASET": RouteSpec(nxt="PROPOSE_METADATA", prv="GET_FILE"),
    "PROPOSE_METADATA": RouteSpec(nxt="CONFIRM_METADATA", prv="LOAD_DATASET"),
    "CONFIRM_METADATA": RouteSpec(nxt="DONE", prv="PROPOSE_METADATA"),
    "DONE": RouteSpec(nxt="DONE", prv="CONFIRM_METADATA"),
}


def _require_control(state: ConversationState) -> ControlState:
    if "control" not in state:
        raise KeyError("ConversationState missing 'control'")
    return cast(ControlState, state["control"]) # pyright: ignore[reportUnnecessaryCast]


def _done_node() -> NodeFn:
    """
    Final stage node.

    SimpleWorkflow will:
      - detect node_message/post_action,
      - emit AIMessage,
      - clear node_message,
      - and keep stage/status consistent with cfg.
    """
    def _fn(state: ConversationState) -> ConversationState:
        c = _require_control(state)
        return {
            **state,
            "control": cast(
                ControlState,
                {
                    **c,
                    "stage": "DONE",
                    "status": "DONE",
                    "post_action": "PRESENT",
                    "node_message": (c.get("node_message") or "Done."),
                    "pending_stage": None,
                },
            ),
        }

    return _fn


# =============================================================================
# Builders (CLI should call build_simple_copilot_app)
# =============================================================================
def build_default_nodes(
    *,
    data_repo: DataRepo,
    llm: LLMService,
    mcp_client: McpClient,  # accepted for signature stability / future nodes
) -> Dict[Stage, NodeFn]:
    """
    Stage -> NodeFn mapping for SimpleWorkflow.

    Notes:
      - PROPOSE_METADATA and CONFIRM_METADATA share the same node instance.
      - mcp_client not used yet in these stages, but keeping it avoids churn later.
    """
    _ = mcp_client  # reserved

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
    return SimpleWorkflow(repo=repo, nodes=nodes, cfg=cfg)


# ✅ this is what your CLI should import
def build_simple_copilot_app(
    *,
    repo: ConversationRepo,
    data_repo: DataRepo,
    llm: LLMService,
    mcp_client: McpClient,
) -> SimpleWorkflow:
    return build_simple_workflow(repo=repo, data_repo=data_repo, llm=llm, mcp_client=mcp_client)


__all__ = [
    "RouteSpec",
    "DEFAULT_ROUTES",
    "build_default_nodes",
    "build_workflow_config",
    "build_simple_workflow",
    "build_simple_copilot_app",
]
