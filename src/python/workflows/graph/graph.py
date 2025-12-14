# src/python/workflows/graph/app.py
from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END  # pyright: ignore[reportMissingTypeStubs]

from python.workflows.state.conversation_state import ConversationState
from python.workflows.graph.router import route_by_stage
from python.workflows.graph.advance_stage import advance_stage_node

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMService

from python.workflows.nodes.load_dataset import make_load_dataset_node
from python.workflows.nodes.propose_metadata_design import make_propose_metadata_node
from python.workflows.nodes.confirm_metadata import make_confirm_metadata_node
from python.workflows.nodes.present_user_message import make_present_user_message_node


def build_copilot_app(*, data_repo: DataRepo, llm: LLMService) -> Any:
    g = StateGraph(ConversationState)

    # Router owns transitions (stage advancement)
    g.add_node("ROUTER", advance_stage_node)

    # Stage nodes (do NOT mutate control.stage)
    g.add_node("LOAD_DATASET", make_load_dataset_node(data_repo))
    g.add_node("PROPOSE_METADATA", make_propose_metadata_node(llm=llm, data_repo=data_repo))
    g.add_node("CONFIRM_METADATA", make_confirm_metadata_node(llm=llm))

    # Stubs for now
    g.add_node("SELECT_ESTIMATOR", lambda s: s)
    g.add_node("FIT_MODEL", lambda s: s)
    g.add_node("PLAN_EFFECTS", lambda s: s)
    g.add_node("RUN_EFFECTS", lambda s: s)
    g.add_node("DONE", lambda s: s)

    # Presenter runs at end of each turn
    g.add_node("PRESENT", make_present_user_message_node(llm))

    g.set_entry_point("ROUTER")

    # ROUTER decides *which stage node to run*, based on control.stage
    g.add_conditional_edges(
        "ROUTER",
        route_by_stage,
        {
            "LOAD_DATASET": "LOAD_DATASET",
            "PROPOSE_METADATA": "PROPOSE_METADATA",
            "CONFIRM_METADATA": "CONFIRM_METADATA",
            "SELECT_ESTIMATOR": "SELECT_ESTIMATOR",
            "FIT_MODEL": "FIT_MODEL",
            "PLAN_EFFECTS": "PLAN_EFFECTS",
            "RUN_EFFECTS": "RUN_EFFECTS",
            "DONE": "DONE",
        },
    )

    # After any stage-node: present exactly one message and end.
    for node in [
        "LOAD_DATASET",
        "PROPOSE_METADATA",
        "CONFIRM_METADATA",
        "SELECT_ESTIMATOR",
        "FIT_MODEL",
        "PLAN_EFFECTS",
        "RUN_EFFECTS",
        "DONE",
    ]:
        g.add_edge(node, "PRESENT")

    g.add_edge("PRESENT", END)

    return g.compile()
