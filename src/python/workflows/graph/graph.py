from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END  # pyright: ignore[reportMissingTypeStubs]

from python.workflows.state.conversation_state import ConversationState
from python.workflows.graph.router import route_by_stage
from python.workflows.graph.ochestrator import advance_stage

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMService

from python.workflows.nodes.load_dataset import make_load_dataset_node
from python.workflows.nodes.propose_metadata_design import make_propose_metadata_node
from python.workflows.nodes.confirm_metadata import make_confirm_metadata_node
from python.workflows.nodes.present_user_message import make_present_user_message_node


def build_copilot_app(*, data_repo: DataRepo, llm: LLMService) -> Any:
    g = StateGraph(ConversationState)

    g.add_node("ROUTER", lambda s: s)

    g.add_node("LOAD_DATASET", make_load_dataset_node(data_repo))
    g.add_node("PROPOSE_METADATA", make_propose_metadata_node(llm=llm, data_repo=data_repo))
    g.add_node("CONFIRM_METADATA", make_confirm_metadata_node(llm=llm))

    # stubs for now
    g.add_node("SELECT_ESTIMATOR", lambda s: s)
    g.add_node("FIT_MODEL", lambda s: s)
    g.add_node("PLAN_EFFECTS", lambda s: s)
    g.add_node("RUN_EFFECTS", lambda s: s)
    g.add_node("DONE", lambda s: s)

    g.add_node("ADVANCE_STAGE", lambda s: {**s, "control": advance_stage(s["control"])})
    g.add_node("PRESENT", make_present_user_message_node(llm))

    g.set_entry_point("ROUTER")

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
        g.add_edge(node, "ADVANCE_STAGE")

    def _pause_or_continue(state: ConversationState) -> str:
        c = state["control"]
        if c["status"] == "ERROR":
            return "PRESENT"
        if c["stage"] == "DONE":
            return "PRESENT"
        if c["need"] != "NONE":
            return "PRESENT"
        return "ROUTER"

    g.add_conditional_edges("ADVANCE_STAGE", _pause_or_continue, {"PRESENT": "PRESENT", "ROUTER": "ROUTER"})
    g.add_edge("PRESENT", END)

    return g.compile()
