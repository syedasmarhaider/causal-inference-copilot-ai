from langgraph.graph import StateGraph # pyright: ignore[reportMissingTypeStubs]

from workflows.state.conversation_state import ConversationState
from workflows.nodes.understand_user_request import (
    understand_user_request,
    entry_router,
)
from python.workflows.nodes.load_dataset import load_and_validate_dataset
from workflows.nodes.infer_initial_metadata_design import infer_initial_metadata_design
from workflows.nodes.list_estimators_via_mcp import list_estimators_via_mcp

graph = StateGraph(ConversationState)

graph.add_node("understand_user_request", understand_user_request)
graph.add_node("load_and_validate_dataset", load_and_validate_dataset)
graph.add_node("infer_initial_metadata_design", infer_initial_metadata_design)
graph.add_node("list_estimators_via_mcp", list_estimators_via_mcp)

graph.set_entry_point("understand_user_request")

graph.add_conditional_edges(
    "understand_user_request",
    entry_router,
    {
        "NEED_DATASET": "load_and_validate_dataset",
        "NEED_METADATA": "infer_initial_metadata_design",
        "READY_FOR_INFERENCE": "list_estimators_via_mcp",
    },
)
