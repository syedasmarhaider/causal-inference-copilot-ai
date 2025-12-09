# TODO: will implemment later, it will take user message and see what needs to be done and which node user is currently at
# and which action user wants to perform# src/python/workflows/nodes/understand_user_request.py
from __future__ import annotations

from typing import Literal

from workflows.state.conversation_state import ConversationState

# Labels used by the router. These will be referenced in the graph wiring.
RouteLabel = Literal["NEED_DATASET", "NEED_METADATA", "READY_FOR_INFERENCE"]


def understand_user_request(state: ConversationState) -> ConversationState:
    """
    Very simple first version.

    - If analysis_goal is not set, assume a full backdoor CATE pipeline.
    - We don't look at the text of the user request yet.
      (LLM-based interpretation can be added later.)

    Routing itself is done by `entry_router` below.
    """
    if "analysis_goal" not in state or not state["analysis_goal"]:
        return {"analysis_goal": "FULL_PIPELINE"}

    # Nothing to change in state
    return {}


def entry_router(state: ConversationState) -> RouteLabel:
    """
    Decide which *phase* to start from, based on what we already have in state.

    For now:
      - No dataset_id      -> we must load + validate dataset.
      - dataset_id only    -> we must infer/persist metadata.
      - dataset_id+metadata -> we can skip straight to estimator / effect planning.

    Later, this can become smarter (look at user message, resume runs, etc.).
    """
    if "dataset_id" not in state:
        return "NEED_DATASET"

    if "metadata" not in state:
        return "NEED_METADATA"

    # We already have dataset + metadata; start from estimator selection / inference
    return "READY_FOR_INFERENCE"
