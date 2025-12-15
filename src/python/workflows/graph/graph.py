# src/python/workflows/graph/graph.py
from __future__ import annotations

import json
import logging
from typing import Any, cast

from langgraph.graph import END, StateGraph  # type: ignore
from langchain_core.messages import BaseMessage, HumanMessage

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMService
from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState, Need, Stage, Status
from python.workflows.nodes.load_dataset import make_load_dataset_node
from python.workflows.nodes.propose_metadata_design import make_propose_metadata_node
from python.workflows.nodes.confirm_metadata import make_confirm_metadata_node
from python.workflows.utils.types import DEFAULT_MODEL_GEMNI
from python.workflows.utils.user_message_builder import build_user_message_with_llm

log = logging.getLogger("python.workflows.graph.graph")

VALID_STAGES: set[Stage] = {
    "GET_FILE",
    "LOAD_DATASET",
    "PROPOSE_METADATA",
    "CONFIRM_METADATA",
    "SELECT_ESTIMATOR",
    "FIT_MODEL",
    "PLAN_EFFECTS",
    "RUN_EFFECTS",
    "DONE",
}

_NEXT_STAGE: dict[Stage, Stage] = {
    "GET_FILE": "LOAD_DATASET",
    "LOAD_DATASET": "PROPOSE_METADATA",
    "PROPOSE_METADATA": "CONFIRM_METADATA",
    "CONFIRM_METADATA": "SELECT_ESTIMATOR",
    "SELECT_ESTIMATOR": "FIT_MODEL",
    "FIT_MODEL": "PLAN_EFFECTS",
    "PLAN_EFFECTS": "RUN_EFFECTS",
    "RUN_EFFECTS": "DONE",
    "DONE": "DONE",
}

_PREV_STAGE: dict[Stage, Stage] = {
    "GET_FILE": "GET_FILE",
    "LOAD_DATASET": "GET_FILE",
    "PROPOSE_METADATA": "LOAD_DATASET",
    "CONFIRM_METADATA": "PROPOSE_METADATA",
    "SELECT_ESTIMATOR": "CONFIRM_METADATA",
    "FIT_MODEL": "SELECT_ESTIMATOR",
    "PLAN_EFFECTS": "FIT_MODEL",
    "RUN_EFFECTS": "PLAN_EFFECTS",
    "DONE": "RUN_EFFECTS",
}


def _control(state: ConversationState) -> ControlState:
    if "control" not in state:
        raise KeyError("ConversationState is missing 'control'")
    return cast(ControlState, state["control"])


def _last_msg_is_human(state: ConversationState) -> bool:
    msgs = cast(list[BaseMessage], state.get("messages", []))
    return bool(msgs) and isinstance(msgs[-1], HumanMessage)


# -----------------------------
# ROUTER (deterministic only)
# -----------------------------
def router_node(state: ConversationState) -> ConversationState:
    """
    Deterministic bookkeeping ONLY:
    - validate stage
    - unlock NEEDS_INPUT if a new human message arrived
    - auto-rewind/advance silently when DONE/ABORTED and nothing to present
    """
    c = _control(state)

    stage: Stage = c["stage"]
    status: Status = c["status"]
    need: Need = c["need"]
    msg = (c.get("node_message") or "").strip()

    if stage not in VALID_STAGES:
        raise ValueError(f"router_node: invalid stage {stage!r}")

    # 1) Unlock when waiting for input and user has replied.
    if need == "NEEDS_INPUT" and _last_msg_is_human(state):
        c2 = cast(ControlState, {**c, "need": "NONE"})
        return {**state, "control": c2}

    # 2) If ABORTED and nothing to present -> rewind immediately (silent)
    if status == "ABORTED" and need == "NONE" and not msg:
        prev_stage = _PREV_STAGE.get(stage, "GET_FILE")
        c2 = cast(ControlState, {**c, "stage": prev_stage, "status": "PENDING"})
        return {**state, "control": c2}

    # 3) If DONE and nothing to present -> advance immediately (silent)
    #    But do NOT auto-advance terminal DONE stage.
    if stage != "DONE" and status == "DONE" and need == "NONE" and not msg:
        nxt = _NEXT_STAGE.get(stage, stage)
        c2 = cast(ControlState, {**c, "stage": nxt, "status": "PENDING"})
        return {**state, "control": c2}

    return state


def route_from_router(state: ConversationState) -> str:
    """
    Turn boundary rules (REST-friendly):
    - NEEDS_INPUT => END immediately
    - PRESENT / PRESENT_AND_USER_INPUT / node_message => go PRESENT (and PRESENT will END)
    - else run the current stage node
    """
    c = _control(state)
    stage: Stage = c["stage"]
    need: Need = c["need"]
    msg = (c.get("node_message") or "").strip()

    if need == "NEEDS_INPUT":
        return "END"

    if need in {"PRESENT", "PRESENT_AND_USER_INPUT"} or msg:
        return "PRESENT"

    return stage if stage in VALID_STAGES else "GET_FILE"


# -----------------------------
# PRESENT (build + flush exactly once, then END)
# -----------------------------
def make_present_node(
    llm: LLMService,
    *,
    model_name: str = DEFAULT_MODEL_GEMNI,
    history_window: int = 12,
) -> Any:
    def present(state: ConversationState) -> ConversationState:
        c = _control(state)

        stage: Stage = c["stage"]
        status: Status = c["status"]
        need: Need = c["need"]

        # Log last_error if present
        err = c.get("last_error")
        if err is not None:
            try:
                log.error(
                    "workflow_error stage=%s conversation_id=%s error=%s",
                    c.get("stage"),
                    str(c.get("conversation_id")),
                    json.dumps(err, ensure_ascii=False),
                )
            except Exception:
                log.error(
                    "workflow_error stage=%s conversation_id=%s error=%r",
                    c.get("stage"),
                    str(c.get("conversation_id")),
                    err,
                )

        # Build assistant message (LLM or template)
        ai = build_user_message_with_llm(
            llm=llm,
            state=state,
            model_name=model_name,
            history_window=history_window,
        )

        prior = cast(list[BaseMessage], state.get("messages", []))
        out_messages = [*prior, ai]

        # After presenting:
        # - clear node_message
        # - if we asked for user input, flip to NEEDS_INPUT
        next_need: Need = "NEEDS_INPUT" if need == "PRESENT_AND_USER_INPUT" else "NONE"

        # Advance/rewind NOW (because PRESENT will END and router won't run again this turn).
        next_stage: Stage = stage
        next_status: Status = status

        pending = cast(Stage | None, c.get("pending_stage")) if "pending_stage" in c else None

        if status == "ABORTED":
            next_stage = _PREV_STAGE.get(stage, "GET_FILE")
            next_status = "PENDING"
        elif pending is not None:
            next_stage = pending
            next_status = "PENDING"
        elif status == "DONE" and stage != "DONE":
            next_stage = _NEXT_STAGE.get(stage, stage)
            next_status = "PENDING"

        c2 = cast(
            ControlState,
            {
                **c,
                "stage": next_stage,
                "status": next_status,
                "need": next_need,
                "node_message": "",          # PRESENT is the only place that clears it
                "pending_stage": None,       # consumed here if used
            },
        )

        return {**state, "control": c2, "messages": out_messages}

    return present


# -----------------------------
# Graph wiring
# -----------------------------
def build_copilot_app(*, data_repo: DataRepo, llm: LLMService) -> Any:
    g = StateGraph(ConversationState)

    g.add_node("ROUTER", router_node)
    g.add_node("PRESENT", make_present_node(llm))

    # GET_FILE (example)
    g.add_node(
        "GET_FILE",
        lambda s: {
            **s,
            "control": {
                **_control(s),
                "status": "PENDING",
                "need": "PRESENT_AND_USER_INPUT",
                "node_message": "Paste the absolute path to your CSV file (ending in .csv).",
            },
        },
    )

    g.add_node("LOAD_DATASET", make_load_dataset_node(data_repo))
    g.add_node("PROPOSE_METADATA", make_propose_metadata_node(llm=llm, data_repo=data_repo))
    g.add_node("CONFIRM_METADATA", make_confirm_metadata_node(llm=llm))

    # stubs
    g.add_node("SELECT_ESTIMATOR", lambda s: {**s, "control": {**_control(s), "status": "DONE"}})
    g.add_node("FIT_MODEL", lambda s: {**s, "control": {**_control(s), "status": "DONE"}})
    g.add_node("PLAN_EFFECTS", lambda s: {**s, "control": {**_control(s), "status": "DONE"}})
    g.add_node("RUN_EFFECTS", lambda s: {**s, "control": {**_control(s), "status": "DONE"}})
    g.add_node(
        "DONE",
        lambda s: {
            **s,
            "control": {**_control(s), "status": "DONE", "need": "PRESENT", "node_message": "All done."},
        },
    )

    g.set_entry_point("ROUTER")

    g.add_conditional_edges(
        "ROUTER",
        route_from_router,
        path_map={
            "GET_FILE": "GET_FILE",
            "LOAD_DATASET": "LOAD_DATASET",
            "PROPOSE_METADATA": "PROPOSE_METADATA",
            "CONFIRM_METADATA": "CONFIRM_METADATA",
            "SELECT_ESTIMATOR": "SELECT_ESTIMATOR",
            "FIT_MODEL": "FIT_MODEL",
            "PLAN_EFFECTS": "PLAN_EFFECTS",
            "RUN_EFFECTS": "RUN_EFFECTS",
            "DONE": "DONE",
            "PRESENT": "PRESENT",
            "END": END,
        },
    )

    # Stage nodes return to ROUTER (same invoke) so we can run silently until we hit a boundary.
    for n in [
        "GET_FILE",
        "LOAD_DATASET",
        "PROPOSE_METADATA",
        "CONFIRM_METADATA",
        "SELECT_ESTIMATOR",
        "FIT_MODEL",
        "PLAN_EFFECTS",
        "RUN_EFFECTS",
        "DONE",
    ]:
        g.add_edge(n, "ROUTER")

    # KEY: PRESENT ends the graph for this turn (REST boundary)
    g.add_edge("PRESENT", END)

    return g.compile()
