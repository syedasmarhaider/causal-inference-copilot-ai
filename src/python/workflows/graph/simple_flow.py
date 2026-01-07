from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ACTION, NEED_STAGE, ControlState, Stage, Status

log = logging.getLogger(__name__)

NodeFn = Callable[[ConversationState], ConversationState]
PresenterFn = Callable[[ConversationState], AIMessage]


def _control(state: ConversationState) -> ControlState:
    if "control" not in state:
        raise KeyError("ConversationState missing 'control'")
    return cast(ControlState, state["control"]) # pyright: ignore[reportUnnecessaryCast]


def _msgs(state: ConversationState) -> list[BaseMessage]:
    return cast(list[BaseMessage], state.get("messages", [])) # pyright: ignore[reportUnnecessaryCast]


def _last_is_human(state: ConversationState) -> bool:
    m = _msgs(state)
    return bool(m) and isinstance(m[-1], HumanMessage)


def _node_message(c: ControlState) -> str:
    return str(c.get("node_message") or "").strip()


def _post_action(c: ControlState) -> ACTION:
    return cast(ACTION, c.get("post_action") or "NONE")


def _status(c: ControlState) -> Status:
    return cast(Status, c.get("status") or "PENDING")


def _should_present(c: ControlState) -> bool:
    # Present if node requested it OR node_message exists.
    a = _post_action(c)
    return bool(_node_message(c)) or a in {"PRESENT", "PRESENT_AND_USER_INPUT"}


def _log_error_if_any(c: ControlState) -> None:
    err = c.get("last_error")
    if err is None:
        return
    try:
        log.error(
            "workflow_error stage=%s conversation_id=%s error=%r",
            c.get("stage"),
            str(c.get("conversation_id")),
            err,
        )
    except Exception:
        log.error("workflow_error error_logging_failed")


@dataclass(frozen=True)
class WorkflowConfig:
    next_stage: Mapping[Stage, Stage]
    prev_stage: Mapping[Stage, Stage]
    valid_stages: set[Stage]
    max_internal_steps: int = 32  # guardrail against bad loops


class SimpleWorkflowApp:
    """
    Deterministic state-machine runner (no LangGraph).

    Semantics (important):
      - NEEDS_INPUT is the ONLY waiting latch.
      - Nodes request presentation via:
          * post_action = PRESENT                 (emit message, keep going)
          * post_action = PRESENT_AND_USER_INPUT  (emit message, then latch NEEDS_INPUT)
        Node may also set node_message; presenter will use it as signal/context.

      - Presenter ALWAYS runs when we present (NLP output),
        with fallback to raw node_message only if presenter fails.

      - One invoke() may:
          * run multiple silent stages
          * emit multiple AI messages
        but stops immediately once NEEDS_INPUT latch is set.
    """

    def __init__(self, *, nodes: Dict[Stage, NodeFn], cfg: WorkflowConfig, presenter: PresenterFn):
        self._nodes = nodes
        self._cfg = cfg
        self._presenter = presenter

    def invoke(self, state: ConversationState) -> ConversationState:
        for _ in range(self._cfg.max_internal_steps):
            c = _control(state)
            stage: Stage = cast(Stage, c["stage"]) # pyright: ignore[reportUnnecessaryCast]

            if stage not in self._cfg.valid_stages:
                raise ValueError(f"Invalid stage {stage!r}")

            # -----------------------------
            # Waiting latch (NEEDS_INPUT)
            # -----------------------------
            if _post_action(c) == "NEEDS_INPUT":
                # If user replied, unlock and continue running.
                if _last_is_human(state):
                    c2 = cast(
                        ControlState,
                        {
                            **c,
                            "post_action": "NONE",
                            "status": "PENDING",
                        },
                    )
                    state = {**state, "control": c2}
                    continue

                # Otherwise stop this turn (do not run nodes).
                return state

            # -----------------------------
            # Run the stage node
            # -----------------------------
            node = self._nodes.get(stage)
            if node is None:
                raise KeyError(f"No node registered for stage {stage!r}")

            state2 = node(state)
            c2 = _control(state2)

            # -----------------------------
            # Present boundary (NLP presenter)
            # -----------------------------
            if _should_present(c2):
                state3 = self._present(state2)

                # If we latched NEEDS_INPUT, stop immediately.
                if _post_action(_control(state3)) == "NEEDS_INPUT":
                    return state3

                # Otherwise continue auto-running next stages in same invoke.
                state = state3
                continue

            # -----------------------------
            # Silent transitions
            # -----------------------------
            st = _status(c2)

            if st == "ABORTED":
                suggested = cast(NEED_STAGE | None, c2.get("post_failure_suggested_stage"))
                fallback = self._cfg.prev_stage.get(stage, stage)
                nxt = cast(Stage, suggested) if suggested is not None else fallback

                c3 = cast(
                    ControlState,
                    {
                        **c2,
                        "stage": nxt,
                        "status": "PENDING",
                        "pending_stage": None,
                    },
                )
                state = {**state2, "control": c3}
                continue

            if stage != "DONE" and st == "DONE":
                pending = cast(Optional[Stage], c2.get("pending_stage"))
                nxt = pending if pending is not None else self._cfg.next_stage.get(stage, stage)

                c3 = cast(
                    ControlState,
                    {
                        **c2,
                        "stage": nxt,
                        "status": "PENDING",
                        "pending_stage": None,
                    },
                )
                state = {**state2, "control": c3}
                continue

            # Nothing else to do this turn.
            return state2

        log.warning("SimpleWorkflowApp.invoke hit max_internal_steps=%s", self._cfg.max_internal_steps)
        return state

    def _present(self, state: ConversationState) -> ConversationState:
        c = _control(state)
        stage: Stage = cast(Stage, c["stage"])
        st: Status = _status(c)
        action: ACTION = _post_action(c)

        _log_error_if_any(c)

        # Always NLP-present (fallback only if presenter fails)
        try:
            ai = self._presenter(state)
            if not isinstance(ai, AIMessage):
                ai = AIMessage(content=str(getattr(ai, "content", "") or ""))
        except Exception:
            # Hard fallback: never lose output
            text = _node_message(c) or "(no message)"
            ai = AIMessage(content=text)

        prior = _msgs(state)
        out_messages = [*prior, ai]

        # Decide next stage/status AFTER presenting
        next_stage: Stage = stage
        next_status: Status = st

        pending = cast(Optional[Stage], c.get("pending_stage"))
        suggested = cast(NEED_STAGE | None, c.get("post_failure_suggested_stage"))

        if st == "ABORTED":
            next_stage = cast(Stage, suggested) if suggested is not None else self._cfg.prev_stage.get(stage, stage)
            next_status = "PENDING"
        elif pending is not None:
            next_stage = pending
            next_status = "PENDING"
        elif st == "DONE" and stage != "DONE":
            next_stage = self._cfg.next_stage.get(stage, stage)
            next_status = "PENDING"

        # Convert PRESENT_AND_USER_INPUT -> NEEDS_INPUT latch
        wants_user = action == "PRESENT_AND_USER_INPUT"
        next_action: ACTION = "NEEDS_INPUT" if wants_user else "NONE"

        c2 = cast(
            ControlState,
            {
                **c,
                "stage": next_stage,
                "status": next_status,
                "post_action": next_action,
                "node_message": "",
                "pending_stage": None,
            },
        )
        return {**state, "control": c2, "messages": out_messages}
