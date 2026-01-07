from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict,  cast

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMService
from python.domain.service.mcp_client import McpClient

from python.workflows.state.conversation_state import ConversationState
from python.workflows.state.control_state import ControlState, Stage

from python.workflows.nodes.get_file import make_get_file_node
from python.workflows.nodes.load_dataset import make_load_dataset_node
from python.workflows.nodes.propose_and_confirm_metadata import make_propose_and_confirm_metadata
from python.workflows.nodes.validate_backdoor import make_validate_backdoor_node


NodeFn = Callable[[ConversationState], ConversationState]


@dataclass(frozen=True)
class _RouteSpec:
    nxt: Stage
    prv: Stage

_DEFAULT_ROUTES: dict[Stage, _RouteSpec] = {
    "GET_FILE": _RouteSpec(nxt="LOAD_DATASET", prv="GET_FILE"),
    "LOAD_DATASET": _RouteSpec(nxt="PROPOSE_METADATA", prv="GET_FILE"),
    "PROPOSE_METADATA": _RouteSpec(nxt="CONFIRM_METADATA", prv="LOAD_DATASET"),
    "CONFIRM_METADATA": _RouteSpec(nxt="VALIDATE_BACKDOOR", prv="PROPOSE_METADATA"),
    "VALIDATE_BACKDOOR": _RouteSpec(nxt="DONE", prv="CONFIRM_METADATA"),
}

def _require_control(state: ConversationState) -> ControlState:
    if "control" not in state:
        raise KeyError("ConversationState missing 'control'")
    return cast(ControlState, state["control"]) # pyright: ignore[reportUnnecessaryCast]


class StaticRouter:
    """
    Static router that contains BOTH:
      - routing logic (based on control.stage + control.status)
      - node initialization (nodes are built in __init__ from deps)

    Usage:
      router = StaticRouter(data_repo=..., llm=..., mcp_client=...)
      node_fn = router.getStage(state)   # returns a callable
      state = node_fn(state)             # executes exactly one node
      # or simply:
      state = router.step(state)

    Routing rules (driven by ControlState):
      1) pending_stage overrides everything
      2) awaiting_user latch prevents movement (stay on current stage)
      3) status:
         - PENDING         => run current stage
         - DONE            => run next stage
         - ABORTED         => run suggested stage if present else prev stage
         - RETRYABLE_ERROR => run suggested stage if present else current stage

    Important invariant:
      The returned NodeFn is wrapped so that control.stage is aligned to the node being executed,
    """

    def __init__(self, *, data_repo: DataRepo, llm: LLMService, mcp_client: McpClient) -> None:
        meta_node = make_propose_and_confirm_metadata(llm=llm, data_repo=data_repo)

        self._nodes: Dict[Stage, NodeFn] = {
            "GET_FILE": make_get_file_node(llm),
            "LOAD_DATASET": make_load_dataset_node(data_repo),
            "PROPOSE_METADATA": meta_node,
            "CONFIRM_METADATA": meta_node,
            "VALIDATE_BACKDOOR": make_validate_backdoor_node(mcp_client),
            "DONE": self._done_node(),
        }

    # ----------------------------
    # Public API
    # ----------------------------

    def getStage(self, state: ConversationState) -> NodeFn:
        """
        Return the NodeFn to execute next (does NOT execute it).
        """
        ctrl = _require_control(state)
        target = self._choose_target(ctrl)

        node = self._nodes.get(target)
        if node is None:
            return self._missing_node_fn(expected_stage=target)

        return self._wrap(node=node, target=target)

    def step(self, state: ConversationState) -> ConversationState:
        """
        Convenience: chooses the next node and executes it once.
        """
        return self.getStage(state)(state)

    # ----------------------------
    # Routing
    # ----------------------------

    def _choose_target(self, ctrl: ControlState) -> Stage:
        cur: Stage = ctrl["stage"]

        pending = ctrl.get("pending_stage", None)
        if pending:
            return pending

        if ctrl.get("awaiting_user", False):
            return cur

        route = _DEFAULT_ROUTES.get(cur)
        if route is None:
            # Unknown stage -> safest is "stay"
            return cur

        status = ctrl["status"]

        if status == "DONE":
            return route.nxt

        if status == "ABORTED":
            return cast(Stage, ctrl.get("post_failure_suggested_stage", None) or route.prv) # pyright: ignore[reportUnnecessaryCast]

        if status == "RETRYABLE_ERROR":
            return cast(Stage, ctrl.get("post_failure_suggested_stage", None) or cur) # pyright: ignore[reportUnnecessaryCast]

        # PENDING (or any unknown status) => run current stage
        return cur

    # ----------------------------
    # Wrapping / execution alignment
    # ----------------------------

    def _wrap(self, *, node: NodeFn, target: Stage) -> NodeFn:
        """
        Ensure control.stage is correct BEFORE executing `node`.
        Also clears pending_stage so overrides don't stick.
        """

        def _run(s: ConversationState) -> ConversationState:
            c = _require_control(s)

            if c["stage"] != target:
                new_c = cast(
                    ControlState,
                    {
                        **c,
                        "stage": target,
                        "status": "PENDING",
                        "pending_stage": None,
                        "awaiting_user": False,
                        "last_error": None,
                        "post_failure_suggested_stage": None,
                    },
                )
            else:
                new_c = cast(ControlState, {**c, "pending_stage": None})

            return node({**s, "control": new_c})

        return _run

    # ----------------------------
    # Node helpers
    # ----------------------------

    def _done_node(self) -> NodeFn:
        def _fn(state: ConversationState) -> ConversationState:
            ctrl = _require_control(state)
            return {
                **state,
                "control": cast(
                    ControlState,
                    {
                        **ctrl,
                        "stage": "DONE",
                        "status": "DONE",
                        "post_action": "PRESENT",
                        "node_message": "✅ Done.",
                        "pending_stage": None,
                    },
                ),
            }

        return _fn

    def _not_implemented(self, stage: Stage) -> NodeFn:
        def _fn(state: ConversationState) -> ConversationState:
            ctrl = _require_control(state)
            return {
                **state,
                "control": cast(
                    ControlState,
                    {
                        **ctrl,
                        "stage": stage,
                        "status": "ABORTED",
                        "post_action": "PRESENT",
                        "post_failure_suggested_stage": ctrl["stage"],
                        "last_error": {"code": "NOT_IMPLEMENTED", "detail": f"{stage} not implemented"},
                        "node_message": f"Stage `{stage}` not implemented yet.",
                        "pending_stage": None,
                    },
                ),
            }

        return _fn

    def _missing_node_fn(self, *, expected_stage: Stage) -> NodeFn:
        def _fn(state: ConversationState) -> ConversationState:
            ctrl = _require_control(state)
            return {
                **state,
                "control": cast(
                    ControlState,
                    {
                        **ctrl,
                        "status": "ABORTED",
                        "post_action": "PRESENT",
                        "post_failure_suggested_stage": ctrl["stage"],
                        "last_error": {
                            "code": "MISSING_NODE_FOR_STAGE",
                            "detail": f"No node registered for stage {expected_stage!r}.",
                        },
                        "node_message": f"Fatal: no node registered for stage `{expected_stage}`.",
                        "pending_stage": None,
                    },
                ),
            }

        return _fn
