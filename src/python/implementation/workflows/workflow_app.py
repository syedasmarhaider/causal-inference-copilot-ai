from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Type
from uuid import UUID

from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.node import Node
from python.domain.workflows.route import Router
from python.domain.workflows.state import State, Status
from python.domain.workflows.tool_factory import ToolFactory

Stage = str


@dataclass(frozen=True)
class WorkflowRequest:
    user_id: UUID
    conversation_id: UUID
    user_message: Optional[str]  # raw user text


@dataclass(frozen=True)
class WorkflowResponse:
    node_message: Optional[str]
    needs_input: bool
    current_stage: Stage
    current_stage_status: Status


class WorkflowApp:
    def __init__(
        self,
        *,
        repo: WorkflowStateRepo,
        router: Router,
        nodes_by_state_name: Mapping[str, Node],
        state_classes_by_name: Mapping[str, Type[State]],
        tool_factory: ToolFactory,
        initial_state_name: str,
        done_state_name: str,
        history_limit: int = 30,
        max_steps_per_call: int = 1,
        persist_assistant_messages: bool = True,
    ) -> None:
        if max_steps_per_call <= 0:
            raise ValueError("max_steps_per_call must be >= 1")

        self._repo = repo
        self._router = router
        self._nodes = dict(nodes_by_state_name)
        self._state_classes = dict(state_classes_by_name)
        self._tool_factory = tool_factory

        self._initial_state_name = initial_state_name
        self._done_state_name = done_state_name
        self._history_limit = history_limit
        self._max_steps_per_call = max_steps_per_call
        self._persist_assistant_messages = persist_assistant_messages

        # sanity: ensure we can build initial/done states
        self._require_state_class(self._initial_state_name)
        self._require_state_class(self._done_state_name)

    def invoke(self, req: WorkflowRequest) -> WorkflowResponse:
        if req.user_message is not None and req.user_message.strip():
            self._repo.append_message(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                message=ChatMessage(role="user", content=req.user_message),
            )

        active_name = self._repo.load_active_state_name(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
        )
        
        if not active_name:
            active_name = self._initial_state_name
            self._repo.store_active_state_name(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                state_name=active_name,
            )

        # 2) load current state payload (or init empty)
        current_state = self._repo.load_state(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            state_name=active_name,
        )
        
        if current_state is None:
            current_state = self._init_empty_state(active_name)
            self._repo.store_state(user_id=req.user_id, conversation_id=req.conversation_id, state=current_state)

        history = self._repo.load_message_history(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            limit=self._history_limit,
        )

        # 4) run up to N nodes
        user_message_for_step: Optional[str] = req.user_message
        last_router_message: 
        

        decision = self._router.decide_next(
                current_state=current_state,
                user_message=user_message_for_step,
                messages_history=history,
        )
            last_router_message = decision.router_message_for_node

            # finished / blocked
            if decision.state_name is None:
                done_state = self._ensure_done_state(req.user_id, req.conversation_id)
                return WorkflowResponse(
                    node_message=last_router_message or done_state.message,
                    needs_input=(done_state.needs_action == "NEEDS_INPUT"),
                    current_stage=done_state.name,
                    current_stage_status=done_state.status,
                )

            state_name_to_run = decision.state_name

            # ensure state exists
            state_to_run = self._repo.load_state(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                state_name=state_name_to_run,
            )
            if state_to_run is None:
                state_to_run = self._init_empty_state(state_name_to_run)
                self._repo.store_state(user_id=req.user_id, conversation_id=req.conversation_id, state=state_to_run)

            # deps for node/state_to_run
            deps = self._load_deps(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                required=state_to_run.required_states_keys(),
            )

            node = self._nodes.get(state_name_to_run)
            if node is None:
                raise KeyError(f"WorkflowApp: no node registered for state_name={state_name_to_run!r}")

            new_state = node.run(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                state=state_to_run,
                tool_factory=self._tool_factory,
                previous_state_dependencies=deps,
                user_message=user_message_for_step,
                router_message=decision.router_message_for_node,
                messages_history=history,
            )

            # persist state + pointer (ONLY NAME)
            self._repo.store_state(user_id=req.user_id, conversation_id=req.conversation_id, state=new_state)
            self._repo.store_active_state_name(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                state_name=new_state.name,
            )

            # optionally append assistant message (what user sees)
            if self._persist_assistant_messages and new_state.message:
                self._repo.append_message(
                    user_id=req.user_id,
                    conversation_id=req.conversation_id,
                    message=ChatMessage(role="assistant", content=new_state.message),
                )
                # refresh history for subsequent steps in same call
                history = self._repo.load_message_history(
                    user_id=req.user_id,
                    conversation_id=req.conversation_id,
                    limit=self._history_limit,
                )

            current_state = new_state
            user_message_for_step = None  # only first node sees the incoming user_message

            # stop if blocked or not auto-advancable
            if current_state.needs_action == "NEEDS_INPUT":
                break
            if current_state.status == "ABORTED":
                break
            if current_state.status != "DONE":
                break
            # else DONE: loop may continue if max_steps_per_call > 1

        return WorkflowResponse(
            node_message=current_state.message or last_router_message,
            needs_input=(current_state.needs_action == "NEEDS_INPUT"),
            current_stage=current_state.name,
            current_stage_status=current_state.status,
        )

    # ------------------------
    # helpers
    # ------------------------

    def _require_state_class(self, state_name: str) -> None:
        if state_name not in self._state_classes:
            raise KeyError(f"WorkflowApp: missing State class for state_name={state_name!r}")

    def _init_empty_state(self, state_name: str) -> State:
        cls = self._state_classes.get(state_name)
        if cls is None:
            raise KeyError(f"WorkflowApp: no State class registered for state_name={state_name!r}")
        st = cls.init_empty()
        if st.name != state_name:
            raise ValueError(f"init_empty() returned name={st.name!r}, expected={state_name!r}")
        return st

    def _load_deps(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        required: Sequence[str],
    ) -> Mapping[str, Any]:
        deps: dict[str, Any] = {}
        for dep_name in required:
            dep_state = self._repo.load_state(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=dep_name,
            )
            if dep_state is not None:
                deps[dep_name] = dep_state
        return deps

    def _ensure_done_state(self, user_id: UUID, conversation_id: UUID) -> State:
        # pointer -> DONE
        self._repo.store_active_state_name(user_id=user_id, conversation_id=conversation_id, state_name=self._done_state_name)

        st = self._repo.load_state(user_id=user_id, conversation_id=conversation_id, state_name=self._done_state_name)
        if st is None:
            st = self._init_empty_state(self._done_state_name)
            self._repo.store_state(user_id=user_id, conversation_id=conversation_id, state=st)
        return st