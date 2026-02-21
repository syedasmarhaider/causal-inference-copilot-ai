from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence
from uuid import UUID

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.service.llm_service import ChatMessage, LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.route import Router
from python.domain.workflows.state import State, Status

from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.nodes.noop_done.noop_done_state import NoopDoneState

from python.implementation.workflows.router.llm_assisted_router import (
    LLMAssistedRouterRouter,
    init_all_nodoes_with_name_as_key,
)


Stage = str

@dataclass(frozen=True)
class WorkflowRequest:
    user_id: UUID
    conversation_id: UUID
    user_message: Optional[str]


@dataclass(frozen=True)
class WorkflowResponse:
    node_message: Optional[str]
    needs_input: bool
    current_stage: Stage
    current_stage_status: Status


# -----------------------------
# Stores (pointer + per-state payloads + history)
# -----------------------------

class StagePointerStore(Protocol):
    def load_current_stage_name(self, *, user_id: UUID, conversation_id: UUID) -> Optional[str]: ...
    def save_current_stage_name(self, *, user_id: UUID, conversation_id: UUID, stage_name: Optional[str]) -> None: ...


class StateStore(Protocol):
    def load_state(self, *, user_id: UUID, conversation_id: UUID, state_name: str) -> Optional[State]: ...
    def save_state(self, *, user_id: UUID, conversation_id: UUID, state: State) -> None: ...


class HistoryStore(Protocol):
    def load_history(self, *, user_id: UUID, conversation_id: UUID, limit: int = 50) -> Sequence[ChatMessage]: ...


# -----------------------------
# Workflow App (ONE node execution per request)
# -----------------------------

class WorkflowApp:
    def __init__(
        self,
        *,
        router: Router,
        nodes_by_name: Mapping[str, Node],  # expects node.name == State.NAME
        pointer_store: StagePointerStore,
        state_store: StateStore,
        history_store: HistoryStore,
        empty_state_factory: callable,      # (state_name: str) -> State
        initial_stage_name: str = LoadDatasetState.NAME,
        done_stage_name: str = NoopDoneState.NAME,
    ) -> None:
        self._router = router
        self._nodes_by_name = nodes_by_name
        self._pointer_store = pointer_store
        self._state_store = state_store
        self._history_store = history_store
        self._empty_state_factory = empty_state_factory
        self._initial_stage_name = initial_stage_name
        self._done_stage_name = done_stage_name

    def handle(self, req: WorkflowRequest) -> WorkflowResponse:
        # 1) load stage pointer (ONLY NAME is stored)
        stage_name = self._pointer_store.load_current_stage_name(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
        )
        if stage_name is None:
            stage_name = self._initial_stage_name
            self._pointer_store.save_current_stage_name(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                stage_name=stage_name,
            )

        # 2) load current state payload (or init empty)
        current_state = self._state_store.load_state(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            state_name=stage_name,
        )
        if current_state is None:
            current_state = self._empty_state_factory(stage_name)
            self._state_store.save_state(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                state=current_state,
            )

        # 3) load history (pass-through; do NOT format)
        history = self._history_store.load_history(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            limit=50,
        )

        # 4) router decides which stage to execute next
        decision = self._router.decide_next(
            current_state=current_state,
            user_message=req.user_message,
            messages_history=history,
        )

        # If router says "finished/blocked"
        if decision.state_name is None:
            # mark workflow done (pointer only)
            self._pointer_store.save_current_stage_name(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                stage_name=self._done_stage_name,
            )
            done_state = self._state_store.load_state(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                state_name=self._done_stage_name,
            ) or self._empty_state_factory(self._done_stage_name)

            self._state_store.save_state(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                state=done_state,
            )

            return WorkflowResponse(
                node_message=decision.router_message_for_node or done_state.message,
                needs_input=(done_state.needs_action == "NEEDS_INPUT"),
                current_stage=done_state.name,
                current_stage_status=done_state.status,
            )

        next_stage = decision.state_name

        # 5) load/ensure the state we are about to execute
        state_to_run = self._state_store.load_state(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            state_name=next_stage,
        )
        if state_to_run is None:
            state_to_run = self._empty_state_factory(next_stage)
            self._state_store.save_state(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                state=state_to_run,
            )

        # 6) load dependencies required by the state-to-run (pass states directly)
        deps: dict[str, object] = {}
        for dep_name in state_to_run.required_states_keys():
            dep_state = self._state_store.load_state(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                state_name=dep_name,
            )
            if dep_state is not None:
                deps[dep_name] = dep_state

        # 7) execute node for that stage
        try:
            node = self._nodes_by_name[next_stage]
        except KeyError as e:
            raise KeyError(f"WorkflowApp: no node registered for stage/state '{next_stage}'") from e

        new_state = node.run(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            state=state_to_run,
            tool_factory=None,  # <-- pass your real ToolFactory instance if you have one
            previous_state_dependencies=deps,
            user_message=req.user_message,
            router_message=decision.router_message_for_node,
            messages_history=history,
        )

        # 8) persist state payload + persist pointer (ONLY NAME)
        self._state_store.save_state(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            state=new_state,
        )
        self._pointer_store.save_current_stage_name(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            stage_name=new_state.name,
        )

        return WorkflowResponse(
            node_message=new_state.message or decision.router_message_for_node,
            needs_input=(new_state.needs_action == "NEEDS_INPUT"),
            current_stage=new_state.name,
            current_stage_status=new_state.status,
        )


# -----------------------------
# Initializer (wires router + nodes; you plug real stores + empty_state_factory)
# -----------------------------

def init_workflow_app(
    *,
    llm: LLMService,
    data_repo: DataRepo,
    models_repo: ModelsRepo,
    pointer_store: StagePointerStore,
    state_store: StateStore,
    history_store: HistoryStore,
    empty_state_factory: callable,  # (state_name: str) -> State
    model_name: Optional[str] = None,
) -> WorkflowApp:
    router = LLMAssistedRouterRouter(llm=llm, model_name=model_name)
    nodes_by_name = init_all_nodoes_with_name_as_key(llm=llm, data_repo=data_repo, models_repo=models_repo)

    return WorkflowApp(
        router=router,
        nodes_by_name=nodes_by_name,
        pointer_store=pointer_store,
        state_store=state_store,
        history_store=history_store,
        empty_state_factory=empty_state_factory,
        initial_stage_name=LoadDatasetState.NAME,
        done_stage_name=NoopDoneState.NAME,
    )