from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Type
from uuid import UUID

from python.domain.repo.data_repo import ArtifactRef, DataRepo
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
    node_message: str
    needs_input: bool
    current_stage: Stage
    current_stage_status: Status
    artifact_ids: Optional[Sequence[str]] = None


class WorkflowApp:
    def __init__(
        self,
        *,
        repo: WorkflowStateRepo,
        data_repo: DataRepo,
        router: Router,
        nodes_by_state_name: Mapping[str, Node],
        state_classes_by_name: Mapping[str, Type[State]],
        tool_factory: ToolFactory,
        history_limit: int = 30,
        max_steps_per_call: int = 1,
    ) -> None:
        if max_steps_per_call <= 0:
            raise ValueError("max_steps_per_call must be >= 1")

        self._repo = repo
        self._data_repo = data_repo
        self._router = router
        self._nodes = dict(nodes_by_state_name)
        self._state_classes = dict(state_classes_by_name)
        self._tool_factory = tool_factory
        self._history_limit = history_limit
        self._max_steps_per_call = max_steps_per_call
    
    def get_artifact(self, user_id: UUID, conversation_id: UUID, artifact_id: UUID) -> ArtifactRef:
        return self._data_repo.get_artifact_ref(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
        ) 
    
    # TODO: this will be removed just now for init data
    def create_init_files(self, user_id: UUID, conversation_id: UUID) -> None: 
        df= self._data_repo.get_csv_data(
            user_id=UUID("486f4975-6cd9-4261-a122-e6b0fc46462d"),
            conversation_id=UUID("486f4975-6cd9-4261-a122-e6b0fc46462d"),
            dataset_id=UUID("486f4975-6cd9-4261-a122-e6b0fc46462d"),
        )
        
        self._data_repo.save_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=UUID("486f4975-6cd9-4261-a122-e6b0fc46462d"),
            df=df,
        )
        
    def handle(self, req: WorkflowRequest) -> WorkflowResponse:
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
            active_name = self._router.get_initial_state_name()
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

        history: list[ChatMessage] = list(self._repo.load_message_history(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            limit=self._history_limit,
        ))
        
        decision = self._router.decide_next(
                current_state=current_state,
                messages_history=history,
           )
    
        last_router_message = decision.router_message_for_node
        
        if last_router_message:
            history.append(ChatMessage(role="system", content=last_router_message))
            self._repo.append_message(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                message=ChatMessage(role="system", content=last_router_message),
            )
        
        #TODO: Temp sol delete later
        # this will not work sometimes on errors and exceptions
        if decision.delete_next_states_names:
            for state_name in decision.delete_next_states_names:
                self._repo.delete_state(
                    user_id=req.user_id,
                    conversation_id=req.conversation_id,
                    state_name=state_name,
                )
                    

        state_name_to_run = decision.state_name 
        state_to_run = self._repo.load_state(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                state_name=state_name_to_run,
            )
        if current_state.status == "ABORTED" or state_to_run is None:
            state_to_run = self._init_empty_state(state_name_to_run)
        
        deps = self._load_deps(
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            required=state_to_run.pre_required_states_names(),
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
            messages_history=history,
            )
        
        self._repo.store_state(user_id=req.user_id, conversation_id=req.conversation_id, state=new_state)
        self._repo.store_active_state_name(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                state_name=new_state.name,
        )
        

        self._repo.append_message(
                    user_id=req.user_id,
                    conversation_id=req.conversation_id,
                    message=ChatMessage(role="assistant", content=new_state.message.txt_message),
            
        )
        
        if new_state.error:
            self._repo.append_message(
                    user_id=req.user_id,
                    conversation_id=req.conversation_id,
                    message=ChatMessage(role="system", content=f"Error returned from node {new_state.name}: {new_state.error}"),
        )
            
        return WorkflowResponse(
            node_message=new_state.message.txt_message,
            needs_input=(new_state.needs_action == "NEEDS_INPUT"),
            current_stage=new_state.name,
            current_stage_status=new_state.status,
            artifact_ids=new_state.message.artifact_ids,
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