from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from python.domain.models.errors import ConversationNotFoundError, StateNotFoundError
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.node import Node
from python.domain.workflows.route import Router
from python.domain.workflows.state import State, Status
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import (
    LoadDatasetState,
)

Stage = str

_MESSAGES_HISTORY_LIMIT = 30

@dataclass(frozen=True)
class WorkflowRequest:
    user_id: UUID
    conversation_id: UUID
    user_message: str | None  # raw user text


@dataclass(frozen=True)
class WorkflowResponse:
    node_message: str
    needs_input: bool
    current_stage: Stage
    current_stage_status: Status
    needs_data: bool = False
    artifact_ids: Sequence[str] | None = None


@dataclass(frozen=True)
class ArtifactResponse:
    mime: str
    content: bytes


class WorkflowApp:
    def __init__(
        self,
        *,
        repo: WorkflowStateRepo,
        data_repo: DataRepo,
        router: Router,
        nodes_by_state_name: Mapping[str, Node],
        state_classes_by_name: Mapping[str, type[State]],
        tool_factory: ToolFactory,
        history_limit: int = _MESSAGES_HISTORY_LIMIT,
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
    
    
    def raise_if_userid_not_relates_to_conversation_id(self, *, user_id: UUID, conversation_id: UUID) -> None:
        if not self._repo.is_conversation_id_for_user_id_exists(user_id=user_id, conversation_id=conversation_id):
            raise ConversationNotFoundError(user_id=user_id, conversation_id=conversation_id)
    
    def create_conversation(self, user_id: UUID) -> UUID:
        conversation_id = uuid4()
        self._repo.save_conversation_id(user_id=user_id, conversation_id=conversation_id)
        return conversation_id    
    
    def list_conversations(self, user_id: UUID) -> Sequence[UUID]:
        return self._repo.get_conversation_ids_for_user(user_id=user_id)  
    
    def get_last_conversation_state(self, *, user_id: UUID, conversation_id: UUID) -> WorkflowResponse | None:
        active_name = self._repo.load_active_state_name(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not active_name:
            return None
        
        state = self._repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=active_name,
        )
        if state is None:
            return None
        
        return WorkflowResponse(
            node_message=state.message.txt_message,
            needs_input=(state.message.action == "NEEDS_INPUT"),
            needs_data=(state.message.action == "NEEDS_DATA"),
            current_stage=state.name,
            current_stage_status=state.status,
            artifact_ids=state.message.artifact_ids,
        )      

    def upload_csv_data(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        csv_bytes: bytes,
    ) -> UUID:
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False) # pyright: ignore[reportUnknownMemberType]
        except Exception as exc:
            raise ValueError(f"Uploaded file is not a valid CSV: {exc}") from exc

        active_name = self._repo.load_active_state_name(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not active_name or active_name != LoadDatasetState.NAME:
            raise ValueError(f"No active conversation found for user_id={user_id} and conversation_id={conversation_id} or state is not at load data set")
        
        dataset_id = LoadDatasetState.INIT_DATA_ID
        self._data_repo.save_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
            df=df,
            overwrite=True,
        )
         
        return dataset_id

    def get_artifact(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
    ) -> ArtifactResponse:
        mime = self._data_repo.get_artifact_mime(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
        )
        content = self._data_repo.get_artifact_bytes(
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
            expected_mime=mime,
        )
        return ArtifactResponse(mime=mime, content=content)
    
    def revert_to_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state_name: str,
    ) -> None:
        state = self._repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=state_name,
        )
        if state is None:
            raise StateNotFoundError(state_name=state_name)
        
        state_names = self._router.get_next_state_names(state_name)
        self._repo.delete_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=state_name,
        )
        for name in state_names:
            self._repo.delete_state(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=name,
            )
        
        self._repo.store_active_state_name(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=state_name,
        )    
        
        self._repo.append_message(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message=ChatMessage(role="system", content=f"User reverted to state {state_name}. and fresh start of this state. Deleted states: {', '.join(state_names)}"),
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
            needs_input=(new_state.message.action == "NEEDS_INPUT"),
            needs_data=(new_state.message.action == "NEEDS_DATA"),
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
