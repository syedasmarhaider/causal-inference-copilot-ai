from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from python.domain.models.errors import ConversationNotFoundError, StateNotFoundError
from python.domain.models.models import (
    ArtifactRef,
    ChatMessage,
)
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.workflows.node import Node
from python.domain.workflows.route import Router
from python.domain.workflows.state import Action, State, Status
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.dataflow_app import (
    DataflowArtifactResponse,
)

_MESSAGES_HISTORY_LIMIT = 15

# TODO: add distributed tnx or locks later

@dataclass(frozen=True)
class WorkflowResponse:
    _current_state: State
    _assistant_messages_override: Sequence[ChatMessage] | None = None
    _current_stage_name_override: str | None = None
    _current_stage_status_override: Status | None = None
    _action_override: Action | None = None

    @property
    def messages(self) -> Sequence[ChatMessage]:
        if self._assistant_messages_override is not None:
            return tuple(self._assistant_messages_override)
        return tuple(_assistant_messages_for_user(self._current_state))

    @property
    def current_stage_name(self) -> str:
        if self._current_stage_name_override is not None:
            return self._current_stage_name_override
        return self._current_state.name()
    
    @property
    def current_stage_status(self) -> Status:
        if self._current_stage_status_override is not None:
            return self._current_stage_status_override
        return self._current_state.status()
    
    @property
    def action(self) -> Action:
        if self._action_override is not None:
            return self._action_override
        return self._current_state.action()
    
    @property
    def artifact_refs(self) -> Sequence[ArtifactRef] | None:
        artifact_refs: list[ArtifactRef] = []
        for message in self.messages:
            artifact_refs.extend(list(message.artifact_refs or ()))
        return artifact_refs or None
    

ArtifactResponse = DataflowArtifactResponse


class WorkflowApp:
    def __init__(
        self,
        *,
        repo: WorkflowStateRepo,
        router: Router,
        nodes_by_state_name: Mapping[str, Node],
        state_classes_by_name: Mapping[str, type[State]],
        tool_factory: ToolFactory | None = None,
        history_limit: int = _MESSAGES_HISTORY_LIMIT,
        max_steps_per_call: int = 1,
    ) -> None:
        if max_steps_per_call <= 0:
            raise ValueError("max_steps_per_call must be >= 1")

        self._repo = repo
        self._router = router
        self._nodes = dict(nodes_by_state_name)
        self._state_classes = dict(state_classes_by_name)
        self._tool_factory = tool_factory
        self._history_limit = history_limit
        self._max_steps_per_call = max_steps_per_call
        self._log = get_app_logger(
            __name__,
            component=self.__class__.__name__,
            log_type="workflow_service",
        )
        self._log.info(
            "workflow app initialized",
            nodes_count=len(self._nodes),
            state_classes_count=len(self._state_classes),
            history_limit=self._history_limit,
            max_steps_per_call=self._max_steps_per_call,
        )

    def raise_if_userid_not_relates_to_conversation_id(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> None:
        if not self._repo.is_conversation_id_for_user_id_exists(
            user_id=user_id,
            conversation_id=conversation_id,
        ):
            self._log.info(
                "conversation ownership check failed",
                user_id=user_id,
                conversation_id=conversation_id,
            )
            raise ConversationNotFoundError(user_id=user_id, conversation_id=conversation_id)

    def create_conversation(self, user_id: UUID) -> UUID:
        conversation_id = uuid4()
        self._repo.save_conversation_id(user_id=user_id, conversation_id=conversation_id)
        self._log.info(
            "conversation created",
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return conversation_id

    def list_conversations(self, user_id: UUID) -> Sequence[UUID]:
        return self._repo.get_conversation_ids_for_user(user_id=user_id)

    def get_last_conversation_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> WorkflowResponse | None:
        active_name = self._repo.load_active_state_name(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not active_name:
            self._log.debug(
                "latest conversation state not found because active state is missing",
                user_id=user_id,
                conversation_id=conversation_id,
            )
            return None

        state = self._repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=active_name,
        )
        if state is None:
            self._log.info(
                "active state name exists but state payload is missing",
                user_id=user_id,
                conversation_id=conversation_id,
                active_state_name=active_name,
            )
            return None

        return WorkflowResponse(_current_state=state)

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
            self._log.info(
                "revert requested for missing state",
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=state_name,
            )
            raise StateNotFoundError(state_name=state_name)

        state_names = self._router.get_next_state_names(state_name)
        self._repo.delete_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=state_name,
        )
        for next_state_name in state_names:
            self._repo.delete_state(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=next_state_name,
            )

        self._repo.store_active_state_name(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=state_name,
        )

        self._repo.append_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message=ChatMessage(
                role="system",
                content=(
                    f"User reverted to state {state_name}. Fresh start of this state. "
                    f"Deleted states: {', '.join(state_names)}"
                ),
            ),
        )
        self._log.info(
            "conversation reverted to state",
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=state_name,
            deleted_states_count=len(state_names),
        )

    def handle(self, 
               user_id: UUID,
               conversation_id: UUID,
               user_message: str | None,
               ) -> WorkflowResponse:
        self._log.debug(
            "workflow handle requested",
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
        )
        if user_message is not None and user_message.strip():
            self._repo.append_message(
                user_id=user_id,
                conversation_id=conversation_id,
                message=ChatMessage(role="user", content=user_message),
            )

        history = list(
            self._repo.load_message_history(
                user_id=user_id,
                conversation_id=conversation_id,
                limit=self._history_limit,
            )
        )

        state_name_to_route = self._repo.load_active_state_name(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        state_name_to_run: str
        state_to_run: State
        pre_state_to_run_status: Status
        active_state_name_before_run = state_name_to_route
        active_state_status_before_run: Status | None = None

        if not state_name_to_route:
            state_name_to_run = self._router.get_initial_state_name()
            self._repo.store_active_state_name(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=state_name_to_run,
            )
            state_to_run = self._load_or_init_state(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=state_name_to_run,
            )
        else:
            current_state = self._load_or_init_state(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=state_name_to_route,
            )
            active_state_status_before_run = current_state.status()
            decision = self._router.decide_next(
                current_state=current_state,
                messages_history=history,
            )
            if decision.state_name is None:
                confirmation_message = (
                    decision.router_confirmation_message_for_user
                    or "I need a bit more clarification before I can route this request."
                )
                # Persist the router clarification so history stays coherent even when execution stops.
                self._repo.append_message(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message=ChatMessage(role="assistant", content=confirmation_message),
                )
                return WorkflowResponse(
                    _current_state=current_state,
                    _assistant_messages_override=[
                        ChatMessage(role="assistant", content=confirmation_message),
                    ],
                    _current_stage_name_override=state_name_to_route,
                    _current_stage_status_override="PENDING",
                    _action_override="NEEDS_INPUT",
                )
                                
            state_name_to_run = decision.state_name
            state_to_run = self._load_or_init_state(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=state_name_to_run,
            )

        deps = self._load_deps(
            user_id=user_id,
            conversation_id=conversation_id,
            required=state_to_run.pre_required_states_names(),
        )
        node = self._nodes.get(state_name_to_run)
        if node is None:
            self._log.error(
                "workflow node is not registered for state",
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=state_name_to_run,
            )
            raise KeyError(f"WorkflowApp: no node registered for state_name={state_name_to_run!r}")

        pre_state_to_run_status = state_to_run.status()
        new_state = node.run(
            user_id=user_id,
            conversation_id=conversation_id,
            state=state_to_run,
            previous_state_dependencies=deps,
            messages_history=history,
        )

        new_state_name = new_state.name()
        new_state_status = new_state.status()

        # If the routed state was already frozen before execution, treat it as a read-only detour and
        # keep the active pointer where it was. The only exception is when the returned state itself
        # is frozen, in which case that frozen state becomes the new active checkpoint.
        active_state_name_after_run = active_state_name_before_run
        active_state_status_after_run = active_state_status_before_run
        if new_state_status == "FREEZED" or pre_state_to_run_status != "FREEZED":
            active_state_name_after_run = new_state_name
            active_state_status_after_run = new_state_status
        elif active_state_name_before_run == new_state_name:
            # Running the currently active frozen state in place rewrites that state's payload, so the
            # response should reflect the newly stored status even though the active pointer does not move.
            active_state_status_after_run = new_state_status
        
        
             
        if new_state.status() == "DONE":
            new_state.set_status_freez()
            new_state_status = new_state.status()
            if active_state_name_after_run == new_state_name:
                active_state_status_after_run = new_state_status
        self._repo.store_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state=new_state,
        )

        if (
            active_state_name_after_run is not None
            and active_state_name_after_run != active_state_name_before_run
        ):
            self._repo.store_active_state_name(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=active_state_name_after_run,
            )

        ordered_history_messages = _ordered_history_messages(new_state)
        if ordered_history_messages:
            self._repo.append_messages(
                user_id=user_id,
                conversation_id=conversation_id,
                messages=ordered_history_messages,
            )

        state_error = new_state.error()
        if state_error is not None:
            self._log.error(
                "node returned workflow error",
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=new_state_name,
                state_status=new_state_status,
                node_error=state_error.error,
            )
        else:
            self._log.debug(
                "workflow node completed",
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=new_state_name,
                state_status=new_state_status,
                assistant_messages_count=len(_assistant_messages_for_user(new_state)),
            )    

        return WorkflowResponse(
            _current_state=new_state,
            _current_stage_name_override=active_state_name_after_run or new_state_name,
            _current_stage_status_override=active_state_status_after_run or new_state_status,
        )

    def _init_empty_state(self, state_name: str) -> State:
        cls = self._state_classes.get(state_name)
        if cls is None:
            raise KeyError(f"WorkflowApp: no State class registered for state_name={state_name!r}")
        state = cls.init_empty()
        state_name_from_state = state.name()
        if state_name_from_state != state_name:
            raise ValueError(
                f"init_empty() returned name={state_name_from_state!r}, expected={state_name!r}"
            )
        return state

    def _load_or_init_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state_name: str,
    ) -> State:
        loaded = self._repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=state_name,
        )
        if loaded is not None:
            return loaded
        return self._init_empty_state(state_name)

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

def _state_messages(state: State) -> Sequence[ChatMessage]:
    value = state.messages()
    return list(value)


def _assistant_messages_for_user(state: State) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="assistant",
            content=message.content,
            artifact_refs=list(message.artifact_refs or ()) or None,
            artifacts=list(message.artifacts or ()) or None,
            id=message.id,
        )
        for message in _state_messages(state)
        if message.role == "assistant"
    ]


def _ordered_history_messages(state: State) -> list[ChatMessage]:
    messages = _state_messages(state)
    assistants = [message for message in messages if message.role == "assistant"]
    systems = [message for message in messages if message.role == "system"]
    return [*assistants, *systems]


__all__ = [
    "ArtifactResponse",
    "WorkflowApp",
    "WorkflowResponse",
]
