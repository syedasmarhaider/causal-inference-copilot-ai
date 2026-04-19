from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

from python.domain.models.errors import ConversationNotFoundError, StateNotFoundError
from python.domain.models.models import ArtifactRef, ChatMessage
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.workflows.node import Action, Status
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.dataflow_app import DataflowArtifactResponse
from python.implementation.workflows.ochestrator.ochestraotor import Ochestrator
from python.implementation.workflows.ochestrator.writable_ochestrator_state import WritableOchestratorState


@dataclass(frozen=True)
class WorkflowResponse:
    messages: Sequence[ChatMessage]
    current_stage_name: str
    current_stage_status: Status
    action: Action
    current_data_id: UUID | None = None
    is_dataset_frozen: bool | None = None

    @property
    def artifact_refs(self) -> Sequence[ArtifactRef] | None:
        refs: list[ArtifactRef] = []
        for message in self.messages:
            refs.extend(list(message.artifact_refs or ()))
        return refs or None


ArtifactResponse = DataflowArtifactResponse

class WorkflowApp:
    def __init__(
        self,
        *,
        repo: WorkflowStateRepo,
        ochestrator: Ochestrator,
    ) -> None:
        self._repo = repo
        self._ochestrator = ochestrator
        self._log = get_app_logger(
            __name__,
            component=self.__class__.__name__,
            log_type="workflow_service",
        )

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Read current state (no execution)
    # ------------------------------------------------------------------

    def load_current_state_info(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> list[str]:
        return self._ochestrator.load_state_info(
            user_id=user_id,
            conversation_id=conversation_id,
        )
    
    
    def load_conversation_messages(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        limit: int = 30,
    ) -> Sequence[ChatMessage]:
        return self._repo.load_message_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=limit,
        )

        
        

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def handle(
        self,
        user_id: UUID,
        conversation_id: UUID,
        user_message: str | None,
    ) -> WorkflowResponse:
        self._log.debug(
            "workflow handle requested",
            user_id=user_id,
            conversation_id=conversation_id,
        )
        
        user_message = user_message or ""

        response = self._ochestrator.answer(
            conversation_id=conversation_id,
            user_id=user_id,
            user_message=ChatMessage(role="user", content=user_message),
        )
        

        return WorkflowResponse(
            messages=response.messages,
            current_stage_name=response.current_state,
            current_stage_status=response.current_status,
            current_data_id=response.current_data_id,
            action=response.action,
            is_dataset_frozen=response.is_dataset_frozen,
        )

    # ------------------------------------------------------------------
    # Revert
    # ------------------------------------------------------------------

    def revert_to_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state_name: str,
    ) -> None:
        ochestrator_state = self._repo.load_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not isinstance(ochestrator_state, WritableOchestratorState):
            raise StateNotFoundError(state_name=state_name)

        # Roll back the global orchestrator state to the recovery point.
        ochestrator_state.roll_back_to_state(state_name)

        # Delete all states at and after the recovery point so they re-execute fresh.
        for name_to_delete in ochestrator_state.get_forward_states_after_node(state_name):
            self._repo.delete_state(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=name_to_delete,
            )
        
        self._repo.delete_state(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=state_name,
        )    
            
        self._repo.store_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state=ochestrator_state,
        )
        self._log.info(
            "conversation reverted",
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=state_name,
        )