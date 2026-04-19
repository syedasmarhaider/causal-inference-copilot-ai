from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

from python.domain.models.errors import ConversationNotFoundError, StateNotFoundError, ValidationError
from python.domain.models.models import ArtifactRef, ChatMessage
from python.domain.repo.workflow_state_repo import Conversation, ConversationType, WorkflowStateRepo
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


@dataclass(frozen=True)
class ConversationResponse:
    conversation_type: ConversationType
    messages: Sequence[ChatMessage]
    states: list[str]
    current_data_id: UUID | None = None
    is_dataset_frozen: bool | None = None


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

    def _raise_if_userid_not_relates_to_conversation_id(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
    ) -> None:
        if conversation_type not in ["causal", "data"]:
            raise ValidationError("conversation_type", f"Invalid conversation type: {conversation_type}")
        
        conversation = Conversation(conversation_id=conversation_id, conversation_type=conversation_type)
        if not self._repo.is_conversation_id_for_user_id_exists(
            user_id=user_id,
            conversation=conversation,
        ):
            raise ConversationNotFoundError(user_id=user_id, conversation_id=conversation.conversation_id)

    def create_conversation(self, user_id: UUID, conversation_type: str) -> UUID:
        conversation_id = uuid4()
        if conversation_type not in ["causal", "data"]:
            raise ValidationError("conversation_type", f"Invalid conversation type: {conversation_type}")
        
        # Depending on how Conversation is defined, you might need ConversationType(conversation_type) here
        conversation = Conversation(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
        )
        
        self._repo.save_conversation(user_id=user_id, conversation=conversation)
        self._log.info(
            "conversation created",
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return conversation_id

    def list_conversations(self, user_id: UUID) -> Sequence[Conversation]:
        return self._repo.get_conversations(user_id=user_id)
    
    def get_current_conversation_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
    ) -> ConversationResponse:
        # FIXED: Pass arguments to match the method signature
        self._raise_if_userid_not_relates_to_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
        )
        
        messages = self._repo.load_message_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=30,
        )
        state_info = self._ochestrator.load_state_info(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return ConversationResponse(
            # FIXED: Cast the string back to the ConversationType enum/type required by the dataclass
            conversation_type=ConversationType(conversation_type),
            messages=messages,
            states=state_info,
        )

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
        conversation_type: str,
        limit: int = 30,
    ) -> Sequence[ChatMessage]:
        self._raise_if_userid_not_relates_to_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
        )
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
        conversation_type: str,
        user_message: str | None,
    ) -> WorkflowResponse:
        self._raise_if_userid_not_relates_to_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
        )
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
        conversation_type: str,
        state_name: str,
    ) -> None:
        self._raise_if_userid_not_relates_to_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
        )
        ochestrator_state = self._repo.load_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not isinstance(ochestrator_state, WritableOchestratorState):
            raise StateNotFoundError(state_name=state_name)

        ochestrator_state.roll_back_to_state(state_name)

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
    