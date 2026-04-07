from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

from python.domain.models.errors import ConversationNotFoundError, StateNotFoundError
from python.domain.models.models import ArtifactRef, ChatMessage
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.workflows.state import Action, Status
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.dataflow_app import DataflowArtifactResponse
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import (
    CausalInferenceState,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_state import (
    CompileAndValidateState,
)
from python.implementation.workflows.nodes.dataset.dataset_state import DatasetState
from python.implementation.workflows.nodes.model_selection.mode_selection_state import (
    ModelSelectionState,
)
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainState
from python.implementation.workflows.nodes.noop_done.noop_done_state import NoopDoneState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)
from python.implementation.workflows.ochestrator.ochestraotor import (
    Ochestrator,
    build_state_name_by_node_name,
)
from python.implementation.workflows.ochestrator.ochestrator_global_state import (
    OchestratorWritableGlobalState,
)


@dataclass(frozen=True)
class WorkflowResponse:
    messages: Sequence[ChatMessage]
    current_stage_name: str
    current_stage_status: Status
    action: Action

    @property
    def artifact_refs(self) -> Sequence[ArtifactRef] | None:
        refs: list[ArtifactRef] = []
        for message in self.messages:
            refs.extend(list(message.artifact_refs or ()))
        return refs or None


ArtifactResponse = DataflowArtifactResponse

# State names ordered to allow forward-deletion on revert.
_STATE_ORDER: tuple[str, ...] = (
    DatasetState.NAME,
    ProtocolDiscussionState.NAME,
    CompileAndValidateState.NAME,
    ModelSelectionState.NAME,
    ModelTrainState.NAME,
    CausalInferenceState.NAME,
    NoopDoneState.NAME,
)


class WorkflowApp:
    def __init__(
        self,
        *,
        repo: WorkflowStateRepo,
        ochestrator: Ochestrator,
    ) -> None:
        self._repo = repo
        self._ochestrator = ochestrator
        self._state_name_by_node_name = build_state_name_by_node_name()
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

    def get_last_conversation_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> WorkflowResponse | None:
        ochestrator_state = self._repo.load_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not isinstance(ochestrator_state, OchestratorWritableGlobalState):
            return None

        last_node_name = ochestrator_state.needs_node_name()
        if not last_node_name:
            return None

        state_name = self._state_name_by_node_name.get(last_node_name)
        if state_name is None:
            return None

        state = self._repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=state_name,
        )
        if state is None:
            return None

        assistant_messages = [msg for msg in state.messages() if msg.role == "assistant"]
        return WorkflowResponse(
            messages=assistant_messages,
            current_stage_name=state_name,
            current_stage_status=state.status(),
            action=state.action(),
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

        # Skip appending empty messages — the orchestrator appends in answer().
        message_text = (user_message or "").strip()
        if not message_text:
            last = self.get_last_conversation_state(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if last is not None:
                return last

        response = self._ochestrator.answer(
            conversation_id=conversation_id,
            user_id=user_id,
            user_message=ChatMessage(role="user", content=message_text),
        )
        
        state = response.state
    
        self._log.debug(
            "workflow handle completed",
            user_id=user_id,
            conversation_id=conversation_id,
            stage_name=state.name(),
            stage_status=state.status(),
        )

        assistant_messages = [msg for msg in response.messages if msg.role == "assistant"]

        return WorkflowResponse(
            messages=assistant_messages,
            current_stage_name=state.name(),
            current_stage_status=state.status(),
            action=state.action(),
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
        if not isinstance(ochestrator_state, OchestratorWritableGlobalState):
            raise StateNotFoundError(state_name=state_name)

        # Roll back the global orchestrator state to the recovery point.
        ochestrator_state.rollback_orchestrator_global_state(recovery_state_name=state_name)

        # Delete all states at and after the recovery point so they re-execute fresh.
        for name_to_delete in _state_names_from(state_name):
            self._repo.delete_state(
                user_id=user_id,
                conversation_id=conversation_id,
                state_name=name_to_delete,
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

def _state_names_from(state_name: str) -> tuple[str, ...]:
    """Return state_name plus all states that come after it in the workflow order."""
    if state_name not in _STATE_ORDER:
        return ()
    idx = _STATE_ORDER.index(state_name)
    return _STATE_ORDER[idx:]


__all__ = [
    "ArtifactResponse",
    "WorkflowApp",
    "WorkflowResponse",
]