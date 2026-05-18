from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID


class WorkflowError(Exception, ABC):
    """Base exception for workflow-related errors."""

    @property
    @abstractmethod
    def code(self) -> str:
        """Machine-readable error code."""


class ConversationNotFoundError(WorkflowError):
    """Raised when a conversation is not found for the given user and conversation ID."""

    @property
    def code(self) -> str:
        return "conversation_not_found"

    def __init__(self, user_id: UUID, conversation_id: UUID):
        self.user_id = user_id
        self.conversation_id = conversation_id
        super().__init__(
            f"No active conversation found for user_id={user_id} and conversation_id={conversation_id}"
        )


class StateNotFoundError(WorkflowError):
    """Raised when a required state is not found."""

    @property
    def code(self) -> str:
        return "state_not_found"

    def __init__(self, state_name: str):
        self.state_name = state_name
        super().__init__(f"State '{state_name}' not found")


class InvalidStateError(WorkflowError):
    """Raised when a state is in an invalid condition."""

    @property
    def code(self) -> str:
        return "invalid_state"

    def __init__(self, state_name: str, reason: str):
        self.state_name = state_name
        self.reason = reason
        super().__init__(f"Invalid state '{state_name}': {reason}")


class StateConflictError(WorkflowError):
    """Raised when a stale state write would overwrite a newer persisted state."""

    @property
    def code(self) -> str:
        return "state_conflict"

    def __init__(
        self,
        *,
        state_name: str,
        expected_counter: int,
        actual_counter: int | None,
    ) -> None:
        self.state_name = state_name
        self.expected_counter = expected_counter
        self.actual_counter = actual_counter
        actual = "missing" if actual_counter is None else str(actual_counter)
        super().__init__(
            f"State '{state_name}' has changed since it was loaded: "
            f"expected update_counter={expected_counter}, actual update_counter={actual}"
        )


class DataUploadError(WorkflowError):
    """Raised when data upload fails."""

    @property
    def code(self) -> str:
        return "data_upload_failed"

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Data upload failed: {reason}")


class ArtifactNotFoundError(WorkflowError):
    """Raised when an artifact is not found."""

    @property
    def code(self) -> str:
        return "artifact_not_found"

    def __init__(self, artifact_id: UUID):
        self.artifact_id = artifact_id
        super().__init__(f"Artifact '{artifact_id}' not found")


class AuthenticationError(WorkflowError):
    """Raised when authentication fails."""

    @property
    def code(self) -> str:
        return "authentication_failed"


class ValidationError(WorkflowError):
    """Raised when input validation fails."""

    @property
    def code(self) -> str:
        return "validation_failed"

    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"Validation error for '{field}': {reason}")


class NodeExecutionError(WorkflowError):
    @property
    def code(self) -> str:
        return "node_execution_error"

    def __init__(self, state_name: str, error: str):
        self.state_name = state_name
        self.error = error
        super().__init__(f"Error '{state_name}': {error}")


class StateDependencyError(NodeExecutionError):
    """Raised when a state transition fails due to unmet dependencies."""

    @property
    def code(self) -> str:
        return "state_dependency_error"

    def __init__(self, from_state: str, to_state: str, missing_dependencies: list[str]):
        detail_message = (
            f"Cannot transition from '{from_state}' to '{to_state}' due to missing dependencies: "
            f"{', '.join(missing_dependencies)}"
        )
        super().__init__(state_name=to_state, error=detail_message)
        self.from_state = from_state
        self.to_state = to_state
        self.missing_dependencies = missing_dependencies
