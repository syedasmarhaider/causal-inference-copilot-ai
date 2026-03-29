
from uuid import UUID


class WorkflowError(Exception):
    """Base exception for workflow-related errors."""
    pass


class ConversationNotFoundError(WorkflowError):
    """Raised when a conversation is not found for the given user and conversation ID."""
    
    def __init__(self, user_id: UUID, conversation_id: UUID):
        self.user_id = user_id
        self.conversation_id = conversation_id
        super().__init__(f"No active conversation found for user_id={user_id} and conversation_id={conversation_id}")


class StateNotFoundError(WorkflowError):
    """Raised when a required state is not found."""
    
    def __init__(self, state_name: str):
        self.state_name = state_name
        super().__init__(f"State '{state_name}' not found")


class InvalidStateError(WorkflowError):
    """Raised when a state is in an invalid condition."""
    
    def __init__(self, state_name: str, reason: str):
        self.state_name = state_name
        self.reason = reason
        super().__init__(f"Invalid state '{state_name}': {reason}")


class DataUploadError(WorkflowError):
    """Raised when data upload fails."""
    
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Data upload failed: {reason}")


class ArtifactNotFoundError(WorkflowError):
    """Raised when an artifact is not found."""
    
    def __init__(self, artifact_id: UUID):
        self.artifact_id = artifact_id
        super().__init__(f"Artifact '{artifact_id}' not found")


class AuthenticationError(WorkflowError):
    """Raised when authentication fails."""
    pass


class ValidationError(WorkflowError):
    """Raised when input validation fails."""
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"Validation error for '{field}': {reason}")
                
class NodeExecutionError(WorkflowError):
    def __init__(self, state_name: str, error: str):
        self.state_name = state_name
        self.error = error
        super().__init__(f"Error '{state_name}': {error}")          
                      
        
