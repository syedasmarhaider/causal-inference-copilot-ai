class NodeExecutionError(Exception):
    def __init__(self, node_name: str, error: str):
        self.node_name = node_name
        self.error = error
        super().__init__(f"Error in node '{node_name}': {error}")

class RecoverableByRouterNodeExecutionError(NodeExecutionError):
    """Indicates an error that occurred during node execution, but the workflow can recover by routing to a different node."""
    pass
        

class NonRecoverableByRouterNodeExecutionError(NodeExecutionError):
    """Indicates an error that occurred during node execution, and the workflow cannot recover."""
    pass
