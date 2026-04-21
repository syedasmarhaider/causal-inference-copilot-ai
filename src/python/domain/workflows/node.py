from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.node_state import NodeState
from python.domain.workflows.ochestrator_state import OchestratorState


Status = Literal["PENDING", "DONE", "ABORTED"]
Action = Literal["NONE", "NEEDS_INPUT", "NEEDS_DATA"]

@dataclass(frozen=True)
class NodeRequest:
    user_id: UUID
    conversation_id: UUID
    node_state: NodeState
    orchestrator_state: OchestratorState
    read_only_messages_history: Sequence[ChatMessage] | None = None

@dataclass(frozen=True)    
class NodeExecutionResult:
    new_node_state: NodeState
    new_orchestrator_state: OchestratorState
    status: Status   
    action: Action
    response_messages: Sequence[ChatMessage] | None = None

class Node(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_info(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        *,
        request: NodeRequest,
    ) -> NodeExecutionResult:
        raise NotImplementedError
