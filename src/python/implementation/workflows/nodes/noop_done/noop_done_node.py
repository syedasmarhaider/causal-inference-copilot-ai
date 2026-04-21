from __future__ import annotations

from typing import ClassVar

from python.domain.models.models import ChatMessage
from python.domain.workflows.node import Node, NodeExecutionResult, NodeRequest
from python.implementation.workflows.nodes.noop_done.noop_done_state import NoopDoneState


class NoopDoneNode(Node):
    NAME: ClassVar[str] = NoopDoneState.NAME

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def get_info(cls) -> str:
        return "No-op terminal node: immediately returns DONE."

    def run(
        self,
        *,
        request: NodeRequest,
    ) -> NodeExecutionResult:
        return NodeExecutionResult(
            new_node_state=NoopDoneState.init_empty(),
            new_orchestrator_state=request.orchestrator_state,
            status="DONE",
            action="NONE",
            response_messages=[
                ChatMessage(role="assistant", content="Workflow is complete.")
            ],
        )
