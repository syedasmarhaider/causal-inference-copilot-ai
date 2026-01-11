from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

@dataclass(frozen=True)
class WorkflowResponse:
    text: str
    needs_input: bool
    conversation_id: UUID