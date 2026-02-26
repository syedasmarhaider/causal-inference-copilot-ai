from __future__ import annotations

from typing import Any, Dict, Protocol
from uuid import UUID

from python.implementation.workflows.tools.causal.causal_command import BaseCommand, BaseResult

class CausalModel(Protocol):
    def get_info(self) -> Dict[str, Any]: ...    
    def execute(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: BaseCommand,
    ) -> BaseResult: ...
