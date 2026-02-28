from __future__ import annotations

from typing import Any, Dict, Protocol, Union
from uuid import UUID

from python.implementation.workflows.tools.causal.causal_command import ATECommand, ATEResult, CATECommand, CATEResult, FitCommand, FitResult

CausalCommand = Union[
    FitCommand,
    ATECommand,
    CATECommand,
]

CausalResult = Union[
    FitResult,
    ATEResult,
    CATEResult,
]  

class CausalModel(Protocol):
    def get_info(self) -> Dict[str, Any]: ...    
    def execute(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: CausalCommand,
    ) -> CausalResult: ...

