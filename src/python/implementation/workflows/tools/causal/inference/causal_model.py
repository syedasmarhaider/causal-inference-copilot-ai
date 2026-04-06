from __future__ import annotations

from typing import Protocol
from uuid import UUID

from python.implementation.workflows.tools.causal.inference.causal_command import (
    ATECommand,
    ATEResult,
    CATECommand,
    CATEResult,
    CommandType,
    FitCommand,
    FitResult,
)

CausalCommand = FitCommand | ATECommand | CATECommand

CausalResult = FitResult | ATEResult | CATEResult


class CausalModel(Protocol):
    def get_info(self) -> str: ...
    def get_command_info(self, command: CommandType) -> str | None: ...
    def execute(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: CausalCommand,
    ) -> CausalResult: ...
