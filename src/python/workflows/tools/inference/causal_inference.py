from __future__ import annotations

from typing import Protocol

from python.workflows.tools.inference.models.causal_command import CausalCommand
from python.workflows.tools.inference.models.causal_result import CausalResult


class CausalInference(Protocol):
    def get_info(self, estimator_fqcn: str) -> str: ...
    def execute(self, command: CausalCommand) -> CausalResult: ...
