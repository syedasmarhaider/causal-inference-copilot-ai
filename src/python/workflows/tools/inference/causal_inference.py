from __future__ import annotations

from typing import Any, Dict, Protocol

from python.workflows.state.inference_ready_state import InferenceReadyState
from python.workflows.tools.inference.models.causal_command import CausalCommand
from python.workflows.tools.inference.models.causal_result import CausalResult


class CausalInference(Protocol):
    # capability advertisement
    def get_info(self, estimator_fqcn: str) -> Dict[str, Any]: ...

    # negotiation contract (knobs only; IR passed in)
    def get_input_requirements(self, *, cmd: str, ir: InferenceReadyState) -> Dict[str, Any]: ...

    # output contract
    def get_output_schema(self, *, cmd: str) -> Dict[str, Any]: ...

    # runtime execution (ALL runtime context passed in here)
    def execute(
        self,
        command: CausalCommand,
        *,
        user_id: Any,
        conversation_id: Any,
        model_id: Any,
        ir: InferenceReadyState,
    ) -> CausalResult: ...
