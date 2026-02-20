from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from python.domain.workflows.state import State
from python.domain.workflows.state_dep import StateDep
from python.implementation.workflows.nodes.compile_inference.compile_inference_state import CompileInferenceState
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState


@dataclass(frozen=True)
class ValidateCompiledInferenceDeps(StateDep):
    compile_protocol: CompileProtocolState
    compile_inference: CompileInferenceState

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "ValidateCompiledInferenceDeps":
        pd = loaded.get(CompileProtocolState.NAME)
        if not isinstance(pd, CompileProtocolState):
            raise ValueError(
                f"ValidateCompiledInferenceDeps: missing/invalid {CompileProtocolState.NAME} (got {type(pd).__name__ if pd else None})"
            )
        ci = loaded.get(CompileInferenceState.NAME)
        if not isinstance(ci, CompileInferenceState):
            raise ValueError(
                f"ValidateCompiledInferenceDeps: missing/invalid {CompileInferenceState.NAME} (got {type(ci).__name__ if ci else None})"
            )
        return cls(compile_protocol=pd, compile_inference=ci)
