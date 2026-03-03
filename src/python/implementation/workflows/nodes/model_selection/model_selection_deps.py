from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Sequence, TypeVar

from python.domain.workflows.state import State

from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_state import ValidateCleanProtocolState



T = TypeVar("T", bound=State)


@dataclass(frozen=True, slots=True)
class ModelSelectionDeps:
    compile_protocol: CompileProtocolState
    clean_protocol: CleanProtocolState
    validate_clean_protocol: ValidateCleanProtocolState

    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (
            LoadDatasetState.NAME,
            CompileProtocolState.NAME,
            CleanProtocolState.NAME,
            ValidateCleanProtocolState.NAME,
        )

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "ModelSelectionDeps":
        def _get(name: str, expected_type: type[T]) -> T:
            st = loaded.get(name)
            if st is None:
                raise ValueError(f"ModelSelectionDeps: missing {name}")
            if not isinstance(st, expected_type):
                raise ValueError(
                    f"ModelSelectionDeps: invalid {name} "
                    f"(expected {expected_type.__name__}, got {type(st).__name__})"
                )
            return st

        cp = _get(CompileProtocolState.NAME, CompileProtocolState)
        cl = _get(CleanProtocolState.NAME, CleanProtocolState)
        vc = _get(ValidateCleanProtocolState.NAME, ValidateCleanProtocolState)

        return cls(
            compile_protocol=cp,
            clean_protocol=cl,
            validate_clean_protocol=vc,
        )