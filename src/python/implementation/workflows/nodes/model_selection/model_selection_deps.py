from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from python.domain.models.errors import StateDependencyError
from python.domain.workflows.state import State
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import (
    CleanProtocolState,
)
from python.implementation.workflows.nodes.load_dataset.load_dataset_state import LoadDatasetState
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_state import (
    ValidateCleanProtocolState,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.utils.validation import ValidationIssueModel

T = TypeVar("T", bound=State)


@dataclass(frozen=True, slots=True)
class ModelSelectionDeps:
    clean_dataset_summary: DatasetSummaryModel
    compiled_causal_spec: CausalSpec
    validation_errors: list[ValidationIssueModel]

    @classmethod
    def pre_required_states_names(cls) -> Sequence[str]:
        return (
            LoadDatasetState.NAME,
            CleanProtocolState.NAME,
            ValidateCleanProtocolState.NAME,
        )

    @classmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> ModelSelectionDeps:
        def _get(name: str, expected_type: type[T]) -> T:
            st = loaded.get(name)
            if st is None:
                raise StateDependencyError(f"ModelSelectionDeps: missing {name}", to_state="ModelSelectionDeps", missing_dependencies=[
                    name,
                ])
            if not isinstance(st, expected_type):
                raise StateDependencyError(f"ModelSelectionDeps: invalid {name} (expected {expected_type.__name__}, got {type(st).__name__})", to_state="ModelSelectionDeps", missing_dependencies=[name])      
            return st

        cl = _get(CleanProtocolState.NAME, CleanProtocolState)
        vc = _get(ValidateCleanProtocolState.NAME, ValidateCleanProtocolState)

        clean_ds_summary = cl.payload.summary
        if clean_ds_summary is None:
            raise StateDependencyError(f"ModelSelectionDeps: {CleanProtocolState.NAME} is not DONE yet (missing clean dataset summary)", to_state="ModelSelectionDeps", missing_dependencies=[CleanProtocolState.NAME])
        compiled_causal_spec = cl.payload.compiled_causal_spec
        if compiled_causal_spec is None:
            raise StateDependencyError(f"ModelSelectionDeps: {CleanProtocolState.NAME} is not DONE yet (missing compiled causal spec)", to_state="ModelSelectionDeps", missing_dependencies=[CleanProtocolState.NAME])
        validation_errors = vc.payload.issues
        return cls(clean_dataset_summary=clean_ds_summary, compiled_causal_spec=compiled_causal_spec, validation_errors=validation_errors)