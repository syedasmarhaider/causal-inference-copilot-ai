from __future__ import annotations

import pytest
from pydantic import ValidationError

from python.implementation.workflows.ochestrator.causal_ochestrator_state import (
    CausalOchestratorState,
)
from python.implementation.workflows.ochestrator.data_ochestrator_state import (
    DataOchestratorState,
)


def test_empty_orchestrator_states_start_with_update_counter_zero() -> None:
    assert CausalOchestratorState.init_empty().get_update_counter() == 0
    assert DataOchestratorState.init_empty().get_update_counter() == 0


def test_orchestrator_state_serializes_update_counter() -> None:
    causal_state = CausalOchestratorState.init_empty()
    data_state = DataOchestratorState.init_empty()

    causal_state.set_update_counter(3)
    data_state.set_update_counter(4)

    assert causal_state.to_json_dict()["update_counter"] == 3
    assert data_state.to_json_dict()["update_counter"] == 4


def test_legacy_payload_defaults_update_counter_to_zero() -> None:
    causal_state = CausalOchestratorState.from_json_dict(
        {"working_dataset_ids": [str(CausalOchestratorState.INIT_DATA_ID)]}
    )
    data_state = DataOchestratorState.from_json_dict(
        {"working_dataset_ids": [str(DataOchestratorState.INIT_DATA_ID)]}
    )

    assert causal_state.get_update_counter() == 0
    assert data_state.get_update_counter() == 0


def test_payload_rejects_invalid_update_counter() -> None:
    with pytest.raises(ValidationError, match=r"update_counter"):
        CausalOchestratorState.from_json_dict({"update_counter": "1"})
    with pytest.raises(ValidationError, match=r"update_counter"):
        DataOchestratorState.from_json_dict({"update_counter": -1})


def test_set_update_counter_rejects_invalid_values() -> None:
    causal_state = CausalOchestratorState.init_empty()
    data_state = DataOchestratorState.init_empty()

    with pytest.raises(ValidationError, match=r"update_counter"):
        causal_state.set_update_counter(-1)
    with pytest.raises(ValidationError, match=r"update_counter"):
        data_state.set_update_counter(-1)
    with pytest.raises(ValidationError, match=r"update_counter"):
        causal_state.set_update_counter("1")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match=r"update_counter"):
        data_state.set_update_counter("1")  # type: ignore[arg-type]
