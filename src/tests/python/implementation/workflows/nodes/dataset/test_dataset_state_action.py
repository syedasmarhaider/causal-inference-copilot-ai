from __future__ import annotations

from python.implementation.workflows.nodes.dataset.dataset_state import (
    DatasetIterationModel,
    DatasetPayloadModel,
    DatasetState,
)


def test_dataset_state_requires_input_by_default_when_dataset_exists() -> None:
    state = DatasetState(
        DatasetPayloadModel(
            dataset_iterations=[DatasetIterationModel(dataset_id=DatasetState.INIT_DATA_ID)],
            user_message="Dataset is ready.",
        )
    )

    assert state.action() == "NEEDS_INPUT"


def test_dataset_state_action_is_none_for_non_interactive_dataset_turn() -> None:
    state = DatasetState(
        DatasetPayloadModel(
            dataset_iterations=[DatasetIterationModel(dataset_id=DatasetState.INIT_DATA_ID)],
            user_message="Applied the confirmed protocol cleaning request.",
            awaiting_user_input=False,
        )
    )

    assert state.action() == "NONE"


def test_dataset_state_still_requests_data_when_no_dataset_exists() -> None:
    state = DatasetState(
        DatasetPayloadModel(
            user_message="Dataset missing.",
            awaiting_user_input=False,
        )
    )

    assert state.action() == "NEEDS_DATA"
