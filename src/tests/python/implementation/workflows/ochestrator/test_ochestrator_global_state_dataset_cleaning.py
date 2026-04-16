from __future__ import annotations

from uuid import uuid4

from python.implementation.workflows.ochestrator.writable_ochestrator_state import (
    OchestratorWritableGlobalState,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


def test_dataset_cleaning_pending_is_set_after_protocol_confirmation_and_cleared_on_freeze() -> (
    None
):
    dataset_id = uuid4()
    summary = DatasetSummaryModel(n_rows=1, profiles=[])
    state = OchestratorWritableGlobalState.init_empty()

    state.set_working_dataset(dataset_id=dataset_id, summary=summary)
    state.set_protocol_discussion("Confirmed protocol discussion.")
    state.mark_dataset_cleaning_pending()

    assert state.get("dataset_cleaning_pending") is True
    assert state.needs_node_name() == "DATASET"

    state.freeze_working_dataset_snapshot(dataset_id=dataset_id, dataset_summary=summary)

    assert state.get("dataset_cleaning_pending") is False
    assert state.get("working_dataset_frozen") is True
