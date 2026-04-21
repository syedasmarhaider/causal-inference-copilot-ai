from __future__ import annotations

import json
from uuid import uuid4

import pandas as pd

from python.adapters.api.schemas import DatasetDiffResponse
from python.implementation.workflows.utils.diff_util import diff_dataframes


def test_dataset_diff_response_serializes_keyed_diff_with_numpy_scalar_keys() -> None:
    older_df = pd.DataFrame(
        {
            "id": pd.Series([1, 2], dtype="int64"),
            "value": [10, 20],
        }
    )
    newer_df = pd.DataFrame(
        {
            "id": pd.Series([1, 2], dtype="int64"),
            "value": [10, 25],
        }
    )

    diff = diff_dataframes(older_df, newer_df, key_columns=["id"])

    response = DatasetDiffResponse(
        conversation_id=uuid4(),
        conversation_type="data",
        previous_dataset_id=uuid4(),
        current_dataset_id=uuid4(),
        diff=diff,
    )

    payload = json.loads(response.model_dump_json())

    assert payload["diff"]["identity_mode"] == "key"
    assert payload["diff"]["row_changes"][0]["row_ref"]["key"] == {"id": 2}
