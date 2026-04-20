from __future__ import annotations

import pandas as pd

from python.implementation.workflows.utils.diff_util import diff_dataframes


def test_diff_dataframes_omits_unchanged_rows_but_keeps_schema() -> None:
    older_df = pd.DataFrame(
        [
            {"id": 1, "age": 40, "city": "A"},
            {"id": 2, "age": 41, "city": "B"},
            {"id": 3, "age": 42, "city": "C"},
        ]
    )
    newer_df = pd.DataFrame(
        [
            {"id": 1, "age": 40, "city": "A"},
            {"id": 2, "age": 45, "city": "B"},
            {"id": 3, "age": 42, "city": "C"},
        ]
    )

    diff = diff_dataframes(older_df, newer_df, key_columns=["id"])

    assert diff.identity_mode == "key"
    assert diff.key_columns == ["id"]
    assert diff.summary.old_row_count == 3
    assert diff.summary.new_row_count == 3
    assert diff.summary.updated_rows == 1
    assert diff.summary.total_changed_rows == 1
    assert diff.summary.total_changed_cells == 1
    assert len(diff.row_changes) == 1
    assert diff.row_changes[0].row_ref.key == {"id": 2}
    assert [cell_change.model_dump() for cell_change in diff.row_changes[0].cell_changes] == [
        {
            "column": "age",
            "op": "modified",
            "old_value": 41,
            "new_value": 45,
        }
    ]


def test_diff_dataframes_omits_null_only_cells_for_inserted_and_deleted_rows() -> None:
    older_df = pd.DataFrame([{"id": 1, "age": None, "city": "A"}])
    newer_df = pd.DataFrame([{"id": 2, "age": None, "city": "B"}])

    diff = diff_dataframes(older_df, newer_df, key_columns=["id"])

    assert [row_change.op for row_change in diff.row_changes] == ["deleted", "inserted"]
    assert [cell_change.model_dump() for cell_change in diff.row_changes[0].cell_changes] == [
        {
            "column": "id",
            "op": "removed",
            "old_value": 1,
            "new_value": None,
        },
        {
            "column": "city",
            "op": "removed",
            "old_value": "A",
            "new_value": None,
        },
    ]
    assert [cell_change.model_dump() for cell_change in diff.row_changes[1].cell_changes] == [
        {
            "column": "id",
            "op": "added",
            "old_value": None,
            "new_value": 2,
        },
        {
            "column": "city",
            "op": "added",
            "old_value": None,
            "new_value": "B",
        },
    ]
