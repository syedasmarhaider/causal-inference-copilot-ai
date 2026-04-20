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

    assert diff.row_changes == []
    assert diff.summary.inserted_rows == 1
    assert diff.summary.deleted_rows == 1
    assert diff.summary.updated_rows == 0
    assert diff.summary.total_changed_rows == 2
    assert diff.summary.total_changed_cells == 4


def test_diff_dataframes_truncates_detailed_updated_rows_but_keeps_summary_counts() -> None:
    older_df = pd.DataFrame(
        [{"id": index, "age": 40} for index in range(5)]
    )
    newer_df = pd.DataFrame(
        [{"id": index, "age": 41} for index in range(5)]
    )

    diff = diff_dataframes(
        older_df,
        newer_df,
        key_columns=["id"],
        max_detailed_row_changes=2,
    )

    assert diff.summary.updated_rows == 5
    assert diff.summary.total_changed_rows == 5
    assert diff.summary.total_changed_cells == 5
    assert len(diff.row_changes) == 2
    assert [row_change.row_ref.key for row_change in diff.row_changes] == [
        {"id": 0},
        {"id": 1},
    ]
