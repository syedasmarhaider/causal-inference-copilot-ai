from __future__ import annotations

from typing import Any, Literal, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class RowRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["key", "position"] = Field(
        description="How the row was identified in the diff. `key` means business-key matching, `position` means zero-based row index matching."
    )
    key: dict[str, Any] | None = Field(
        default=None,
        description="Key-column values for the row when `mode` is `key`.",
    )
    position: int | None = Field(
        default=None,
        description="Zero-based row position when `mode` is `position`.",
    )


class CellChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column name where the change happened.")
    op: Literal["added", "removed", "modified"] = Field(
        description="Cell-level operation relative to the older dataset version."
    )
    old_value: Any = Field(
        default=None,
        description="Value from the previous dataset version. Null when the cell was newly added.",
    )
    new_value: Any = Field(
        default=None,
        description="Value from the current dataset version. Null when the cell was removed.",
    )


class RowChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_ref: RowRef = Field(description="Reference that identifies the changed row.")
    op: Literal["inserted", "deleted", "updated"] = Field(
        description="Row-level operation relative to the previous dataset version."
    )
    cell_changes: list[CellChange] = Field(
        default_factory=list,
        description="Cell-level changes for this row. Inserted and deleted rows typically include one entry per visible column.",
    )


class ColumnTypeChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(description="Column whose inferred dataframe dtype changed.")
    old_dtype: str = Field(description="Dtype in the previous dataset version.")
    new_dtype: str = Field(description="Dtype in the current dataset version.")


class SchemaDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns_added: list[str] = Field(
        default_factory=list,
        description="Columns present only in the current dataset version.",
    )
    columns_removed: list[str] = Field(
        default_factory=list,
        description="Columns present only in the previous dataset version.",
    )
    column_type_changes: list[ColumnTypeChange] = Field(
        default_factory=list,
        description="Shared columns whose inferred dataframe dtype changed between versions.",
    )


class DiffSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    old_row_count: int = Field(description="Number of rows in the previous dataset version.")
    new_row_count: int = Field(description="Number of rows in the current dataset version.")
    inserted_rows: int = Field(description="Count of rows newly introduced in the current dataset.")
    deleted_rows: int = Field(description="Count of rows removed from the previous dataset.")
    updated_rows: int = Field(description="Count of matched rows with one or more changed cells.")
    total_changed_rows: int = Field(description="Total count of inserted, deleted, and updated rows.")
    total_changed_cells: int = Field(description="Total count of emitted cell-level changes across all row changes.")


class DataFrameDiff(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "identity_mode": "key",
                "key_columns": ["patient_id"],
                "schema_diff": {
                    "columns_added": ["bmi"],
                    "columns_removed": [],
                    "column_type_changes": [
                        {
                            "column": "age",
                            "old_dtype": "int64",
                            "new_dtype": "float64",
                        }
                    ],
                },
                "row_changes": [
                    {
                        "row_ref": {
                            "mode": "key",
                            "key": {"patient_id": 101},
                            "position": None,
                        },
                        "op": "updated",
                        "cell_changes": [
                            {
                                "column": "age",
                                "op": "modified",
                                "old_value": 44,
                                "new_value": 45,
                            },
                            {
                                "column": "bmi",
                                "op": "added",
                                "old_value": None,
                                "new_value": 27.1,
                            },
                        ],
                    }
                ],
                "summary": {
                    "old_row_count": 100,
                    "new_row_count": 101,
                    "inserted_rows": 1,
                    "deleted_rows": 0,
                    "updated_rows": 1,
                    "total_changed_rows": 2,
                    "total_changed_cells": 3,
                },
            }
        },
    )

    identity_mode: Literal["key", "position"] = Field(
        description="How rows were matched across versions. `key` uses `key_columns`; `position` compares rows by zero-based index."
    )
    key_columns: list[str] = Field(
        default_factory=list,
        description="Key columns used for row matching. Empty when positional comparison was used.",
    )
    schema_diff: SchemaDiff = Field(
        description="Schema-level changes between the previous and current dataset versions."
    )
    row_changes: list[RowChange] = Field(
        description="Only changed rows are included. Unchanged rows are omitted."
    )
    summary: DiffSummary = Field(
        description="Compact counts summarizing the diff result."
    )


def diff_dataframes(
    older_df: pd.DataFrame,
    newer_df: pd.DataFrame,
    key_columns: Sequence[str] | None = None,
) -> DataFrameDiff:
    """
    Return only the diff from older_df -> newer_df.

    Modes:
    - keyed mode: if key_columns is provided
    - positional mode: if key_columns is None or empty

    Rules:
    - older_df is the baseline / source of truth
    - unchanged rows are omitted
    - equal nulls are treated as unchanged
    - in keyed mode, changed keys are treated as delete + insert
    """

    old = older_df.copy()
    new = newer_df.copy()

    resolved_key_columns = list(key_columns or [])
    identity_mode: Literal["key", "position"] = (
        "key" if resolved_key_columns else "position"
    )

    schema_diff = _build_schema_diff(old, new)

    if identity_mode == "key":
        row_changes = _diff_keyed(
            old=old,
            new=new,
            key_columns=resolved_key_columns,
            schema_diff=schema_diff,
        )
    else:
        row_changes = _diff_by_position(
            old=old,
            new=new,
            schema_diff=schema_diff,
        )

    inserted_rows = sum(1 for r in row_changes if r.op == "inserted")
    deleted_rows = sum(1 for r in row_changes if r.op == "deleted")
    updated_rows = sum(1 for r in row_changes if r.op == "updated")
    total_changed_cells = sum(len(r.cell_changes) for r in row_changes)

    summary = DiffSummary(
        old_row_count=len(old),
        new_row_count=len(new),
        inserted_rows=inserted_rows,
        deleted_rows=deleted_rows,
        updated_rows=updated_rows,
        total_changed_rows=len(row_changes),
        total_changed_cells=total_changed_cells,
    )

    return DataFrameDiff(
        identity_mode=identity_mode,
        key_columns=resolved_key_columns,
        schema_diff=schema_diff,
        row_changes=row_changes,
        summary=summary,
    )


def _build_schema_diff(old: pd.DataFrame, new: pd.DataFrame) -> SchemaDiff:
    old_cols = list(old.columns)
    new_cols = list(new.columns)

    old_set = set(old_cols)
    new_set = set(new_cols)

    columns_added = [c for c in new_cols if c not in old_set]
    columns_removed = [c for c in old_cols if c not in new_set]

    shared = [c for c in old_cols if c in new_set]
    column_type_changes: list[ColumnTypeChange] = []

    for col in shared:
        old_dtype = str(old[col].dtype)
        new_dtype = str(new[col].dtype)
        if old_dtype != new_dtype:
            column_type_changes.append(
                ColumnTypeChange(
                    column=col,
                    old_dtype=old_dtype,
                    new_dtype=new_dtype,
                )
            )

    return SchemaDiff(
        columns_added=columns_added,
        columns_removed=columns_removed,
        column_type_changes=column_type_changes,
    )


def _diff_keyed(
    old: pd.DataFrame,
    new: pd.DataFrame,
    key_columns: list[str],
    schema_diff: SchemaDiff,
) -> list[RowChange]:
    if not key_columns:
        raise ValueError("key_columns must be non-empty in keyed mode.")

    missing_in_old = [c for c in key_columns if c not in old.columns]
    missing_in_new = [c for c in key_columns if c not in new.columns]

    if missing_in_old:
        raise ValueError(f"Missing key columns in older_df: {missing_in_old}")
    if missing_in_new:
        raise ValueError(f"Missing key columns in newer_df: {missing_in_new}")

    _assert_no_null_keys(old, key_columns, "older_df")
    _assert_no_null_keys(new, key_columns, "newer_df")
    _assert_no_duplicate_keys(old, key_columns, "older_df")
    _assert_no_duplicate_keys(new, key_columns, "newer_df")

    old_map = _build_key_to_row_map(old, key_columns)
    new_map = _build_key_to_row_map(new, key_columns)

    old_keys = set(old_map.keys())
    new_keys = set(new_map.keys())

    inserted_keys = new_keys - old_keys
    deleted_keys = old_keys - new_keys
    shared_keys = old_keys & new_keys

    row_changes: list[RowChange] = []

    for key in sorted(deleted_keys):
        old_row = old_map[key]
        row_changes.append(
            RowChange(
                row_ref=RowRef(mode="key", key=_key_tuple_to_dict(key_columns, key)),
                op="deleted",
                cell_changes=_build_deleted_row_cell_changes(old_row),
            )
        )

    for key in sorted(inserted_keys):
        new_row = new_map[key]
        row_changes.append(
            RowChange(
                row_ref=RowRef(mode="key", key=_key_tuple_to_dict(key_columns, key)),
                op="inserted",
                cell_changes=_build_inserted_row_cell_changes(new_row),
            )
        )

    for key in sorted(shared_keys):
        old_row = old_map[key]
        new_row = new_map[key]

        cell_changes = _build_cell_changes_for_matched_rows(
            old_row=old_row,
            new_row=new_row,
            schema_diff=schema_diff,
        )

        if cell_changes:
            row_changes.append(
                RowChange(
                    row_ref=RowRef(mode="key", key=_key_tuple_to_dict(key_columns, key)),
                    op="updated",
                    cell_changes=cell_changes,
                )
            )

    return row_changes


def _diff_by_position(
    old: pd.DataFrame,
    new: pd.DataFrame,
    schema_diff: SchemaDiff,
) -> list[RowChange]:
    old_reset = old.reset_index(drop=True)
    new_reset = new.reset_index(drop=True)

    row_changes: list[RowChange] = []
    max_len = max(len(old_reset), len(new_reset))

    for pos in range(max_len):
        has_old = pos < len(old_reset)
        has_new = pos < len(new_reset)

        if has_old and not has_new:
            old_row = old_reset.iloc[pos]
            row_changes.append(
                RowChange(
                    row_ref=RowRef(mode="position", position=pos),
                    op="deleted",
                    cell_changes=_build_deleted_row_cell_changes(old_row),
                )
            )
            continue

        if has_new and not has_old:
            new_row = new_reset.iloc[pos]
            row_changes.append(
                RowChange(
                    row_ref=RowRef(mode="position", position=pos),
                    op="inserted",
                    cell_changes=_build_inserted_row_cell_changes(new_row),
                )
            )
            continue

        old_row = old_reset.iloc[pos]
        new_row = new_reset.iloc[pos]

        cell_changes = _build_cell_changes_for_matched_rows(
            old_row=old_row,
            new_row=new_row,
            schema_diff=schema_diff,
        )

        if cell_changes:
            row_changes.append(
                RowChange(
                    row_ref=RowRef(mode="position", position=pos),
                    op="updated",
                    cell_changes=cell_changes,
                )
            )

    return row_changes


def _build_key_to_row_map(
    df: pd.DataFrame,
    key_columns: Sequence[str],
) -> dict[tuple[Any, ...], pd.Series]:
    result: dict[tuple[Any, ...], pd.Series] = {}
    for _, row in df.iterrows():
        key = tuple(row[col] for col in key_columns)
        result[key] = row
    return result


def _key_tuple_to_dict(
    key_columns: Sequence[str],
    key_tuple: tuple[Any, ...],
) -> dict[str, Any]:
    return {col: value for col, value in zip(key_columns, key_tuple, strict=True)}


def _assert_no_duplicate_keys(
    df: pd.DataFrame,
    key_columns: Sequence[str],
    df_name: str,
) -> None:
    dup_mask = df.duplicated(subset=list(key_columns), keep=False)
    if dup_mask.any():
        duplicate_examples = df.loc[dup_mask, list(key_columns)].head(5).to_dict("records")
        raise ValueError(
            f"{df_name} contains duplicate keys for columns {list(key_columns)}. "
            f"Examples: {duplicate_examples}"
        )


def _assert_no_null_keys(
    df: pd.DataFrame,
    key_columns: Sequence[str],
    df_name: str,
) -> None:
    null_mask = df[list(key_columns)].isna().any(axis=1)
    if null_mask.any():
        examples = df.loc[null_mask, list(key_columns)].head(5).to_dict("records")
        raise ValueError(
            f"{df_name} contains null keys for columns {list(key_columns)}. "
            f"Examples: {examples}"
        )


def _build_deleted_row_cell_changes(old_row: pd.Series) -> list[CellChange]:
    changes: list[CellChange] = []
    for col in old_row.index:
        changes.append(
            CellChange(
                column=str(col),
                op="removed",
                old_value=_normalize_scalar(old_row[col]),
                new_value=None,
            )
        )
    return changes


def _build_inserted_row_cell_changes(new_row: pd.Series) -> list[CellChange]:
    changes: list[CellChange] = []
    for col in new_row.index:
        changes.append(
            CellChange(
                column=str(col),
                op="added",
                old_value=None,
                new_value=_normalize_scalar(new_row[col]),
            )
        )
    return changes


def _build_cell_changes_for_matched_rows(
    old_row: pd.Series,
    new_row: pd.Series,
    schema_diff: SchemaDiff,
) -> list[CellChange]:
    changes: list[CellChange] = []

    old_cols = set(map(str, old_row.index))
    new_cols = set(map(str, new_row.index))

    shared_cols = sorted(old_cols & new_cols)
    added_cols = schema_diff.columns_added
    removed_cols = schema_diff.columns_removed

    for col in shared_cols:
        old_value = old_row[col]
        new_value = new_row[col]

        if not _values_equal(old_value, new_value):
            changes.append(
                CellChange(
                    column=col,
                    op="modified",
                    old_value=_normalize_scalar(old_value),
                    new_value=_normalize_scalar(new_value),
                )
            )

    for col in added_cols:
        if col in new_cols:
            new_value = new_row[col]
            if not _is_null_like(new_value):
                changes.append(
                    CellChange(
                        column=col,
                        op="added",
                        old_value=None,
                        new_value=_normalize_scalar(new_value),
                    )
                )
            else:
                # if you want every added column emitted even when null, remove this condition
                pass

    for col in removed_cols:
        if col in old_cols:
            old_value = old_row[col]
            if not _is_null_like(old_value):
                changes.append(
                    CellChange(
                        column=col,
                        op="removed",
                        old_value=_normalize_scalar(old_value),
                        new_value=None,
                    )
                )
            else:
                # if you want every removed column emitted even when null, remove this condition
                pass

    return changes


def _values_equal(a: Any, b: Any) -> bool:
    if _is_null_like(a) and _is_null_like(b):
        return True
    return a == b


def _is_null_like(value: Any) -> bool:
    return pd.isna(value)


def _normalize_scalar(value: Any) -> Any:
    if _is_null_like(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value    
