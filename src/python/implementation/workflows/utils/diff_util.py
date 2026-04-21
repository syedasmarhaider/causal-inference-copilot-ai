from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MAX_DETAILED_ROW_CHANGES = 500
MAX_UNKEYED_ALIGNMENT_MATRIX_CELLS = 200_000


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
        description=(
            "Only changed cells for this row-level change. "
            "For updates, unchanged cells are omitted. "
            "For inserts and deletes, non-null row values are emitted by default."
        ),
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
    total_changed_cells: int = Field(description="Total count of cell-level changes across the full diff.")


class DataFrameDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
        description=(
            "Detailed row-level changes. Unchanged rows are omitted. "
            "For large diffs this list may be truncated, while `summary` still reflects the full diff."
        )
    )
    summary: DiffSummary = Field(
        description="Compact counts summarizing the diff result."
    )


@dataclass(frozen=True)
class _IndexedRow:
    position: int
    row: pd.Series
    signature: tuple[Any, ...]


def diff_dataframes(
    older_df: pd.DataFrame,
    newer_df: pd.DataFrame,
    key_columns: Sequence[str] | None = None,
    max_detailed_row_changes: int | None = DEFAULT_MAX_DETAILED_ROW_CHANGES,
) -> DataFrameDiff:
    """
    Return only the diff from older_df -> newer_df.

    Modes:
    - keyed mode: if key_columns is provided
    - schema-compatible no-key mode: if key_columns is None or empty

    Rules:
    - older_df is the baseline / source of truth
    - unchanged rows are omitted
    - equal nulls are treated as unchanged
    - in keyed mode, changed keys are treated as delete + insert
    - in no-key mode, exact full-row matches are cancelled first before update inference
    """
    if max_detailed_row_changes is not None and max_detailed_row_changes < 0:
        raise ValueError("max_detailed_row_changes must be >= 0 or None.")

    old = older_df.copy()
    new = newer_df.copy()

    _assert_unique_columns(old, "older_df")
    _assert_unique_columns(new, "newer_df")

    resolved_key_columns = list(key_columns or [])
    _assert_unique_key_columns(resolved_key_columns)

    identity_mode: Literal["key", "position"] = (
        "key" if resolved_key_columns else "position"
    )

    schema_diff = _build_schema_diff(old, new)

    if identity_mode == "key":
        row_changes = _diff_keyed(
            old=old,
            new=new,
            key_columns=resolved_key_columns,
        )
    else:
        row_changes = _diff_without_keys(
            old=old,
            new=new,
        )

    inserted_rows = sum(1 for row_change in row_changes if row_change.op == "inserted")
    deleted_rows = sum(1 for row_change in row_changes if row_change.op == "deleted")
    updated_rows = sum(1 for row_change in row_changes if row_change.op == "updated")
    total_changed_cells = sum(len(row_change.cell_changes) for row_change in row_changes)

    summary = DiffSummary(
        old_row_count=len(old),
        new_row_count=len(new),
        inserted_rows=inserted_rows,
        deleted_rows=deleted_rows,
        updated_rows=updated_rows,
        total_changed_rows=len(row_changes),
        total_changed_cells=total_changed_cells,
    )

    detailed_row_changes = _compact_row_changes(
        row_changes=row_changes,
        max_items=max_detailed_row_changes,
    )

    return DataFrameDiff(
        identity_mode=identity_mode,
        key_columns=resolved_key_columns,
        schema_diff=schema_diff,
        row_changes=detailed_row_changes,
        summary=summary,
    )


def _assert_unique_columns(df: pd.DataFrame, df_name: str) -> None:
    if df.columns.has_duplicates:
        duplicate_columns = list(df.columns[df.columns.duplicated(keep=False)])
        raise ValueError(
            f"{df_name} contains duplicate column names, which are not supported by this diff model. "
            f"Duplicate columns: {duplicate_columns}"
        )


def _assert_unique_key_columns(key_columns: Sequence[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []

    for column in key_columns:
        if column in seen:
            duplicates.append(column)
        seen.add(column)

    if duplicates:
        raise ValueError(f"key_columns contains duplicates: {duplicates}")


def _build_schema_diff(old: pd.DataFrame, new: pd.DataFrame) -> SchemaDiff:
    old_cols = list(old.columns)
    new_cols = list(new.columns)

    old_set = set(old_cols)
    new_set = set(new_cols)

    columns_added = [str(column) for column in new_cols if column not in old_set]
    columns_removed = [str(column) for column in old_cols if column not in new_set]

    shared_columns = [column for column in old_cols if column in new_set]
    column_type_changes: list[ColumnTypeChange] = []

    for column in shared_columns:
        old_dtype = str(old[column].dtype)
        new_dtype = str(new[column].dtype)
        if old_dtype != new_dtype:
            column_type_changes.append(
                ColumnTypeChange(
                    column=str(column),
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
) -> list[RowChange]:
    if not key_columns:
        raise ValueError("key_columns must be non-empty in keyed mode.")

    missing_in_old = [column for column in key_columns if column not in old.columns]
    missing_in_new = [column for column in key_columns if column not in new.columns]

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

    old_key_set = set(old_map.keys())
    new_key_set = set(new_map.keys())

    deleted_keys = [key for key in old_map.keys() if key not in new_key_set]
    inserted_keys = [key for key in new_map.keys() if key not in old_key_set]
    shared_keys = [key for key in new_map.keys() if key in old_key_set]

    row_changes: list[RowChange] = []

    for key in deleted_keys:
        old_row = old_map[key]
        row_changes.append(
            RowChange(
                row_ref=RowRef(mode="key", key=_key_tuple_to_dict(key_columns, key)),
                op="deleted",
                cell_changes=_build_deleted_row_cell_changes(old_row),
            )
        )

    for key in inserted_keys:
        new_row = new_map[key]
        row_changes.append(
            RowChange(
                row_ref=RowRef(mode="key", key=_key_tuple_to_dict(key_columns, key)),
                op="inserted",
                cell_changes=_build_inserted_row_cell_changes(new_row),
            )
        )

    for key in shared_keys:
        old_row = old_map[key]
        new_row = new_map[key]

        cell_changes = _build_cell_changes_for_matched_rows(
            old_row=old_row,
            new_row=new_row,
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


def _diff_without_keys(
    old: pd.DataFrame,
    new: pd.DataFrame,
) -> list[RowChange]:
    """
    No-key diff algorithm.

    Important:
    - Public schema still uses `identity_mode="position"` for compatibility.
    - Internally, position is NOT treated as row identity.
    - Exact full-row matches are cancelled first.
    - Leftover unmatched rows are aligned conservatively to infer updates.
    """
    old_reset = old.reset_index(drop=True)
    new_reset = new.reset_index(drop=True)

    canonical_columns = _ordered_union_columns(old_reset.columns, new_reset.columns)

    old_rows = _build_indexed_rows(old_reset, canonical_columns)
    new_rows = _build_indexed_rows(new_reset, canonical_columns)

    matched_old_positions, matched_new_positions = _match_exact_rows_by_signature(
        old_rows=old_rows,
        new_rows=new_rows,
    )

    remaining_old = [
        indexed_row
        for indexed_row in old_rows
        if indexed_row.position not in matched_old_positions
    ]
    remaining_new = [
        indexed_row
        for indexed_row in new_rows
        if indexed_row.position not in matched_new_positions
    ]

    return _align_unmatched_rows_without_keys(
        remaining_old=remaining_old,
        remaining_new=remaining_new,
        canonical_columns=canonical_columns,
    )


def _ordered_union_columns(
    old_columns: Sequence[Any],
    new_columns: Sequence[Any],
) -> list[Any]:
    result: list[Any] = list(old_columns)
    old_set = set(old_columns)

    for column in new_columns:
        if column not in old_set:
            result.append(column)

    return result


def _build_indexed_rows(
    df: pd.DataFrame,
    canonical_columns: Sequence[Any],
) -> list[_IndexedRow]:
    indexed_rows: list[_IndexedRow] = []

    for position, (_, row) in enumerate(df.iterrows()):
        indexed_rows.append(
            _IndexedRow(
                position=position,
                row=row,
                signature=_build_row_signature(row, canonical_columns),
            )
        )

    return indexed_rows


def _build_row_signature(
    row: pd.Series,
    canonical_columns: Sequence[Any],
) -> tuple[Any, ...]:
    return tuple(
        _freeze_for_signature(_get_row_value_or_none(row, column))
        for column in canonical_columns
    )


def _freeze_for_signature(value: Any) -> Any:
    normalized = _normalize_scalar(value)

    if isinstance(normalized, list):
        return tuple(_freeze_for_signature(item) for item in normalized)

    if isinstance(normalized, tuple):
        return tuple(_freeze_for_signature(item) for item in normalized)

    if isinstance(normalized, dict):
        return tuple(
            (str(key), _freeze_for_signature(val))
            for key, val in sorted(normalized.items(), key=lambda item: str(item[0]))
        )

    if isinstance(normalized, set):
        return tuple(sorted(_freeze_for_signature(item) for item in normalized))

    try:
        hash(normalized)
        return normalized
    except Exception:
        return repr(normalized)


def _match_exact_rows_by_signature(
    old_rows: Sequence[_IndexedRow],
    new_rows: Sequence[_IndexedRow],
) -> tuple[set[int], set[int]]:
    old_buckets: dict[tuple[Any, ...], deque[_IndexedRow]] = defaultdict(deque)
    new_buckets: dict[tuple[Any, ...], deque[_IndexedRow]] = defaultdict(deque)

    for indexed_row in old_rows:
        old_buckets[indexed_row.signature].append(indexed_row)

    for indexed_row in new_rows:
        new_buckets[indexed_row.signature].append(indexed_row)

    matched_old_positions: set[int] = set()
    matched_new_positions: set[int] = set()

    shared_signatures = set(old_buckets.keys()) & set(new_buckets.keys())

    for signature in shared_signatures:
        old_queue = old_buckets[signature]
        new_queue = new_buckets[signature]

        while old_queue and new_queue:
            old_item = old_queue.popleft()
            new_item = new_queue.popleft()
            matched_old_positions.add(old_item.position)
            matched_new_positions.add(new_item.position)

    return matched_old_positions, matched_new_positions


def _align_unmatched_rows_without_keys(
    remaining_old: Sequence[_IndexedRow],
    remaining_new: Sequence[_IndexedRow],
    canonical_columns: Sequence[Any],
) -> list[RowChange]:
    if not remaining_old and not remaining_new:
        return []

    pair_matrix_cells = len(remaining_old) * len(remaining_new)
    if pair_matrix_cells > MAX_UNKEYED_ALIGNMENT_MATRIX_CELLS:
        return _emit_strict_unmatched_changes(
            remaining_old=remaining_old,
            remaining_new=remaining_new,
        )

    update_eligibility_cache: dict[tuple[int, int], bool] = {}

    def can_update(i: int, j: int) -> bool:
        cache_key = (i, j)
        if cache_key not in update_eligibility_cache:
            update_eligibility_cache[cache_key] = _rows_can_be_treated_as_update(
                old_row=remaining_old[i].row,
                new_row=remaining_new[j].row,
                canonical_columns=canonical_columns,
            )
        return update_eligibility_cache[cache_key]

    old_count = len(remaining_old)
    new_count = len(remaining_new)

    dp: list[list[int]] = [[0] * (new_count + 1) for _ in range(old_count + 1)]
    choice: list[list[str | None]] = [[None] * (new_count + 1) for _ in range(old_count + 1)]

    for i in range(old_count, -1, -1):
        for j in range(new_count, -1, -1):
            if i == old_count and j == new_count:
                dp[i][j] = 0
                continue

            if i == old_count:
                dp[i][j] = new_count - j
                choice[i][j] = "insert"
                continue

            if j == new_count:
                dp[i][j] = old_count - i
                choice[i][j] = "delete"
                continue

            best_cost = 1 + dp[i + 1][j]
            best_choice = "delete"

            insert_cost = 1 + dp[i][j + 1]
            if insert_cost < best_cost:
                best_cost = insert_cost
                best_choice = "insert"

            if can_update(i, j):
                update_cost = 1 + dp[i + 1][j + 1]
                if update_cost < best_cost or (update_cost == best_cost and best_choice != "update"):
                    best_cost = update_cost
                    best_choice = "update"

            dp[i][j] = best_cost
            choice[i][j] = best_choice

    row_changes: list[RowChange] = []
    i = 0
    j = 0

    while i < old_count or j < new_count:
        if i == old_count:
            new_item = remaining_new[j]
            row_changes.append(
                RowChange(
                    row_ref=RowRef(mode="position", position=new_item.position),
                    op="inserted",
                    cell_changes=_build_inserted_row_cell_changes(new_item.row),
                )
            )
            j += 1
            continue

        if j == new_count:
            old_item = remaining_old[i]
            row_changes.append(
                RowChange(
                    row_ref=RowRef(mode="position", position=old_item.position),
                    op="deleted",
                    cell_changes=_build_deleted_row_cell_changes(old_item.row),
                )
            )
            i += 1
            continue

        step = choice[i][j]

        if step == "update":
            old_item = remaining_old[i]
            new_item = remaining_new[j]

            cell_changes = _build_cell_changes_for_matched_rows(
                old_row=old_item.row,
                new_row=new_item.row,
            )

            if cell_changes:
                row_changes.append(
                    RowChange(
                        row_ref=RowRef(mode="position", position=old_item.position),
                        op="updated",
                        cell_changes=cell_changes,
                    )
                )

            i += 1
            j += 1
            continue

        if step == "insert":
            new_item = remaining_new[j]
            row_changes.append(
                RowChange(
                    row_ref=RowRef(mode="position", position=new_item.position),
                    op="inserted",
                    cell_changes=_build_inserted_row_cell_changes(new_item.row),
                )
            )
            j += 1
            continue

        old_item = remaining_old[i]
        row_changes.append(
            RowChange(
                row_ref=RowRef(mode="position", position=old_item.position),
                op="deleted",
                cell_changes=_build_deleted_row_cell_changes(old_item.row),
            )
        )
        i += 1

    return row_changes


def _emit_strict_unmatched_changes(
    remaining_old: Sequence[_IndexedRow],
    remaining_new: Sequence[_IndexedRow],
) -> list[RowChange]:
    row_changes: list[RowChange] = []

    for old_item in remaining_old:
        row_changes.append(
            RowChange(
                row_ref=RowRef(mode="position", position=old_item.position),
                op="deleted",
                cell_changes=_build_deleted_row_cell_changes(old_item.row),
            )
        )

    for new_item in remaining_new:
        row_changes.append(
            RowChange(
                row_ref=RowRef(mode="position", position=new_item.position),
                op="inserted",
                cell_changes=_build_inserted_row_cell_changes(new_item.row),
            )
        )

    return row_changes


def _rows_can_be_treated_as_update(
    old_row: pd.Series,
    new_row: pd.Series,
    canonical_columns: Sequence[Any],
) -> bool:
    """
    Conservative heuristic for no-key update inference.

    Exact full-row matches were already removed.
    We only infer an update when there is enough shared evidence between rows.
    """
    matches = 0
    compared = 0

    for column in canonical_columns:
        old_value = _get_row_value_or_none(old_row, column)
        new_value = _get_row_value_or_none(new_row, column)

        if _is_null_like(old_value) and _is_null_like(new_value):
            continue

        compared += 1
        if _values_equal(old_value, new_value):
            matches += 1

    if compared <= 1:
        return False

    if matches == 0:
        return False

    if compared == 2:
        return matches >= 1

    similarity = matches / compared
    return matches >= 2 and similarity >= 0.5


def _build_key_to_row_map(
    df: pd.DataFrame,
    key_columns: Sequence[str],
) -> dict[tuple[Any, ...], pd.Series]:
    result: dict[tuple[Any, ...], pd.Series] = {}

    for _, row in df.iterrows():
        key = tuple(row[column] for column in key_columns)
        result[key] = row

    return result


def _key_tuple_to_dict(
    key_columns: Sequence[str],
    key_tuple: tuple[Any, ...],
) -> dict[str, Any]:
    return {
        column: _normalize_scalar(value)
        for column, value in zip(key_columns, key_tuple, strict=True)
    }


def _assert_no_duplicate_keys(
    df: pd.DataFrame,
    key_columns: Sequence[str],
    df_name: str,
) -> None:
    duplicate_mask = df.duplicated(subset=list(key_columns), keep=False)
    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, list(key_columns)].head(5).to_dict("records")
        raise ValueError(
            f"{df_name} contains duplicate keys for columns {list(key_columns)}. "
            f"Examples: {examples}"
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

    for column in old_row.index:
        old_value = old_row[column]
        if _is_null_like(old_value):
            continue

        changes.append(
            CellChange(
                column=str(column),
                op="removed",
                old_value=_normalize_scalar(old_value),
                new_value=None,
            )
        )

    return changes


def _build_inserted_row_cell_changes(new_row: pd.Series) -> list[CellChange]:
    changes: list[CellChange] = []

    for column in new_row.index:
        new_value = new_row[column]
        if _is_null_like(new_value):
            continue

        changes.append(
            CellChange(
                column=str(column),
                op="added",
                old_value=None,
                new_value=_normalize_scalar(new_value),
            )
        )

    return changes


def _build_cell_changes_for_matched_rows(
    old_row: pd.Series,
    new_row: pd.Series,
) -> list[CellChange]:
    changes: list[CellChange] = []

    old_columns = list(old_row.index)
    new_columns = list(new_row.index)

    old_column_set = set(old_columns)
    new_column_set = set(new_columns)

    shared_columns = [column for column in old_columns if column in new_column_set]
    added_columns = [column for column in new_columns if column not in old_column_set]
    removed_columns = [column for column in old_columns if column not in new_column_set]

    for column in shared_columns:
        old_value = old_row[column]
        new_value = new_row[column]

        if not _values_equal(old_value, new_value):
            changes.append(
                CellChange(
                    column=str(column),
                    op="modified",
                    old_value=_normalize_scalar(old_value),
                    new_value=_normalize_scalar(new_value),
                )
            )

    for column in added_columns:
        new_value = new_row[column]
        if not _is_null_like(new_value):
            changes.append(
                CellChange(
                    column=str(column),
                    op="added",
                    old_value=None,
                    new_value=_normalize_scalar(new_value),
                )
            )

    for column in removed_columns:
        old_value = old_row[column]
        if not _is_null_like(old_value):
            changes.append(
                CellChange(
                    column=str(column),
                    op="removed",
                    old_value=_normalize_scalar(old_value),
                    new_value=None,
                )
            )

    return changes


def _compact_row_changes(
    row_changes: Sequence[RowChange],
    *,
    max_items: int | None,
) -> list[RowChange]:
    compacted: list[RowChange] = []

    for row_change in row_changes:
        normalized_row_change = row_change

        if row_change.op == "updated":
            compact_cell_changes: list[CellChange] = []

            for cell_change in row_change.cell_changes:
                if cell_change.op == "modified" and _values_equal(
                    cell_change.old_value,
                    cell_change.new_value,
                ):
                    continue
                compact_cell_changes.append(cell_change)

            if not compact_cell_changes:
                continue

            normalized_row_change = RowChange(
                row_ref=row_change.row_ref,
                op=row_change.op,
                cell_changes=compact_cell_changes,
            )

        compacted.append(normalized_row_change)

        if max_items is not None and len(compacted) >= max_items:
            break

    return compacted


def _get_row_value_or_none(row: pd.Series, column: Any) -> Any:
    if column in row.index:
        return row[column]
    return None


def _values_equal(a: Any, b: Any) -> bool:
    if _is_null_like(a) and _is_null_like(b):
        return True

    if _is_null_like(a) or _is_null_like(b):
        return False

    try:
        return _freeze_for_signature(a) == _freeze_for_signature(b)
    except Exception:
        return False


def _is_null_like(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except Exception:
        return False

    if isinstance(result, bool):
        return result

    if hasattr(result, "item"):
        try:
            return bool(result.item())
        except Exception:
            return False

    return False


def _normalize_scalar(value: Any) -> Any:
    if _is_null_like(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value

    return value