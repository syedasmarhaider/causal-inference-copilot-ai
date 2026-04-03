from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingError,
    DatasetProfilingTool,
)


@dataclass
class _BrokenColumnAccessFrame:
    data: dict[str, pd.Series]

    @property
    def columns(self) -> list[str]:
        return list(self.data.keys()) + ["broken"]

    @property
    def shape(self) -> tuple[int, int]:
        return (2, 2)

    @property
    def dtypes(self) -> dict[str, str]:
        return {"good": "int64", "broken": "object"}

    def __getitem__(self, key: str) -> pd.Series:
        if key == "broken":
            raise KeyError("broken column")
        return self.data[key]


class _NoColumnsObject:
    pass


def test_extract_dataset_summary_profiles_mixed_dataframe_in_column_order() -> None:
    tool = DatasetProfilingTool()
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, None],
            "when": ["2026-01-01", "2026-01-03", None],
            "flag": ["yes", "no", None],
            "cat": ["a", "a", "b"],
            "obj": [{"a": 1}, {"b": 2}, None],
        }
    )

    summary = tool.extract_dataset_summary(df, max_categories=1, sample_distinct=2)

    assert summary.n_rows == 3
    assert [profile.name for profile in summary.profiles] == ["num", "when", "flag", "cat", "obj"]
    assert [profile.inferred_kind for profile in summary.profiles] == [
        "NUMERIC",
        "DATETIME",
        "BOOLEAN",
        "CATEGORICAL",
        "OTHER",
    ]

    num_profile = summary.profiles[0]
    assert num_profile.n_missing == 1
    assert num_profile.distinct_count == 2
    assert num_profile.missing_rate == pytest.approx(1 / 3)
    assert num_profile.summary.min == 1.0
    assert num_profile.summary.max == 2.0
    assert num_profile.summary.quantiles is not None

    when_profile = summary.profiles[1]
    assert when_profile.summary.min == "2026-01-01"
    assert when_profile.summary.max == "2026-01-03"

    flag_profile = summary.profiles[2]
    assert flag_profile.summary.counts == {"yes": 1, "no": 1}

    cat_profile = summary.profiles[3]
    assert [item.model_dump() for item in cat_profile.summary.top_categories] == [
        {"value": "a", "count": 2}
    ]
    assert cat_profile.summary.other_count == 1

    obj_profile = summary.profiles[4]
    assert obj_profile.summary.distinct_values_sample == ["{'a': 1}", "{'b': 2}"]


def test_extract_dataset_summary_can_disable_quantiles() -> None:
    summary = DatasetProfilingTool().extract_dataset_summary(
        pd.DataFrame({"num": [1.0, 2.0, 3.0]}),
        compute_quantiles=False,
    )

    assert summary.profiles[0].inferred_kind == "NUMERIC"
    assert summary.profiles[0].summary.quantiles is None


def test_dataset_summary_json_roundtrip_preserves_profiles() -> None:
    tool = DatasetProfilingTool()
    summary = tool.extract_dataset_summary(
        pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]}),
    )

    payload = tool.dataset_summary_to_json(summary, indent=2)
    restored = tool.dataset_summary_from_json(payload)

    assert restored.model_dump() == summary.model_dump()
    assert "\n" in payload


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"max_categories": 0}, "max_categories must be > 0."),
        ({"sample_distinct": 0}, "sample_distinct must be > 0."),
    ],
)
def test_extract_dataset_summary_rejects_invalid_params(
    kwargs: dict[str, int],
    reason: str,
) -> None:
    with pytest.raises(DatasetProfilingError, match=reason) as exc_info:
        DatasetProfilingTool().extract_dataset_summary(
            pd.DataFrame({"x": [1]}),
            **kwargs,
        )

    assert exc_info.value.details.reason == reason


def test_extract_dataset_summary_zero_columns_behavior_depends_on_strict_mode() -> None:
    tool = DatasetProfilingTool()
    empty_df = pd.DataFrame()

    with pytest.raises(DatasetProfilingError, match=r"zero columns"):
        tool.extract_dataset_summary(empty_df, strict=True)

    non_strict = tool.extract_dataset_summary(empty_df, strict=False)
    assert non_strict.n_rows == 0
    assert non_strict.profiles == []


def test_extract_dataset_summary_empty_column_name_behavior_depends_on_strict_mode() -> None:
    tool = DatasetProfilingTool()
    df = pd.DataFrame([[1, 2]], columns=["", "good"])

    with pytest.raises(DatasetProfilingError, match=r"empty column name"):
        tool.extract_dataset_summary(df, strict=True)

    summary = tool.extract_dataset_summary(df, strict=False)
    assert [profile.name for profile in summary.profiles] == ["good"]


def test_extract_dataset_summary_non_strict_fallbacks_on_column_access_error() -> None:
    tool = DatasetProfilingTool()
    frame = _BrokenColumnAccessFrame(data={"good": pd.Series([1, 2])})

    summary = tool.extract_dataset_summary(frame, strict=False)

    assert [profile.name for profile in summary.profiles] == ["good", "broken"]
    assert summary.profiles[0].inferred_kind == "NUMERIC"
    assert summary.profiles[1].inferred_kind == "OTHER"
    assert summary.profiles[1].note == "Profiling failed for this column in non-strict mode."


def test_extract_dataset_summary_strict_raises_on_column_access_error() -> None:
    tool = DatasetProfilingTool()
    frame = _BrokenColumnAccessFrame(data={"good": pd.Series([1, 2])})

    with pytest.raises(DatasetProfilingError, match=r"Could not access column via df\[col\]"):
        tool.extract_dataset_summary(frame, strict=True)


def test_extract_dataset_summary_rejects_non_dataframe_like_objects() -> None:
    with pytest.raises(DatasetProfilingError, match=r"no 'columns' attribute"):
        DatasetProfilingTool().extract_dataset_summary(_NoColumnsObject())  # type: ignore[arg-type]


def test_extract_dataset_summary_handles_zero_rows_without_division_errors() -> None:
    summary = DatasetProfilingTool().extract_dataset_summary(
        pd.DataFrame({"num": pd.Series([], dtype="float64")}),
    )

    assert summary.n_rows == 0
    assert len(summary.profiles) == 1
    assert summary.profiles[0].n_missing == 0
    assert summary.profiles[0].missing_rate == 0.0
    assert summary.profiles[0].distinct_count == 0


def test_extract_dataset_summary_other_summary_handles_unhashable_list_values() -> None:
    summary = DatasetProfilingTool().extract_dataset_summary(
        pd.DataFrame({"obj": [[1, 2], [3, 4], None]}),
        sample_distinct=2,
    )

    assert len(summary.profiles) == 1
    assert summary.profiles[0].inferred_kind == "OTHER"
    assert summary.profiles[0].summary.distinct_values_sample == ["[1, 2]", "[3, 4]"]


def test_extract_dataset_summary_categorical_summary_truncates_and_counts_remaining_values() -> None:
    summary = DatasetProfilingTool().extract_dataset_summary(
        pd.DataFrame({"cat": ["a", "a", "b", "c", "d"]}),
        max_categories=2,
    )

    profile = summary.profiles[0]
    assert profile.inferred_kind == "CATEGORICAL"
    assert [item.model_dump() for item in profile.summary.top_categories] == [
        {"value": "a", "count": 2},
        {"value": "b", "count": 1},
    ]
    assert profile.summary.other_count == 2
