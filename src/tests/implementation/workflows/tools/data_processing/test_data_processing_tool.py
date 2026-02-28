from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import ValidationError
from python.implementation.workflows.tools.data_processing.data_processing_tool import (  # noqa: E501
    DataProcessingTool,
    InclusionRuleModel,
)


@pytest.fixture()
def tool() -> DataProcessingTool:
    return DataProcessingTool()


@pytest.fixture()
def base_df() -> pd.DataFrame:
    # Keep values as strings because InclusionRuleModel.values is NonEmptyStr
    # (and your implementation does ZERO coercion).
    return pd.DataFrame(
        {
            "country": ["DE", "AT", "DE", None, "FR", "DE"],
            "status": ["active", "inactive", None, "active", "active", "inactive"],
            "age_s": ["010", "018", "020", "025", None, "030"],  # string numbers, safe lexicographic
        },
        index=[10, 11, 12, 13, 14, 15],
    )


def test_no_rules_returns_same_object_when_copy_false(tool: DataProcessingTool, base_df: pd.DataFrame) -> None:
    out = tool.apply_inclusion_rules(base_df, [], copy=False)
    assert out is base_df


def test_no_rules_returns_copy_when_copy_true(tool: DataProcessingTool, base_df: pd.DataFrame) -> None:
    out = tool.apply_inclusion_rules(base_df, [], copy=True, deep_copy=True)
    assert out is not base_df
    assert_frame_equal(out, base_df)


def test_missing_column_raises_keyerror(tool: DataProcessingTool, base_df: pd.DataFrame) -> None:
    rules = [InclusionRuleModel(column="missing_col", op="==", values=["x"])]
    with pytest.raises(KeyError) as e:
        tool.apply_inclusion_rules(base_df, rules)

    msg = str(e.value)
    assert "missing_col" in msg
    assert "Available columns" in msg


@pytest.mark.parametrize("op", ["==", ">=", "<=", ">", "<"])
def test_scalar_ops_require_exactly_one_value(tool: DataProcessingTool, base_df: pd.DataFrame, op: str) -> None:
    # 0 values
    rules0 = [InclusionRuleModel(column="country", op=op, values=[])]
    with pytest.raises(ValueError, match="requires exactly 1 value"):
        tool.apply_inclusion_rules(base_df, rules0)

    # 2 values
    rules2 = [InclusionRuleModel(column="country", op=op, values=["DE", "AT"])]
    with pytest.raises(ValueError, match="requires exactly 1 value"):
        tool.apply_inclusion_rules(base_df, rules2)


@pytest.mark.parametrize("op", ["in", "not_in"])
def test_in_ops_require_nonempty_list(tool: DataProcessingTool, base_df: pd.DataFrame, op: str) -> None:
    rules = [InclusionRuleModel(column="country", op=op, values=[])]
    with pytest.raises(ValueError, match="requires a non-empty values list"):
        tool.apply_inclusion_rules(base_df, rules)


def test_rules_are_anded_together(tool: DataProcessingTool, base_df: pd.DataFrame) -> None:
    # country in {DE, AT} AND status == active
    rules = [
        InclusionRuleModel(column="country", op="in", values=["DE", "AT"]),
        InclusionRuleModel(column="status", op="==", values=["active"]),
    ]
    out = tool.apply_inclusion_rules(base_df, rules, copy=True, deep_copy=True)

    # Manual expectation:
    # idx 10: DE + active -> keep
    # idx 11: AT + inactive -> drop
    # idx 12: DE + None -> NA excluded -> drop
    # idx 13: country None -> NA excluded -> drop
    # idx 14: FR -> drop
    # idx 15: DE + inactive -> drop
    assert list(out.index) == [10]
    assert out.loc[10, "country"] == "DE"
    assert out.loc[10, "status"] == "active"


def test_na_excluded_for_in(tool: DataProcessingTool, base_df: pd.DataFrame) -> None:
    rules = [InclusionRuleModel(column="country", op="in", values=["DE", "AT", "FR"])]
    out = tool.apply_inclusion_rules(base_df, rules, copy=True, deep_copy=True)

    # index 13 has country None -> must be excluded even though "in" would otherwise be False anyway
    assert 13 not in out.index
    assert set(out["country"].unique()) <= {"DE", "AT", "FR"}


def test_na_excluded_for_not_in(tool: DataProcessingTool, base_df: pd.DataFrame) -> None:
    # This is the critical case: NA must NOT be included for not_in.
    rules = [InclusionRuleModel(column="status", op="not_in", values=["inactive"])]
    out = tool.apply_inclusion_rules(base_df, rules, copy=True, deep_copy=True)

    # status NA at index 12 must be excluded (not accidentally included)
    assert 12 not in out.index
    # inactive rows at 11 and 15 excluded
    assert 11 not in out.index
    assert 15 not in out.index
    # active rows retained where non-NA
    assert set(out.index) == {10, 13, 14}


def test_string_comparison_works_lexicographically(tool: DataProcessingTool, base_df: pd.DataFrame) -> None:
    # age_s >= "018" in lexicographic order; we zero-padded to make it behave like numeric order
    rules = [InclusionRuleModel(column="age_s", op=">=", values=["018"])]
    out = tool.apply_inclusion_rules(base_df, rules, copy=True, deep_copy=True)

    # age_s values: 010, 018, 020, 025, None, 030 -> keep 018,020,025,030; exclude 010 and NA
    assert list(out.index) == [11, 12, 13, 15]


def test_invalid_comparison_raises_typeerror_with_context(tool: DataProcessingTool) -> None:
    # Numeric column compared to string threshold should fail and be wrapped.
    df = pd.DataFrame({"age": [10, 20, 30, None]}, index=[0, 1, 2, 3])
    rules = [InclusionRuleModel(column="age", op=">=", values=["18"])]

    with pytest.raises(TypeError) as e:
        tool.apply_inclusion_rules(df, rules)

    msg = str(e.value)
    assert "Failed applying inclusion rule" in msg
    assert "column='age'" in msg
    assert "op='>='" in msg
    assert "values=['18']" in msg
    assert "series_dtype" in msg


def test_empty_result_preserves_columns(tool: DataProcessingTool, base_df: pd.DataFrame) -> None:
    rules = [InclusionRuleModel(column="country", op="==", values=["ZZZ"])]
    out = tool.apply_inclusion_rules(base_df, rules, copy=True, deep_copy=True)

    assert out.shape[0] == 0
    assert list(out.columns) == list(base_df.columns)


def test_input_df_not_mutated(tool: DataProcessingTool, base_df: pd.DataFrame) -> None:
    before = base_df.copy(deep=True)
    rules = [InclusionRuleModel(column="country", op="in", values=["DE"])]
    _ = tool.apply_inclusion_rules(base_df, rules, copy=True, deep_copy=True)

    assert_frame_equal(base_df, before)


def test_deep_copy_isolation_mutating_output_does_not_change_input(tool: DataProcessingTool, base_df: pd.DataFrame) -> None:
    rules = [InclusionRuleModel(column="country", op="==", values=["DE"])]
    out = tool.apply_inclusion_rules(base_df, rules, copy=True, deep_copy=True)

    assert out is not base_df
    assert len(out) > 0

    # Mutate the output and ensure input unchanged.
    out.iloc[0, out.columns.get_loc("country")] = "MUTATED"
    assert base_df.loc[out.index[0], "country"] == "DE"


def test_copy_false_still_does_not_mutate_input_on_filter(tool: DataProcessingTool, base_df: pd.DataFrame) -> None:
    # Filtering itself must not mutate parent df; copy flag only affects the returned frame.
    before = base_df.copy(deep=True)
    rules = [InclusionRuleModel(column="country", op="==", values=["DE"])]
    _ = tool.apply_inclusion_rules(base_df, rules, copy=False)

    assert_frame_equal(base_df, before)


def test_pydantic_rejects_empty_string_values() -> None:
    # NonEmptyStr should reject empty string (or whitespace, depending on your NonEmptyStr impl).
    with pytest.raises(ValidationError):
        InclusionRuleModel(column="country", op="==", values=[""])

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_randomized_equivalence_for_membership_ops(tool: DataProcessingTool, seed: int) -> None:
    rng = np.random.default_rng(seed)
    countries = np.array(["DE", "AT", "FR", "IT", None], dtype=object)
    status = np.array(["active", "inactive", None], dtype=object)

    df = pd.DataFrame(
        {
            "country": rng.choice(countries, size=200, replace=True),
            "status": rng.choice(status, size=200, replace=True),
        }
    )

    rules = [
        InclusionRuleModel(column="country", op="in", values=["DE", "AT"]),
        InclusionRuleModel(column="status", op="not_in", values=["inactive"]),
    ]

    out = tool.apply_inclusion_rules(df, rules, copy=True, deep_copy=True)

    # Manual spec: non_na enforced for BOTH rules, ANDed.
    m = (
        df["country"].notna()
        & df["country"].isin(["DE", "AT"])
        & df["status"].notna()
        & ~df["status"].isin(["inactive"])
    )
    expected = df.loc[m].copy(deep=True)

    assert_frame_equal(out.reset_index(drop=True), expected.reset_index(drop=True))        