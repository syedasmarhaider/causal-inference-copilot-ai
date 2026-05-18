from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from python.implementation.workflows.tools.simple_data_transformation_tool.simple_data_transformation_tool import (
    SimpleDataTransformationSpec,
    SimpleDataTransformationTool,
)


def test_tool_identity_and_info() -> None:
    tool = SimpleDataTransformationTool()

    assert tool.get_tool_name() == "SIMPLE_DATA_TRANSFORMATION"
    assert "deterministic dataframe column transformations" in tool.get_tool_info()


def test_transform_sets_static_value_and_casts_type_without_mutating_input() -> None:
    df = pd.DataFrame({"flag": ["old", "old"], "score": ["1.5", "2.0"]})
    spec = SimpleDataTransformationSpec.model_validate(
        {
            "columns": [
                {"column": "flag", "value": "yes", "target_dtype": "boolean"},
                {"column": "score", "target_dtype": "float"},
            ]
        }
    )
    tool = SimpleDataTransformationTool()

    result = tool.transform(dataframe=df, specification=spec)

    assert result["flag"].tolist() == [True, True]
    assert str(result["flag"].dtype) == "boolean"
    assert result["score"].tolist() == [1.5, 2.0]
    assert str(result["score"].dtype) == "float64"
    assert df["flag"].tolist() == ["old", "old"]


def test_transform_applies_replacements_fill_and_nullable_integer_cast() -> None:
    df = pd.DataFrame({"group": ["control", "treated", None, "unknown"]})
    tool = SimpleDataTransformationTool()

    result = tool.transform(
        dataframe=df,
        specification={
            "columns": [
                {
                    "column": "group",
                    "replacements": [
                        {"from_value": "control", "to_value": 0},
                        {"from_value": "treated", "to_value": 1},
                        {"from_value": "unknown", "to_value": None},
                    ],
                    "fill_value": -1,
                    "target_dtype": "integer",
                }
            ]
        },
    )

    assert result["group"].tolist() == [0, 1, -1, -1]
    assert str(result["group"].dtype) == "int64"


def test_transform_casts_datetime_with_coercion() -> None:
    df = pd.DataFrame({"started_at": ["2026-01-01", "bad-date"]})
    tool = SimpleDataTransformationTool()

    result = tool.transform(
        dataframe=df,
        specification={
            "columns": [
                {
                    "column": "started_at",
                    "target_dtype": "datetime",
                    "errors": "coerce",
                }
            ]
        },
    )

    assert result["started_at"].iloc[0] == pd.Timestamp("2026-01-01")
    assert pd.isna(result["started_at"].iloc[1])


def test_integer_cast_rejects_fractional_values_by_default() -> None:
    tool = SimpleDataTransformationTool()

    with pytest.raises(ValueError, match=r"fractional values"):
        tool.transform(
            dataframe=pd.DataFrame({"age": [1.2]}),
            specification={"columns": [{"column": "age", "target_dtype": "integer"}]},
        )


def test_transform_can_mutate_input_when_copy_false() -> None:
    df = pd.DataFrame({"active": ["yes", "no"]})
    tool = SimpleDataTransformationTool()

    result = tool.transform(
        dataframe=df,
        specification={"columns": [{"column": "active", "target_dtype": "boolean"}]},
        copy=False,
    )

    assert result is df
    assert df["active"].tolist() == [True, False]
    assert str(df["active"].dtype) == "boolean"


def test_spec_rejects_empty_noop_and_duplicate_columns() -> None:
    with pytest.raises(ValidationError, match=r"must define target_dtype"):
        SimpleDataTransformationSpec.model_validate({"columns": [{"column": "age"}]})

    with pytest.raises(ValidationError, match=r"duplicate column transformations"):
        SimpleDataTransformationSpec.model_validate(
            {
                "columns": [
                    {"column": "age", "target_dtype": "integer"},
                    {"column": "age", "target_dtype": "float"},
                ]
            }
        )


def test_transform_rejects_missing_dataframe_column() -> None:
    tool = SimpleDataTransformationTool()

    with pytest.raises(KeyError, match=r"Column not found"):
        tool.transform(
            dataframe=pd.DataFrame({"age": [1]}),
            specification={"columns": [{"column": "missing", "target_dtype": "integer"}]},
        )
