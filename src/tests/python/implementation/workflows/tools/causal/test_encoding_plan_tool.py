from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from python.implementation.workflows.tools.encoding.encoding_plan import (
    TransformPlan,
)
from python.implementation.workflows.tools.encoding.encoding_plan_tool import (
    EncodingPlanTool,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


def _numeric_profile(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "float64",
        "n_rows": 10,
        "n_missing": 0,
        "missing_rate": 0.0,
        "distinct_count": 10,
        "inferred_kind": "NUMERIC",
        "summary": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.1, "quantiles": None},
    }


def _categorical_profile(name: str, values: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "object",
        "n_rows": 10,
        "n_missing": 0,
        "missing_rate": 0.0,
        "distinct_count": len(values),
        "inferred_kind": "CATEGORICAL",
        "summary": {
            "top_categories": [{"value": value, "count": 5} for value in values],
            "other_count": 0,
        },
    }


def _boolean_profile(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "bool",
        "n_rows": 10,
        "n_missing": 0,
        "missing_rate": 0.0,
        "distinct_count": 2,
        "inferred_kind": "BOOLEAN",
        "summary": {"counts": {"True": 6, "False": 4}},
    }


def _datetime_profile(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "datetime64[ns]",
        "n_rows": 10,
        "n_missing": 0,
        "missing_rate": 0.0,
        "distinct_count": 10,
        "inferred_kind": "DATETIME",
        "summary": {"min": "2026-01-01T00:00:00", "max": "2026-01-10T00:00:00"},
    }


def _summary_model(*profiles: dict[str, Any]) -> DatasetSummaryModel:
    return DatasetSummaryModel.model_validate({"n_rows": 10, "profiles": list(profiles)})


def _plan_payload(*, columns: list[dict[str, Any]]) -> dict[str, Any]:
    return {"columns": columns}


def test_encoding_plan_tool_identity_and_info() -> None:
    tool = EncodingPlanTool()

    assert tool.get_tool_name() == "ENCODING_PLAN"
    assert "covariate and effect_modifier columns only" in tool.get_tool_info()


def test_build_encoding_schema_binds_eligible_columns_from_roles_without_schema_bloat() -> None:
    summary = _summary_model(
        _numeric_profile("age"),
        _categorical_profile("segment", ["A", "B"]),
    )
    tool = EncodingPlanTool()

    schema = tool.build_encoding_schema(
        data_summary=summary,
        covariate_columns=["age"],
        effect_modifier_columns=["segment"],
    )

    assert schema is not TransformPlan
    assert schema.SUMMARY_FIELD_NAMES == ("age", "segment")
    assert schema.SUMMARY_FIELD_KINDS == {
        "age": "NUMERIC",
        "segment": "CATEGORICAL",
    }
    assert schema.ELIGIBLE_COLUMNS == ("age", "segment")
    assert schema.EXPECTED_ROLE_BY_COLUMN == {
        "age": "covariate",
        "segment": "effect_modifier",
    }

    schema_json = str(schema.model_json_schema())
    assert "SUMMARY_FIELD_NAMES" not in schema_json
    assert "ELIGIBLE_COLUMNS" not in schema_json
    assert "EXPECTED_ROLE_BY_COLUMN" not in schema_json


def test_build_encoding_schema_rejects_invalid_role_inputs() -> None:
    summary = _summary_model(_numeric_profile("age"))
    tool = EncodingPlanTool()

    with pytest.raises(ValueError, match=r"overlap"):
        tool.build_encoding_schema(
            data_summary=summary,
            covariate_columns=["age"],
            effect_modifier_columns=["age"],
        )

    with pytest.raises(ValueError, match=r"At least one covariate or effect_modifier column is required"):
        tool.build_encoding_schema(data_summary=summary)

    with pytest.raises(ValueError, match=r"not present in dataset_summary"):
        tool.build_encoding_schema(
            data_summary=summary,
            covariate_columns=["missing_col"],
        )


def test_validate_encoding_payload_returns_summary_bound_plan() -> None:
    summary = _summary_model(
        _numeric_profile("age"),
        _categorical_profile("segment", ["A", "B"]),
    )
    tool = EncodingPlanTool()

    model = tool.validate_encoding_payload(
        payload=_plan_payload(
            columns=[
                {
                    "column": "age",
                    "role": "covariate",
                    "encoding": {"preset": "num_standard"},
                },
                {
                    "column": "segment",
                    "role": "effect_modifier",
                    "encoding": {"preset": "cat_onehot"},
                },
            ]
        ),
        data_summary=summary,
        covariate_columns=["age"],
        effect_modifier_columns=["segment"],
    )

    assert isinstance(model, TransformPlan)
    assert type(model) is not TransformPlan
    assert [column.column for column in model.columns] == ["age", "segment"]


def test_validate_encoding_payload_rejects_unknown_dataset_summary_columns() -> None:
    summary = _summary_model(_numeric_profile("age"))
    tool = EncodingPlanTool()

    with pytest.raises(ValidationError, match=r"unknown dataset_summary columns"):
        tool.validate_encoding_payload(
            payload=_plan_payload(
                columns=[
                    {
                        "column": "missing_col",
                        "role": "covariate",
                        "encoding": {"preset": "num_standard"},
                    }
                ]
            ),
            data_summary=summary,
            covariate_columns=["age"],
        )


def test_validate_encoding_payload_rejects_non_eligible_and_missing_columns() -> None:
    summary = _summary_model(
        _numeric_profile("age"),
        _categorical_profile("segment", ["A", "B"]),
        _numeric_profile("score"),
    )
    tool = EncodingPlanTool()

    with pytest.raises(ValidationError, match=r"non-eligible columns"):
        tool.validate_encoding_payload(
            payload=_plan_payload(
                columns=[
                    {
                        "column": "score",
                        "role": "covariate",
                        "encoding": {"preset": "num_standard"},
                    }
                ]
            ),
            data_summary=summary,
            covariate_columns=["age"],
            effect_modifier_columns=["segment"],
        )

    with pytest.raises(ValidationError, match=r"missing eligible columns"):
        tool.validate_encoding_payload(
            payload=_plan_payload(
                columns=[
                    {
                        "column": "age",
                        "role": "covariate",
                        "encoding": {"preset": "num_standard"},
                    }
                ]
            ),
            data_summary=summary,
            covariate_columns=["age"],
            effect_modifier_columns=["segment"],
        )


def test_validate_encoding_payload_rejects_wrong_roles() -> None:
    summary = _summary_model(
        _numeric_profile("age"),
        _categorical_profile("segment", ["A", "B"]),
    )
    tool = EncodingPlanTool()

    with pytest.raises(ValidationError, match=r"wrong roles"):
        tool.validate_encoding_payload(
            payload=_plan_payload(
                columns=[
                    {
                        "column": "age",
                        "role": "effect_modifier",
                        "encoding": {"preset": "num_standard"},
                    },
                    {
                        "column": "segment",
                        "role": "effect_modifier",
                        "encoding": {"preset": "cat_onehot"},
                    },
                ]
            ),
            data_summary=summary,
            covariate_columns=["age"],
            effect_modifier_columns=["segment"],
        )


def test_validate_encoding_payload_rejects_preset_kind_incompatibility() -> None:
    summary = _summary_model(
        _numeric_profile("age"),
        _datetime_profile("visit_time"),
        _boolean_profile("flag"),
    )
    tool = EncodingPlanTool()

    with pytest.raises(ValidationError, match=r"type and preset incompatibilities"):
        tool.validate_encoding_payload(
            payload=_plan_payload(
                columns=[
                    {
                        "column": "age",
                        "role": "covariate",
                        "encoding": {"preset": "cat_onehot"},
                    }
                ]
            ),
            data_summary=summary,
            covariate_columns=["age"],
        )

    model = tool.validate_encoding_payload(
        payload=_plan_payload(
            columns=[
                {
                    "column": "flag",
                    "role": "covariate",
                    "encoding": {"preset": "num_standard"},
                },
                {
                    "column": "visit_time",
                    "role": "effect_modifier",
                    "encoding": {"preset": "datetime_epoch_seconds"},
                },
            ]
        ),
        data_summary=summary,
        covariate_columns=["flag"],
        effect_modifier_columns=["visit_time"],
    )

    assert [column.column for column in model.columns] == ["flag", "visit_time"]


def test_validate_encoding_payload_rejects_mapping_values_not_supported_by_exact_summary() -> None:
    summary = _summary_model(
        _categorical_profile("segment", ["A", "B"]),
        _boolean_profile("flag"),
    )
    tool = EncodingPlanTool()

    with pytest.raises(ValidationError, match=r"map_binary mapping contains values not supported"):
        tool.validate_encoding_payload(
            payload=_plan_payload(
                columns=[
                    {
                        "column": "segment",
                        "role": "covariate",
                        "encoding": {
                            "preset": "map_binary",
                            "mapping": {"A": 1.0, "C": 0.0},
                            "allow_unknown": True,
                            "unknown_value": -1.0,
                        },
                    }
                ]
            ),
            data_summary=summary,
            covariate_columns=["segment"],
        )

    with pytest.raises(ValidationError, match=r"map_ordinal order contains values not supported"):
        tool.validate_encoding_payload(
            payload=_plan_payload(
                columns=[
                    {
                        "column": "flag",
                        "role": "effect_modifier",
                        "encoding": {
                            "preset": "map_ordinal",
                            "order": ["True", "Maybe"],
                            "allow_unknown": True,
                            "unknown_value": -1,
                        },
                    }
                ]
            ),
            data_summary=summary,
            effect_modifier_columns=["flag"],
        )


def test_validate_encoding_payload_structured_returns_success_payload() -> None:
    summary = _summary_model(_numeric_profile("age"))
    tool = EncodingPlanTool()

    model_dict, issues = tool.validate_encoding_payload_structured(
        payload=_plan_payload(
            columns=[
                {
                    "column": "age",
                    "role": "covariate",
                    "encoding": {"preset": "num_standard"},
                }
            ]
        ),
        data_summary=summary,
        covariate_columns=["age"],
    )

    assert issues == []
    assert model_dict is not None
    assert model_dict["columns"][0]["column"] == "age"
    assert model_dict["columns"][0]["encoding"]["preset"] == "num_standard"


def test_validate_encoding_payload_structured_returns_issues_instead_of_raising() -> None:
    summary = _summary_model(_numeric_profile("age"))
    tool = EncodingPlanTool()

    model_dict, issues = tool.validate_encoding_payload_structured(
        payload=_plan_payload(
            columns=[
                {
                    "column": "age",
                    "role": "covariate",
                    "encoding": {"preset": "cat_onehot"},
                }
            ]
        ),
        data_summary=summary,
        covariate_columns=["age"],
    )

    assert model_dict is None
    assert issues
    assert issues[0]["path"] == ""
    assert "type and preset incompatibilities" in str(issues[0]["message"])


def test_validate_encoding_payload_structured_converts_non_mapping_payload_to_issues() -> None:
    summary = _summary_model(_numeric_profile("age"))
    tool = EncodingPlanTool()

    model_dict, issues = tool.validate_encoding_payload_structured(  # type: ignore[arg-type]
        payload=["not", "a", "mapping"],
        data_summary=summary,
        covariate_columns=["age"],
    )

    assert model_dict is None
    assert issues
    assert issues[0]["path"] == ""
    assert "valid dictionary" in str(issues[0]["message"]).lower()


def test_post_validate_encoding_plan_revalidates_generic_plan_against_summary() -> None:
    summary = _summary_model(_numeric_profile("age"))
    generic_plan = TransformPlan.model_validate(
        _plan_payload(
            columns=[
                {
                    "column": "missing_col",
                    "role": "covariate",
                    "encoding": {"preset": "num_standard"},
                }
            ]
        )
    )
    tool = EncodingPlanTool()

    with pytest.raises(ValidationError, match=r"unknown dataset_summary columns"):
        tool.post_validate_encoding_plan(
            plan=generic_plan,
            data_summary=summary,
            covariate_columns=["age"],
        )


def test_post_validate_encoding_plan_returns_summary_bound_plan() -> None:
    summary = _summary_model(
        _numeric_profile("age"),
        _categorical_profile("segment", ["A", "B"]),
    )
    generic_plan = TransformPlan.model_validate(
        _plan_payload(
            columns=[
                {
                    "column": "age",
                    "role": "covariate",
                    "encoding": {"preset": "num_standard"},
                },
                {
                    "column": "segment",
                    "role": "effect_modifier",
                    "encoding": {"preset": "cat_onehot"},
                },
            ]
        )
    )
    tool = EncodingPlanTool()

    model = tool.post_validate_encoding_plan(
        plan=generic_plan,
        data_summary=summary,
        covariate_columns=["age"],
        effect_modifier_columns=["segment"],
    )

    assert isinstance(model, TransformPlan)
    assert type(model) is not TransformPlan
    assert [column.column for column in model.columns] == ["age", "segment"]
