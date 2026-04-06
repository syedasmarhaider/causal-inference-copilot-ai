from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.specs.causal_specs_tool import (
    CausalSpecsTool,
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


def _spec_payload(
    *,
    treatment_column: str = "treatment",
    treated: str = "drug",
    control: str = "placebo",
    outcome_kind: str = "continuous",
    outcome_column: str = "outcome",
    event: str = "1",
    non_event: str = "0",
    covariates: list[str] | None = None,
    effect_modifiers: list[str] | None = None,
    experiment_type: str = "OBSERVATIONAL",
) -> dict[str, Any]:
    if outcome_kind == "binary":
        outcome_spec: dict[str, Any] = {
            "kind": "binary",
            "column": outcome_column,
            "event": event,
            "non_event": non_event,
        }
    else:
        outcome_spec = {
            "kind": "continuous",
            "column": outcome_column,
            "unit": "score",
        }

    return {
        "treatment_spec": {
            "kind": "binary",
            "column": treatment_column,
            "treated": treated,
            "control": control,
        },
        "outcome_spec": outcome_spec,
        "covariates": covariates if covariates is not None else ["age"],
        "effect_modifiers": effect_modifiers if effect_modifiers is not None else ["segment"],
        "experiment_type": experiment_type,
    }


def test_tool_identity_and_info() -> None:
    tool = CausalSpecsTool()

    assert tool.get_tool_name() == "CAUSAL_BACKDOOR_SPEC"
    assert "dataset-summary-bound validation" in tool.get_tool_info()


def test_build_backdoor_schema_binds_summary_headers_without_bloating_json_schema() -> None:
    summary = _summary_model(
        _categorical_profile("treatment", ["drug", "placebo"]),
        _numeric_profile("outcome"),
        _numeric_profile("age"),
        _categorical_profile("segment", ["A", "B"]),
    )
    tool = CausalSpecsTool()

    schema = tool.build_backdoor_schema(data_summary=summary)

    assert schema is not CausalSpec
    assert schema.SUMMARY_FIELD_NAMES == ("treatment", "outcome", "age", "segment")
    assert schema.SUMMARY_FIELD_KINDS == {
        "treatment": "CATEGORICAL",
        "outcome": "NUMERIC",
        "age": "NUMERIC",
        "segment": "CATEGORICAL",
    }

    schema_json = str(schema.model_json_schema())
    assert "SUMMARY_FIELD_NAMES" not in schema_json
    assert "SUMMARY_FIELD_KINDS" not in schema_json
    assert "SUMMARY_KNOWN_VALUES" not in schema_json


def test_validate_backdoor_payload_returns_summary_bound_model() -> None:
    summary = _summary_model(
        _categorical_profile("treatment", ["drug", "placebo"]),
        _numeric_profile("outcome"),
        _numeric_profile("age"),
        _categorical_profile("segment", ["A", "B"]),
    )
    tool = CausalSpecsTool()

    model = tool.validate_backdoor_payload(
        payload=_spec_payload(),
        data_summary=summary,
    )

    assert isinstance(model, CausalSpec)
    assert type(model) is not CausalSpec
    assert model.treatment_spec.column == "treatment"
    assert model.outcome_spec.column == "outcome"
    assert model.covariates == ["age"]
    assert model.effect_modifiers == ["segment"]


def test_validate_backdoor_payload_allows_observational_without_covariates_at_this_stage() -> None:
    summary = _summary_model(
        _categorical_profile("treatment", ["drug", "placebo"]),
        _numeric_profile("outcome"),
    )
    tool = CausalSpecsTool()

    model = tool.validate_backdoor_payload(
        payload=_spec_payload(
            covariates=[],
            effect_modifiers=[],
            experiment_type="OBSERVATIONAL",
        ),
        data_summary=summary,
    )

    assert model.experiment_type == "OBSERVATIONAL"
    assert model.covariates == []
    assert model.effect_modifiers == []


def test_validate_backdoor_payload_rejects_unknown_columns() -> None:
    summary = _summary_model(
        _categorical_profile("treatment", ["drug", "placebo"]),
        _numeric_profile("outcome"),
        _numeric_profile("age"),
    )
    tool = CausalSpecsTool()

    with pytest.raises(ValidationError, match=r"unknown dataset_summary columns"):
        tool.validate_backdoor_payload(
            payload=_spec_payload(effect_modifiers=["missing_col"]),
            data_summary=summary,
        )


def test_validate_backdoor_payload_rejects_role_overlap() -> None:
    summary = _summary_model(
        _categorical_profile("treatment", ["drug", "placebo"]),
        _numeric_profile("outcome"),
        _numeric_profile("age"),
    )
    tool = CausalSpecsTool()

    with pytest.raises(ValidationError, match=r"covariates and effect_modifiers overlap"):
        tool.validate_backdoor_payload(
            payload=_spec_payload(covariates=["age"], effect_modifiers=["age"]),
            data_summary=summary,
        )


def test_validate_backdoor_payload_rejects_non_numeric_continuous_outcome() -> None:
    summary = _summary_model(
        _categorical_profile("treatment", ["drug", "placebo"]),
        _categorical_profile("outcome", ["high", "low"]),
        _numeric_profile("age"),
    )
    tool = CausalSpecsTool()

    with pytest.raises(ValidationError, match=r"continuous outcome requires NUMERIC"):
        tool.validate_backdoor_payload(
            payload=_spec_payload(
                outcome_kind="continuous",
                outcome_column="outcome",
                effect_modifiers=[],
            ),
            data_summary=summary,
        )


def test_validate_backdoor_payload_rejects_datetime_treatment_column() -> None:
    summary = _summary_model(
        _datetime_profile("treatment"),
        _numeric_profile("outcome"),
        _numeric_profile("age"),
    )
    tool = CausalSpecsTool()

    with pytest.raises(ValidationError, match=r"binary treatment column cannot be DATETIME"):
        tool.validate_backdoor_payload(
            payload=_spec_payload(effect_modifiers=[]),
            data_summary=summary,
        )


def test_validate_backdoor_payload_rejects_literals_not_supported_by_exact_summary() -> None:
    summary = _summary_model(
        _boolean_profile("treatment"),
        _categorical_profile("outcome", ["yes", "no"]),
        _numeric_profile("age"),
    )
    tool = CausalSpecsTool()

    with pytest.raises(ValidationError, match=r"treatment literals are not supported"):
        tool.validate_backdoor_payload(
            payload=_spec_payload(
                treated="drug",
                control="placebo",
                outcome_kind="binary",
                outcome_column="outcome",
                event="yes",
                non_event="no",
                effect_modifiers=[],
            ),
            data_summary=summary,
        )


def test_validate_backdoor_payload_structured_returns_success_payload() -> None:
    summary = _summary_model(
        _categorical_profile("treatment", ["drug", "placebo"]),
        _numeric_profile("outcome"),
        _numeric_profile("age"),
        _categorical_profile("segment", ["A", "B"]),
    )
    tool = CausalSpecsTool()

    model_dict, issues = tool.validate_backdoor_payload_structured(
        payload=_spec_payload(),
        data_summary=summary,
    )

    assert issues == []
    assert model_dict is not None
    assert model_dict["experiment_type"] == "OBSERVATIONAL"
    assert model_dict["treatment_spec"]["column"] == "treatment"
    assert model_dict["outcome_spec"]["column"] == "outcome"


def test_validate_backdoor_payload_structured_returns_issues_for_semantic_errors() -> None:
    summary = _summary_model(
        _categorical_profile("treatment", ["drug", "placebo"]),
        _numeric_profile("outcome"),
        _numeric_profile("age"),
    )
    tool = CausalSpecsTool()

    model_dict, issues = tool.validate_backdoor_payload_structured(
        payload=_spec_payload(effect_modifiers=["missing_col"]),
        data_summary=summary,
    )

    assert model_dict is None
    assert issues
    assert issues[0]["path"] == ""
    assert "unknown dataset_summary columns" in str(issues[0]["message"])


def test_validate_backdoor_payload_structured_converts_non_mapping_payload_to_issues() -> None:
    summary = _summary_model(
        _categorical_profile("treatment", ["drug", "placebo"]),
        _numeric_profile("outcome"),
        _numeric_profile("age"),
    )
    tool = CausalSpecsTool()

    model_dict, issues = tool.validate_backdoor_payload_structured(  # type: ignore[arg-type]
        payload=["not", "a", "mapping"],
        data_summary=summary,
    )

    assert model_dict is None
    assert issues
    assert issues[0]["path"] == ""
    assert "valid dictionary" in str(issues[0]["message"]).lower()


def test_post_validate_backdoor_spec_revalidates_generic_model_against_summary() -> None:
    summary = _summary_model(
        _categorical_profile("treatment", ["drug", "placebo"]),
        _numeric_profile("outcome"),
        _numeric_profile("age"),
    )
    generic_model = CausalSpec.model_validate(_spec_payload(effect_modifiers=["unknown_modifier"]))
    tool = CausalSpecsTool()

    with pytest.raises(ValidationError, match=r"unknown dataset_summary columns"):
        tool.post_validate_backdoor_spec(
            causal_spec=generic_model,
            data_summary=summary,
        )


def test_post_validate_backdoor_spec_returns_summary_bound_model() -> None:
    summary = _summary_model(
        _categorical_profile("treatment", ["drug", "placebo"]),
        _numeric_profile("outcome"),
        _numeric_profile("age"),
        _categorical_profile("segment", ["A", "B"]),
    )
    generic_model = CausalSpec.model_validate(_spec_payload())
    tool = CausalSpecsTool()

    model = tool.post_validate_backdoor_spec(
        causal_spec=generic_model,
        data_summary=summary,
    )

    assert isinstance(model, CausalSpec)
    assert type(model) is not CausalSpec
    assert model.effect_modifiers == ["segment"]
