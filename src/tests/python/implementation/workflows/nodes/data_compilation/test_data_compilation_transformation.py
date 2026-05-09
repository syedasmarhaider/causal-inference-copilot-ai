from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from python.domain.service.llm_service import ChatMessage, LLMConfig
from python.implementation.workflows.nodes.data_compilation.data_compilation_transformation import (
    ColumnTransformationSuggestionList,
    TransformationResult,
    transform,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan_tool import (
    EncodingPlanTool,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)


def _build_dataframe() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(60):
        rows.append(
            {
                "patient_id": f"p{index + 1}",
                "treatment": "drug" if index % 2 == 0 else "control",
                "outcome": "event" if index % 3 == 0 else "non_event",
                "age": 30 + index,
                "isex": 1 if index % 2 == 0 else 2,
                "segment": "A" if index % 3 == 0 else "B",
                "visit_date": f"2024-01-{(index % 28) + 1:02d}",
            }
        )
    return pd.DataFrame(rows)


def _build_summary(df: pd.DataFrame) -> DatasetSummaryModel:
    return DatasetProfilingTool().extract_dataset_summary(
        df,
        max_categories=20,
        sample_distinct=20,
        compute_quantiles=False,
        strict=True,
    )


def _object_mutation_summary(*, inferred_kind: str = "BOOLEAN") -> DatasetSummaryModel:
    if inferred_kind == "NUMERIC":
        profile = {
            "name": "pik3ca_mut",
            "dtype": "object",
            "n_rows": 60,
            "n_missing": 0,
            "missing_rate": 0.0,
            "distinct_count": 3,
            "inferred_kind": "NUMERIC",
            "summary": {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "quantiles": None,
            },
        }
    else:
        profile = {
            "name": "pik3ca_mut",
            "dtype": "object",
            "n_rows": 60,
            "n_missing": 0,
            "missing_rate": 0.0,
            "distinct_count": 3,
            "inferred_kind": "BOOLEAN",
            "summary": {"counts": {"0": 58, "E545K": 1, "H1047R": 1}},
        }
    return DatasetSummaryModel.model_validate({"n_rows": 60, "profiles": [profile]})


def _cleaned_mutation_summary() -> DatasetSummaryModel:
    return DatasetSummaryModel.model_validate(
        {
            "n_rows": 60,
            "profiles": [
                {
                    "name": "pik3ca_mut",
                    "dtype": "int64",
                    "n_rows": 60,
                    "n_missing": 0,
                    "missing_rate": 0.0,
                    "distinct_count": 2,
                    "inferred_kind": "NUMERIC",
                    "summary": {
                        "min": 0.0,
                        "max": 1.0,
                        "mean": 0.2,
                        "std": 0.4,
                        "quantiles": None,
                    },
                },
                {
                    "name": "tp53_mut",
                    "dtype": "bool",
                    "n_rows": 60,
                    "n_missing": 0,
                    "missing_rate": 0.0,
                    "distinct_count": 2,
                    "inferred_kind": "BOOLEAN",
                    "summary": {"counts": {"True": 12, "False": 48}},
                },
            ],
        }
    )


def _causal_spec_payload(
    *,
    covariates: list[str] | None = None,
    effect_modifiers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "treatment_spec": {
            "kind": "binary",
            "column": "treatment",
            "treated": "drug",
            "control": "control",
        },
        "outcome_spec": {
            "kind": "binary",
            "column": "outcome",
            "event": "event",
            "non_event": "non_event",
        },
        "covariates": covariates if covariates is not None else ["age"],
        "effect_modifiers": (effect_modifiers if effect_modifiers is not None else ["isex"]),
        "experiment_type": "OBSERVATIONAL",
        "id_col": "patient_id",
    }


@dataclass
class _FakeLLM:
    json_outputs: list[Any] = field(default_factory=list)
    generate_json_calls: list[dict[str, Any]] = field(default_factory=list)

    def generate_json(
        self,
        *,
        schema: type[Any],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> Any:
        self.generate_json_calls.append(
            {
                "schema": schema,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "config": config,
                "history": history,
                "max_attempts": max_attempts,
            }
        )
        if not self.json_outputs:
            raise AssertionError("unexpected generate_json call")
        next_output = self.json_outputs.pop(0)
        if isinstance(next_output, Exception):
            raise next_output
        if isinstance(next_output, dict):
            return schema.model_validate(next_output)
        return next_output


def test_transform_returns_none_when_no_protocol_scope_columns_exist() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM()

    result = transform(
        transformation_instructions="",
        causal_spec=CausalSpec.model_validate(
            _causal_spec_payload(covariates=[], effect_modifiers=[])
        ),
        data_summary=_build_summary(dataframe),
        llm=llm,
    )

    assert isinstance(result, TransformationResult)
    assert result.transformation_plan is None
    assert result.transformation_suggestions is None
    assert llm.generate_json_calls == []


def test_transform_does_not_offer_identifier_column_as_eligible_feature() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                        "preferred_type": "NUMERIC",
                        "preferred_type_reason": "Age is already numeric.",
                    },
                    {
                        "column": "isex",
                        "role": "effect_modifier",
                        "preset": "passthrough",
                        "preferred_type": "CATEGORICAL",
                        "preferred_type_reason": "The numeric codes act like categories.",
                    },
                ]
            }
        ]
    )

    _ = transform(
        transformation_instructions="",
        causal_spec=CausalSpec.model_validate(_causal_spec_payload()),
        data_summary=_build_summary(dataframe),
        llm=llm,
    )

    prompt_payload = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))
    assert prompt_payload["compiled_causal_specification"]["id_col"] == "patient_id"
    assert prompt_payload["eligible_columns"] == ["age", "isex"]
    assert "patient_id" not in prompt_payload["eligible_columns"]


def test_transform_builds_type_driven_plan_and_saves_preferred_type_suggestions() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                        "preferred_type": "NUMERIC",
                        "preferred_type_reason": "Age is already stored as numeric values.",
                    },
                    {
                        "column": "isex",
                        "role": "effect_modifier",
                        "preset": "passthrough",
                        "preferred_type": "CATEGORICAL",
                        "preferred_type_reason": "The numeric codes look like category labels and would be cleaner as explicit categories.",
                    },
                ]
            }
        ]
    )

    result = transform(
        transformation_instructions="",
        causal_spec=CausalSpec.model_validate(_causal_spec_payload()),
        data_summary=_build_summary(dataframe),
        llm=llm,
    )

    assert result.transformation_plan is not None
    assert [column.column for column in result.transformation_plan.columns] == ["age", "isex"]
    assert [column.encoding.preset for column in result.transformation_plan.columns] == [
        "passthrough",
        "passthrough",
    ]
    assert result.transformation_suggestions is not None
    assert isinstance(result.transformation_suggestions, ColumnTransformationSuggestionList)
    assert [
        suggestion.preferred_type for suggestion in result.transformation_suggestions.suggestions
    ] == ["NUMERIC", "CATEGORICAL"]


def test_transform_uses_numeric_preset_when_numeric_change_is_requested() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "column": "age",
                        "role": "covariate",
                        "preset": "num_standard",
                        "preferred_type": "NUMERIC",
                        "preferred_type_reason": "Age should remain numeric.",
                    },
                    {
                        "column": "isex",
                        "role": "effect_modifier",
                        "preset": "passthrough",
                        "preferred_type": "CATEGORICAL",
                        "preferred_type_reason": "The codes would be clearer as explicit category labels.",
                    },
                ]
            }
        ]
    )

    result = transform(
        transformation_instructions="Standardize age before modeling.",
        causal_spec=CausalSpec.model_validate(_causal_spec_payload()),
        data_summary=_build_summary(dataframe),
        llm=llm,
    )

    assert result.transformation_plan is not None
    assert [column.encoding.preset for column in result.transformation_plan.columns] == [
        "num_standard",
        "passthrough",
    ]


def test_transform_only_allows_cat_onehot_for_categorical_columns() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                        "preferred_type": "NUMERIC",
                        "preferred_type_reason": "Age is already numeric.",
                    },
                    {
                        "column": "segment",
                        "role": "effect_modifier",
                        "preset": "cat_onehot",
                        "preferred_type": "CATEGORICAL",
                        "preferred_type_reason": "Segment is already categorical.",
                    },
                ]
            }
        ]
    )

    result = transform(
        transformation_instructions="",
        causal_spec=CausalSpec.model_validate(_causal_spec_payload(effect_modifiers=["segment"])),
        data_summary=_build_summary(dataframe),
        llm=llm,
    )

    assert result.transformation_plan is not None
    assert [column.encoding.preset for column in result.transformation_plan.columns] == [
        "passthrough",
        "cat_onehot",
    ]


def test_transform_rejects_object_boolean_passthrough_without_cleaning_cast() -> None:
    dataframe = _build_dataframe()
    dataframe["flag"] = ["Yes" if index % 2 == 0 else "No" for index in range(len(dataframe))]
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "column": "flag",
                        "role": "covariate",
                        "preset": "passthrough",
                        "preferred_type": "BOOLEAN",
                        "preferred_type_reason": "Flag is a boolean-like Yes/No indicator.",
                    }
                ]
            },
            {
                "columns": [
                    {
                        "column": "flag",
                        "role": "covariate",
                        "preset": "passthrough",
                        "preferred_type": "BOOLEAN",
                        "preferred_type_reason": "Flag is a boolean-like Yes/No indicator.",
                    }
                ]
            },
        ]
    )

    with pytest.raises(ValueError, match="current kind 'CATEGORICAL'"):
        transform(
            transformation_instructions="",
            causal_spec=CausalSpec.model_validate(
                _causal_spec_payload(covariates=["flag"], effect_modifiers=[])
            ),
            data_summary=_build_summary(dataframe),
            llm=llm,
            encoding_plan_tool=EncodingPlanTool(),
        )

    assert len(llm.generate_json_calls) == 2


def test_transform_only_allows_datetime_epoch_seconds_for_datetime_columns() -> None:
    dataframe = _build_dataframe()
    dataframe["visit_date"] = pd.to_datetime(dataframe["visit_date"])
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "column": "visit_date",
                        "role": "covariate",
                        "preset": "datetime_epoch_seconds",
                        "preferred_type": "DATETIME",
                        "preferred_type_reason": "Visit date is already stored as a datetime-compatible field.",
                    }
                ]
            }
        ]
    )

    result = transform(
        transformation_instructions="",
        causal_spec=CausalSpec.model_validate(
            _causal_spec_payload(covariates=["visit_date"], effect_modifiers=[])
        ),
        data_summary=_build_summary(dataframe),
        llm=llm,
    )

    assert result.transformation_plan is not None
    assert [column.encoding.preset for column in result.transformation_plan.columns] == [
        "datetime_epoch_seconds"
    ]


def test_transform_prompt_payload_uses_stored_kind_and_preserves_inferred_kind() -> None:
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "column": "pik3ca_mut",
                        "role": "covariate",
                        "preset": "cat_onehot",
                        "preferred_type": "BOOLEAN",
                        "preferred_type_reason": "Mutation strings are categorical until cleaning casts them.",
                    }
                ]
            }
        ]
    )

    result = transform(
        transformation_instructions="",
        causal_spec=CausalSpec.model_validate(
            _causal_spec_payload(covariates=["pik3ca_mut"], effect_modifiers=[])
        ),
        data_summary=_object_mutation_summary(),
        llm=llm,
    )

    assert result.transformation_plan is not None
    prompt_payload = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))
    [column_payload] = prompt_payload["scoped_dataset_summary"]["columns"]
    assert column_payload["name"] == "pik3ca_mut"
    assert column_payload["dtype"] == "object"
    assert column_payload["kind"] == "CATEGORICAL"
    assert column_payload["inferred_kind"] == "BOOLEAN"


def test_transform_retries_object_mutation_passthrough_as_stored_kind_incompatibility() -> None:
    invalid_output = {
        "columns": [
            {
                "column": "pik3ca_mut",
                "role": "covariate",
                "preset": "passthrough",
                "preferred_type": "BOOLEAN",
                "preferred_type_reason": "Mutation strings look like a present/absent flag.",
            }
        ]
    }
    llm = _FakeLLM(json_outputs=[invalid_output, invalid_output])

    with pytest.raises(ValueError, match="current kind 'CATEGORICAL'"):
        transform(
            transformation_instructions="",
            causal_spec=CausalSpec.model_validate(
                _causal_spec_payload(covariates=["pik3ca_mut"], effect_modifiers=[])
            ),
            data_summary=_object_mutation_summary(),
            llm=llm,
        )

    assert len(llm.generate_json_calls) == 2
    retry_payload = json.loads(str(llm.generate_json_calls[1]["user_prompt"]))
    assert "current kind 'CATEGORICAL'" in retry_payload["retry_note"]


def test_transform_allows_numeric_and_boolean_presets_after_cleaning_casts_mutations() -> None:
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "column": "pik3ca_mut",
                        "role": "covariate",
                        "preset": "num_standard",
                        "preferred_type": "NUMERIC",
                        "preferred_type_reason": "Cleaning recoded the mutation to numeric 0/1.",
                    },
                    {
                        "column": "tp53_mut",
                        "role": "effect_modifier",
                        "preset": "passthrough",
                        "preferred_type": "BOOLEAN",
                        "preferred_type_reason": "Cleaning recoded the mutation to bool.",
                    },
                ]
            }
        ]
    )

    result = transform(
        transformation_instructions="",
        causal_spec=CausalSpec.model_validate(
            _causal_spec_payload(covariates=["pik3ca_mut"], effect_modifiers=["tp53_mut"])
        ),
        data_summary=_cleaned_mutation_summary(),
        llm=llm,
    )

    assert result.transformation_plan is not None
    assert [column.encoding.preset for column in result.transformation_plan.columns] == [
        "num_standard",
        "passthrough",
    ]


def test_transform_rejects_preset_that_is_incompatible_with_current_kind() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                        "preferred_type": "NUMERIC",
                        "preferred_type_reason": "Age is already numeric.",
                    },
                    {
                        "column": "isex",
                        "role": "effect_modifier",
                        "preset": "cat_onehot",
                        "preferred_type": "CATEGORICAL",
                        "preferred_type_reason": "The codes would be clearer as explicit categories.",
                    },
                ]
            },
            {
                "columns": [
                    {
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                        "preferred_type": "NUMERIC",
                        "preferred_type_reason": "Age is already numeric.",
                    },
                    {
                        "column": "isex",
                        "role": "effect_modifier",
                        "preset": "cat_onehot",
                        "preferred_type": "CATEGORICAL",
                        "preferred_type_reason": "The codes would be clearer as explicit categories.",
                    },
                ]
            },
        ]
    )

    with pytest.raises(ValueError, match="preset 'cat_onehot' is not allowed"):
        transform(
            transformation_instructions="",
            causal_spec=CausalSpec.model_validate(_causal_spec_payload()),
            data_summary=_build_summary(dataframe),
            llm=llm,
        )


def test_transform_retries_batch_with_retry_note_then_succeeds() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                        "preferred_type": "NUMERIC",
                        "preferred_type_reason": "Age is already numeric.",
                    }
                ]
            },
            {
                "columns": [
                    {
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                        "preferred_type": "NUMERIC",
                        "preferred_type_reason": "Age is already numeric.",
                    },
                    {
                        "column": "isex",
                        "role": "effect_modifier",
                        "preset": "num_standard",
                        "preferred_type": "CATEGORICAL",
                        "preferred_type_reason": "The codes would be clearer as explicit categories.",
                    },
                ]
            },
        ]
    )

    result = transform(
        transformation_instructions="",
        causal_spec=CausalSpec.model_validate(_causal_spec_payload()),
        data_summary=_build_summary(dataframe),
        llm=llm,
    )

    assert result.transformation_plan is not None
    assert result.transformation_suggestions is not None
    assert len(llm.generate_json_calls) == 2
    second_batch_payload = json.loads(str(llm.generate_json_calls[1]["user_prompt"]))
    assert "retry_note" in second_batch_payload
    assert "preferred_type" in str(llm.generate_json_calls[0]["system_prompt"])


def test_transform_requires_non_empty_preferred_type_reason() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                        "preferred_type": "NUMERIC",
                        "preferred_type_reason": "",
                    },
                    {
                        "column": "isex",
                        "role": "effect_modifier",
                        "preset": "passthrough",
                        "preferred_type": "CATEGORICAL",
                        "preferred_type_reason": "The codes would be clearer as explicit categories.",
                    },
                ]
            },
            {
                "columns": [
                    {
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                        "preferred_type": "NUMERIC",
                        "preferred_type_reason": "",
                    },
                    {
                        "column": "isex",
                        "role": "effect_modifier",
                        "preset": "passthrough",
                        "preferred_type": "CATEGORICAL",
                        "preferred_type_reason": "The codes would be clearer as explicit categories.",
                    },
                ]
            },
        ]
    )

    with pytest.raises(ValueError, match="batch transformation draft failed after retry"):
        transform(
            transformation_instructions="",
            causal_spec=CausalSpec.model_validate(_causal_spec_payload()),
            data_summary=_build_summary(dataframe),
            llm=llm,
        )
    assert len(llm.generate_json_calls) == 2
