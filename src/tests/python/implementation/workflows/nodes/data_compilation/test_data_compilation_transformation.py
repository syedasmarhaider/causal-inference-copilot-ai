from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from python.domain.service.llm_service import ChatMessage, LLMConfig
from python.implementation.workflows.nodes.data_compilation.data_compilation_transformation import (
    DatasetRepairPlan,
    TransformationResult,
    transform,
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
                "treatment": "drug" if index % 2 == 0 else "control",
                "outcome": "event" if index % 3 == 0 else "non_event",
                "age": 30 + index,
                "isex": 1 if index % 2 == 0 else 2,
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
        "effect_modifiers": (
            effect_modifiers if effect_modifiers is not None else ["isex"]
        ),
        "experiment_type": "OBSERVATIONAL",
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
    assert result.required_dataset_changes is None
    assert llm.generate_json_calls == []


def test_transform_builds_passthrough_plan_for_covariates_and_effect_modifiers_only() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "decision": "plan",
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                    },
                    {
                        "decision": "plan",
                        "column": "isex",
                        "role": "effect_modifier",
                        "preset": "passthrough",
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

    assert isinstance(result, TransformationResult)
    assert result.required_dataset_changes is None
    assert result.transformation_plan is not None
    assert [column.column for column in result.transformation_plan.columns] == ["age", "isex"]
    assert [column.role for column in result.transformation_plan.columns] == [
        "covariate",
        "effect_modifier",
    ]
    assert [column.encoding.preset for column in result.transformation_plan.columns] == [
        "passthrough",
        "passthrough",
    ]


def test_transform_uses_real_preset_when_grounded_change_is_requested() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "decision": "plan",
                        "column": "age",
                        "role": "covariate",
                        "preset": "num_standard",
                    },
                    {
                        "decision": "plan",
                        "column": "isex",
                        "role": "effect_modifier",
                        "preset": "passthrough",
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

    assert result.required_dataset_changes is None
    assert result.transformation_plan is not None
    assert [column.encoding.preset for column in result.transformation_plan.columns] == [
        "num_standard",
        "passthrough",
    ]


def test_transform_returns_strict_dataset_change_blocker_for_numeric_coded_category() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "decision": "plan",
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                    },
                    {
                        "decision": "dataset_change",
                        "column": "isex",
                        "role": "effect_modifier",
                        "problem": "numeric_coded_category",
                        "action": "normalize_categorical_representation",
                        "reason": "This effect modifier appears to need categorical handling, but the current numeric coding does not ground a safe categorical encoding plan.",
                        "repair_instruction": "Replace numeric codes 1 and 2 with explicit category labels such as male and female in the source dataset before recompilation.",
                        "user_explanation": "Update the dataset values first, then rerun compilation so categorical encoding can be grounded safely.",
                    },
                ]
            }
        ]
    )

    result = transform(
        transformation_instructions="Treat isex as a categorical effect modifier.",
        causal_spec=CausalSpec.model_validate(_causal_spec_payload()),
        data_summary=_build_summary(dataframe),
        llm=llm,
    )

    assert result.transformation_plan is None
    assert result.required_dataset_changes is not None
    assert isinstance(result.required_dataset_changes, DatasetRepairPlan)
    assert len(result.required_dataset_changes.actions) == 1
    action = result.required_dataset_changes.actions[0]
    assert action.column == "isex"
    assert action.role == "effect_modifier"
    assert action.problem == "numeric_coded_category"
    assert action.action == "normalize_categorical_representation"
    assert "Replace numeric codes 1 and 2" in action.repair_instruction
    assert action.user_explanation is not None


def test_transform_retries_batch_with_repair_request_then_succeeds() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "decision": "plan",
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                    }
                ]
            },
            {
                "columns": [
                    {
                        "decision": "plan",
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                    },
                    {
                        "decision": "plan",
                        "column": "isex",
                        "role": "effect_modifier",
                        "preset": "num_standard",
                    }
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
    assert result.required_dataset_changes is None
    assert len(llm.generate_json_calls) == 2
    second_batch_payload = json.loads(str(llm.generate_json_calls[1]["user_prompt"]))
    assert "repair_request" in second_batch_payload
    assert "columns array" in str(llm.generate_json_calls[0]["system_prompt"])


def test_transform_raises_when_batch_retry_is_still_incompatible() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "decision": "plan",
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                    }
                ]
            },
            {
                "columns": [
                    {
                        "decision": "plan",
                        "column": "age",
                        "role": "covariate",
                        "preset": "passthrough",
                    }
                ]
            }
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
