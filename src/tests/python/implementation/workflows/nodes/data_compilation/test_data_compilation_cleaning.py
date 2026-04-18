from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from python.domain.service.llm_service import ChatMessage, LLMConfig
from python.implementation.workflows.nodes.data_compilation.data_compilation_cleaning import (
    CleaningResult,
    cleaning,
)
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import CausalSpecDraft
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)


def _build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "extra": "drop-me",
                "isex": 1,
                "outcome": "event",
                "treatment": "drug",
                "age": 45,
            },
            {
                "extra": "drop-me-too",
                "isex": 2,
                "outcome": "non_event",
                "treatment": "control",
                "age": 61,
            },
        ]
    )


def _build_summary(dataframe: pd.DataFrame) -> DatasetSummaryModel:
    return DatasetProfilingTool().extract_dataset_summary(
        dataframe,
        max_categories=200,
        sample_distinct=200,
        compute_quantiles=False,
        strict=True,
    )


def _draft() -> CausalSpecDraft:
    return CausalSpecDraft.model_validate(
        {
            "treatment_column": "treatment",
            "outcome_column": "outcome",
            "covariates": ["age"],
            "effect_modifiers": ["isex"],
        }
    )


def _causal_spec_payload(
    *,
    treatment_column: str = "treatment",
    outcome_column: str = "outcome",
    covariates: list[str] | None = None,
    effect_modifiers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "treatment_spec": {
            "kind": "binary",
            "column": treatment_column,
            "treated": "drug",
            "control": "control",
        },
        "outcome_spec": {
            "kind": "binary",
            "column": outcome_column,
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


@dataclass
class _FakeDataManipulationTool:
    responses: list[pd.DataFrame] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def manipulate(
        self,
        *,
        dataframe: pd.DataFrame,
        table_name: str,
        data_summary: str,
        instructions: str,
        retry_attempts: int = 3,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "dataframe_columns": list(dataframe.columns),
                "table_name": table_name,
                "data_summary": data_summary,
                "instructions": instructions,
                "retry_attempts": retry_attempts,
            }
        )
        if self.responses:
            return self.responses.pop(0).copy()
        return dataframe.copy()


def test_cleaning_narrows_input_dataframe_to_draft_scope_and_preserves_order() -> None:
    dataframe = _build_dataframe()
    result = cleaning(
        protocol_discussion=None,
        cleaning_instructions="",
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=_FakeLLM(json_outputs=[_causal_spec_payload()]),
    )

    assert isinstance(result, CleaningResult)
    assert list(result.pd_cleaned.columns) == ["treatment", "outcome", "age", "isex"]
    assert "extra" not in result.cleaned_data_summary.model_dump_json()


def test_cleaning_fails_immediately_when_input_dataframe_missing_draft_column() -> None:
    dataframe = _build_dataframe().drop(columns=["age"])

    with pytest.raises(ValueError, match="input dataframe does not satisfy draft causal spec"):
        cleaning(
            protocol_discussion="Confirmed protocol discussion",
            cleaning_instructions="Keep only protocol columns.",
            draft_causal_spec=_draft(),
            data_summary=_build_summary(dataframe),
            to_clean_df=dataframe,
            datasetProfilingTool=DatasetProfilingTool(),
            dataManipulationTool=_FakeDataManipulationTool(),
            llm=_FakeLLM(json_outputs=[_causal_spec_payload()]),
        )


def test_cleaning_runs_manipulation_when_effective_instructions_are_present() -> None:
    dataframe = _build_dataframe()
    data_manipulation_tool = _FakeDataManipulationTool()
    llm = _FakeLLM(json_outputs=[_causal_spec_payload()])

    result = cleaning(
        protocol_discussion="Treatment is binary and age is a baseline covariate.",
        cleaning_instructions="Normalize only grounded values.",
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert isinstance(result, CleaningResult)
    assert len(data_manipulation_tool.calls) == 1
    assert data_manipulation_tool.calls[0]["dataframe_columns"] == [
        "treatment",
        "outcome",
        "age",
        "isex",
    ]
    assert "Confirmed protocol cleaning instructions:" in data_manipulation_tool.calls[0]["instructions"]
    assert "Confirmed protocol discussion:" in data_manipulation_tool.calls[0]["instructions"]
    assert "Preserve exactly these columns" in data_manipulation_tool.calls[0]["instructions"]


def test_cleaning_skips_manipulation_when_effective_instructions_are_empty() -> None:
    dataframe = _build_dataframe()
    data_manipulation_tool = _FakeDataManipulationTool()
    llm = _FakeLLM(json_outputs=[_causal_spec_payload()])

    result = cleaning(
        protocol_discussion=None,
        cleaning_instructions="   ",
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert isinstance(result, CleaningResult)
    assert data_manipulation_tool.calls == []
    assert list(result.pd_cleaned.columns) == ["treatment", "outcome", "age", "isex"]


def test_cleaning_fails_when_manipulation_drops_required_draft_column() -> None:
    dataframe = _build_dataframe()
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[dataframe.loc[:, ["treatment", "outcome", "isex"]]]
    )

    with pytest.raises(ValueError, match="cleaned dataframe does not satisfy draft causal spec"):
        cleaning(
            protocol_discussion="Confirmed protocol discussion",
            cleaning_instructions="Apply the protocol cleaning.",
            draft_causal_spec=_draft(),
            data_summary=_build_summary(dataframe),
            to_clean_df=dataframe,
            datasetProfilingTool=DatasetProfilingTool(),
            dataManipulationTool=data_manipulation_tool,
            llm=_FakeLLM(json_outputs=[_causal_spec_payload()]),
        )


def test_cleaning_retries_compile_when_first_compiled_spec_mismatches_draft() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            _causal_spec_payload(covariates=["isex"], effect_modifiers=["age"]),
            _causal_spec_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Confirmed protocol discussion",
        cleaning_instructions="   ",
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=llm,
    )

    assert isinstance(result, CleaningResult)
    assert len(llm.generate_json_calls) == 2
    second_call_payload = json.loads(str(llm.generate_json_calls[1]["user_prompt"]))
    assert "compile_feedback" in second_call_payload
    assert result.causal.model_dump(mode="json") == llm.generate_json_calls[1]["schema"].model_validate(
        _causal_spec_payload()
    ).model_dump(mode="json")


def test_cleaning_fails_when_compiled_spec_still_mismatches_after_retry() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            _causal_spec_payload(covariates=["isex"], effect_modifiers=["age"]),
            _causal_spec_payload(covariates=["isex"], effect_modifiers=["age"]),
        ]
    )

    with pytest.raises(ValueError, match="compiled causal spec does not match draft causal spec after retry"):
        cleaning(
            protocol_discussion="Confirmed protocol discussion",
            cleaning_instructions="",
            draft_causal_spec=_draft(),
            data_summary=_build_summary(dataframe),
            to_clean_df=dataframe,
            datasetProfilingTool=DatasetProfilingTool(),
            dataManipulationTool=_FakeDataManipulationTool(),
            llm=llm,
        )


def test_cleaning_compiles_without_protocol_discussion() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(json_outputs=[_causal_spec_payload()])

    result = cleaning(
        protocol_discussion=None,
        cleaning_instructions="Keep protocol columns only.",
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=llm,
    )

    first_call_payload = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))
    assert isinstance(result, CleaningResult)
    assert "protocol_discussion" not in first_call_payload
    assert list(result.pd_cleaned.columns) == ["treatment", "outcome", "age", "isex"]
