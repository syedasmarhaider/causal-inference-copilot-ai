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
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    ID_COL_AUTO_FILL,
    CausalSpecDraft,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)


def _build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "extra": "drop-me",
                "patient_id": "p1",
                "isex": 1,
                "outcome": "event",
                "treatment": "drug",
                "age": 45,
            },
            {
                "extra": "drop-me-too",
                "patient_id": "p2",
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


def _draft_with_identifier(identifier_column: str) -> CausalSpecDraft:
    return CausalSpecDraft.model_validate(
        {
            "id_col": identifier_column,
            "treatment_column": "treatment",
            "outcome_column": "outcome",
            "covariates": ["age"],
            "effect_modifiers": ["isex"],
        }
    )


def _semantic_payload(
    *,
    treated: str = "drug",
    control: str = "control",
    outcome_kind: str = "binary",
    event: str = "event",
    non_event: str = "non_event",
    unit: str | None = None,
    clip_min: float | None = None,
    clip_max: float | None = None,
    experiment_type: str = "OBSERVATIONAL",
) -> dict[str, Any]:
    outcome: dict[str, Any]
    if outcome_kind == "continuous":
        outcome = {
            "kind": "continuous",
            "unit": unit,
            "clip_min": clip_min,
            "clip_max": clip_max,
        }
    else:
        outcome = {
            "kind": "binary",
            "event": event,
            "non_event": non_event,
        }

    return {
        "treatment": {
            "treated": treated,
            "control": control,
        },
        "outcome": outcome,
        "experiment_type": experiment_type,
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
        llm=_FakeLLM(json_outputs=[_semantic_payload()]),
    )

    assert isinstance(result, CleaningResult)
    assert list(result.pd_cleaned.columns) == [
        ID_COL_AUTO_FILL,
        "treatment",
        "outcome",
        "age",
        "isex",
    ]
    assert result.causal.id_col == ID_COL_AUTO_FILL
    assert result.pd_cleaned[ID_COL_AUTO_FILL].tolist() == [1, 2]
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
            llm=_FakeLLM(json_outputs=[_semantic_payload()]),
        )


def test_cleaning_runs_manipulation_when_effective_instructions_are_present() -> None:
    dataframe = _build_dataframe()
    data_manipulation_tool = _FakeDataManipulationTool()
    llm = _FakeLLM(json_outputs=[_semantic_payload()])

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
    llm = _FakeLLM(json_outputs=[_semantic_payload()])

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
    assert list(result.pd_cleaned.columns) == [
        ID_COL_AUTO_FILL,
        "treatment",
        "outcome",
        "age",
        "isex",
    ]


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
            llm=_FakeLLM(json_outputs=[_semantic_payload()]),
        )


def test_cleaning_retries_compile_when_first_semantic_compile_is_invalid() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            _semantic_payload(treated="rx", control="control"),
            _semantic_payload(),
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
    assert result.causal.treatment_spec.column == "treatment"
    assert result.causal.outcome_spec.column == "outcome"
    assert result.causal.id_col == ID_COL_AUTO_FILL
    assert result.causal.covariates == ["age"]
    assert result.causal.effect_modifiers == ["isex"]
    assert result.causal.experiment_type == "OBSERVATIONAL"


def test_cleaning_fails_when_semantic_compile_is_still_invalid_after_retry() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            _semantic_payload(treated="rx", control="control"),
            _semantic_payload(treated="rx", control="control"),
        ]
    )

    with pytest.raises(ValueError, match="compiled causal spec semantics remained invalid after retry"):
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
    llm = _FakeLLM(json_outputs=[_semantic_payload()])

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
    assert list(result.pd_cleaned.columns) == [
        ID_COL_AUTO_FILL,
        "treatment",
        "outcome",
        "age",
        "isex",
    ]


def test_cleaning_preserves_explicit_identifier_column_and_compiles_it_into_causal_spec() -> None:
    dataframe = _build_dataframe()

    result = cleaning(
        protocol_discussion=None,
        cleaning_instructions="",
        draft_causal_spec=_draft_with_identifier("patient_id"),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=_FakeLLM(json_outputs=[_semantic_payload()]),
    )

    assert list(result.pd_cleaned.columns) == [
        "patient_id",
        "treatment",
        "outcome",
        "age",
        "isex",
    ]
    assert result.pd_cleaned["patient_id"].tolist() == ["p1", "p2"]
    assert result.causal.id_col == "patient_id"
