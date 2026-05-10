from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.implementation.workflows.nodes.data_compilation.data_compilation_cleaning import (
    clean,
)
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    ID_COL_AUTO_FILL,
    CausalSpecDraft,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)


def _dataframe() -> pd.DataFrame:
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


def _summary(dataframe: pd.DataFrame) -> DatasetSummaryModel:
    return DatasetProfilingTool().extract_dataset_summary(
        dataframe,
        max_categories=200,
        sample_distinct=200,
        compute_quantiles=False,
        strict=True,
    )


def _draft() -> CausalSpecDraft:
    return CausalSpecDraft(
        treatment_column="treatment",
        outcome_column="outcome",
        covariates=["age"],
        effect_modifiers=["isex"],
    )


@dataclass
class _FakeLLM:
    generate_calls: list[dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
    ) -> LLMResponse:
        self.generate_calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "config": config,
                "history": history,
            }
        )
        return LLMResponse(content="Return the dataframe unchanged while preserving columns.")


@dataclass
class _FakeManipulationTool:
    responses: list[pd.DataFrame] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def manipulate(
        self,
        *,
        dataframe: pd.DataFrame,
        table_name: str,
        data_summary: str,
        instructions: str,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "dataframe": dataframe.copy(),
                "table_name": table_name,
                "data_summary": data_summary,
                "instructions": instructions,
            }
        )
        if self.responses:
            return self.responses.pop(0).copy()
        return dataframe.copy()


def test_clean_is_draft_driven_and_projects_to_compiled_scope() -> None:
    dataframe = _dataframe()
    llm = _FakeLLM()
    manipulation_tool = _FakeManipulationTool()

    result = clean(
        data=dataframe,
        data_summary=_summary(dataframe),
        draft=_draft(),
        data_maupulation_tools=manipulation_tool,
        data_profiling_tools=DatasetProfilingTool(),
        llm=llm,
    )

    assert list(result.pd_cleaned.columns) == [
        ID_COL_AUTO_FILL,
        "treatment",
        "outcome",
        "age",
        "isex",
    ]
    assert result.causal.treatment_spec.column == "treatment"
    assert result.causal.outcome_spec.column == "outcome"
    assert result.effective_draft is not None
    assert result.effective_draft.id_col == ID_COL_AUTO_FILL
    assert "Removed columns: extra, patient_id" in result.summary_str
    assert len(manipulation_tool.calls) == 1
    assert len(llm.generate_calls) == 1


def test_clean_adds_missingness_indicator_for_imputed_feature_column() -> None:
    dataframe = _dataframe()
    dataframe.loc[0, "age"] = None
    typed_data = dataframe.drop(columns=["extra"]).copy()
    typed_data.insert(0, ID_COL_AUTO_FILL, [1, 2])
    imputed_data = typed_data.copy()
    imputed_data["age"] = imputed_data["age"].fillna(53)
    llm = _FakeLLM()
    manipulation_tool = _FakeManipulationTool(responses=[typed_data, imputed_data])

    result = clean(
        data=dataframe,
        data_summary=_summary(dataframe),
        draft=_draft(),
        data_maupulation_tools=manipulation_tool,
        data_profiling_tools=DatasetProfilingTool(),
        llm=llm,
    )

    assert "age__missing" in result.pd_cleaned.columns
    assert result.pd_cleaned["age__missing"].tolist() == [1, 0]
    assert result.effective_draft is not None
    assert "age__missing" in result.effective_draft.covariates
    assert "age__missing" in result.summary_str
    assert len(manipulation_tool.calls) == 2


def test_clean_passes_revised_feedback_to_planners() -> None:
    dataframe = _dataframe()
    llm = _FakeLLM()
    manipulation_tool = _FakeManipulationTool()

    clean(
        data=dataframe,
        data_summary=_summary(dataframe),
        draft=_draft(),
        data_maupulation_tools=manipulation_tool,
        data_profiling_tools=DatasetProfilingTool(),
        llm=llm,
        revised_instructions="Reclean age as a numeric baseline covariate.",
    )

    assert llm.generate_calls
    datatype_payload = json.loads(llm.generate_calls[0]["user_prompt"])
    assert datatype_payload["revised_instructions"] == (
        "Reclean age as a numeric baseline covariate."
    )
