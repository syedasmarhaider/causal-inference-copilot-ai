from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.implementation.workflows.nodes.data_compilation.data_compilation_cleaning import (
    clean,
    compile_causal_spec_from_draft,
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
    json_outputs: list[dict[str, Any]] = field(default_factory=list)
    generate_json_calls: list[dict[str, Any]] = field(default_factory=list)

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
        payload = (
            self.json_outputs.pop(0)
            if self.json_outputs
            else {
                "treatment_treated": "drug",
                "treatment_control": "control",
                "outcome_event": "event",
                "outcome_non_event": "non_event",
                "negative_control_event": None,
                "negative_control_non_event": None,
            }
        )
        return schema.model_validate(payload)


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


def test_compile_causal_spec_keeps_one_as_treated_when_zero_is_majority() -> None:
    dataframe = pd.DataFrame(
        {
            ID_COL_AUTO_FILL: range(1, 5467),
            "treatment": ["0"] * 5464 + ["1", "1"],
            "outcome": ["0", "1"] * 2733,
            "age": [50] * 5466,
        }
    )
    llm = _FakeLLM(
        json_outputs=[
            {
                "treatment_treated": "1",
                "treatment_control": "0",
                "outcome_event": "1",
                "outcome_non_event": "0",
                "negative_control_event": None,
                "negative_control_non_event": None,
            }
        ]
    )

    causal_spec = compile_causal_spec_from_draft(
        dataset_summary=_summary(dataframe),
        previous_draft=CausalSpecDraft(
            id_col=ID_COL_AUTO_FILL,
            treatment_column="treatment",
            outcome_column="outcome",
            covariates=["age"],
        ),
        llm=llm,
    )

    assert causal_spec.treatment_spec.treated == "1"
    assert causal_spec.treatment_spec.control == "0"
    prompt_payload = json.loads(llm.generate_json_calls[0]["user_prompt"])
    assert "negative_control_outcome" not in prompt_payload["binary_roles"]


def test_compile_causal_spec_uses_llm_outcome_event_when_non_event_is_majority() -> None:
    dataframe = pd.DataFrame(
        {
            ID_COL_AUTO_FILL: range(1, 5467),
            "treatment": ["drug", "control"] * 2733,
            "outcome": ["0"] * 5464 + ["1", "1"],
            "age": [50] * 5466,
        }
    )
    llm = _FakeLLM(
        json_outputs=[
            {
                "treatment_treated": "drug",
                "treatment_control": "control",
                "outcome_event": "1",
                "outcome_non_event": "0",
                "negative_control_event": None,
                "negative_control_non_event": None,
            }
        ]
    )

    causal_spec = compile_causal_spec_from_draft(
        dataset_summary=_summary(dataframe),
        previous_draft=CausalSpecDraft(
            id_col=ID_COL_AUTO_FILL,
            treatment_column="treatment",
            outcome_column="outcome",
            covariates=["age"],
        ),
        llm=llm,
    )

    assert causal_spec.outcome_spec.kind == "binary"
    assert causal_spec.outcome_spec.event == "1"
    assert causal_spec.outcome_spec.non_event == "0"


def test_compile_causal_spec_uses_optional_negative_control_binary_roles() -> None:
    dataframe = pd.DataFrame(
        {
            ID_COL_AUTO_FILL: range(1, 7),
            "treatment": ["drug", "control"] * 3,
            "outcome": ["event", "non_event"] * 3,
            "negative_control": ["negative_non_event"] * 4 + ["negative_event"] * 2,
            "age": [50] * 6,
        }
    )
    llm = _FakeLLM(
        json_outputs=[
            {
                "treatment_treated": "drug",
                "treatment_control": "control",
                "outcome_event": "event",
                "outcome_non_event": "non_event",
                "negative_control_event": "negative_event",
                "negative_control_non_event": "negative_non_event",
            }
        ]
    )

    causal_spec = compile_causal_spec_from_draft(
        dataset_summary=_summary(dataframe),
        previous_draft=CausalSpecDraft(
            id_col=ID_COL_AUTO_FILL,
            treatment_column="treatment",
            outcome_column="outcome",
            negative_control_outcome="negative_control",
            covariates=["age"],
        ),
        llm=llm,
    )

    assert causal_spec.negative_control_outcome is not None
    assert causal_spec.negative_control_outcome.kind == "binary"
    assert causal_spec.negative_control_outcome.event == "negative_event"
    assert causal_spec.negative_control_outcome.non_event == "negative_non_event"


def test_compile_causal_spec_rejects_llm_role_value_not_observed() -> None:
    dataframe = pd.DataFrame(
        {
            ID_COL_AUTO_FILL: range(1, 5),
            "treatment": ["drug", "control"] * 2,
            "outcome": ["event", "non_event"] * 2,
            "age": [50] * 4,
        }
    )
    llm = _FakeLLM(
        json_outputs=[
            {
                "treatment_treated": "drug",
                "treatment_control": "placebo",
                "outcome_event": "event",
                "outcome_non_event": "non_event",
                "negative_control_event": None,
                "negative_control_non_event": None,
            }
        ]
    )

    with pytest.raises(ValueError, match="not observed for treatment"):
        compile_causal_spec_from_draft(
            dataset_summary=_summary(dataframe),
            previous_draft=CausalSpecDraft(
                id_col=ID_COL_AUTO_FILL,
                treatment_column="treatment",
                outcome_column="outcome",
                covariates=["age"],
            ),
            llm=llm,
        )


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

    assert "age_missing" in result.pd_cleaned.columns
    assert "age__missing" not in result.pd_cleaned.columns
    assert result.pd_cleaned["age_missing"].tolist() == [1, 0]
    assert result.effective_draft is not None
    assert "age_missing" in result.effective_draft.covariates
    assert "age_missing" in result.summary_str
    assert len(manipulation_tool.calls) == 2


def test_clean_revised_instructions_do_not_auto_add_missingness_indicator() -> None:
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
        revised_instructions=(
            "Simplification: remove redundant missing indicators for effect modifiers "
            "and covariates unless they are clinically necessary."
        ),
    )

    assert "age_missing" not in result.pd_cleaned.columns
    assert result.effective_draft is not None
    assert "age_missing" not in result.effective_draft.covariates
    imputation_payload = json.loads(llm.generate_calls[-1]["user_prompt"])
    assert "not added automatically" in imputation_payload["missing_indicator_policy"]


def test_clean_revised_instructions_keep_missingness_indicator_when_llm_adds_it() -> None:
    dataframe = _dataframe()
    dataframe.loc[0, "age"] = None
    typed_data = dataframe.drop(columns=["extra"]).copy()
    typed_data.insert(0, ID_COL_AUTO_FILL, [1, 2])
    imputed_data = typed_data.copy()
    imputed_data["age"] = imputed_data["age"].fillna(53)
    imputed_data["age_missing"] = [1, 0]
    llm = _FakeLLM()
    manipulation_tool = _FakeManipulationTool(responses=[typed_data, imputed_data])

    result = clean(
        data=dataframe,
        data_summary=_summary(dataframe),
        draft=_draft(),
        data_maupulation_tools=manipulation_tool,
        data_profiling_tools=DatasetProfilingTool(),
        llm=llm,
        revised_instructions="Keep a missingness indicator for age because it is clinically useful.",
    )

    assert "age_missing" in result.pd_cleaned.columns
    assert result.pd_cleaned["age_missing"].tolist() == [1, 0]
    assert result.effective_draft is not None
    assert "age_missing" in result.effective_draft.covariates


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
