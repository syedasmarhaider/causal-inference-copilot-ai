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
from python.implementation.workflows.tools.simple_data_transformation_tool.simple_data_transformation_tool import (
    SimpleDataTransformationSpec,
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


def _empty_simple_plan() -> dict[str, Any]:
    return {"columns": []}


def _empty_manipulation_plan() -> dict[str, Any]:
    return {"instructions": None}


def _column_names_from_prompt_summary(summary: dict[str, Any]) -> list[str]:
    return [str(column["name"]) for column in summary["columns"]]


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
                "dataframe": dataframe.copy(),
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


@dataclass
class _FakeSimpleDataTransformationTool:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def transform(
        self,
        *,
        dataframe: pd.DataFrame,
        specification: SimpleDataTransformationSpec | dict[str, Any],
        copy: bool = True,
    ) -> pd.DataFrame:
        spec = (
            specification
            if isinstance(specification, SimpleDataTransformationSpec)
            else SimpleDataTransformationSpec.model_validate(specification)
        )
        self.calls.append(
            {
                "dataframe": dataframe.copy(),
                "specification": spec.model_dump(mode="json"),
                "copy": copy,
            }
        )
        result = dataframe.copy(deep=True) if copy else dataframe
        for column_spec in spec.columns:
            column = str(column_spec.column)
            if column_spec.has_value:
                result[column] = column_spec.value
            for replacement in column_spec.replacements:
                result[column] = result[column].replace(
                    {replacement.from_value: replacement.to_value}
                )
            if column_spec.has_fill_value:
                result[column] = result[column].where(
                    ~result[column].isna(),
                    column_spec.fill_value,
                )
            if column_spec.target_dtype == "integer":
                result[column] = pd.to_numeric(result[column]).astype("int64")
            if column_spec.target_dtype == "float":
                result[column] = pd.to_numeric(result[column]).astype("float64")
        return result


def test_cleaning_narrows_input_dataframe_to_draft_scope_and_preserves_order() -> None:
    dataframe = _build_dataframe()
    result = cleaning(
        protocol_discussion=None,
        cleaning_instructions="",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        simpleDataTransformationTool=_FakeSimpleDataTransformationTool(),
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=_FakeLLM(
            json_outputs=[
                _empty_simple_plan(),
                _empty_manipulation_plan(),
                _semantic_payload(),
            ]
        ),
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

    with pytest.raises(
        ValueError,
        match="input dataframe is missing required column\\(s\\): age",
    ):
        cleaning(
            protocol_discussion="Confirmed protocol discussion",
            cleaning_instructions="Keep only protocol columns.",
            review_recompile_request=None,
            draft_causal_spec=_draft(),
            data_summary=_build_summary(dataframe),
            to_clean_df=dataframe,
            datasetProfilingTool=DatasetProfilingTool(),
            simpleDataTransformationTool=_FakeSimpleDataTransformationTool(),
            dataManipulationTool=_FakeDataManipulationTool(),
            llm=_FakeLLM(json_outputs=[_semantic_payload()]),
        )


def test_cleaning_runs_manipulation_when_effective_instructions_are_present() -> None:
    dataframe = _build_dataframe()
    data_manipulation_tool = _FakeDataManipulationTool()
    llm = _FakeLLM(
        json_outputs=[
            _empty_simple_plan(),
            {"instructions": "Normalize only grounded values."},
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Treatment is binary and age is a baseline covariate.",
        cleaning_instructions="Normalize only grounded values.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        simpleDataTransformationTool=_FakeSimpleDataTransformationTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert isinstance(result, CleaningResult)
    assert len(data_manipulation_tool.calls) == 1
    assert data_manipulation_tool.calls[0]["dataframe_columns"] == [
        "extra",
        "patient_id",
        "isex",
        "outcome",
        "treatment",
        "age",
        ID_COL_AUTO_FILL,
    ]
    assert data_manipulation_tool.calls[0]["instructions"] == ("Normalize only grounded values.")


def test_cleaning_applies_simple_transform_before_sql_and_manually_drops_extra_columns() -> None:
    dataframe = _build_dataframe()
    dataframe["age"] = ["45", "61"]
    simple_transform_tool = _FakeSimpleDataTransformationTool()
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            pd.DataFrame(
                {
                    ID_COL_AUTO_FILL: [1, 2],
                    "treatment": ["drug", "control"],
                    "outcome": ["event", "non_event"],
                    "age": [45, 61],
                    "isex": [1, 2],
                    "sql_extra": ["drop", "drop"],
                }
            )
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            {
                "columns": [
                    {
                        "column": "age",
                        "target_dtype": "integer",
                    }
                ]
            },
            {"instructions": "Apply downstream SQL cleaning."},
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Age is a numeric baseline covariate.",
        cleaning_instructions="Cast age to an integer before downstream cleaning.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        simpleDataTransformationTool=simple_transform_tool,
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert len(simple_transform_tool.calls) == 1
    assert simple_transform_tool.calls[0]["specification"]["columns"][0]["column"] == "age"
    assert len(data_manipulation_tool.calls) == 1
    sql_input = data_manipulation_tool.calls[0]["dataframe"]
    assert str(sql_input["age"].dtype) == "int64"
    assert data_manipulation_tool.calls[0]["instructions"] == ("Apply downstream SQL cleaning.")
    simple_prompt = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))
    manipulation_prompt = json.loads(str(llm.generate_json_calls[1]["user_prompt"]))
    assert ID_COL_AUTO_FILL not in _column_names_from_prompt_summary(
        simple_prompt["source_dataset_summary"]
    )
    assert ID_COL_AUTO_FILL in _column_names_from_prompt_summary(
        simple_prompt["prepared_dataset_summary"]
    )
    transformed_age_profile = next(
        column
        for column in manipulation_prompt["transformed_dataset_summary"]["columns"]
        if column["name"] == "age"
    )
    assert transformed_age_profile["dtype"] == "int64"
    assert json.loads(data_manipulation_tool.calls[0]["data_summary"])
    assert list(result.pd_cleaned.columns) == [
        ID_COL_AUTO_FILL,
        "treatment",
        "outcome",
        "age",
        "isex",
    ]
    assert "sql_extra" not in result.pd_cleaned.columns


def test_cleaning_skips_manipulation_when_effective_instructions_are_empty() -> None:
    dataframe = _build_dataframe()
    data_manipulation_tool = _FakeDataManipulationTool()
    simple_transform_tool = _FakeSimpleDataTransformationTool()
    llm = _FakeLLM(
        json_outputs=[
            _empty_simple_plan(),
            _empty_manipulation_plan(),
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion=None,
        cleaning_instructions="   ",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        simpleDataTransformationTool=simple_transform_tool,
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert isinstance(result, CleaningResult)
    assert len(llm.generate_json_calls) == 3
    assert simple_transform_tool.calls == []
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

    with pytest.raises(
        ValueError,
        match="cleaned dataframe is missing required column\\(s\\): " f"{ID_COL_AUTO_FILL}, age",
    ):
        cleaning(
            protocol_discussion="Confirmed protocol discussion",
            cleaning_instructions="Apply the protocol cleaning.",
            review_recompile_request=None,
            draft_causal_spec=_draft(),
            data_summary=_build_summary(dataframe),
            to_clean_df=dataframe,
            datasetProfilingTool=DatasetProfilingTool(),
            simpleDataTransformationTool=_FakeSimpleDataTransformationTool(),
            dataManipulationTool=data_manipulation_tool,
            llm=_FakeLLM(
                json_outputs=[
                    _empty_simple_plan(),
                    {"instructions": "Apply the protocol cleaning."},
                    _semantic_payload(),
                ]
            ),
        )


def test_cleaning_retries_compile_when_first_semantic_compile_is_invalid() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            _empty_simple_plan(),
            _empty_manipulation_plan(),
            _semantic_payload(treated="rx", control="control"),
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Confirmed protocol discussion",
        cleaning_instructions="   ",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        simpleDataTransformationTool=_FakeSimpleDataTransformationTool(),
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=llm,
    )

    assert isinstance(result, CleaningResult)
    assert len(llm.generate_json_calls) == 4
    fourth_call_payload = json.loads(str(llm.generate_json_calls[3]["user_prompt"]))
    assert "compile_feedback" in fourth_call_payload
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
            _empty_simple_plan(),
            _empty_manipulation_plan(),
            _semantic_payload(treated="rx", control="control"),
            _semantic_payload(treated="rx", control="control"),
        ]
    )

    with pytest.raises(
        ValueError, match="compiled causal spec semantics remained invalid after retry"
    ):
        cleaning(
            protocol_discussion="Confirmed protocol discussion",
            cleaning_instructions="",
            review_recompile_request=None,
            draft_causal_spec=_draft(),
            data_summary=_build_summary(dataframe),
            to_clean_df=dataframe,
            datasetProfilingTool=DatasetProfilingTool(),
            simpleDataTransformationTool=_FakeSimpleDataTransformationTool(),
            dataManipulationTool=_FakeDataManipulationTool(),
            llm=llm,
        )


def test_cleaning_compiles_without_protocol_discussion() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(
        json_outputs=[
            _empty_simple_plan(),
            _empty_manipulation_plan(),
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion=None,
        cleaning_instructions="Keep protocol columns only.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        simpleDataTransformationTool=_FakeSimpleDataTransformationTool(),
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=llm,
    )

    semantic_call_payload = json.loads(str(llm.generate_json_calls[-1]["user_prompt"]))
    assert isinstance(result, CleaningResult)
    assert "protocol_discussion" not in semantic_call_payload
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
        review_recompile_request=None,
        draft_causal_spec=_draft_with_identifier("patient_id"),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        simpleDataTransformationTool=_FakeSimpleDataTransformationTool(),
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=_FakeLLM(
            json_outputs=[
                _empty_simple_plan(),
                _empty_manipulation_plan(),
                _semantic_payload(),
            ]
        ),
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


@pytest.mark.parametrize(
    "identifier_values",
    [
        [None, "p2"],
        ["p1", "p1"],
    ],
)
def test_cleaning_generates_auto_id_when_explicit_identifier_is_unusable(
    identifier_values: list[str | None],
) -> None:
    dataframe = _build_dataframe()
    dataframe["patient_id"] = identifier_values

    result = cleaning(
        protocol_discussion=None,
        cleaning_instructions="",
        review_recompile_request=None,
        draft_causal_spec=_draft_with_identifier("patient_id"),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        simpleDataTransformationTool=_FakeSimpleDataTransformationTool(),
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=_FakeLLM(
            json_outputs=[
                _empty_simple_plan(),
                _empty_manipulation_plan(),
                _semantic_payload(),
            ]
        ),
    )

    assert list(result.pd_cleaned.columns) == [
        ID_COL_AUTO_FILL,
        "treatment",
        "outcome",
        "age",
        "isex",
    ]
    assert result.pd_cleaned[ID_COL_AUTO_FILL].tolist() == [1, 2]
    assert result.causal.id_col == ID_COL_AUTO_FILL


def test_cleaning_generates_auto_id_when_explicit_identifier_is_missing() -> None:
    dataframe = _build_dataframe().drop(columns=["patient_id"])

    result = cleaning(
        protocol_discussion=None,
        cleaning_instructions="",
        review_recompile_request=None,
        draft_causal_spec=_draft_with_identifier("patient_id"),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        simpleDataTransformationTool=_FakeSimpleDataTransformationTool(),
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=_FakeLLM(
            json_outputs=[
                _empty_simple_plan(),
                _empty_manipulation_plan(),
                _semantic_payload(),
            ]
        ),
    )

    assert list(result.pd_cleaned.columns) == [
        ID_COL_AUTO_FILL,
        "treatment",
        "outcome",
        "age",
        "isex",
    ]
    assert result.causal.id_col == ID_COL_AUTO_FILL


def test_cleaning_records_sql_missingness_resolution_without_missingness_plan() -> None:
    dataframe = _build_dataframe()
    dataframe.loc[0, "age"] = None
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            dataframe.assign(age=[53.0, 61.0], **{ID_COL_AUTO_FILL: [1, 2]}),
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _empty_simple_plan(),
            {
                "instructions": (
                    "Review-time recompilation request: Handle the baseline age gap "
                    "before review. Resolve all remaining protocol-scope missingness "
                    "in SQL. - Column 'age' (covariate): 1 missing value(s) remain."
                )
            },
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Keep the treatment and outcome grounded.",
        cleaning_instructions="Resolve missingness before compilation.",
        review_recompile_request="Handle the baseline age gap before review.",
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        simpleDataTransformationTool=_FakeSimpleDataTransformationTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert len(llm.generate_json_calls) == 3
    simple_transform_payload = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))
    assert "missingness_plan" not in simple_transform_payload
    manipulation_payload = json.loads(str(llm.generate_json_calls[1]["user_prompt"]))
    assert manipulation_payload["required_column_missing_counts"]["age"] == 1
    assert data_manipulation_tool.calls
    assert "Review-time recompilation request:" in (data_manipulation_tool.calls[0]["instructions"])
    assert "Resolve all remaining protocol-scope missingness in SQL" in (
        data_manipulation_tool.calls[0]["instructions"]
    )
    age_decision = next(
        decision for decision in result.missingness_decisions.decisions if decision.column == "age"
    )
    assert age_decision.resolution == "impute"
    assert age_decision.missing_count_before == 1
    assert age_decision.missing_count_after == 0


def test_cleaning_leaves_missingness_row_drops_for_sql() -> None:
    dataframe = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "extra": "drop",
                        "patient_id": "p0",
                        "isex": 1,
                        "outcome": "event",
                        "treatment": None,
                        "age": 40,
                    }
                ]
            ),
            _build_dataframe(),
        ],
        ignore_index=True,
    )
    simple_transform_tool = _FakeSimpleDataTransformationTool()
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            _build_dataframe()
            .assign(**{ID_COL_AUTO_FILL: [2, 3]})
            .loc[:, [ID_COL_AUTO_FILL, "treatment", "outcome", "age", "isex"]]
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _empty_simple_plan(),
            {
                "instructions": (
                    "Drop rows with missing treatment. - Column 'treatment' "
                    "(treatment): 1 missing value(s) remain."
                )
            },
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Treatment must be observed.",
        cleaning_instructions="Drop rows with missing treatment.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        simpleDataTransformationTool=simple_transform_tool,
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert simple_transform_tool.calls == []
    assert data_manipulation_tool.calls
    assert "Drop rows with missing treatment." in (data_manipulation_tool.calls[0]["instructions"])
    assert "- Column 'treatment' (treatment): 1 missing value(s) remain." in (
        data_manipulation_tool.calls[0]["instructions"]
    )
    treatment_decision = next(
        decision
        for decision in result.missingness_decisions.decisions
        if decision.column == "treatment"
    )
    assert treatment_decision.resolution == "drop_rows"
    assert treatment_decision.missing_count_after == 0


def test_cleaning_fails_when_protocol_scope_missingness_remains_after_cleaning() -> None:
    dataframe = _build_dataframe()
    dataframe.loc[0, "age"] = None
    llm = _FakeLLM(
        json_outputs=[
            _empty_simple_plan(),
            _empty_manipulation_plan(),
        ]
    )

    with pytest.raises(
        ValueError,
        match="cleaned dataframe still contains protocol-scope missing values: age=1",
    ):
        cleaning(
            protocol_discussion="Confirmed protocol discussion",
            cleaning_instructions="Resolve missingness before compilation.",
            review_recompile_request=None,
            draft_causal_spec=_draft(),
            data_summary=_build_summary(dataframe),
            to_clean_df=dataframe,
            datasetProfilingTool=DatasetProfilingTool(),
            simpleDataTransformationTool=_FakeSimpleDataTransformationTool(),
            dataManipulationTool=_FakeDataManipulationTool(),
            llm=llm,
        )
