from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from python.domain.repo.analytics_repo import AnalyticsSQLResult
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
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


def _build_threshold_missingness_dataframe() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index in range(100):
        rows.append(
            {
                "extra": f"drop-{index}",
                "patient_id": f"p{index + 1}",
                "isex": None if index < 50 else 1 + (index % 2),
                "outcome": "event" if index % 2 == 0 else "non_event",
                "treatment": "drug" if index % 2 == 0 else "control",
                "age": None if index < 50 else 40 + index,
            }
        )
    return pd.DataFrame(rows)


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


def _draft_with_negative_control() -> CausalSpecDraft:
    return CausalSpecDraft.model_validate(
        {
            "treatment_column": "treatment",
            "outcome_column": "outcome",
            "negative_control_outcome": "negative_control",
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
    negative_control_outcome: dict[str, Any] | None = None,
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
        "negative_control_outcome": negative_control_outcome,
        "experiment_type": experiment_type,
    }


def _done_instruction(reason: str = "No manipulation needed.") -> dict[str, Any]:
    return {
        "action": "done",
        "instruction": None,
        "reason": reason,
    }


def _run_instruction(instruction: str, reason: str = "Apply grounded cleaning.") -> dict[str, Any]:
    return {
        "action": "run_instruction",
        "instruction": instruction,
        "reason": reason,
    }


def _column_names_from_compact_summary(summary: dict[str, Any]) -> list[str]:
    return [str(column["column"]) for column in summary["columns"]]


@dataclass
class _FakeLLM:
    json_outputs: list[Any] = field(default_factory=list)
    text_outputs: list[str | Exception] = field(default_factory=list)
    generate_json_calls: list[dict[str, Any]] = field(default_factory=list)
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
        if self.text_outputs:
            next_output = self.text_outputs.pop(0)
            if isinstance(next_output, Exception):
                raise next_output
            return LLMResponse(content=next_output)
        return LLMResponse(content="Generated comprehensive cleaning system prompt.")

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
        if "action" in getattr(schema, "model_fields", {}):
            return self._next_instruction_step(schema)

        if not self.json_outputs:
            raise AssertionError("unexpected generate_json call")
        next_output = self.json_outputs.pop(0)
        if isinstance(next_output, Exception):
            raise next_output
        if isinstance(next_output, dict):
            return schema.model_validate(next_output)
        return next_output

    def _next_instruction_step(self, schema: type[Any]) -> Any:
        if not self.json_outputs:
            return schema.model_validate(_done_instruction())
        next_output = self.json_outputs[0]
        if isinstance(next_output, Exception):
            raise self.json_outputs.pop(0)
        if not isinstance(next_output, dict):
            return self.json_outputs.pop(0)
        if "action" in next_output:
            return schema.model_validate(self.json_outputs.pop(0))
        if "treatment" in next_output and "outcome" in next_output:
            return schema.model_validate(_done_instruction())
        return schema.model_validate(self.json_outputs.pop(0))


@dataclass
class _FakeAnalyticsRepo:
    responses: list[pd.DataFrame | Exception] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def execute_sql(
        self,
        *,
        dataframe: pd.DataFrame,
        request: object,
    ) -> AnalyticsSQLResult:
        self.calls.append(
            {
                "dataframe": dataframe.copy(),
                "dataframe_columns": list(dataframe.columns),
                "request": request,
                "table_name": request.table_name,
                "statements": tuple(request.statements),
            }
        )
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            output_df = response.copy()
        else:
            output_df = dataframe.copy()
        return AnalyticsSQLResult(
            table_name=request.table_name,
            executed_statements=tuple(request.statements),
            columns=tuple(str(column) for column in output_df.columns),
            row_count=int(len(output_df)),
            has_result_set=True,
            elapsed_ms=1.0,
            dataframe=output_df,
        )


@dataclass
class _FakeDataManipulationTool:
    responses: list[pd.DataFrame | Exception] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    analytics_repo: _FakeAnalyticsRepo | None = None

    def __post_init__(self) -> None:
        if self.analytics_repo is None:
            self.analytics_repo = _FakeAnalyticsRepo(responses=self.responses)

    def manipulate(
        self,
        *,
        dataframe: pd.DataFrame,
        table_name: str,
        data_summary: str,
        instructions: str,
        retry_attempts: int = 3,
    ) -> pd.DataFrame:
        statements = tuple(
            statement.strip() for statement in instructions.splitlines() if statement.strip()
        ) or (instructions,)
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
        if self.analytics_repo is not None:
            self.analytics_repo.calls.append(
                {
                    "dataframe": dataframe.copy(),
                    "dataframe_columns": list(dataframe.columns),
                    "request": None,
                    "table_name": table_name,
                    "statements": statements,
                }
            )
        if self.responses:
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response.copy()
        return dataframe.copy()


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


def test_cleaning_preserves_negative_control_outcome_and_compiles_final_spec() -> None:
    dataframe = _build_dataframe()
    dataframe["negative_control"] = ["neg_event", "neg_non_event"]

    result = cleaning(
        protocol_discussion=(
            "Confirmed protocol discussion. Negative-control outcome is negative_control."
        ),
        cleaning_instructions="Keep protocol columns only.",
        review_recompile_request=None,
        draft_causal_spec=_draft_with_negative_control(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=_FakeLLM(
            json_outputs=[
                _semantic_payload(
                    negative_control_outcome={
                        "kind": "binary",
                        "event": "neg_event",
                        "non_event": "neg_non_event",
                    }
                ),
            ]
        ),
    )

    assert list(result.pd_cleaned.columns) == [
        ID_COL_AUTO_FILL,
        "treatment",
        "outcome",
        "negative_control",
        "age",
        "isex",
    ]
    assert result.causal.negative_control_outcome is not None
    assert result.causal.negative_control_outcome.column == "negative_control"


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
            dataManipulationTool=_FakeDataManipulationTool(),
            llm=_FakeLLM(json_outputs=[_semantic_payload()]),
        )


def test_cleaning_runs_manipulation_when_effective_instructions_are_present() -> None:
    dataframe = _build_dataframe()
    data_manipulation_tool = _FakeDataManipulationTool()
    llm = _FakeLLM(
        json_outputs=[
            _run_instruction(
                "Normalize only grounded values while preserving the full working dataset.",
                reason="Normalize only grounded values.",
            ),
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
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert isinstance(result, CleaningResult)
    assert data_manipulation_tool.analytics_repo is not None
    assert len(data_manipulation_tool.analytics_repo.calls) == 1
    assert data_manipulation_tool.analytics_repo.calls[0]["dataframe_columns"] == [
        "extra",
        "patient_id",
        "isex",
        "outcome",
        "treatment",
        "age",
        ID_COL_AUTO_FILL,
    ]
    assert data_manipulation_tool.analytics_repo.calls[0]["statements"] == (
        "Normalize only grounded values while preserving the full working dataset.",
    )


def test_cleaning_runs_transformation_instruction_before_cleanup_and_drops_extra_columns() -> None:
    dataframe = _build_dataframe()
    dataframe["age"] = ["45", "61"]
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
            _run_instruction(
                "Cast age to integer values while keeping every current row and column.",
                reason="Age is a numeric covariate but is stored as text.",
            ),
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
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert data_manipulation_tool.analytics_repo is not None
    assert len(data_manipulation_tool.analytics_repo.calls) == 1
    transform_payload = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))
    assert _column_names_from_compact_summary(
        transform_payload["compact_current_dataset_summary"]
    ) == [ID_COL_AUTO_FILL, "treatment", "outcome", "age", "isex"]
    transform_age_profile = next(
        column
        for column in transform_payload["compact_current_dataset_summary"]["columns"]
        if column["column"] == "age"
    )
    assert transform_age_profile["dtype"] == "object"
    assert "age" in data_manipulation_tool.calls[0]["instructions"]
    assert result.pd_cleaned["age"].tolist() == [45, 61]
    assert data_manipulation_tool.analytics_repo.calls[0]["table_name"] == (
        "protocol_scope_df"
    )
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
    llm = _FakeLLM(json_outputs=[_semantic_payload()])

    result = cleaning(
        protocol_discussion=None,
        cleaning_instructions="   ",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert isinstance(result, CleaningResult)
    assert len(llm.generate_json_calls) == 4
    assert data_manipulation_tool.analytics_repo is not None
    assert data_manipulation_tool.analytics_repo.calls == []
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
        responses=[
            dataframe.loc[:, ["treatment", "outcome", "isex"]],
            dataframe.loc[:, ["treatment", "outcome", "isex"]],
            dataframe.loc[:, ["treatment", "outcome", "isex"]],
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "adaptive data manipulation cleaning failed: cleanup_2 data manipulation "
            "output dataframe is missing required column\\(s\\): "
            f"{ID_COL_AUTO_FILL}, age"
        ),
    ):
        cleaning(
            protocol_discussion="Confirmed protocol discussion",
            cleaning_instructions="Apply the protocol cleaning.",
            review_recompile_request=None,
            draft_causal_spec=_draft(),
            data_summary=_build_summary(dataframe),
            to_clean_df=dataframe,
            datasetProfilingTool=DatasetProfilingTool(),
            dataManipulationTool=data_manipulation_tool,
            llm=_FakeLLM(
                json_outputs=[
                    _run_instruction("Return a dataframe missing required draft columns."),
                    _run_instruction("Retry but still return a dataframe missing required draft columns."),
                    _run_instruction("Retry one more time with the same invalid projection."),
                ]
            ),
        )


def test_cleaning_allows_sql_helper_columns_and_drops_them_after_projection() -> None:
    dataframe = _build_dataframe()
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            dataframe.assign(
                **{
                    ID_COL_AUTO_FILL: [1, 2],
                    "mutation_count_missing": [False, False],
                }
            )
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _run_instruction(
                "Create a helper missingness flag if needed, then return the full working dataset.",
                reason="Create helper missingness flags before final projection.",
            ),
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Confirmed protocol discussion",
        cleaning_instructions="Use helper flags if needed.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert "mutation_count_missing" not in result.pd_cleaned.columns
    assert data_manipulation_tool.analytics_repo is not None
    assert data_manipulation_tool.analytics_repo.calls[0]["statements"] == (
        "Create a helper missingness flag if needed, then return the full working dataset.",
    )


def test_cleaning_passes_one_instruction_to_data_manipulation() -> None:
    dataframe = _build_dataframe()
    first_output = dataframe.assign(**{ID_COL_AUTO_FILL: [1, 2], "age_missing": [False, False]})
    second_output = first_output.assign(age=[50, 70])
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            second_output,
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _run_instruction(
                "Create helper flags and use them to impute age while preserving required columns.",
                reason="Create flags before imputation.",
            ),
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Confirmed protocol discussion",
        cleaning_instructions="Create flags before imputation.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert data_manipulation_tool.analytics_repo is not None
    assert len(data_manipulation_tool.analytics_repo.calls) == 1
    assert "age_missing" not in data_manipulation_tool.analytics_repo.calls[0]["dataframe"].columns
    assert data_manipulation_tool.analytics_repo.calls[0]["statements"] == (
        "Create helper flags and use them to impute age while preserving required columns.",
    )
    assert result.pd_cleaned["age"].tolist() == [50, 70]


def test_cleaning_adds_missingness_indicators_for_imputed_feature_columns() -> None:
    dataframe = _build_threshold_missingness_dataframe()
    imputed = dataframe.assign(
        age=[50 + index for index in range(len(dataframe))],
        isex=[1 + (index % 2) for index in range(len(dataframe))],
        **{ID_COL_AUTO_FILL: list(range(1, len(dataframe) + 1))},
    )
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            imputed,
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _done_instruction(),
            _run_instruction(
                "Impute retained covariate and effect modifier missingness.",
                reason="Resolve retained feature missingness with imputation.",
            ),
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Age is a covariate and isex is an effect modifier.",
        cleaning_instructions="Impute retained feature missingness.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert result.pd_cleaned["age__missing"].tolist() == [1] * 50 + [0] * 50
    assert result.pd_cleaned["isex__missing"].tolist() == [1] * 50 + [0] * 50
    assert result.causal.covariates == ["age", "age__missing"]
    assert result.causal.effect_modifiers == ["isex", "isex__missing"]


def test_cleaning_does_not_add_feature_indicator_below_missingness_threshold() -> None:
    dataframe = _build_dataframe()
    dataframe.loc[0, "age"] = None
    imputed = dataframe.assign(
        age=[50, 61],
        **{ID_COL_AUTO_FILL: [1, 2]},
    )
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            imputed,
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _done_instruction(),
            _run_instruction(
                "Impute retained age missingness.",
                reason="Resolve retained feature missingness with imputation.",
            ),
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Age is a covariate.",
        cleaning_instructions="Impute retained feature missingness.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert "age__missing" not in result.pd_cleaned.columns
    assert result.causal.covariates == ["age"]


def test_cleaning_does_not_add_feature_indicator_for_protocol_filtered_rows() -> None:
    dataframe = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "extra": "drop",
                        "patient_id": "p0",
                        "isex": 1,
                        "outcome": "event",
                        "treatment": "drug",
                        "age": None,
                    }
                ]
            ),
            _build_dataframe(),
        ],
        ignore_index=True,
    )
    filtered = _build_dataframe().assign(**{ID_COL_AUTO_FILL: [2, 3]})
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            filtered,
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _run_instruction(
                "Apply the protocol population filter before missingness handling.",
                reason="Protocol excludes the row before missingness handling.",
            ),
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Exclude the row outside the protocol population.",
        cleaning_instructions="Filter protocol-ineligible rows before handling missingness.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert "age__missing" not in result.pd_cleaned.columns
    assert result.causal.covariates == ["age"]


def test_cleaning_does_not_add_feature_indicator_when_missing_feature_rows_are_dropped() -> None:
    dataframe = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "extra": "drop",
                        "patient_id": "p0",
                        "isex": 1,
                        "outcome": "event",
                        "treatment": "drug",
                        "age": None,
                    }
                ]
            ),
            _build_dataframe(),
        ],
        ignore_index=True,
    )
    retained = _build_dataframe().assign(**{ID_COL_AUTO_FILL: [2, 3]})
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            retained,
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _done_instruction(),
            _run_instruction(
                "Drop retained rows with missing age.",
                reason="Age missingness is resolved by row filtering.",
            ),
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Drop rows with unresolved feature missingness.",
        cleaning_instructions="Drop rows with missing age.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert "age__missing" not in result.pd_cleaned.columns
    assert result.causal.covariates == ["age"]
    age_decision = next(
        decision for decision in result.missingness_decisions.decisions if decision.column == "age"
    )
    assert age_decision.resolution == "drop_rows"


def test_cleaning_does_not_add_missingness_indicators_for_treatment_or_outcome() -> None:
    dataframe = _build_dataframe()
    dataframe.loc[0, "treatment"] = None
    dataframe.loc[1, "outcome"] = None
    imputed = dataframe.assign(
        treatment=["drug", "control"],
        outcome=["event", "non_event"],
        **{ID_COL_AUTO_FILL: [1, 2]},
    )
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            imputed,
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _done_instruction(),
            _run_instruction(
                "Resolve required treatment and outcome missingness.",
                reason="Treatment and outcome must be observed for compilation.",
            ),
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Treatment and outcome are required protocol columns.",
        cleaning_instructions="Resolve treatment and outcome missingness.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert "treatment__missing" not in result.pd_cleaned.columns
    assert "outcome__missing" not in result.pd_cleaned.columns
    assert result.causal.covariates == ["age"]
    assert result.causal.effect_modifiers == ["isex"]


def test_cleaning_feeds_validation_feedback_to_next_instruction_planner() -> None:
    dataframe = _build_dataframe()
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            RuntimeError("Catalog Error: Table with name missing_flags does not exist"),
            dataframe.assign(**{ID_COL_AUTO_FILL: [1, 2]}),
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _done_instruction(),
            _run_instruction(
                "Use missing_flags to return the cleaned working dataset.",
                reason="Invalid first attempt.",
            ),
            _run_instruction(
                "Return the full current working dataset from protocol_scope_df.",
                reason="Retry with valid source table.",
            ),
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Confirmed protocol discussion",
        cleaning_instructions="Retry invalid SQL plans.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    retry_payload = json.loads(str(llm.generate_json_calls[2]["user_prompt"]))
    assert "validation_feedback" in retry_payload
    assert "missing_flags" in retry_payload["validation_feedback"]["error"]
    assert retry_payload["validation_feedback"]["stage"] == "cleanup_1"
    assert result.pd_cleaned[ID_COL_AUTO_FILL].tolist() == [1, 2]


def test_cleaning_fails_when_manipulation_corrupts_identifier_after_retries() -> None:
    dataframe = _build_dataframe()
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            dataframe.assign(**{ID_COL_AUTO_FILL: [1, 1]}),
            dataframe.assign(**{ID_COL_AUTO_FILL: [1, 1]}),
            dataframe.assign(**{ID_COL_AUTO_FILL: [1, 1]}),
        ]
    )

    with pytest.raises(
        ValueError,
        match="adaptive data manipulation cleaning failed:.*duplicate effective identifier values",
    ):
        cleaning(
            protocol_discussion="Confirmed protocol discussion",
            cleaning_instructions="Apply SQL cleaning.",
            review_recompile_request=None,
            draft_causal_spec=_draft(),
            data_summary=_build_summary(dataframe),
            to_clean_df=dataframe,
            datasetProfilingTool=DatasetProfilingTool(),
            dataManipulationTool=data_manipulation_tool,
            llm=_FakeLLM(
                json_outputs=[
                    _run_instruction("Return cleaned data but accidentally duplicate IDs."),
                    _run_instruction("Retry while still duplicating IDs."),
                    _run_instruction("Final retry still duplicates IDs."),
                ]
            ),
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
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=llm,
    )

    assert isinstance(result, CleaningResult)
    assert len(llm.generate_json_calls) == 5
    fourth_call_payload = json.loads(str(llm.generate_json_calls[4]["user_prompt"]))
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
            dataManipulationTool=_FakeDataManipulationTool(),
            llm=llm,
        )


def test_cleaning_compiles_without_protocol_discussion() -> None:
    dataframe = _build_dataframe()
    llm = _FakeLLM(json_outputs=[_semantic_payload()])

    result = cleaning(
        protocol_discussion=None,
        cleaning_instructions="Keep protocol columns only.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
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
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=_FakeLLM(json_outputs=[_semantic_payload()]),
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
        dataManipulationTool=_FakeDataManipulationTool(),
        llm=_FakeLLM(json_outputs=[_semantic_payload()]),
    )

    assert list(result.pd_cleaned.columns) == [
        ID_COL_AUTO_FILL,
        "treatment",
        "outcome",
        "age",
        "isex",
    ]
    assert result.causal.id_col == ID_COL_AUTO_FILL


def test_cleaning_records_data_manipulation_missingness_resolution() -> None:
    dataframe = _build_dataframe()
    dataframe.loc[0, "age"] = None
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            dataframe.assign(age=[53.0, 61.0], **{ID_COL_AUTO_FILL: [1, 2]}),
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _done_instruction(),
            _run_instruction(
                (
                    "Handle the baseline age gap before review by resolving age "
                    "missingness while preserving protocol-scope columns."
                ),
                reason="Review-time recompilation request prioritizes age missingness.",
            ),
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
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert len(llm.generate_json_calls) == 5
    transform_payload = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))
    assert "missingness_plan" not in transform_payload
    assert transform_payload["high_priority_review_recompile_request"] == (
        "Handle the baseline age gap before review."
    )
    manipulation_payload = json.loads(str(llm.generate_json_calls[1]["user_prompt"]))
    assert manipulation_payload["high_priority_review_recompile_request"] == (
        "Handle the baseline age gap before review."
    )
    assert manipulation_payload["required_column_missing_counts"]["age"] == 1
    assert "compact_current_dataset_summary" in manipulation_payload
    assert "executed_cleaning_instructions" in manipulation_payload
    assert data_manipulation_tool.analytics_repo is not None
    assert data_manipulation_tool.analytics_repo.calls
    assert data_manipulation_tool.analytics_repo.calls[0]["statements"] == (
        "Handle the baseline age gap before review by resolving age missingness while preserving protocol-scope columns.",
    )
    age_decision = next(
        decision for decision in result.missingness_decisions.decisions if decision.column == "age"
    )
    assert age_decision.resolution == "impute"
    assert age_decision.missing_count_before == 1
    assert age_decision.missing_count_after == 0


def test_cleaning_resolves_missingness_with_row_drop_instruction() -> None:
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
    data_manipulation_tool = _FakeDataManipulationTool(
        responses=[
            _build_dataframe()
            .assign(**{ID_COL_AUTO_FILL: [2, 3]})
            .loc[:, [ID_COL_AUTO_FILL, "treatment", "outcome", "age", "isex"]]
        ]
    )
    llm = _FakeLLM(
        json_outputs=[
            _done_instruction(),
            _run_instruction(
                "Drop rows with missing treatment because treatment must be observed.",
                reason="Treatment must be observed.",
            ),
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
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert data_manipulation_tool.analytics_repo is not None
    assert data_manipulation_tool.analytics_repo.calls
    assert data_manipulation_tool.analytics_repo.calls[0]["statements"] == (
        "Drop rows with missing treatment because treatment must be observed.",
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
            _done_instruction(),
            _done_instruction(),
            _done_instruction(),
        ]
    )

    with pytest.raises(
        ValueError,
        match="adaptive data manipulation cleaning failed: missingness step returned done while protocol-scope missingness remains",
    ):
        cleaning(
            protocol_discussion="Confirmed protocol discussion",
            cleaning_instructions="Resolve missingness before compilation.",
            review_recompile_request=None,
            draft_causal_spec=_draft(),
            data_summary=_build_summary(dataframe),
            to_clean_df=dataframe,
            datasetProfilingTool=DatasetProfilingTool(),
            dataManipulationTool=_FakeDataManipulationTool(),
            llm=llm,
        )


def test_cleaning_repairs_binary_outcome_literals_before_backdoor_validation() -> None:
    dataframe = pd.concat(
        [
            _build_dataframe(),
            pd.DataFrame(
                [
                    {
                        "extra": "drop-third",
                        "patient_id": "p3",
                        "isex": 1,
                        "outcome": "unknown",
                        "treatment": "drug",
                        "age": 72,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    repaired_df = dataframe.assign(
        outcome=["event", "non_event", "non_event"],
        **{ID_COL_AUTO_FILL: [1, 2, 3]},
    ).loc[:, [ID_COL_AUTO_FILL, "treatment", "outcome", "age", "isex"]]
    data_manipulation_tool = _FakeDataManipulationTool(responses=[repaired_df])
    llm = _FakeLLM(
        json_outputs=[
            _semantic_payload(),
            _semantic_payload(),
        ]
    )

    result = cleaning(
        protocol_discussion="Outcome must be binary event versus non_event.",
        cleaning_instructions="Map the outcome into the protocol binary values.",
        review_recompile_request=None,
        draft_causal_spec=_draft(),
        data_summary=_build_summary(dataframe),
        to_clean_df=dataframe,
        datasetProfilingTool=DatasetProfilingTool(),
        dataManipulationTool=data_manipulation_tool,
        llm=llm,
    )

    assert result.pd_cleaned["outcome"].tolist() == ["event", "non_event", "non_event"]
    assert data_manipulation_tool.calls
    repair_instruction = data_manipulation_tool.calls[-1]["instructions"]
    assert "High-priority final semantic consistency repair" in repair_instruction
    assert "Binary outcome column contains values outside" in repair_instruction
    assert "unknown" in repair_instruction
