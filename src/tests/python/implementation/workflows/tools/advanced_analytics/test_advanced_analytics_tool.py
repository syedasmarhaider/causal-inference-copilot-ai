from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.implementation.workflows.tools.advanced_analytics.advanced_analytics_models import (
    AnalyticsPlanModel,
)
from python.implementation.workflows.tools.advanced_analytics.advanced_analytics_tool import (
    AdvancedAnalyticsTool,
)
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel


def _numeric_profile(
    name: str, *, n_rows: int, distinct_count: int | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "float64",
        "n_rows": n_rows,
        "n_missing": 0,
        "missing_rate": 0.0,
        "distinct_count": distinct_count if distinct_count is not None else n_rows,
        "inferred_kind": "NUMERIC",
        "summary": {"min": 1.0, "max": 2.0, "mean": 1.5, "std": 0.5, "quantiles": None},
    }


def _categorical_profile(name: str, *, n_rows: int, distinct_count: int) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "object",
        "n_rows": n_rows,
        "n_missing": 0,
        "missing_rate": 0.0,
        "distinct_count": distinct_count,
        "inferred_kind": "CATEGORICAL",
        "summary": {
            "top_categories": [{"value": "a", "count": 1}, {"value": "b", "count": 1}],
            "other_count": 0,
        },
    }


def _summary_model(*profiles: dict[str, Any], n_rows: int) -> DatasetSummaryModel:
    return DatasetSummaryModel.model_validate({"n_rows": n_rows, "profiles": list(profiles)})


@dataclass
class _FakeLLMService:
    plans: list[object]
    calls: list[dict[str, object]] = field(default_factory=list)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
    ) -> LLMResponse:
        del system_prompt, user_prompt, config, history
        raise NotImplementedError

    def generate_json(
        self,
        *,
        schema: type[AnalyticsPlanModel],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> AnalyticsPlanModel:
        self.calls.append(
            {
                "schema": schema,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "config": config,
                "history": history,
                "max_attempts": max_attempts,
            }
        )

        if not self.plans:
            raise AssertionError("unexpected generate_json call")

        next_plan = self.plans.pop(0)
        payload = next_plan.model_dump() if isinstance(next_plan, BaseModel) else next_plan
        return schema.model_validate(payload)


def test_propensity_score_accepts_string_binary_treatment_and_reports_mapping() -> None:
    dataframe = pd.DataFrame(
        [
            {"istatus": "control", "age": 23, "sex": "F"},
            {"istatus": "treated", "age": 25, "sex": "M"},
            {"istatus": "control", "age": 28, "sex": "F"},
            {"istatus": "treated", "age": 31, "sex": "M"},
            {"istatus": "control", "age": 35, "sex": "F"},
            {"istatus": "treated", "age": 37, "sex": "M"},
            {"istatus": "control", "age": 41, "sex": "M"},
            {"istatus": "treated", "age": 44, "sex": "F"},
        ]
    )
    summary = _summary_model(
        _categorical_profile("istatus", n_rows=len(dataframe), distinct_count=2),
        _numeric_profile("age", n_rows=len(dataframe)),
        _categorical_profile("sex", n_rows=len(dataframe), distinct_count=2),
        n_rows=len(dataframe),
    )
    llm = _FakeLLMService(
        plans=[
            {
                "analysis_type": "propensity_score",
                "treatment": "istatus",
                "covariates": ["age", "sex"],
            }
        ]
    )

    result = AdvancedAnalyticsTool(llm=llm).analyze(
        dataframe=dataframe,
        data_summary=summary,
        user_request="Estimate propensity scores for istatus using age and sex.",
    )

    assert result.analysis_type == "propensity_score"
    assert result.tables["treatment_levels"] == {"0": "control", "1": "treated"}
    assert result.metrics["treatment_levels"] == {"0": "control", "1": "treated"}
    assert 0.0 <= result.metrics["auc"] <= 1.0
    assert "positive class='treated'" in result.summary


def test_propensity_score_rejects_non_binary_treatment_with_clear_error() -> None:
    dataframe = pd.DataFrame(
        [
            {"istatus": "low", "age": 23, "sex": "F"},
            {"istatus": "medium", "age": 25, "sex": "M"},
            {"istatus": "high", "age": 28, "sex": "F"},
        ]
    )
    summary = _summary_model(
        _categorical_profile("istatus", n_rows=len(dataframe), distinct_count=3),
        _numeric_profile("age", n_rows=len(dataframe)),
        _categorical_profile("sex", n_rows=len(dataframe), distinct_count=2),
        n_rows=len(dataframe),
    )
    llm = _FakeLLMService(
        plans=[
            {
                "analysis_type": "propensity_score",
                "treatment": "istatus",
                "covariates": ["age", "sex"],
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"Treatment column 'istatus' must be binary after dropping missing values",
    ):
        _ = AdvancedAnalyticsTool(llm=llm).analyze(
            dataframe=dataframe,
            data_summary=summary,
            user_request="Estimate propensity scores for istatus using age and sex.",
        )


def test_ttest_requires_at_least_two_observations_per_group() -> None:
    dataframe = pd.DataFrame(
        [
            {"score": 10.0, "group": "A"},
            {"score": 20.0, "group": "B"},
        ]
    )
    summary = _summary_model(
        _numeric_profile("score", n_rows=len(dataframe), distinct_count=2),
        _categorical_profile("group", n_rows=len(dataframe), distinct_count=2),
        n_rows=len(dataframe),
    )
    llm = _FakeLLMService(
        plans=[
            {
                "analysis_type": "ttest",
                "numeric_column": "score",
                "group_column": "group",
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"t-test requires at least 2 observations in each group",
    ):
        _ = AdvancedAnalyticsTool(llm=llm).analyze(
            dataframe=dataframe,
            data_summary=summary,
            user_request="Run a t-test on score by group.",
        )
