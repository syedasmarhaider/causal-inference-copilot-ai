from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.implementation.workflows.nodes.node_service.plot_specs_service.plot_specs_service import (
    PlotSpecsPlan,
    PlotSpecsService,
)


@dataclass
class _FakeLLMService:
    plan: PlotSpecsPlan
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
        schema: type[PlotSpecsPlan],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> PlotSpecsPlan:
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
        return self.plan


def test_generate_specs_injects_values_from_dataframe() -> None:
    llm = _FakeLLMService(
        plan=PlotSpecsPlan.model_validate(
            {
                "charts": [
                    {
                        "title": "Age vs Outcome",
                        "rationale": "Relationship chart",
                        "spec": {
                            "mark": "line",
                            "encoding": {
                                "x": {"field": "age", "type": "quantitative"},
                                "y": {"field": "outcome", "type": "quantitative"},
                            },
                        },
                    }
                ]
            }
        )
    )
    service = PlotSpecsService(llm=llm)

    df = pd.DataFrame([{"age": 40, "outcome": 1.0}, {"age": 41, "outcome": 1.2}])
    specs = service.generate_specs(
        dataframe=df,
        data_summary='{"n_rows": 2, "profiles": [{"name": "age"}, {"name": "outcome"}]}',
        user_intent="show trend between age and outcome",
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec["$schema"] == "https://vega.github.io/schema/vega-lite/v5.json"
    assert spec["data"]["values"] == [
        {"age": 40, "outcome": 1.0},
        {"age": 41, "outcome": 1.2},
    ]
    assert len(llm.calls) == 1
    assert "show trend between age and outcome" in str(llm.calls[0]["user_prompt"])


def test_generate_specs_rejects_llm_data_values_in_template() -> None:
    llm = _FakeLLMService(
        plan=PlotSpecsPlan.model_validate(
            {
                "charts": [
                    {
                        "spec": {
                            "mark": "bar",
                            "data": {"values": [{"x": 1}]},
                            "encoding": {"x": {"field": "x", "type": "quantitative"}},
                        }
                    }
                ]
            }
        )
    )
    service = PlotSpecsService(llm=llm)

    with pytest.raises(ValueError, match=r"must not contain data.values"):
        _ = service.generate_specs(
            dataframe=pd.DataFrame([{"x": 1}]),
            data_summary='{"n_rows": 1, "profiles": [{"name": "x"}]}',
            user_intent="bar chart",
        )


def test_generate_specs_rejects_unknown_fields_from_template() -> None:
    llm = _FakeLLMService(
        plan=PlotSpecsPlan.model_validate(
            {
                "charts": [
                    {
                        "spec": {
                            "mark": "point",
                            "encoding": {"x": {"field": "missing_col", "type": "quantitative"}},
                        }
                    }
                ]
            }
        )
    )
    service = PlotSpecsService(llm=llm)

    with pytest.raises(ValueError, match=r"unknown dataframe fields"):
        _ = service.generate_specs(
            dataframe=pd.DataFrame([{"x": 1}]),
            data_summary='{"n_rows": 1, "profiles": [{"name": "x"}]}',
            user_intent="scatter",
        )


def test_generate_specs_converts_datetime_to_iso_strings() -> None:
    llm = _FakeLLMService(
        plan=PlotSpecsPlan.model_validate(
            {
                "charts": [
                    {
                        "spec": {
                            "mark": "line",
                            "encoding": {
                                "x": {"field": "ts", "type": "temporal"},
                                "y": {"field": "value", "type": "quantitative"},
                            },
                        }
                    }
                ]
            }
        )
    )
    service = PlotSpecsService(llm=llm)

    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01T00:00:00", "2026-01-02T00:00:00"]),
            "value": [10, 12],
        }
    )
    specs = service.generate_specs(
        dataframe=df,
        data_summary='{"n_rows": 2, "profiles": [{"name": "ts"}, {"name": "value"}]}',
        user_intent="line plot over time",
    )

    assert specs[0]["data"]["values"][0]["ts"] == "2026-01-01T00:00:00"
    assert specs[0]["data"]["values"][1]["ts"] == "2026-01-02T00:00:00"


@pytest.mark.parametrize(
    ("data_summary", "user_intent", "error_pattern"),
    [
        ("", "plot", r"data_summary must be non-empty"),
        ('{"n_rows": 1}', "", r"user_intent must be non-empty"),
    ],
)
def test_generate_specs_validates_required_inputs(
    data_summary: str,
    user_intent: str,
    error_pattern: str,
) -> None:
    llm = _FakeLLMService(
        plan=PlotSpecsPlan.model_validate(
            {
                "charts": [
                    {
                        "spec": {
                            "mark": "bar",
                            "encoding": {"x": {"field": "x", "type": "quantitative"}},
                        }
                    }
                ]
            }
        )
    )
    service = PlotSpecsService(llm=llm)

    with pytest.raises(ValueError, match=error_pattern):
        _ = service.generate_specs(
            dataframe=pd.DataFrame([{"x": 1}]),
            data_summary=data_summary,
            user_intent=user_intent,
        )

