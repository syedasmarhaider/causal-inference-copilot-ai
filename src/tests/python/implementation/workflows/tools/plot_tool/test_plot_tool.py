from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel
from python.implementation.workflows.tools.plot_tool.plot_tool import PlotSpecsPlan, PlotTool


def _plan_payload(*, charts: list[dict[str, Any]]) -> dict[str, Any]:
    return {"charts": charts}


def _numeric_profile(name: str, *, n_rows: int = 2) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "int64",
        "n_rows": n_rows,
        "n_missing": 0,
        "missing_rate": 0.0,
        "distinct_count": n_rows,
        "inferred_kind": "NUMERIC",
        "summary": {"min": 1.0, "max": 2.0, "mean": 1.5, "std": 0.5, "quantiles": None},
    }


def _datetime_profile(name: str, *, n_rows: int = 2) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "datetime64[ns]",
        "n_rows": n_rows,
        "n_missing": 0,
        "missing_rate": 0.0,
        "distinct_count": n_rows,
        "inferred_kind": "DATETIME",
        "summary": {"min": "2026-01-01T00:00:00", "max": "2026-01-02T00:00:00"},
    }


def _categorical_profile(name: str, *, n_rows: int = 2) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": "object",
        "n_rows": n_rows,
        "n_missing": 0,
        "missing_rate": 0.0,
        "distinct_count": 2,
        "inferred_kind": "CATEGORICAL",
        "summary": {
            "top_categories": [{"value": "A", "count": 1}, {"value": "B", "count": 1}],
            "other_count": 0,
        },
    }


def _summary_json(*profiles: dict[str, Any], n_rows: int = 2) -> str:
    return json.dumps({"n_rows": n_rows, "profiles": list(profiles)})


def _summary_model(*profiles: dict[str, Any], n_rows: int = 2) -> DatasetSummaryModel:
    return DatasetSummaryModel.model_validate_json(_summary_json(*profiles, n_rows=n_rows))


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

        last_validation_error: ValidationError | None = None

        for _ in range(max_attempts):
            if not self.plans:
                raise AssertionError("unexpected generate_json call")

            next_plan = self.plans.pop(0)
            if isinstance(next_plan, Exception):
                raise next_plan

            payload = next_plan.model_dump() if isinstance(next_plan, BaseModel) else next_plan
            try:
                return schema.model_validate(payload)
            except ValidationError as exc:
                last_validation_error = exc

        raise RuntimeError(
            f"Failed JSON schema={schema.__name__} after {max_attempts} attempts. "
            f"Last error: {last_validation_error}"
        )


def test_plot_plan_schema_binds_summary_headers_without_bloating_json_schema() -> None:
    summary = DatasetSummaryModel.model_validate_json(
        _summary_json(_numeric_profile("age"), _numeric_profile("outcome"))
    )
    schema = PlotSpecsPlan.for_summary(summary)

    assert schema is not PlotSpecsPlan
    assert schema.ALLOWED_FIELD_NAMES == ("age", "outcome")
    assert schema.FIELD_KINDS == {"age": "NUMERIC", "outcome": "NUMERIC"}

    schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
    assert "ALLOWED_FIELD_NAMES" not in schema_json
    assert "FIELD_KINDS" not in schema_json
    assert "age" not in schema_json
    assert "outcome" not in schema_json

    plan = schema.model_validate(
        _plan_payload(
            charts=[
                {
                    "title": "Age vs Outcome",
                    "spec": {
                        "mark": "line",
                        "encoding": {
                            "x": {"field": "age", "type": "quantitative"},
                            "y": {"field": "outcome", "type": "quantitative"},
                        },
                    },
                }
            ]
        )
    )
    assert len(plan.charts) == 1


def test_plot_plan_schema_rejects_unknown_summary_headers() -> None:
    summary = DatasetSummaryModel.model_validate_json(_summary_json(_numeric_profile("x")))
    schema = PlotSpecsPlan.for_summary(summary)

    with pytest.raises(ValidationError, match=r"unknown data_summary fields"):
        schema.model_validate(
            _plan_payload(
                charts=[
                    {
                        "spec": {
                            "mark": "point",
                            "encoding": {"x": {"field": "missing_col", "type": "quantitative"}},
                        }
                    }
                ]
            )
        )


def test_plot_plan_schema_rejects_summary_type_mismatch() -> None:
    summary = DatasetSummaryModel.model_validate_json(_summary_json(_categorical_profile("label")))
    schema = PlotSpecsPlan.for_summary(summary)

    with pytest.raises(ValidationError, match=r"declared quantitative but data_summary inferred kind is CATEGORICAL"):
        schema.model_validate(
            _plan_payload(
                charts=[
                    {
                        "spec": {
                            "mark": "bar",
                            "encoding": {"x": {"field": "label", "type": "quantitative"}},
                        }
                    }
                ]
            )
        )


def test_plot_plan_schema_rejects_forbidden_template_values() -> None:
    summary = DatasetSummaryModel.model_validate_json(_summary_json(_numeric_profile("x")))
    schema = PlotSpecsPlan.for_summary(summary)

    with pytest.raises(ValidationError, match=r"must not contain data.values"):
        schema.model_validate(
            _plan_payload(
                charts=[
                    {
                        "spec": {
                            "mark": "bar",
                            "data": {"values": [{"x": 1}]},
                            "encoding": {"x": {"field": "x", "type": "quantitative"}},
                        }
                    }
                ]
            )
        )


def test_plot_plan_schema_rejects_missing_visual_definition() -> None:
    summary = DatasetSummaryModel.model_validate_json(_summary_json(_numeric_profile("x")))
    schema = PlotSpecsPlan.for_summary(summary)

    with pytest.raises(ValidationError, match=r"must define a visual grammar"):
        schema.model_validate(
            _plan_payload(
                charts=[
                    {
                        "spec": {
                            "encoding": {"x": {"field": "x", "type": "quantitative"}},
                        }
                    }
                ]
            )
        )


def test_generate_specs_injects_values_and_title_from_dataframe() -> None:
    llm = _FakeLLMService(
        plans=[
            _plan_payload(
                charts=[
                    {
                        "title": "Age vs Outcome",
                        "spec": {
                            "mark": "line",
                            "encoding": {
                                "x": {"field": "age", "type": "quantitative"},
                                "y": {"field": "outcome", "type": "quantitative"},
                            },
                        },
                    }
                ]
            )
        ]
    )
    tool = PlotTool(llm=llm)

    df = pd.DataFrame([{"age": 40, "outcome": 1.0}, {"age": 41, "outcome": 1.2}])
    specs = tool.generate_specs(
        dataframe=df,
        data_summary=_summary_model(_numeric_profile("age"), _numeric_profile("outcome")),
        user_intent="show trend between age and outcome",
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec["$schema"] == "https://vega.github.io/schema/vega-lite/v5.json"
    assert spec["title"] == "Age vs Outcome"
    assert spec["data"]["values"] == [
        {"age": 40, "outcome": 1.0},
        {"age": 41, "outcome": 1.2},
    ]
    assert len(llm.calls) == 1
    assert "show trend between age and outcome" in str(llm.calls[0]["user_prompt"])
    assert llm.calls[0]["schema"] is not PlotSpecsPlan
    assert llm.calls[0]["schema"].ALLOWED_FIELD_NAMES == ("age", "outcome")


def test_generate_specs_uses_summary_headers_for_internal_retry_not_dataframe_columns() -> None:
    llm = _FakeLLMService(
        plans=[
            _plan_payload(
                charts=[
                    {
                        "spec": {
                            "mark": "point",
                            "encoding": {"x": {"field": "hidden", "type": "quantitative"}},
                        }
                    }
                ]
            ),
            _plan_payload(
                charts=[
                    {
                        "spec": {
                            "mark": "point",
                            "encoding": {"x": {"field": "x", "type": "quantitative"}},
                        }
                    }
                ]
            ),
        ]
    )
    tool = PlotTool(llm=llm)

    specs = tool.generate_specs(
        dataframe=pd.DataFrame([{"x": 1, "hidden": 99}]),
        data_summary=_summary_model(_numeric_profile("x")),
        user_intent="scatter",
        max_attempts=2,
    )

    assert len(llm.calls) == 1
    assert llm.calls[0]["max_attempts"] == 2
    assert specs[0]["data"]["values"] == [{"x": 1}]


def test_generate_specs_uses_llm_internal_retry_for_forbidden_template_values() -> None:
    llm = _FakeLLMService(
        plans=[
            _plan_payload(
                charts=[
                    {
                        "spec": {
                            "mark": "bar",
                            "data": {"values": [{"x": 1}]},
                            "encoding": {"x": {"field": "x", "type": "quantitative"}},
                        }
                    }
                ]
            ),
            _plan_payload(
                charts=[
                    {
                        "spec": {
                            "mark": "bar",
                            "encoding": {"x": {"field": "x", "type": "quantitative"}},
                        }
                    }
                ]
            ),
        ]
    )
    tool = PlotTool(llm=llm)

    specs = tool.generate_specs(
        dataframe=pd.DataFrame([{"x": 1}]),
        data_summary=_summary_model(_numeric_profile("x")),
        user_intent="bar chart",
        max_attempts=2,
    )

    assert len(llm.calls) == 1
    assert llm.calls[0]["max_attempts"] == 2
    assert specs[0]["data"]["values"] == [{"x": 1}]


def test_generate_specs_uses_llm_internal_retry_for_summary_type_mismatch() -> None:
    llm = _FakeLLMService(
        plans=[
            _plan_payload(
                charts=[
                    {
                        "spec": {
                            "mark": "bar",
                            "encoding": {"x": {"field": "label", "type": "quantitative"}},
                        }
                    }
                ]
            ),
            _plan_payload(
                charts=[
                    {
                        "spec": {
                            "mark": "bar",
                            "encoding": {"x": {"field": "label", "type": "nominal"}},
                        }
                    }
                ]
            ),
        ]
    )
    tool = PlotTool(llm=llm)

    specs = tool.generate_specs(
        dataframe=pd.DataFrame([{"label": "A"}]),
        data_summary=_summary_model(_categorical_profile("label"), n_rows=1),
        user_intent="show label distribution",
        max_attempts=2,
    )

    assert len(llm.calls) == 1
    assert llm.calls[0]["max_attempts"] == 2
    assert specs[0]["data"]["values"] == [{"label": "A"}]


def test_generate_specs_raises_runtime_error_after_retry_budget_is_exhausted() -> None:
    llm = _FakeLLMService(
        plans=[
            _plan_payload(
                charts=[
                    {
                        "spec": {
                            "mark": "point",
                            "encoding": {"x": {"field": "missing_col", "type": "quantitative"}},
                        }
                    }
                ]
            )
        ]
    )
    tool = PlotTool(llm=llm)

    with pytest.raises(RuntimeError, match=r"Failed JSON schema=PlotSpecsPlanForFields_1 after 1 attempts"):
        _ = tool.generate_specs(
            dataframe=pd.DataFrame([{"x": 1}]),
            data_summary=_summary_model(_numeric_profile("x")),
            user_intent="scatter",
            max_attempts=1,
        )

def test_generate_specs_rejects_non_model_data_summary() -> None:
    tool = PlotTool(llm=_FakeLLMService(plans=[]))

    with pytest.raises(AttributeError, match=r"profiles"):
        _ = tool.generate_specs(
            dataframe=pd.DataFrame([{"x": 1}]),
            data_summary=_summary_json(_numeric_profile("x")),
            user_intent="plot x",
        )


def test_generate_specs_rejects_summary_header_not_in_dataframe() -> None:
    tool = PlotTool(
        llm=_FakeLLMService(
            plans=[
                _plan_payload(
                    charts=[
                        {
                            "spec": {
                                "mark": "bar",
                                "encoding": {"x": {"field": "x", "type": "quantitative"}},
                            }
                        }
                    ]
                )
            ]
        )
    )

    with pytest.raises(ValueError, match=r"data_summary references unknown dataframe headers: \['y'\]"):
        _ = tool.generate_specs(
            dataframe=pd.DataFrame([{"x": 1}]),
            data_summary=_summary_model(_numeric_profile("x"), _numeric_profile("y")),
            user_intent="plot x",
        )


def test_generate_specs_converts_datetime_to_iso_strings() -> None:
    llm = _FakeLLMService(
        plans=[
            _plan_payload(
                charts=[
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
            )
        ]
    )
    tool = PlotTool(llm=llm)

    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2026-01-01T00:00:00", "2026-01-02T00:00:00"]),
            "value": [10, 12],
        }
    )
    specs = tool.generate_specs(
        dataframe=df,
        data_summary=_summary_model(_datetime_profile("ts"), _numeric_profile("value")),
        user_intent="line plot over time",
    )

    assert specs[0]["data"]["values"][0]["ts"] == "2026-01-01T00:00:00"
    assert specs[0]["data"]["values"][1]["ts"] == "2026-01-02T00:00:00"


def test_generate_specs_warns_threshold_but_injects_all_values() -> None:
    llm = _FakeLLMService(
        plans=[
            _plan_payload(
                charts=[
                    {
                        "spec": {
                            "mark": "bar",
                            "encoding": {"x": {"field": "x", "type": "quantitative"}},
                        }
                    }
                ]
            )
        ]
    )
    tool = PlotTool(llm=llm, warn_max_rows_for_values=1)

    specs = tool.generate_specs(
        dataframe=pd.DataFrame([{"x": 1}, {"x": 2}]),
        data_summary=_summary_model(_numeric_profile("x")),
        user_intent="show x",
    )

    assert specs[0]["data"]["values"] == [{"x": 1}, {"x": 2}]


def test_generate_specs_rejects_quantitative_encoding_when_values_are_not_numeric() -> None:
    llm = _FakeLLMService(
        plans=[
            _plan_payload(
                charts=[
                    {
                        "spec": {
                            "mark": "bar",
                            "encoding": {"x": {"field": "label", "type": "quantitative"}},
                        }
                    }
                ]
            )
        ]
    )
    tool = PlotTool(llm=llm)

    with pytest.raises(ValueError, match=r"declared quantitative"):
        _ = tool.generate_specs(
            dataframe=pd.DataFrame([{"label": "not-a-number"}]),
            data_summary=_summary_model(_numeric_profile("label"), n_rows=1),
            user_intent="plot label",
        )


def test_generate_specs_preserves_existing_spec_title_over_plan_title() -> None:
    llm = _FakeLLMService(
        plans=[
            _plan_payload(
                charts=[
                    {
                        "title": "Plan Title",
                        "spec": {
                            "title": "Spec Title",
                            "mark": "bar",
                            "encoding": {"x": {"field": "x", "type": "quantitative"}},
                        },
                    }
                ]
            )
        ]
    )
    tool = PlotTool(llm=llm)

    specs = tool.generate_specs(
        dataframe=pd.DataFrame([{"x": 1}]),
        data_summary=_summary_model(_numeric_profile("x"), n_rows=1),
        user_intent="plot x",
    )

    assert specs[0]["title"] == "Spec Title"


@pytest.mark.parametrize(
    ("data_summary", "user_intent", "dataframe", "error_pattern"),
    [
        ("", "plot", pd.DataFrame([{"x": 1}]), r"profiles"),
        (_summary_model(_numeric_profile("x")), "", pd.DataFrame([{"x": 1}]), r"user_intent must be non-empty"),
        (_summary_model(_numeric_profile("x"), n_rows=0), "plot", pd.DataFrame(), r"dataframe must have at least one column"),
    ],
)
def test_generate_specs_validates_required_inputs(
    data_summary: DatasetSummaryModel | str,
    user_intent: str,
    dataframe: pd.DataFrame,
    error_pattern: str,
) -> None:
    tool = PlotTool(
        llm=_FakeLLMService(
            plans=[
                _plan_payload(
                    charts=[
                        {
                            "spec": {
                                "mark": "bar",
                                "encoding": {"x": {"field": "x", "type": "quantitative"}},
                            }
                        }
                    ]
                )
            ]
        )
    )

    expected_error = AttributeError if isinstance(data_summary, str) else ValueError
    with pytest.raises(expected_error, match=error_pattern):
        _ = tool.generate_specs(
            dataframe=dataframe,
            data_summary=data_summary,
            user_intent=user_intent,
        )


def test_plot_tool_uses_configurable_warning_threshold() -> None:
    tool = PlotTool(
        llm=_FakeLLMService(plans=[]),
        warn_max_rows_for_values=7,
    )

    assert tool.warn_max_rows_for_values == 7
