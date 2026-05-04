from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID, uuid4

import pandas as pd
import pytest

from python.domain.models.errors import StateDependencyError
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse, LLMService
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.causal_inference.causal_inference_deps import (
    CausalInferenceDeps,
)
from python.implementation.workflows.nodes.causal_inference.causal_inference_node import (
    CausalInferenceNode,
    _extract_explicit_column_mentions,
    _requests_effect_graph,
)
from python.implementation.workflows.nodes.causal_inference.causal_inference_prompts import (
    get_causal_inference_node_info,
)
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import (
    CausalInferencePayloadModel,
    CausalInferenceState,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_state import (
    CompileAndValidatePayloadModel,
    CompileAndValidateState,
)
from python.implementation.workflows.nodes.dataset.dataset_state import (
    DatasetIterationModel,
    DatasetPayloadModel,
    DatasetState,
)
from python.implementation.workflows.nodes.model_selection.mode_selection_state import (
    ConfirmedModelSelectionPayload,
    ModelSelectionPayload,
    ModelSelectionState,
)
from python.implementation.workflows.nodes.model_train.model_train_state import (
    ModelTrainPayloadModel,
    ModelTrainState,
)
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import (
    TransformPlan,
)
from python.implementation.workflows.tools.causal.inference.causal_command import (
    ATESuccess,
    CATESuccess,
)
from python.implementation.workflows.tools.causal.inference.causal_model_factory_tool import (
    CausalModelFactoryTool,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.tools.plot_tool.plot_tool import PlotTool


def _build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "treatment": ["drug", "control", "drug", "control"],
            "outcome": [1.2, 0.4, 1.0, 0.6],
            "age": [61, 55, 70, 49],
            "sex": ["F", "M", "F", "M"],
        }
    )


def _build_inference_ready_spec() -> InferenceReadyCausalSpec:
    dataframe = _build_dataframe()
    summary = DatasetProfilingTool().extract_dataset_summary(
        dataframe,
        max_categories=10,
        sample_distinct=10,
        compute_quantiles=False,
        strict=True,
    )
    causal_spec = CausalSpec.model_validate(
        {
            "treatment_spec": {
                "kind": "binary",
                "column": "treatment",
                "treated": "drug",
                "control": "control",
            },
            "outcome_spec": {
                "kind": "continuous",
                "column": "outcome",
                "unit": "score",
            },
            "covariates": ["age"],
            "effect_modifiers": ["sex"],
            "experiment_type": "OBSERVATIONAL",
            "id_col": "patient_id",
        }
    )
    plan = TransformPlan.model_validate(
        {
            "columns": [
                {
                    "column": "age",
                    "role": "covariate",
                    "encoding": {"preset": "num_standard"},
                },
                {
                    "column": "sex",
                    "role": "effect_modifier",
                    "encoding": {
                        "preset": "map_binary",
                        "mapping": {"F": 0.0, "M": 1.0},
                        "allow_unknown": False,
                        "missing": "error",
                    },
                },
            ]
        }
    )
    return InferenceReadyCausalSpec(
        causal_spec=causal_spec,
        transformation_plan=plan,
        data_summary=summary,
    )


def _compile_state(*, dataset_id: UUID | None = None) -> CompileAndValidateState:
    spec = _build_inference_ready_spec()
    _ = dataset_id
    return CompileAndValidateState(
        CompileAndValidatePayloadModel(
            compiled_causal_spec=spec.causal_spec,
            transformation_plan=spec.transformation_plan,
            inference_ready_causal_spec=spec,
            phase="CONFIRMED",
            assistant_message="Compile confirmed.",
        )
    )


def _dataset_state(*, dataset_id: UUID | None = None) -> DatasetState:
    spec = _build_inference_ready_spec()
    resolved_dataset_id = dataset_id or uuid4()
    return DatasetState(
        DatasetPayloadModel(
            dataset_iterations=[DatasetIterationModel(dataset_id=resolved_dataset_id)],
            latest_summary=spec.data_summary,
        )
    )


def _selection_state(*, model_name: str = "econml.dml.LinearDML") -> ModelSelectionState:
    return ModelSelectionState(
        ModelSelectionPayload(
            confirmed_model_selection=ConfirmedModelSelectionPayload(
                selected_model=model_name,
                reasoning="Good fit.",
            ),
            assistant_message="Model confirmed.",
        )
    )


def _train_state(*, trained_model_id: UUID | None = None) -> ModelTrainState:
    return ModelTrainState(
        ModelTrainPayloadModel(
            trained_model_id=trained_model_id or uuid4(),
            assistant_message="Training done.",
        )
    )


@dataclass
class _FakeDataRepo(DataRepo):
    dataframe: pd.DataFrame
    raise_on_load: Exception | None = None
    saved_json_calls: list[dict[str, Any]] = field(default_factory=list)

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        start: int = 0,
        limit: int | None = None,
    ) -> pd.DataFrame:
        _ = user_id, conversation_id, dataset_id
        if self.raise_on_load is not None:
            raise self.raise_on_load
        dataframe = self.dataframe.iloc[start:].copy()
        if limit is None:
            return dataframe
        return dataframe.head(limit).copy()

    def save_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        df: pd.DataFrame,
        *,
        overwrite: bool = True,
        include_index: bool = False,
    ) -> None:
        raise NotImplementedError

    def get_json_data(self, user_id: UUID, conversation_id: UUID, dataset_id: UUID) -> str:
        raise NotImplementedError

    def save_json_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        json_data: str,
        *,
        overwrite: bool = True,
    ) -> None:
        self.saved_json_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "dataset_id": dataset_id,
                "json_data": json_data,
                "overwrite": overwrite,
            }
        )

    def save_artifact(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        content: bytes,
        *,
        mime: str,
        overwrite: bool = True,
    ) -> None:
        raise NotImplementedError

    def get_artifact_mime(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
    ) -> str:
        raise NotImplementedError

    def get_artifact_bytes(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        *,
        expected_mime: str | None = None,
    ) -> bytes:
        raise NotImplementedError


@dataclass
class _FakeLLM(LLMService):
    generate_content: str = "ATE summary."
    generate_json_results: list[Any] = field(default_factory=list)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
    ) -> LLMResponse:
        _ = system_prompt, user_prompt, config, history
        return LLMResponse(content=self.generate_content)

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
        _ = system_prompt, user_prompt, config, history, max_attempts
        if not self.generate_json_results:
            raise AssertionError("No fake JSON result configured")
        result = self.generate_json_results.pop(0)
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
        return result


@dataclass
class _FakeCausalModel:
    results: list[object]
    commands: list[object] = field(default_factory=list)

    def get_info(self) -> str:
        return "fake causal model"

    def get_command_info(self, command: str) -> str | None:
        _ = command
        return None

    def execute(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: object,
    ) -> object:
        _ = user_id, conversation_id
        self.commands.append(command)
        if not self.results:
            raise AssertionError("No fake model result configured")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@dataclass
class _FakeModelFactory:
    model: Any | None
    requested_models: list[str] = field(default_factory=list)

    def resolve(self, estimator_fqcn: str) -> Any | None:
        self.requested_models.append(estimator_fqcn)
        return self.model


@dataclass
class _FakeToolFactory(ToolFactory):
    model_factory: Any
    data_manipulation_tool: Any
    plot_tool: Any
    profiling_tool: Any

    def get_tool_names(self) -> list[str]:
        return [
            CausalModelFactoryTool.NAME,
            DataManipulationTool.NAME,
            PlotTool.NAME,
            DatasetProfilingTool.NAME,
        ]

    def get_tool_info(self, name: str) -> str:
        raise NotImplementedError

    def get_tools_info(self) -> dict[str, str]:
        raise NotImplementedError

    def has_tool(self, name: str) -> bool:
        return name in {
            CausalModelFactoryTool.NAME,
            DataManipulationTool.NAME,
            PlotTool.NAME,
            DatasetProfilingTool.NAME,
        }

    def get_tool(self, name: str) -> Any:
        if name == CausalModelFactoryTool.NAME:
            return self.model_factory
        if name == DataManipulationTool.NAME:
            return self.data_manipulation_tool
        if name == PlotTool.NAME:
            return self.plot_tool
        if name == DatasetProfilingTool.NAME:
            return self.profiling_tool
        raise KeyError(name)


@dataclass
class _FakeDataManipulationTool:
    result_dataframe: pd.DataFrame
    calls: list[dict[str, object]] = field(default_factory=list)

    def manipulate(
        self,
        *,
        dataframe: pd.DataFrame,
        table_name: str,
        data_summary: str,
        instructions: str,
        retry_attempts: int | None = None,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "dataframe": dataframe.copy(),
                "table_name": table_name,
                "data_summary": data_summary,
                "instructions": instructions,
                "retry_attempts": retry_attempts,
            }
        )
        return self.result_dataframe.copy()


@dataclass
class _FakePlotTool:
    specs: list[dict[str, object]] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)

    def generate_specs(
        self,
        *,
        dataframe: pd.DataFrame,
        data_summary: Any,
        user_intent: str,
    ) -> list[dict[str, object]]:
        self.calls.append(
            {
                "dataframe": dataframe.copy(),
                "data_summary": data_summary.model_dump(mode="python"),
                "user_intent": user_intent,
            }
        )
        return [dict(spec) for spec in self.specs]


def test_causal_inference_state_roundtrip_stays_pending_and_does_not_store_dep_ids() -> None:
    assert "caches ate" in get_causal_inference_node_info().lower()

    state = CausalInferenceState.init_empty()
    assert state.status() == "PENDING"
    assert state.error() is None
    assert state.messages()[0].role == "assistant"

    populated = CausalInferenceState(
        CausalInferencePayloadModel(
            ate_result_raw_json_str='{"estimate": 0.5}',
            latest_cate_result_raw_json_str='{"graph_rows": []}',
            latest_cate_request_summary="women vs men",
            assistant_message="Here is the latest result.",
            system_message="dataset graph handoff",
            message_artifact_refs=[
                {
                    "id": uuid4(),
                    "kind": "data",
                    "format": "json",
                    "artifact_meta": {"kind": "chart_spec"},
                }
            ],
            error_message="ignored",
        )
    )
    assert populated.status() == "PENDING"
    assert populated.error() is None

    dumped = populated.to_json_dict()
    assert "dataset_id" not in dumped
    assert "trained_model_id" not in dumped
    assert "selected_model" not in dumped

    restored = CausalInferenceState.from_json_dict(dumped)
    assert restored.payload.model_dump(mode="json") == populated.payload.model_dump(mode="json")


def test_causal_inference_deps_require_confirmed_compile_model_selection_and_training() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state(model_name="econml.dml.CausalForestDML")
    train_state = _train_state()
    dataset_state = _dataset_state()

    deps = CausalInferenceDeps.from_loaded(
        {
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
            ModelTrainState.NAME: train_state,
        }
    )

    assert deps.dataset_id == dataset_state.payload.dataset_iterations[-1].dataset_id
    assert deps.dataset_summary == dataset_state.payload.latest_summary
    assert deps.selected_model == "econml.dml.CausalForestDML"
    assert deps.trained_model_id == train_state.payload.trained_model_id

    with pytest.raises(StateDependencyError):
        CausalInferenceDeps.from_loaded(
            {
                DatasetState.NAME: dataset_state,
                CompileAndValidateState.NAME: CompileAndValidateState(
                    compile_state.payload.model_copy(update={"phase": "REVIEW_READY"})
                ),
                ModelSelectionState.NAME: selection_state,
                ModelTrainState.NAME: train_state,
            }
        )


def test_causal_inference_initial_run_computes_and_caches_ate_without_copying_dep_ids() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state()
    train_state = _train_state()
    dataset_state = _dataset_state()
    ate_success = ATESuccess(
        run_id=uuid4(),
        started_at=None,
        finished_at=None,
        warnings=["Observe residual confounding risk."],
        meta={},
        fitted_model_id=train_state.payload.trained_model_id,
        contrast={"treated": "drug", "control": "control"},
        ate=[{"ate": 0.5, "ate_interval": [0.1, 0.9]}],
    )
    fake_model = _FakeCausalModel(results=[ate_success])
    fake_factory = _FakeModelFactory(model=fake_model)
    node = CausalInferenceNode(
        llm=_FakeLLM(generate_content="Clinical ATE summary."),
        data_repo=_FakeDataRepo(dataframe=_build_dataframe()),
        tool_factory=_FakeToolFactory(
            model_factory=fake_factory,
            data_manipulation_tool=_FakeDataManipulationTool(result_dataframe=pd.DataFrame()),
            plot_tool=_FakePlotTool(),
            profiling_tool=DatasetProfilingTool(),
        ),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=CausalInferenceState.init_empty(),
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
            ModelTrainState.NAME: train_state,
        },
        messages_history=[
            ChatMessage(role="user", content="What is the average treatment effect?")
        ],
    )

    assert isinstance(result, CausalInferenceState)
    assert result.status() == "PENDING"
    assert result.payload.ate_result_raw_json_str is not None
    assert result.payload.latest_cate_result_raw_json_str is None
    assert result.payload.assistant_message == "Clinical ATE summary."
    assert result.payload.system_message is None
    assert result.error() is None
    assert "dataset_id" not in result.to_json_dict()
    assert "trained_model_id" not in result.to_json_dict()
    assert "selected_model" not in result.to_json_dict()
    assert fake_factory.requested_models == ["econml.dml.LinearDML"]
    assert len(fake_model.commands) == 1
    assert fake_model.commands[0].model_name == "econml.dml.LinearDML"


def test_causal_inference_dataset_graph_handoff_stays_pending() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state()
    train_state = _train_state()
    dataset_state = _dataset_state()
    fake_model = _FakeCausalModel(results=[])
    node = CausalInferenceNode(
        llm=_FakeLLM(
            generate_json_results=[
                {
                    "action": "handoff_dataset_graph",
                    "assistant_message": "I will hand this chart request to dataset visualization.",
                    "dataset_graph_request": "Plot the raw age distribution by treatment arm.",
                }
            ]
        ),
        data_repo=_FakeDataRepo(dataframe=_build_dataframe()),
        tool_factory=_FakeToolFactory(
            model_factory=_FakeModelFactory(model=fake_model),
            data_manipulation_tool=_FakeDataManipulationTool(result_dataframe=pd.DataFrame()),
            plot_tool=_FakePlotTool(),
            profiling_tool=DatasetProfilingTool(),
        ),
    )
    state = CausalInferenceState(
        CausalInferencePayloadModel(
            ate_result_raw_json_str=json.dumps(
                {"estimate": 0.5, "interval": {"lower": 0.1, "upper": 0.9}}
            ),
            assistant_message="Old answer.",
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=state,
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
            ModelTrainState.NAME: train_state,
        },
        messages_history=[
            ChatMessage(role="user", content="Show me a histogram of age by treatment.")
        ],
    )

    assert result.status() == "PENDING"
    assert result.error() is None
    assert (
        result.payload.assistant_message
        == "I will hand this chart request to dataset visualization."
    )
    assert result.payload.system_message is not None
    handoff = json.loads(result.payload.system_message)
    assert handoff["handoff_target"] == "DATASET"
    assert handoff["graph_scope"] == "data"
    assert fake_model.commands == []


def test_causal_inference_cate_graph_uses_data_manipulation_and_plot_tools() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state()
    train_state = _train_state()
    dataset_state = _dataset_state()
    cate_result_women = CATESuccess(
        run_id=uuid4(),
        started_at=None,
        finished_at=None,
        warnings=[],
        meta={},
        fitted_model_id=train_state.payload.trained_model_id,
        x_cols=["sex"],
        effects={"cate": [0.4], "cate_interval": [[0.1], [0.7]]},
    )
    cate_result_men = CATESuccess(
        run_id=uuid4(),
        started_at=None,
        finished_at=None,
        warnings=[],
        meta={},
        fitted_model_id=train_state.payload.trained_model_id,
        x_cols=["sex"],
        effects={"cate": [0.2], "cate_interval": [[-0.1], [0.5]]},
    )
    fake_model = _FakeCausalModel(results=[cate_result_women, cate_result_men])
    fake_data_manip = _FakeDataManipulationTool(
        result_dataframe=pd.DataFrame(
            [
                {"group_key": "women", "sex": "F"},
                {"group_key": "men", "sex": "M"},
            ]
        )
    )
    fake_plot_tool = _FakePlotTool(
        specs=[
            {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "mark": "bar",
                "encoding": {
                    "x": {"field": "group_key", "type": "nominal"},
                    "y": {"field": "cate", "type": "quantitative"},
                },
            }
        ]
    )
    node = CausalInferenceNode(
        llm=_FakeLLM(
            generate_content="Clinical subgroup summary.",
            generate_json_results=[
                {
                    "action": "generate_cate_graph",
                    "cate_request_summary": "Compare treatment effects between women and men.",
                }
            ],
        ),
        data_repo=_FakeDataRepo(dataframe=_build_dataframe()),
        tool_factory=_FakeToolFactory(
            model_factory=_FakeModelFactory(model=fake_model),
            data_manipulation_tool=fake_data_manip,
            plot_tool=fake_plot_tool,
            profiling_tool=DatasetProfilingTool(),
        ),
    )
    state = CausalInferenceState(
        CausalInferencePayloadModel(
            ate_result_raw_json_str=json.dumps(
                {"estimate": 0.5, "interval": {"lower": 0.1, "upper": 0.9}}
            )
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=state,
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
            ModelTrainState.NAME: train_state,
        },
        messages_history=[
            ChatMessage(
                role="user", content="Compare treatment effects between women and men as a graph."
            )
        ],
    )

    assert result.status() == "PENDING"
    assert result.error() is None
    assert result.payload.latest_cate_result_raw_json_str is not None
    assert result.payload.message_artifact_refs
    assert len(fake_data_manip.calls) == 1
    assert "group_key" in str(fake_data_manip.calls[0]["instructions"])
    assert list(fake_data_manip.calls[0]["dataframe"].columns) == list(_build_dataframe().columns)
    assert len(fake_plot_tool.calls) == 1
    plotted_df = fake_plot_tool.calls[0]["dataframe"]
    assert isinstance(plotted_df, pd.DataFrame)
    assert {"group_key", "cate", "cate_lower", "cate_upper"}.issubset(plotted_df.columns)
    assert len(fake_model.commands) == 2
    assert list(fake_model.commands[0].inputs.x_rows.columns) == ["sex"]
    payload = json.loads(result.payload.latest_cate_result_raw_json_str)
    assert payload["requested_filter_columns"] == []
    assert payload["non_effect_modifier_filter_columns"] == []


def test_causal_inference_cate_chart_request_generates_graph_even_when_routed_compute() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state()
    train_state = _train_state()
    dataset_state = _dataset_state()
    fake_model = _FakeCausalModel(
        results=[
            CATESuccess(
                run_id=uuid4(),
                started_at=None,
                finished_at=None,
                warnings=[],
                meta={},
                fitted_model_id=train_state.payload.trained_model_id,
                x_cols=["sex"],
                effects={"cate": [0.4], "cate_interval": [[0.1], [0.7]]},
            ),
            CATESuccess(
                run_id=uuid4(),
                started_at=None,
                finished_at=None,
                warnings=[],
                meta={},
                fitted_model_id=train_state.payload.trained_model_id,
                x_cols=["sex"],
                effects={"cate": [0.2], "cate_interval": [[-0.1], [0.5]]},
            ),
        ]
    )
    fake_data_manip = _FakeDataManipulationTool(
        result_dataframe=pd.DataFrame(
            [
                {"group_key": "age <= 50", "sex": "F"},
                {"group_key": "age > 50 and age <= 74", "sex": "M"},
            ]
        )
    )
    fake_plot_tool = _FakePlotTool(
        specs=[
            {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "mark": "boxplot",
                "encoding": {
                    "x": {"field": "group_key", "type": "nominal"},
                    "y": {"field": "cate", "type": "quantitative"},
                },
            }
        ]
    )
    node = CausalInferenceNode(
        llm=_FakeLLM(
            generate_content="Clinical subgroup summary without fake text chart.",
            generate_json_results=[
                {
                    "action": "compute_cate",
                    "cate_request_summary": (
                        "Estimate subgroup treatment effects across age groups and draw box plots."
                    ),
                }
            ],
        ),
        data_repo=_FakeDataRepo(dataframe=_build_dataframe()),
        tool_factory=_FakeToolFactory(
            model_factory=_FakeModelFactory(model=fake_model),
            data_manipulation_tool=fake_data_manip,
            plot_tool=fake_plot_tool,
            profiling_tool=DatasetProfilingTool(),
        ),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=CausalInferenceState(
            CausalInferencePayloadModel(
                ate_result_raw_json_str=json.dumps(
                    {"estimate": 0.5, "interval": {"lower": 0.1, "upper": 0.9}}
                )
            )
        ),
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
            ModelTrainState.NAME: train_state,
        },
        messages_history=[
            ChatMessage(
                role="user",
                content=(
                    "Estimate subgroup treatment effects across age groups and also draw "
                    "box plot charts."
                ),
            )
        ],
    )

    assert result.status() == "PENDING"
    assert result.error() is None
    assert result.payload.message_artifact_refs
    assert len(fake_plot_tool.calls) == 1
    plotted_df = fake_plot_tool.calls[0]["dataframe"]
    assert isinstance(plotted_df, pd.DataFrame)
    assert "effect_row" in plotted_df.columns
    assert "box plot" in str(fake_plot_tool.calls[0]["user_intent"]).lower()


def test_requests_effect_graph_detects_ite_visualization() -> None:
    assert _requests_effect_graph(
        user_request="Plot ITE estimates for selected patients.",
        request_summary="Individual treatment effect chart",
    )


def test_causal_inference_cate_allows_age_filter_with_disclaimer() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state()
    train_state = _train_state()
    dataset_state = _dataset_state()
    fake_model = _FakeCausalModel(
        results=[
            CATESuccess(
                run_id=uuid4(),
                started_at=None,
                finished_at=None,
                warnings=[],
                meta={},
                fitted_model_id=train_state.payload.trained_model_id,
                x_cols=["sex"],
                effects={"cate": [0.4], "cate_interval": [[0.1], [0.7]]},
            )
        ]
    )
    fake_data_manip = _FakeDataManipulationTool(
        result_dataframe=pd.DataFrame([{"group_key": "older", "sex": "F"}])
    )
    node = CausalInferenceNode(
        llm=_FakeLLM(
            generate_content="Clinical subgroup summary.",
            generate_json_results=[
                {
                    "action": "compute_cate",
                    "cate_request_summary": "Estimate treatment effects for age >= 60.",
                }
            ],
        ),
        data_repo=_FakeDataRepo(dataframe=_build_dataframe()),
        tool_factory=_FakeToolFactory(
            model_factory=_FakeModelFactory(model=fake_model),
            data_manipulation_tool=fake_data_manip,
            plot_tool=_FakePlotTool(),
            profiling_tool=DatasetProfilingTool(),
        ),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=CausalInferenceState(
            CausalInferencePayloadModel(
                ate_result_raw_json_str=json.dumps(
                    {"estimate": 0.5, "interval": {"lower": 0.1, "upper": 0.9}}
                )
            )
        ),
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
            ModelTrainState.NAME: train_state,
        },
        messages_history=[ChatMessage(role="user", content="Estimate CATE for age >= 60.")],
    )

    assert result.status() == "PENDING"
    assert result.error() is None
    assert len(fake_model.commands) == 1
    assert list(fake_model.commands[0].inputs.x_rows.columns) == ["sex"]
    payload = json.loads(cast(str, result.payload.latest_cate_result_raw_json_str))
    assert payload["requested_filter_columns"] == ["age"]
    assert payload["non_effect_modifier_filter_columns"] == ["age"]
    assert payload["effect_modifier_columns"] == ["sex"]
    assert "filtered using age" in cast(str, result.payload.assistant_message).lower()
    assert "confirmed effect modifiers: sex" in cast(
        str, result.payload.assistant_message
    ).lower()


def test_causal_inference_cate_allows_identifier_filter_with_disclaimer() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state()
    train_state = _train_state()
    dataset_state = _dataset_state()
    fake_model = _FakeCausalModel(
        results=[
            CATESuccess(
                run_id=uuid4(),
                started_at=None,
                finished_at=None,
                warnings=[],
                meta={},
                fitted_model_id=train_state.payload.trained_model_id,
                x_cols=["sex"],
                effects={"cate": [0.4], "cate_interval": [[0.1], [0.7]]},
            )
        ]
    )
    fake_data_manip = _FakeDataManipulationTool(
        result_dataframe=pd.DataFrame([{"group_key": "patient", "sex": "F"}])
    )
    node = CausalInferenceNode(
        llm=_FakeLLM(
            generate_content="Clinical subgroup summary.",
            generate_json_results=[
                {
                    "action": "compute_cate",
                    "cate_request_summary": "Estimate treatment effects for patient_id p1.",
                }
            ],
        ),
        data_repo=_FakeDataRepo(dataframe=_build_dataframe()),
        tool_factory=_FakeToolFactory(
            model_factory=_FakeModelFactory(model=fake_model),
            data_manipulation_tool=fake_data_manip,
            plot_tool=_FakePlotTool(),
            profiling_tool=DatasetProfilingTool(),
        ),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=CausalInferenceState(
            CausalInferencePayloadModel(
                ate_result_raw_json_str=json.dumps(
                    {"estimate": 0.5, "interval": {"lower": 0.1, "upper": 0.9}}
                )
            )
        ),
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
            ModelTrainState.NAME: train_state,
        },
        messages_history=[
            ChatMessage(role="user", content="Estimate CATE for patient_id p1.")
        ],
    )

    assert result.status() == "PENDING"
    assert result.error() is None
    assert len(fake_model.commands) == 1
    assert list(fake_model.commands[0].inputs.x_rows.columns) == ["sex"]
    payload = json.loads(cast(str, result.payload.latest_cate_result_raw_json_str))
    assert payload["requested_filter_columns"] == ["patient_id"]
    assert payload["non_effect_modifier_filter_columns"] == ["patient_id"]
    assert "filtered using patient_id" in cast(str, result.payload.assistant_message).lower()


def test_causal_inference_invalid_cate_selection_stays_pending() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state()
    train_state = _train_state()
    dataset_state = _dataset_state()
    fake_data_manip = _FakeDataManipulationTool(
        result_dataframe=pd.DataFrame(
            [
                {"group_key": "women", "age": 61, "sex": "F", "patient_id": "p1"},
            ]
        )
    )
    node = CausalInferenceNode(
        llm=_FakeLLM(
            generate_content=(
                "The final returned dataframe must contain only group_key plus the confirmed "
                "effect modifiers."
            ),
            generate_json_results=[
                {
                    "action": "compute_cate",
                    "cate_request_summary": "Estimate treatment effects among women.",
                }
            ],
        ),
        data_repo=_FakeDataRepo(dataframe=_build_dataframe()),
        tool_factory=_FakeToolFactory(
            model_factory=_FakeModelFactory(model=_FakeCausalModel(results=[])),
            data_manipulation_tool=fake_data_manip,
            plot_tool=_FakePlotTool(),
            profiling_tool=DatasetProfilingTool(),
        ),
    )
    state = CausalInferenceState(
        CausalInferencePayloadModel(
            ate_result_raw_json_str=json.dumps(
                {"estimate": 0.5, "interval": {"lower": 0.1, "upper": 0.9}}
            )
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=state,
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
            ModelTrainState.NAME: train_state,
        },
        messages_history=[
            ChatMessage(role="user", content="Estimate treatment effects among women.")
        ],
    )

    assert result.status() == "PENDING"
    assert result.error() is None
    assert result.payload.latest_cate_result_raw_json_str is None
    assert "final returned dataframe must contain only group_key" in cast(
        str, result.payload.assistant_message
    ).lower()


def test_extract_explicit_column_mentions_is_case_insensitive() -> None:
    matches = _extract_explicit_column_mentions(
        texts=["Compare AGE and patient_ID cohorts."],
        available_columns=["age", "patient_id", "sex"],
    )

    assert matches == ["age", "patient_id"]


def test_extract_explicit_column_mentions_does_not_match_substrings() -> None:
    matches = _extract_explicit_column_mentions(
        texts=["Compare stage 2 patients only."],
        available_columns=["age", "stage"],
    )

    assert matches == ["stage"]


def test_extract_explicit_column_mentions_prefers_longer_overlapping_names() -> None:
    matches = _extract_explicit_column_mentions(
        texts=["Estimate effects for age_group high."],
        available_columns=["age", "age_group"],
    )

    assert matches == ["age_group"]


def test_causal_inference_dataset_load_failure_stays_pending_with_system_detail() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state()
    train_state = _train_state()
    dataset_state = _dataset_state()
    node = CausalInferenceNode(
        llm=_FakeLLM(),
        data_repo=_FakeDataRepo(
            dataframe=_build_dataframe(),
            raise_on_load=RuntimeError("storage unavailable"),
        ),
        tool_factory=_FakeToolFactory(
            model_factory=_FakeModelFactory(model=_FakeCausalModel(results=[])),
            data_manipulation_tool=_FakeDataManipulationTool(result_dataframe=pd.DataFrame()),
            plot_tool=_FakePlotTool(),
            profiling_tool=DatasetProfilingTool(),
        ),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=CausalInferenceState.init_empty(),
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
            ModelTrainState.NAME: train_state,
        },
        messages_history=None,
    )

    assert result.status() == "PENDING"
    assert result.error() is None
    assert "could not load the cleaned dataset" in (result.payload.assistant_message or "").lower()
    assert "dataset load failed" in (result.payload.system_message or "").lower()
