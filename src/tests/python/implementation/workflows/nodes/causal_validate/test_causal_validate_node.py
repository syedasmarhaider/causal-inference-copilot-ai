from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from python.domain.models.models import ChatMessage
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import LLMConfig, LLMResponse, LLMService
from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.nodes.causal_validate.causal_validate_node import (
    CausalValidateNode,
)
from python.implementation.workflows.nodes.causal_validate.causal_validate_state import (
    CausalValidateState,
)
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.inference.causal_command import (
    FitCommand,
    ValidateCommand,
    ValidateSuccess,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.tools.plot_tool.plot_tool import PlotTool

_CAUSAL_MODEL_FACTORY_TOOL_NAME = "CAUSAL_MODEL_FACTORY"


def _dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4"],
            "treatment": ["drug", "control", "drug", "control"],
            "outcome": [1.2, 0.4, 1.0, 0.6],
            "age": [61, 55, 70, 49],
            "sex": ["F", "M", "F", "M"],
        }
    )


def _inference_ready_spec() -> InferenceReadyCausalSpec:
    dataframe = _dataframe()
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
            "outcome_spec": {"kind": "continuous", "column": "outcome", "unit": "score"},
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


@dataclass
class _FakeDataRepo(DataRepo):
    datasets: dict[UUID, pd.DataFrame]
    json_artifacts: dict[UUID, str] = field(default_factory=dict)

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        start: int = 0,
        limit: int | None = None,
    ) -> pd.DataFrame:
        _ = user_id, conversation_id
        dataframe = self.datasets[dataset_id].iloc[start:].copy()
        return dataframe if limit is None else dataframe.head(limit).copy()

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
        _ = user_id, conversation_id, overwrite, include_index
        self.datasets[dataset_id] = df.copy()

    def get_json_data(self, user_id: UUID, conversation_id: UUID, dataset_id: UUID) -> str:
        _ = user_id, conversation_id
        return self.json_artifacts[dataset_id]

    def save_json_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        json_data: str,
        *,
        overwrite: bool = True,
    ) -> None:
        _ = user_id, conversation_id, overwrite
        self.json_artifacts[dataset_id] = json_data

    def save_artifact(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        content: bytes,
        *,
        mime: Any,
        overwrite: bool = True,
    ) -> None:
        raise NotImplementedError

    def get_artifact_mime(self, user_id: UUID, conversation_id: UUID, artifact_id: UUID) -> Any:
        raise NotImplementedError

    def get_artifact_bytes(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        *,
        expected_mime: Any = None,
    ) -> bytes:
        raise NotImplementedError


@dataclass
class _FakeLLM(LLMService):
    json_results: list[dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
    ) -> LLMResponse:
        _ = system_prompt, user_prompt, config, history
        return LLMResponse(content="Validation summary.")

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
        return schema.model_validate(self.json_results.pop(0))


@dataclass
class _FakeCausalModel:
    result: ValidateSuccess
    commands: list[object] = field(default_factory=list)

    def get_info(self) -> str:
        return "fake model"

    def get_command_info(self, command: str) -> str | None:
        _ = command
        return None

    def execute(self, *, user_id: UUID, conversation_id: UUID, command: object) -> object:
        _ = user_id, conversation_id
        self.commands.append(command)
        return self.result


@dataclass
class _FakeModelFactory:
    model: _FakeCausalModel

    def resolve(self, estimator_fqcn: str) -> _FakeCausalModel:
        _ = estimator_fqcn
        return self.model


@dataclass
class _FakeOrchestratorState:
    values: dict[str, Any]

    def get(self, key: str) -> Any:
        return self.values[key]


@dataclass
class _FakeDataManipulationTool:
    calls: list[pd.DataFrame] = field(default_factory=list)

    def manipulate(
        self,
        *,
        dataframe: pd.DataFrame,
        table_name: str,
        data_summary: str,
        instructions: str,
        retry_attempts: int | None = None,
    ) -> pd.DataFrame:
        _ = table_name, data_summary, instructions, retry_attempts
        self.calls.append(dataframe.copy())
        return dataframe.loc[:, ["patient_id", "effect_row", "cate_oof"]].head(2).copy()


@dataclass
class _FakePlotTool:
    calls: list[pd.DataFrame] = field(default_factory=list)

    def generate_specs(
        self,
        *,
        dataframe: pd.DataFrame,
        data_summary: Any,
        user_intent: str,
    ) -> list[dict[str, object]]:
        _ = data_summary, user_intent
        self.calls.append(dataframe.copy())
        return [{"mark": "point", "encoding": {}}]


@dataclass
class _FakeToolFactory:
    model_factory: _FakeModelFactory
    data_manipulation_tool: _FakeDataManipulationTool
    plot_tool: _FakePlotTool

    def get_tool(self, name: str) -> Any:
        if name == _CAUSAL_MODEL_FACTORY_TOOL_NAME:
            return self.model_factory
        if name == DataManipulationTool.NAME:
            return self.data_manipulation_tool
        if name == PlotTool.NAME:
            return self.plot_tool
        if name == DatasetProfilingTool.NAME:
            return DatasetProfilingTool()
        raise KeyError(name)


def _orchestrator_state(
    *, dataset_id: UUID, spec: InferenceReadyCausalSpec
) -> _FakeOrchestratorState:
    return _FakeOrchestratorState(
        values={
            "working_dataset_id": dataset_id,
            "latest_dataset_summary": spec.data_summary,
            "causal_spec": spec.causal_spec,
            "data_transformation_plan": spec.transformation_plan,
            "selected_model": "econml.dml.LinearDML",
            "trained_model_id": uuid4(),
        }
    )


def test_causal_validate_caches_oof_rows_then_queries_without_revalidating() -> None:
    dataframe = _dataframe()
    spec = _inference_ready_spec()
    source_dataset_id = uuid4()
    fake_repo = _FakeDataRepo(datasets={source_dataset_id: dataframe})
    validation_success = ValidateSuccess(
        run_id=uuid4(),
        started_at=None,
        finished_at=None,
        warnings=[],
        meta={"outer_cv_folds": 2, "outer_cv_n_jobs": 2, "row_count": 4},
        validation_dataframe=pd.DataFrame(
            {
                "effect_row": [4, 2, 1, 3],
                "outer_fold": [1, 2, 1, 2],
                "cate_oof": [0.4, 0.2, 0.1, 0.3],
                "cate_oof_lower": [0.1, -0.1, -0.2, 0.0],
                "cate_oof_upper": [0.7, 0.5, 0.4, 0.6],
                "dr_outcome_oof": [1.0, 0.8, 1.1, 0.7],
            }
        ),
        dr_test_summary=pd.DataFrame({"fold": [1, 2], "metric": [0.1, 0.2]}),
    )
    fake_model = _FakeCausalModel(result=validation_success)
    fake_data_manipulation = _FakeDataManipulationTool()
    fake_plot = _FakePlotTool()
    node = CausalValidateNode(
        llm=_FakeLLM(
            json_results=[
                {
                    "action": "query_patient_validation",
                    "request_summary": "Show the two highest held-out CATEs.",
                },
                {
                    "action": "generate_validation_graph",
                    "request_summary": "Plot the two highest held-out CATEs.",
                    "query_target": "patient_validation",
                },
            ]
        ),
        data_repo=fake_repo,
        tools_factory=_FakeToolFactory(
            model_factory=_FakeModelFactory(model=fake_model),
            data_manipulation_tool=fake_data_manipulation,
            plot_tool=fake_plot,
        ),
    )
    orchestrator_state = _orchestrator_state(dataset_id=source_dataset_id, spec=spec)
    user_id = uuid4()
    conversation_id = uuid4()

    initial = node.run(
        request=NodeRequest(
            user_id=user_id,
            conversation_id=conversation_id,
            node_state=CausalValidateState.init_empty(),
            orchestrator_state=orchestrator_state,
            read_only_messages_history=[ChatMessage(role="user", content="Run validation.")],
        )
    )

    assert isinstance(initial.new_node_state, CausalValidateState)
    initial_payload = initial.new_node_state.payload
    assert initial_payload.validation_dataset_id is not None
    assert initial_payload.dr_test_summary_dataset_id is not None
    assert isinstance(fake_model.commands[0], ValidateCommand)
    assert isinstance(fake_model.commands[0].fit_command, FitCommand)
    cached_rows = fake_repo.datasets[initial_payload.validation_dataset_id]
    assert cached_rows["patient_id"].tolist() == ["p1", "p2", "p3", "p4"]
    assert cached_rows["cate_oof"].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert cached_rows["effect_row"].tolist() == [1, 2, 3, 4]

    follow_up = node.run(
        request=NodeRequest(
            user_id=user_id,
            conversation_id=conversation_id,
            node_state=initial.new_node_state,
            orchestrator_state=orchestrator_state,
            read_only_messages_history=[
                ChatMessage(role="user", content="Show the two highest held-out CATEs.")
            ],
        )
    )

    assert isinstance(follow_up.new_node_state, CausalValidateState)
    assert len(fake_model.commands) == 1
    assert len(fake_data_manipulation.calls) == 1
    assert follow_up.new_node_state.payload.latest_query_result_raw_json_str is not None
    assert len(follow_up.response_messages or []) == 1

    graph = node.run(
        request=NodeRequest(
            user_id=user_id,
            conversation_id=conversation_id,
            node_state=follow_up.new_node_state,
            orchestrator_state=orchestrator_state,
            read_only_messages_history=[
                ChatMessage(role="user", content="Plot the two highest held-out CATEs.")
            ],
        )
    )

    assert len(fake_model.commands) == 1
    assert len(fake_data_manipulation.calls) == 2
    assert len(fake_plot.calls) == 1
    graph_refs = list((graph.response_messages or [])[0].artifact_refs or [])
    assert len(graph_refs) == 1
    assert graph_refs[0]["kind"] == "graph"
    assert graph_refs[0]["id"] in fake_repo.json_artifacts

    changed_training_state = _orchestrator_state(dataset_id=source_dataset_id, spec=spec)
    revalidated = node.run(
        request=NodeRequest(
            user_id=user_id,
            conversation_id=conversation_id,
            node_state=graph.new_node_state,
            orchestrator_state=changed_training_state,
            read_only_messages_history=[ChatMessage(role="user", content="Run validation.")],
        )
    )

    assert len(fake_model.commands) == 2
    assert isinstance(revalidated.new_node_state, CausalValidateState)
    assert (
        revalidated.new_node_state.payload.validation_dataset_id
        != initial_payload.validation_dataset_id
    )
