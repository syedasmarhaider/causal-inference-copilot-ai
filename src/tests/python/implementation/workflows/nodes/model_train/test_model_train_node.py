from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from python.domain.repo.data_repo import DataRepo, ImageMime
from python.domain.workflows.node import NodeRequest
from python.domain.workflows.ochestrator_state import OchestratorState
from python.domain.workflows.tool import Tool
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.model_train.model_train_node import (
    ModelTrainNode,
)
from python.implementation.workflows.nodes.model_train.model_train_state import (
    ModelTrainPayloadModel,
    ModelTrainState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import (
    TransformPlan,
)
from python.implementation.workflows.tools.causal.inference.causal_command import (
    CommandFailure,
    ErrorInfo,
    FitCommand,
    FitSuccess,
)
from python.implementation.workflows.tools.causal.inference.causal_model_factory_tool import (
    CausalModelFactoryTool,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)


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


@dataclass(frozen=True)
class _TrainingInputs:
    dataset_id: UUID
    dataset_summary: Any
    causal_spec: CausalSpec
    transformation_plan: TransformPlan
    selected_model: str


def _build_training_inputs(
    *,
    dataset_id: UUID | None = None,
    selected_model: str = "econml.dml.CausalForestDML",
) -> _TrainingInputs:
    df = _build_dataframe()
    summary = DatasetProfilingTool().extract_dataset_summary(
        df,
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
    transformation_plan = TransformPlan.model_validate(
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
    return _TrainingInputs(
        dataset_id=dataset_id or uuid4(),
        dataset_summary=summary,
        causal_spec=causal_spec,
        transformation_plan=transformation_plan,
        selected_model=selected_model,
    )


@dataclass
class _FakeDataRepo(DataRepo):
    dataframe: pd.DataFrame
    loaded_dataset_ids: list[UUID] = field(default_factory=list)

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        start: int = 0,
        limit: int | None = None,
    ) -> pd.DataFrame:
        del user_id, conversation_id
        self.loaded_dataset_ids.append(dataset_id)
        dataframe = self.dataframe.iloc[start:].copy()
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
        raise NotImplementedError

    def get_json_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
    ) -> str:
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
        raise NotImplementedError

    def save_artifact(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        content: bytes,
        *,
        mime: ImageMime,
        overwrite: bool = True,
    ) -> None:
        raise NotImplementedError

    def get_artifact_mime(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
    ) -> ImageMime:
        raise NotImplementedError

    def get_artifact_bytes(
        self,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        *,
        expected_mime: ImageMime | None = None,
    ) -> bytes:
        raise NotImplementedError


@dataclass
class _FakeCausalModel:
    results: list[object]
    commands: list[FitCommand] = field(default_factory=list)

    def execute(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: FitCommand,
    ) -> object:
        del user_id, conversation_id
        self.commands.append(command)
        if not self.results:
            raise AssertionError("No fake fit result configured")
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

    def get_tool_names(self) -> list[str]:
        return [CausalModelFactoryTool.NAME, DatasetProfilingTool.NAME]

    def get_tool_info(self, name: str) -> str:
        return name

    def get_tools_info(self) -> dict[str, str]:
        return {name: name for name in self.get_tool_names()}

    def has_tool(self, name: str) -> bool:
        return name in self.get_tool_names()

    def get_tool(self, name: str) -> Tool:
        if name == CausalModelFactoryTool.NAME:
            return self.model_factory
        if name == DatasetProfilingTool.NAME:
            return DatasetProfilingTool()
        raise KeyError(name)


@dataclass
class _FakeOrchestratorState(OchestratorState):
    values: dict[str, Any]
    set_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    update_counter: int = 0

    def name(self) -> str:
        return "FAKE_OCHESTRATOR_STATE"

    def get_update_counter(self) -> int:
        return self.update_counter

    def set_update_counter(self, value: int) -> None:
        self.update_counter = value

    def get(self, key: str) -> Any:
        return self.values[key]

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.set_calls.append((key, dict(value)))
        self.values.update(value)

    def get_current_node_name(self) -> str:
        return ModelTrainState.NAME

    def get_current_node_companion_names(self, node_name: str) -> list[str]:
        del node_name
        return []

    def get_completed_and_last_pending_nodes(self) -> list[str]:
        return []

    def rocover_failure(self, current_failed_node: str) -> None:
        del current_failed_node

    def get_forward_states_after_node(self, node_name: str) -> list[str]:
        del node_name
        return []

    def roll_back_to_state(self, state_name: str) -> None:
        del state_name

    def get_working_dataset_id_and_frozen_status(self) -> tuple[UUID | None, bool]:
        return self.values.get("working_dataset_id"), False

    def get_ochestration_prompt(self) -> str:
        return ""

    def to_json_dict(self) -> dict[str, Any]:
        return dict(self.values)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> _FakeOrchestratorState:
        return cls(values=dict(payload))

    @classmethod
    def init_empty(cls) -> _FakeOrchestratorState:
        return cls(values={})


@dataclass(frozen=True)
class _EstimatorWithParams:
    alpha: float
    label: str = "estimator"

    def get_params(self, *, deep: bool = True) -> dict[str, Any]:
        return {"alpha": self.alpha, "deep": deep, "label": self.label}


def _orchestrator_state(inputs: _TrainingInputs) -> _FakeOrchestratorState:
    return _FakeOrchestratorState(
        values={
            "working_dataset_id": inputs.dataset_id,
            "latest_dataset_summary": inputs.dataset_summary,
            "causal_spec": inputs.causal_spec,
            "data_transformation_plan": inputs.transformation_plan,
            "selected_model": inputs.selected_model,
        }
    )


def _node_with_model(
    *,
    fake_model: _FakeCausalModel,
    data_repo: _FakeDataRepo | None = None,
) -> ModelTrainNode:
    return ModelTrainNode(
        llm=None,  # type: ignore[arg-type]
        data_repo=data_repo or _FakeDataRepo(dataframe=_build_dataframe()),
        tools_factory=_FakeToolFactory(model_factory=_FakeModelFactory(model=fake_model)),
    )


def test_model_train_state_roundtrips_training_spec_and_resets_on_signature_change() -> None:
    training_spec = {
        "selected_model": "econml.dml.LinearDML",
        "fit": {"backend": "fake", "used_init_kwargs": {"alpha": 0.1}},
    }
    state = ModelTrainState(
        ModelTrainPayloadModel(
            training_signature="sig-1",
            trained_model_id=uuid4(),
            training_warnings=["warn"],
            training_spec=training_spec,
            assistant_message="Training completed.",
        )
    )

    restored = ModelTrainState.from_json_dict(state.to_json_dict())

    assert restored.payload.model_dump(mode="json") == state.payload.model_dump(mode="json")
    reset_payload = restored.payload.reset_for_signature(training_signature="sig-2")
    assert reset_payload.training_signature == "sig-2"
    assert reset_payload.trained_model_id is None
    assert reset_payload.training_warnings == []
    assert reset_payload.training_spec is None


def test_model_train_success_stores_training_spec_in_orchestrator_state() -> None:
    inputs = _build_training_inputs(selected_model="econml.dml.CausalForestDML")
    started_at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    finished_at = datetime(2026, 1, 1, 12, 5, tzinfo=UTC)
    estimator = _EstimatorWithParams(alpha=0.25)
    fit_result = FitSuccess(
        run_id=uuid4(),
        started_at=started_at,
        finished_at=finished_at,
        warnings=["Convergence warning"],
        meta={
            "backend": "fake-backend",
            "columns": {"y": ["outcome"], "t": ["treatment"]},
            "used_init_kwargs": {
                "model_y": estimator,
                "model_t": [_EstimatorWithParams(alpha=0.5, label="candidate")],
                "discrete_treatment": True,
            },
        },
        fitted_model_id=uuid4(),
        artifacts={"n": 4, "x_shape": [4, 1], "w_shape": [4, 1]},
    )
    fake_model = _FakeCausalModel(results=[fit_result])
    data_repo = _FakeDataRepo(dataframe=_build_dataframe())
    orchestrator_state = _orchestrator_state(inputs)
    node = _node_with_model(fake_model=fake_model, data_repo=data_repo)

    result = node.run(
        request=NodeRequest(
            user_id=uuid4(),
            conversation_id=uuid4(),
            node_state=ModelTrainState.init_empty(),
            orchestrator_state=orchestrator_state,
            read_only_messages_history=None,
        )
    )

    assert result.status == "DONE"
    assert isinstance(result.new_node_state, ModelTrainState)
    payload = result.new_node_state.payload
    assert payload.trained_model_id == fit_result.fitted_model_id
    assert payload.training_warnings == ["Convergence warning"]
    assert payload.training_spec is not None

    assert orchestrator_state.set_calls == [
        (
            ModelTrainState.NAME,
            {
                "trained_model_id": fit_result.fitted_model_id,
                "training_warnings": ["Convergence warning"],
                "training_spec": payload.training_spec,
                "training_error_message": None,
            },
        )
    ]

    training_spec = payload.training_spec
    assert sorted(training_spec.keys()) == ["fit"]
    assert training_spec["fit"]["attempts"] == 1
    assert training_spec["fit"]["backend"] == "fake-backend"
    assert training_spec["fit"]["columns"] == {
        "y": ["outcome"],
        "t": ["treatment"],
    }
    assert training_spec["fit"]["artifacts"] == {
        "n": 4,
        "x_shape": [4, 1],
        "w_shape": [4, 1],
    }
    assert training_spec["fit"]["warnings"] == ["Convergence warning"]
    assert training_spec["fit"]["started_at"] == started_at.isoformat()
    assert training_spec["fit"]["finished_at"] == finished_at.isoformat()

    used_init_kwargs = training_spec["fit"]["used_init_kwargs"]
    assert used_init_kwargs["model_y"]["type"].endswith("._EstimatorWithParams")
    assert used_init_kwargs["model_y"]["params"] == {
        "alpha": 0.25,
        "deep": False,
        "label": "estimator",
    }
    assert used_init_kwargs["model_t"][0]["params"]["label"] == "candidate"
    assert used_init_kwargs["discrete_treatment"] is True
    json.dumps(training_spec)

    assert data_repo.loaded_dataset_ids == [inputs.dataset_id]
    assert len(fake_model.commands) == 1
    assert fake_model.commands[0].model_name == "econml.dml.CausalForestDML"


def test_model_train_success_after_retry_records_attempt_count() -> None:
    inputs = _build_training_inputs()
    fit_result = FitSuccess(
        run_id=uuid4(),
        started_at=None,
        finished_at=None,
        warnings=[],
        meta={"backend": "fake-backend", "columns": {}, "used_init_kwargs": {}},
        fitted_model_id=uuid4(),
    )
    fake_model = _FakeCausalModel(
        results=[
            CommandFailure(
                run_id=uuid4(),
                started_at=None,
                finished_at=None,
                warnings=[],
                meta={},
                error=ErrorInfo(
                    code="TRANSIENT",
                    message="temporary failure",
                    details={},
                ),
            ),
            fit_result,
        ]
    )

    result = _node_with_model(fake_model=fake_model).run(
        request=NodeRequest(
            user_id=uuid4(),
            conversation_id=uuid4(),
            node_state=ModelTrainState.init_empty(),
            orchestrator_state=_orchestrator_state(inputs),
            read_only_messages_history=None,
        )
    )

    assert result.status == "DONE"
    assert isinstance(result.new_node_state, ModelTrainState)
    assert result.new_node_state.payload.training_spec is not None
    assert result.new_node_state.payload.training_spec["fit"]["attempts"] == 2
    assert len(fake_model.commands) == 2


def test_model_train_failure_does_not_keep_stale_training_spec() -> None:
    inputs = _build_training_inputs()
    signature_payload = ModelTrainPayloadModel(
        training_signature="stale-signature",
        training_spec={"selected_model": "old"},
    )
    fake_model = _FakeCausalModel(
        results=[
            CommandFailure(
                run_id=uuid4(),
                started_at=None,
                finished_at=None,
                warnings=[],
                meta={},
                error=ErrorInfo(code="ESTIMATOR_ERROR", message="fit failed", details={}),
            ),
            CommandFailure(
                run_id=uuid4(),
                started_at=None,
                finished_at=None,
                warnings=[],
                meta={},
                error=ErrorInfo(code="ESTIMATOR_ERROR", message="fit failed", details={}),
            ),
        ]
    )

    result = _node_with_model(fake_model=fake_model).run(
        request=NodeRequest(
            user_id=uuid4(),
            conversation_id=uuid4(),
            node_state=ModelTrainState(signature_payload),
            orchestrator_state=_orchestrator_state(inputs),
            read_only_messages_history=None,
        )
    )

    assert result.status == "ABORTED"
    assert isinstance(result.new_node_state, ModelTrainState)
    assert result.new_node_state.payload.trained_model_id is None
    assert result.new_node_state.payload.training_spec is None
    assert result.new_node_state.payload.error_message == "fit failed"
    assert result.new_orchestrator_state.set_calls[-1] == (
        ModelTrainState.NAME,
        {
            "trained_model_id": None,
            "training_warnings": [],
            "training_spec": None,
            "training_error_message": "fit failed",
        },
    )
