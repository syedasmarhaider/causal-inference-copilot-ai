from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest

from python.domain.models.errors import StateDependencyError
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse, LLMService
from python.domain.workflows.tool_factory import ToolFactory
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
from python.implementation.workflows.nodes.model_train.model_train_deps import (
    ModelTrainDeps,
)
from python.implementation.workflows.nodes.model_train.model_train_node import (
    ModelTrainNode,
)
from python.implementation.workflows.nodes.model_train.model_train_prompts import (
    get_model_train_node_info,
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
            "treatment": ["drug", "control", "drug", "control"],
            "outcome": [1.2, 0.4, 1.0, 0.6],
            "age": [61, 55, 70, 49],
            "sex": ["F", "M", "F", "M"],
        }
    )


def _build_inference_ready_spec() -> InferenceReadyCausalSpec:
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
            assistant_message="Confirmed compile review",
        )
    )


def _dataset_state(*, dataset_id: UUID | None = None) -> DatasetState:
    resolved_dataset_id = dataset_id or uuid4()
    return DatasetState(
        DatasetPayloadModel(
            dataset_iterations=[DatasetIterationModel(dataset_id=resolved_dataset_id)],
        )
    )


def _selection_state(*, model_name: str = "econml.dml.LinearDML") -> ModelSelectionState:
    return ModelSelectionState(
        ModelSelectionPayload(
            confirmed_model_selection=ConfirmedModelSelectionPayload(
                selected_model=model_name,
                reasoning="Best fit for the current protocol.",
            ),
            assistant_message="Confirmed model.",
        )
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
        _ = user_id
        _ = conversation_id
        self.loaded_dataset_ids.append(dataset_id)
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
        raise NotImplementedError

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
class _FakeCausalModel:
    results: list[object]
    commands: list[FitCommand] = field(default_factory=list)

    def get_info(self) -> str:
        return "fake model"

    def get_command_info(self, command: str) -> str | None:
        _ = command
        return None

    def execute(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: FitCommand,
    ) -> object:
        _ = user_id
        _ = conversation_id
        self.commands.append(command)
        if not self.results:
            raise AssertionError("No fake fit result configured")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@dataclass
class _FakeLLM(LLMService):
    generate_content: str = "LLM failure summary."

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
        _ = schema, system_prompt, user_prompt, config, history, max_attempts
        raise AssertionError("generate_json should not be called in this test")


@dataclass
class _FakeToolFactory(ToolFactory):
    model_factory: Any

    def get_tool_names(self) -> list[str]:
        return [CausalModelFactoryTool.NAME]

    def get_tool_info(self, name: str) -> str:
        raise NotImplementedError

    def get_tools_info(self) -> dict[str, str]:
        raise NotImplementedError

    def has_tool(self, name: str) -> bool:
        return name == CausalModelFactoryTool.NAME

    def get_tool(self, name: str) -> Any:
        if name != CausalModelFactoryTool.NAME:
            raise KeyError(name)
        return self.model_factory


@dataclass
class _FakeModelFactory:
    model: Any | None
    requested_models: list[str] = field(default_factory=list)

    def resolve(self, estimator_fqcn: str) -> Any | None:
        self.requested_models.append(estimator_fqcn)
        return self.model


def test_model_train_info_state_and_roundtrip() -> None:
    assert "confirmed inference-ready causal specification" in get_model_train_node_info().lower()

    state = ModelTrainState.init_empty()
    assert state.status() == "PENDING"
    assert state.messages()[0].role == "assistant"

    done = ModelTrainState(
        ModelTrainPayloadModel(
            dataset_id=uuid4(),
            training_signature="sig-1",
            trained_model_id=uuid4(),
            assistant_message="Training completed.",
        )
    )
    assert done.status() == "DONE"

    failed = ModelTrainState(
        ModelTrainPayloadModel(
            error_message="fit failed",
            assistant_message="Training failed.",
        )
    )
    assert failed.status() == "ABORTED"
    assert failed.error() is not None

    restored = ModelTrainState.from_json_dict(done.to_json_dict())
    assert restored.payload.model_dump(mode="json") == done.payload.model_dump(mode="json")


def test_model_train_deps_require_confirmed_compile_and_selected_model() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state()
    dataset_state = _dataset_state()

    deps = ModelTrainDeps.from_loaded(
        {
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
        }
    )
    assert deps.dataset_id == dataset_state.payload.dataset_iterations[-1].dataset_id
    assert deps.selected_model == "econml.dml.LinearDML"
    assert deps.inference_ready_spec.causal_spec.treatment_spec.column == "treatment"

    with pytest.raises(StateDependencyError):
        ModelTrainDeps.from_loaded(
            {
                DatasetState.NAME: dataset_state,
                CompileAndValidateState.NAME: CompileAndValidateState(
                    compile_state.payload.model_copy(update={"phase": "REVIEW_READY"})
                ),
                ModelSelectionState.NAME: selection_state,
            }
        )

    with pytest.raises(StateDependencyError):
        ModelTrainDeps.from_loaded(
            {
                DatasetState.NAME: dataset_state,
                CompileAndValidateState.NAME: compile_state,
                ModelSelectionState.NAME: ModelSelectionState.init_empty(),
            }
        )


def test_model_train_success_builds_fit_command_from_confirmed_spec() -> None:
    dataset_id = uuid4()
    compile_state = _compile_state(dataset_id=dataset_id)
    selection_state = _selection_state(model_name="econml.dml.CausalForestDML")
    dataset_state = _dataset_state(dataset_id=dataset_id)
    fit_result = FitSuccess(
        run_id=uuid4(),
        started_at=None,
        finished_at=None,
        warnings=["Convergence warning"],
        meta={},
        fitted_model_id=uuid4(),
    )
    fake_model = _FakeCausalModel(results=[fit_result])
    fake_factory = _FakeModelFactory(model=fake_model)
    data_repo = _FakeDataRepo(dataframe=_build_dataframe())
    node = ModelTrainNode(
        llm=None,
        data_repo=data_repo,
        tool_factory=_FakeToolFactory(model_factory=fake_factory),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=ModelTrainState.init_empty(),
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
        },
        messages_history=[ChatMessage(role="user", content="Train it.")],
    )

    assert isinstance(result, ModelTrainState)
    assert result.status() == "DONE"
    assert result.payload.dataset_id == dataset_id
    assert result.payload.training_signature is not None
    assert result.payload.trained_model_id == fit_result.fitted_model_id
    assert result.payload.training_warnings == ["Convergence warning"]
    assert "training completed successfully" in (result.payload.assistant_message or "").lower()
    assert data_repo.loaded_dataset_ids == [dataset_id]
    assert fake_factory.requested_models == ["econml.dml.CausalForestDML"]
    assert len(fake_model.commands) == 1
    fit_command = fake_model.commands[0]
    assert fit_command.model_name == "econml.dml.CausalForestDML"
    assert fit_command.inference_ready_spec == compile_state.payload.inference_ready_causal_spec
    assert fit_command.df.equals(_build_dataframe())


def test_model_train_retries_once_then_returns_aborted_state_on_command_failure() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state()
    dataset_state = _dataset_state()
    failure = CommandFailure(
        run_id=uuid4(),
        started_at=None,
        finished_at=None,
        warnings=["Check positivity"],
        meta={},
        error=ErrorInfo(code="ESTIMATOR_ERROR", message="fit failed"),
    )
    fake_model = _FakeCausalModel(results=[failure, failure])
    node = ModelTrainNode(
        llm=None,
        data_repo=_FakeDataRepo(dataframe=_build_dataframe()),
        tool_factory=_FakeToolFactory(model_factory=_FakeModelFactory(model=fake_model)),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=ModelTrainState.init_empty(),
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
        },
        messages_history=None,
    )

    assert isinstance(result, ModelTrainState)
    assert result.status() == "ABORTED"
    assert result.payload.trained_model_id is None
    assert result.payload.training_warnings == []
    assert result.payload.error_message == "fit failed"
    assert "training failed" in (result.payload.assistant_message or "").lower()
    assert "attempted 2 times" in (result.payload.assistant_message or "").lower()
    assert len(fake_model.commands) == 2


def test_model_train_succeeds_on_second_attempt() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state()
    dataset_state = _dataset_state()
    fit_result = FitSuccess(
        run_id=uuid4(),
        started_at=None,
        finished_at=None,
        warnings=[],
        meta={},
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
                error=ErrorInfo(code="TRANSIENT", message="temporary failure"),
            ),
            fit_result,
        ]
    )
    node = ModelTrainNode(
        llm=None,
        data_repo=_FakeDataRepo(dataframe=_build_dataframe()),
        tool_factory=_FakeToolFactory(model_factory=_FakeModelFactory(model=fake_model)),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=ModelTrainState.init_empty(),
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
        },
        messages_history=None,
    )

    assert isinstance(result, ModelTrainState)
    assert result.status() == "DONE"
    assert result.payload.trained_model_id == fit_result.fitted_model_id
    assert "after 2 attempts" in (result.payload.assistant_message or "").lower()
    assert len(fake_model.commands) == 2


def test_model_train_failure_uses_llm_summary_for_user_message() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state()
    dataset_state = _dataset_state()
    failure = CommandFailure(
        run_id=uuid4(),
        started_at=None,
        finished_at=None,
        warnings=["Check positivity"],
        meta={},
        error=ErrorInfo(code="ESTIMATOR_ERROR", message="fit failed"),
    )
    fake_model = _FakeCausalModel(results=[failure, failure])
    node = ModelTrainNode(
        llm=_FakeLLM(generate_content="Training failed because the estimator saw invalid inputs."),
        data_repo=_FakeDataRepo(dataframe=_build_dataframe()),
        tool_factory=_FakeToolFactory(model_factory=_FakeModelFactory(model=fake_model)),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=ModelTrainState.init_empty(),
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
        },
        messages_history=None,
    )

    assert isinstance(result, ModelTrainState)
    assert result.status() == "ABORTED"
    assert result.payload.assistant_message is not None
    assert "invalid inputs" in result.payload.assistant_message.lower()
    assert "attempted 2 times" in result.payload.assistant_message.lower()


def test_model_train_reuses_existing_fit_for_same_inputs() -> None:
    compile_state = _compile_state()
    selection_state = _selection_state()
    dataset_state = _dataset_state()
    fit_result = FitSuccess(
        run_id=uuid4(),
        started_at=None,
        finished_at=None,
        warnings=[],
        meta={},
        fitted_model_id=uuid4(),
    )
    fake_model = _FakeCausalModel(
        results=[
            fit_result,
            RuntimeError("should not be called"),
        ]
    )
    data_repo = _FakeDataRepo(dataframe=_build_dataframe())
    node = ModelTrainNode(
        llm=None,
        data_repo=data_repo,
        tool_factory=_FakeToolFactory(model_factory=_FakeModelFactory(model=fake_model)),
    )

    first_result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=ModelTrainState.init_empty(),
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
        },
        messages_history=None,
    )
    assert isinstance(first_result, ModelTrainState)
    assert first_result.payload.trained_model_id == fit_result.fitted_model_id

    second_result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        state=first_result,
        previous_state_dependencies={
            DatasetState.NAME: dataset_state,
            CompileAndValidateState.NAME: compile_state,
            ModelSelectionState.NAME: selection_state,
        },
        messages_history=None,
    )

    assert isinstance(second_result, ModelTrainState)
    assert second_result.payload.trained_model_id == fit_result.fitted_model_id
    assert data_repo.loaded_dataset_ids == [dataset_state.payload.dataset_iterations[-1].dataset_id]
    assert len(fake_model.commands) == 1
