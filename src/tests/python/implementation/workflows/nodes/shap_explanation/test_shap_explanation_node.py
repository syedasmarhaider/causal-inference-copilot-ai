from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from python.domain.models.models import ChatMessage
from python.domain.repo.data_repo import DataRepo, ImageMime
from python.domain.repo.models_repo import ModelRecord, ModelsRepo
from python.domain.service.llm_service import LLMConfig, LLMResponse, LLMService
from python.domain.workflows.node import NodeRequest
from python.domain.workflows.ochestrator_state import OchestratorState
from python.domain.workflows.tool import Tool
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.shap_explanation.shap_explanation_node import (
    ShapExplanationNode,
)
from python.implementation.workflows.nodes.shap_explanation.shap_explanation_state import (
    ShapExplanationPayloadModel,
    ShapExplanationState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)


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


@dataclass(frozen=True)
class _Inputs:
    dataset_id: UUID
    dataset_summary: Any
    causal_spec: CausalSpec
    transformation_plan: TransformPlan
    selected_model: str
    trained_model_id: UUID


def _inputs() -> _Inputs:
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
    return _Inputs(
        dataset_id=uuid4(),
        dataset_summary=summary,
        causal_spec=causal_spec,
        transformation_plan=transformation_plan,
        selected_model="econml.dml.LinearDML",
        trained_model_id=uuid4(),
    )


@dataclass
class _FakeDataRepo(DataRepo):
    dataframes: dict[UUID, pd.DataFrame]
    loaded_dataset_ids: list[UUID] = field(default_factory=list)
    saved_csv_calls: list[dict[str, Any]] = field(default_factory=list)

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
        dataframe = self.dataframes[dataset_id].iloc[start:].copy()
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
        del user_id, conversation_id
        self.dataframes[dataset_id] = df.copy()
        self.saved_csv_calls.append(
            {
                "dataset_id": dataset_id,
                "df": df.copy(),
                "overwrite": overwrite,
                "include_index": include_index,
            }
        )

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
class _FakeModelsRepo(ModelsRepo):
    records: dict[UUID, Any]
    load_calls: list[UUID] = field(default_factory=list)

    def save_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
        model: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def load_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> ModelRecord | None:
        del user_id, conversation_id
        self.load_calls.append(model_id)
        model = self.records.get(model_id)
        if model is None:
            return None
        return ModelRecord(model_id=model_id, model=model, metadata={})

    def model_exists(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> bool:
        del user_id, conversation_id
        return model_id in self.records

    def delete_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> None:
        del user_id, conversation_id
        self.records.pop(model_id, None)


@dataclass
class _FakeEstimator:
    values: np.ndarray
    calls: int = 0

    def shap_values(self, X: Any) -> np.ndarray:
        assert list(X.columns) == ["sex"]
        self.calls += 1
        return self.values


@dataclass
class _FakeDataManipulationTool:
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
        return pd.DataFrame([{"column": "shap_sex", "mean_abs_shap": 0.1075, "mean_shap": 0.0775}])


@dataclass
class _FakeToolFactory(ToolFactory):
    data_manipulation_tool: Any
    profiling_tool: Any

    def get_tool_names(self) -> list[str]:
        return [DataManipulationTool.NAME, DatasetProfilingTool.NAME]

    def get_tool_info(self, name: str) -> str:
        return name

    def get_tools_info(self) -> dict[str, str]:
        return {name: name for name in self.get_tool_names()}

    def has_tool(self, name: str) -> bool:
        return name in self.get_tool_names()

    def get_tool(self, name: str) -> Tool:
        if name == DataManipulationTool.NAME:
            return self.data_manipulation_tool
        if name == DatasetProfilingTool.NAME:
            return self.profiling_tool
        raise KeyError(name)


@dataclass
class _FakeLLM(LLMService):
    content: str = (
        "Clinically, sex is the strongest effect modifier in this SHAP summary. "
        "Its mean absolute SHAP value indicates it contributes most to heterogeneity "
        "in the model's estimated treatment-effect contrast."
    )
    generate_calls: list[dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
    ) -> LLMResponse:
        self.generate_calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "config": config,
                "history": list(history or []),
            }
        )
        return LLMResponse(content=self.content)

    def generate_json(
        self,
        *,
        schema: type[Any],
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
        max_attempts: int = 3,
    ) -> Any:
        raise NotImplementedError


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
        return ShapExplanationState.NAME

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
        return self.values.get("working_dataset_id"), True

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


def _orchestrator_state(inputs: _Inputs) -> _FakeOrchestratorState:
    return _FakeOrchestratorState(
        values={
            "working_dataset_id": inputs.dataset_id,
            "latest_dataset_summary": inputs.dataset_summary,
            "causal_spec": inputs.causal_spec,
            "data_transformation_plan": inputs.transformation_plan,
            "selected_model": inputs.selected_model,
            "trained_model_id": inputs.trained_model_id,
        }
    )


def _node(
    *,
    data_repo: _FakeDataRepo,
    models_repo: _FakeModelsRepo,
    data_tool: _FakeDataManipulationTool,
    llm: _FakeLLM | None = None,
) -> ShapExplanationNode:
    return ShapExplanationNode(
        llm=llm or _FakeLLM(),
        data_repo=data_repo,
        models_repo=models_repo,
        tools_factory=_FakeToolFactory(
            data_manipulation_tool=data_tool,
            profiling_tool=DatasetProfilingTool(),
        ),
    )


def test_shap_explanation_state_roundtrips() -> None:
    dataset_id = uuid4()
    state = ShapExplanationState(
        ShapExplanationPayloadModel(
            source_signature="abc",
            shap_values_dataset_id=dataset_id,
            shap_values_summary={"status": "COMPLETED"},
            assistant_message="done",
        )
    )

    restored = ShapExplanationState.from_json_dict(state.to_json_dict())

    assert restored.payload.source_signature == "abc"
    assert restored.payload.shap_values_dataset_id == dataset_id
    assert restored.payload.shap_values_summary == {"status": "COMPLETED"}


def test_shap_explanation_node_calculates_separate_shap_csv_and_updates_state() -> None:
    inputs = _inputs()
    estimator = _FakeEstimator(values=np.asarray([[-0.05], [0.15], [-0.01], [0.22]], dtype=float))
    data_repo = _FakeDataRepo(dataframes={inputs.dataset_id: _dataframe()})
    models_repo = _FakeModelsRepo(records={inputs.trained_model_id: estimator})
    data_tool = _FakeDataManipulationTool()
    llm = _FakeLLM()
    orchestrator_state = _orchestrator_state(inputs)

    result = _node(data_repo=data_repo, models_repo=models_repo, data_tool=data_tool, llm=llm).run(
        request=NodeRequest(
            user_id=uuid4(),
            conversation_id=uuid4(),
            node_state=ShapExplanationState.init_empty(),
            orchestrator_state=orchestrator_state,
            read_only_messages_history=[
                ChatMessage(role="user", content="Which feature is most important by SHAP?")
            ],
        )
    )

    assert result.status == "PENDING"
    assert result.action == "NEEDS_INPUT"
    assert estimator.calls == 1
    assert orchestrator_state.set_calls[0][0] == ShapExplanationState.NAME
    shap_dataset_id = orchestrator_state.values["shap_values_dataset_id"]
    shap_df = data_repo.dataframes[shap_dataset_id]
    assert list(shap_df.columns) == ["patient_id", "sex", "shap_sex"]
    assert list(shap_df["shap_sex"]) == [-0.05, 0.15, -0.01, 0.22]
    assert "cate" not in shap_df.columns
    assert "effect_row" not in shap_df.columns
    assert len(data_tool.calls) == 1
    assert "separate feature-importance artifact" in data_tool.calls[0]["instructions"]
    assert isinstance(result.new_node_state, ShapExplanationState)
    assert result.response_messages
    assert result.response_messages[0].content == llm.content
    assert "Query result preview" not in result.response_messages[0].content
    assert len(llm.generate_calls) == 1
    assert "clinician" in (llm.generate_calls[0]["system_prompt"] or "").casefold()
    assert "shap_summary" in llm.generate_calls[0]["user_prompt"]
    refs = result.response_messages[0].artifact_refs or []
    assert [ref["artifact_meta"]["kind"] for ref in refs] == [
        "shap_values",
        "shap_query_result",
    ]


def test_shap_explanation_node_reuses_cached_shap_csv() -> None:
    inputs = _inputs()
    estimator = _FakeEstimator(values=np.asarray([[-0.05], [0.15], [-0.01], [0.22]], dtype=float))
    data_repo = _FakeDataRepo(dataframes={inputs.dataset_id: _dataframe()})
    models_repo = _FakeModelsRepo(records={inputs.trained_model_id: estimator})
    data_tool = _FakeDataManipulationTool()
    llm = _FakeLLM()
    node = _node(data_repo=data_repo, models_repo=models_repo, data_tool=data_tool, llm=llm)
    orchestrator_state = _orchestrator_state(inputs)
    request_base = {
        "user_id": uuid4(),
        "conversation_id": uuid4(),
        "orchestrator_state": orchestrator_state,
        "read_only_messages_history": [
            ChatMessage(role="user", content="Show feature importance from SHAP")
        ],
    }

    first = node.run(
        request=NodeRequest(
            node_state=ShapExplanationState.init_empty(),
            **request_base,
        )
    )
    second = node.run(
        request=NodeRequest(
            node_state=first.new_node_state,
            **request_base,
        )
    )

    assert first.status == "PENDING"
    assert second.status == "PENDING"
    assert estimator.calls == 1
    assert len(models_repo.load_calls) == 1
    assert len(data_tool.calls) == 2
    assert len(llm.generate_calls) == 2
