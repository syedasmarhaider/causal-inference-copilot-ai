from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from python.domain.repo.data_repo import DataRepo, ImageMime
from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse, LLMService
from python.domain.workflows.node import NodeRequest
from python.domain.workflows.ochestrator_state import OchestratorState
from python.domain.workflows.tool import Tool
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.causal_inference.causal_inference_node import (
    CausalInferenceNode,
    _ResolvedInferenceContext,
    _build_cached_cate_query_instructions,
    _build_cached_cate_query_payload,
    _source_signature,
)
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import (
    CausalInferencePayloadModel,
    CausalInferenceState,
)
from python.implementation.workflows.tools.causal.common.inference_ready_causal_spec import (
    InferenceReadyCausalSpec,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.inference.causal_model_factory_tool import (
    CausalModelFactoryTool,
)
from python.implementation.workflows.tools.causal.inference.cate_cache import (
    CATE_COLUMN,
    CATE_LOWER_COLUMN,
    CATE_REVERSE_COLUMN,
    CATE_REVERSE_LOWER_COLUMN,
    CATE_REVERSE_UPPER_COLUMN,
    CATE_T0_COLUMN,
    CATE_T1_COLUMN,
    CATE_UPPER_COLUMN,
    EFFECT_ROW_COLUMN,
    build_all_row_cate_dataframe,
    summarize_all_row_cate_dataframe,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.tools.plot_tool.plot_tool import PlotTool


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


def _resolved_context(
    dataframe: pd.DataFrame,
    *,
    all_row_cate_dataset_id: UUID | None = None,
    all_row_cate_summary: dict[str, object] | None = None,
) -> _ResolvedInferenceContext:
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
    return _ResolvedInferenceContext(
        dataset_id=uuid4(),
        dataset_summary=summary,
        selected_model="econml.dml.LinearDML",
        trained_model_id=uuid4(),
        inference_ready_spec=InferenceReadyCausalSpec(
            causal_spec=causal_spec,
            transformation_plan=plan,
            data_summary=summary,
        ),
        all_row_cate_dataset_id=all_row_cate_dataset_id,
        all_row_cate_summary=all_row_cate_summary,
    )


@dataclass
class _FakeLLM(LLMService):
    generate_content: str = "Cached CATE summary."
    generate_json_results: list[Any] = field(default_factory=list)

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: Sequence[ChatMessage] | None,
    ) -> LLMResponse:
        del system_prompt, user_prompt, config, history
        return LLMResponse(content=self.generate_content)

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
        del system_prompt, user_prompt, config, history, max_attempts
        if not self.generate_json_results:
            raise AssertionError("No fake JSON result configured")
        result = self.generate_json_results.pop(0)
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
        return result


@dataclass
class _FakeDataRepo(DataRepo):
    dataframes: dict[UUID, pd.DataFrame]
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
    commands: list[object] = field(default_factory=list)

    def execute(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        command: object,
    ) -> object:
        del user_id, conversation_id
        self.commands.append(command)
        raise AssertionError("Cached CATE follow-up should not execute a model command")


@dataclass
class _FakeModelFactory:
    model: Any

    def resolve(self, estimator_fqcn: str) -> Any:
        del estimator_fqcn
        return self.model


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
class _FakeToolFactory(ToolFactory):
    model_factory: Any
    data_manipulation_tool: Any
    profiling_tool: Any

    def get_tool_names(self) -> list[str]:
        return [
            CausalModelFactoryTool.NAME,
            DataManipulationTool.NAME,
            PlotTool.NAME,
            DatasetProfilingTool.NAME,
        ]

    def get_tool_info(self, name: str) -> str:
        return name

    def get_tools_info(self) -> dict[str, str]:
        return {name: name for name in self.get_tool_names()}

    def has_tool(self, name: str) -> bool:
        return name in self.get_tool_names()

    def get_tool(self, name: str) -> Tool:
        if name == CausalModelFactoryTool.NAME:
            return self.model_factory
        if name == DataManipulationTool.NAME:
            return self.data_manipulation_tool
        if name == PlotTool.NAME:
            return object()  # type: ignore[return-value]
        if name == DatasetProfilingTool.NAME:
            return self.profiling_tool
        raise KeyError(name)


@dataclass
class _FakeOrchestratorState(OchestratorState):
    values: dict[str, Any]

    def name(self) -> str:
        return "FAKE_OCHESTRATOR_STATE"

    def get_update_counter(self) -> int:
        return 0

    def set_update_counter(self, value: int) -> None:
        del value

    def get(self, key: str) -> Any:
        return self.values[key]

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.values.update(value)

    def get_current_node_name(self) -> str:
        return CausalInferenceState.NAME

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


def test_cached_all_row_cate_dataframe_exposes_required_query_columns() -> None:
    dataframe = _dataframe()
    query_df = build_all_row_cate_dataframe(
        dataframe=dataframe,
        cate_values=np.asarray([0.1, 0.6, 0.2, 0.8], dtype=float),
        lower_values=np.asarray([-0.1, 0.3, 0.0, 0.5], dtype=float),
        upper_values=np.asarray([0.3, 0.9, 0.4, 1.1], dtype=float),
        for_treatment={"t0": 0.0, "t1": 1.0},
    )

    assert list(query_df["patient_id"]) == ["p1", "p2", "p3", "p4"]
    assert list(query_df[EFFECT_ROW_COLUMN]) == [1, 2, 3, 4]
    assert list(query_df[CATE_COLUMN]) == [0.1, 0.6, 0.2, 0.8]
    assert list(query_df[CATE_LOWER_COLUMN]) == [-0.1, 0.3, 0.0, 0.5]
    assert list(query_df[CATE_UPPER_COLUMN]) == [0.3, 0.9, 0.4, 1.1]
    assert list(query_df[CATE_REVERSE_COLUMN]) == [-0.1, -0.6, -0.2, -0.8]
    assert list(query_df[CATE_REVERSE_LOWER_COLUMN]) == [-0.3, -0.9, -0.4, -1.1]
    assert list(query_df[CATE_REVERSE_UPPER_COLUMN]) == [0.1, -0.3, -0.0, -0.5]
    assert list(query_df[CATE_T0_COLUMN]) == [0.0, 0.0, 0.0, 0.0]
    assert list(query_df[CATE_T1_COLUMN]) == [1.0, 1.0, 1.0, 1.0]


def test_cached_cate_query_instructions_use_duckdb_without_recomputing_cate() -> None:
    instructions = _build_cached_cate_query_instructions(
        request_summary="Which sex has the highest average benefit?",
        effect_modifier_columns=["sex"],
        identifier_column="patient_id",
        all_row_cate_summary={"status": "COMPLETED", "row_count": 4},
    )

    assert "DuckDB SQL" in instructions
    assert "Do not recompute CATE" in instructions
    assert "`cate`" in instructions
    assert "`cate_reverse`" in instructions
    assert "mean_cate" in instructions
    assert "Which sex has the highest average benefit?" in instructions


def test_cached_cate_query_payload_uses_query_result_as_primary_answer() -> None:
    dataframe = _dataframe()
    cate_dataset_id = uuid4()
    cached_df = build_all_row_cate_dataframe(
        dataframe=dataframe,
        cate_values=np.asarray([0.1, 0.6, 0.2, 0.8], dtype=float),
        lower_values=np.asarray([-0.1, 0.3, 0.0, 0.5], dtype=float),
        upper_values=np.asarray([0.3, 0.9, 0.4, 1.1], dtype=float),
        for_treatment={"t0": 0.0, "t1": 1.0},
    )
    cate_summary = summarize_all_row_cate_dataframe(
        dataframe=cached_df,
        dataset_id=cate_dataset_id,
        effect_modifier_columns=["sex"],
        for_treatment={"t0": 0.0, "t1": 1.0},
    )
    query_result_df = pd.DataFrame(
        [{"sex": "M", "row_count": 2, "mean_cate": 0.7, "max_cate": 0.8}]
    )

    payload = _build_cached_cate_query_payload(
        request_summary="Which type of patients are benefiting most?",
        resolved=_resolved_context(
            dataframe,
            all_row_cate_dataset_id=cate_dataset_id,
            all_row_cate_summary=cate_summary,
        ),
        identifier_column="patient_id",
        requested_filter_columns=["sex"],
        non_effect_modifier_filter_columns=[],
        query_result_df=query_result_df,
    )

    assert payload["analysis_kind"] == "cached_cate_query"
    assert payload["all_row_cate_summary"]["dataset_id"] == str(cate_dataset_id)
    assert payload["effect_modifier_columns"] == ["sex"]
    assert payload["query_result"]["columns"] == ["sex", "row_count", "mean_cate", "max_cate"]
    assert payload["query_result"]["records"] == [
        {"sex": "M", "row_count": 2, "mean_cate": 0.7, "max_cate": 0.8}
    ]


def test_causal_inference_followup_loads_cached_cate_and_does_not_execute_cate() -> None:
    dataframe = _dataframe()
    cate_dataset_id = uuid4()
    cached_df = build_all_row_cate_dataframe(
        dataframe=dataframe,
        cate_values=np.asarray([0.1, 0.6, 0.2, 0.8], dtype=float),
        lower_values=np.asarray([-0.1, 0.3, 0.0, 0.5], dtype=float),
        upper_values=np.asarray([0.3, 0.9, 0.4, 1.1], dtype=float),
        for_treatment={"t0": 0.0, "t1": 1.0},
    )
    cate_summary = summarize_all_row_cate_dataframe(
        dataframe=cached_df,
        dataset_id=cate_dataset_id,
        effect_modifier_columns=["sex"],
        for_treatment={"t0": 0.0, "t1": 1.0},
    )
    resolved = _resolved_context(
        dataframe,
        all_row_cate_dataset_id=cate_dataset_id,
        all_row_cate_summary=cate_summary,
    )
    data_repo = _FakeDataRepo(
        dataframes={
            resolved.dataset_id: dataframe,
            cate_dataset_id: cached_df,
        }
    )
    fake_model = _FakeCausalModel()
    fake_data_manip = _FakeDataManipulationTool(
        result_dataframe=pd.DataFrame(
            [{"sex": "M", "row_count": 2, "mean_cate": 0.7, "max_cate": 0.8}]
        )
    )
    node = CausalInferenceNode(
        llm=_FakeLLM(
            generate_content="Cached CATE summary.",
            generate_json_results=[
                {
                    "action": "compute_cate",
                    "cate_request_summary": "Which sex has the highest average benefit?",
                }
            ],
        ),
        data_repo=data_repo,
        tools_factory=_FakeToolFactory(
            model_factory=_FakeModelFactory(model=fake_model),
            data_manipulation_tool=fake_data_manip,
            profiling_tool=DatasetProfilingTool(),
        ),
    )
    state = CausalInferenceState(
        CausalInferencePayloadModel(
            source_signature=_source_signature(resolved=resolved),
            ate_result_raw_json_str=json.dumps(
                {"estimate": 0.5, "interval": {"lower": 0.1, "upper": 0.9}}
            ),
            assistant_message="Initial ATE summary.",
        )
    )
    orchestrator_state = _FakeOrchestratorState(
        values={
            "working_dataset_id": resolved.dataset_id,
            "latest_dataset_summary": resolved.dataset_summary,
            "causal_spec": resolved.inference_ready_spec.causal_spec,
            "data_transformation_plan": resolved.inference_ready_spec.transformation_plan,
            "selected_model": resolved.selected_model,
            "trained_model_id": resolved.trained_model_id,
            "all_row_cate_dataset_id": resolved.all_row_cate_dataset_id,
            "all_row_cate_summary": resolved.all_row_cate_summary,
        }
    )

    result = node.run(
        request=NodeRequest(
            user_id=uuid4(),
            conversation_id=uuid4(),
            node_state=state,
            orchestrator_state=orchestrator_state,
            read_only_messages_history=[
                ChatMessage(role="user", content="Which type of patients are benefiting most?")
            ],
        )
    )

    assert result.status == "PENDING"
    assert fake_model.commands == []
    assert data_repo.loaded_dataset_ids == [resolved.dataset_id, cate_dataset_id]
    assert len(fake_data_manip.calls) == 1
    query_dataframe = fake_data_manip.calls[0]["dataframe"]
    assert isinstance(query_dataframe, pd.DataFrame)
    assert list(query_dataframe[CATE_COLUMN]) == [0.1, 0.6, 0.2, 0.8]
    assert "Do not recompute CATE" in str(fake_data_manip.calls[0]["instructions"])
    assert isinstance(result.new_node_state, CausalInferenceState)
    cate_payload = json.loads(
        result.new_node_state.payload.latest_cate_result_raw_json_str or "{}"
    )
    assert cate_payload["analysis_kind"] == "cached_cate_query"
    assert cate_payload["query_result"]["records"][0]["sex"] == "M"
