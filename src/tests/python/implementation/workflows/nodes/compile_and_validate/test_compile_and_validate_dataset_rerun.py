from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig
from python.domain.workflows.ochestrator_state import ReadOnlyOchestratorState
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_node import (
    CompileAndValidateNode,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_state import (
    CompileAndValidatePayloadModel,
    CompileAndValidateState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.encoding.encoding_plan_tool import (
    EncodingPlanTool,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.specs.causal_specs_tool import (
    CausalSpecsTool,
)
from python.implementation.workflows.tools.causal.validation.validation_backdoor_tool import (
    ValidationBackdoorTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
    DatasetSummaryModel,
)


def _build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"treatment": "drug", "outcome": 1.0, "age": 41, "sex": "F"},
            {"treatment": "control", "outcome": 0.4, "age": 55, "sex": "M"},
            {"treatment": "drug", "outcome": 1.3, "age": 63, "sex": "F"},
            {"treatment": "control", "outcome": 0.2, "age": 49, "sex": "M"},
        ]
    )


def _build_summary(df: pd.DataFrame) -> DatasetSummaryModel:
    return DatasetProfilingTool().extract_dataset_summary(
        df,
        max_categories=10,
        sample_distinct=10,
        compute_quantiles=False,
        strict=True,
    )


@dataclass
class _FakeLLM:
    json_outputs: list[Any] = field(default_factory=list)
    generate_json_calls: list[dict[str, Any]] = field(default_factory=list)

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
        next_output = self.json_outputs.pop(0)
        if isinstance(next_output, dict):
            return schema.model_validate(next_output)
        return next_output


@dataclass
class _FakeDataRepo(DataRepo):
    dataframe: pd.DataFrame

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        _ = user_id
        _ = conversation_id
        _ = dataset_id
        if limit is None:
            return self.dataframe.copy()
        return self.dataframe.head(limit).copy()

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
class _FakeToolFactory(ToolFactory):
    tool_by_name: dict[str, Any]

    def get_tool_names(self) -> list[str]:
        return sorted(self.tool_by_name)

    def get_tool_info(self, name: str) -> str:
        return self.get_tool(name).get_tool_info()

    def get_tools_info(self) -> dict[str, str]:
        return {name: tool.get_tool_info() for name, tool in self.tool_by_name.items()}

    def has_tool(self, name: str) -> bool:
        return name in self.tool_by_name

    def get_tool(self, name: str) -> Any:
        return self.tool_by_name[name]


class _StubOchestratorState(ReadOnlyOchestratorState):
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = values

    def get(self, key: str) -> Any:
        return self._values.get(key)


def _tool_factory() -> _FakeToolFactory:
    return _FakeToolFactory(
        tool_by_name={
            CausalSpecsTool.NAME: CausalSpecsTool(),
            EncodingPlanTool.NAME: EncodingPlanTool(),
            ValidationBackdoorTool.NAME: ValidationBackdoorTool(),
        }
    )


def test_compile_and_validate_reruns_full_pipeline_when_dataset_id_changes() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    current_dataset_id = uuid4()
    previous_dataset_id = uuid4()
    llm = _FakeLLM(
        json_outputs=[
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
                "experiment_type": "RCT",
            },
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
            },
            {
                "assistant_message": "Recompiled review summary.",
            },
        ]
    )
    node = CompileAndValidateNode(
        llm=llm,
        data_repo=_FakeDataRepo(dataframe=dataframe),
        tool_factory=_tool_factory(),
    )

    stale_state = CompileAndValidateState(
        CompileAndValidatePayloadModel(
            dataset_id=previous_dataset_id,
            phase="REVIEW_READY",
            assistant_message="Please confirm the old compiled setup.",
            compiled_causal_spec=CausalSpec.model_validate(
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
                    "experiment_type": "RCT",
                }
            ),
            transformation_plan=TransformPlan.model_validate(
                {
                    "columns": [
                        {
                            "column": "age",
                            "role": "covariate",
                            "encoding": {"preset": "num_standard"},
                        }
                    ]
                }
            ),
        )
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        readonly_orchestrator_state=_StubOchestratorState(
            {
                "working_dataset_id": current_dataset_id,
                "working_dataset_summary": dataset_summary,
                "protocol_discussion": "Confirmed protocol discussion.",
            }
        ),
        messages_history=[ChatMessage(role="user", content="yes")],
        state=stale_state,
    )

    assert result.payload.phase == "REVIEW_READY"
    assert result.payload.dataset_id == current_dataset_id
    assert result.payload.compiled_causal_spec is not None
    assert result.payload.transformation_plan is not None
    assert result.payload.inference_ready_causal_spec is not None
    assert result.payload.assistant_message == "Recompiled review summary."
    assert len(llm.generate_json_calls) == 3


def test_compile_and_validate_state_pending_reset_keeps_dataset_binding_only() -> None:
    dataset_id = uuid4()
    state = CompileAndValidateState(
        CompileAndValidatePayloadModel(
            dataset_id=dataset_id,
            phase="CONFIRMED",
            assistant_message="Confirmed.",
            system_message="System.",
            error_message="boom",
            compiled_causal_spec=CausalSpec.model_validate(
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
                    "experiment_type": "RCT",
                }
            ),
            transformation_plan=TransformPlan.model_validate(
                {
                    "columns": [
                        {
                            "column": "age",
                            "role": "covariate",
                            "encoding": {"preset": "num_standard"},
                        }
                    ]
                }
            ),
        )
    )

    state.set_status_pending()

    assert state.payload.dataset_id == dataset_id
    assert state.payload.phase == "INIT"
    assert state.payload.compiled_causal_spec is None
    assert state.payload.transformation_plan is None
    assert state.payload.inference_ready_causal_spec is None
    assert state.payload.validation_issues == []
    assert state.payload.assistant_message is None
    assert state.payload.system_message is None
    assert state.payload.error_message is None
