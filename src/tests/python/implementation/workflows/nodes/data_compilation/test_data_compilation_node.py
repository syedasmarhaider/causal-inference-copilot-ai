from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig
from python.domain.workflows.node import NodeRequest
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.data_compilation.data_compilation_node import (
    DataCompilationNode,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_state import (
    DataCompilationState,
)
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_state import (
    DataManupulationState,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)
from python.implementation.workflows.ochestrator.writable_ochestrator_state import (
    WritableOchestratorState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan_tool import (
    EncodingPlanTool,
)
from python.implementation.workflows.tools.causal.specs.causal_specs_tool import (
    CausalSpecsTool,
)
from python.implementation.workflows.tools.causal.validation.validation_backdoor_tool import (
    ValidationBackdoorTool,
)
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)


def _build_dataframe() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(60):
        rows.append(
            {
                "treatment": "drug" if index % 2 == 0 else "control",
                "outcome": "event" if index % 3 == 0 else "non_event",
                "age": 30 + index,
                "isex": 1 if index % 2 == 0 else 2,
            }
        )
    return pd.DataFrame(rows)


def _build_summary(df: pd.DataFrame):
    return DatasetProfilingTool().extract_dataset_summary(
        df,
        max_categories=20,
        sample_distinct=20,
        compute_quantiles=False,
        strict=True,
    )


def _causal_spec_payload() -> dict[str, Any]:
    return {
        "treatment_spec": {
            "kind": "binary",
            "column": "treatment",
            "treated": "drug",
            "control": "control",
        },
        "outcome_spec": {
            "kind": "binary",
            "column": "outcome",
            "event": "event",
            "non_event": "non_event",
        },
        "covariates": ["age"],
        "effect_modifiers": ["isex"],
        "experiment_type": "RCT",
    }


def _bad_transform_draft() -> dict[str, Any]:
    return {
        "columns": [
            {
                "column": "age",
                "role": "covariate",
                "preset": "num_standard",
            },
            {
                "column": "isex",
                "role": "effect_modifier",
                "preset": "cat_onehot",
            },
        ]
    }


def _good_transform_draft() -> dict[str, Any]:
    return {
        "columns": [
            {
                "column": "age",
                "role": "covariate",
                "preset": "num_standard",
            },
            {
                "column": "isex",
                "role": "effect_modifier",
                "preset": "num_standard",
            },
        ]
    }


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
        if not self.json_outputs:
            raise AssertionError("unexpected generate_json call")
        next_output = self.json_outputs.pop(0)
        if isinstance(next_output, Exception):
            raise next_output
        if isinstance(next_output, dict):
            return schema.model_validate(next_output)
        return next_output


@dataclass
class _InMemoryDataRepo(DataRepo):
    dataframes: dict[UUID, pd.DataFrame]

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        _ = user_id
        _ = conversation_id
        dataframe = self.dataframes[dataset_id]
        if limit is None:
            return dataframe.copy()
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
        _ = user_id
        _ = conversation_id
        _ = overwrite
        _ = include_index
        self.dataframes[dataset_id] = df.copy()

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
class _FakeDataManipulationTool:
    responses: list[pd.DataFrame] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    NAME: str = DataManipulationTool.NAME

    def get_tool_info(self) -> str:
        return "Fake data manipulation tool for tests."

    def manipulate(
        self,
        *,
        dataframe: pd.DataFrame,
        table_name: str,
        data_summary: str,
        instructions: str,
        retry_attempts: int = 3,
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "table_name": table_name,
                "data_summary": data_summary,
                "instructions": instructions,
                "retry_attempts": retry_attempts,
            }
        )
        if self.responses:
            return self.responses.pop(0).copy()
        return dataframe.copy()


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


def _tool_factory(*, data_manipulation_tool: _FakeDataManipulationTool) -> _FakeToolFactory:
    return _FakeToolFactory(
        tool_by_name={
            DatasetProfilingTool.NAME: DatasetProfilingTool(),
            CausalSpecsTool.NAME: CausalSpecsTool(),
            EncodingPlanTool.NAME: EncodingPlanTool(),
            ValidationBackdoorTool.NAME: ValidationBackdoorTool(),
            DataManipulationTool.NAME: data_manipulation_tool,
        }
    )


def _build_orchestrator_state(*, dataset_id: UUID, dataset_summary: Any) -> WritableOchestratorState:
    state = WritableOchestratorState.init_empty()
    state.set(
        DataManupulationState.NAME,
        {
            "working_dataset_id": dataset_id,
            "latest_dataset_summary": dataset_summary,
        },
    )
    state.set(
        ProtocolDiscussionState.NAME,
        {
            "protocol_discussion": "Confirmed protocol discussion.",
            "protocol_cleaning_instructions": "Normalize only grounded values.",
        },
    )
    return state


def test_data_compilation_node_enters_action_required_for_incompatible_transform_plan() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM(
        json_outputs=[
            _causal_spec_payload(),
            _bad_transform_draft(),
        ]
    )
    data_repo = _InMemoryDataRepo(dataframes={dataset_id: dataframe.copy()})
    node = DataCompilationNode(
        data_repo=data_repo,
        llm=llm,
        tools_factory=_tool_factory(data_manipulation_tool=_FakeDataManipulationTool()),
    )
    orchestrator_state = _build_orchestrator_state(
        dataset_id=dataset_id,
        dataset_summary=dataset_summary,
    )

    result = node.run(
        request=NodeRequest(
            user_id=uuid4(),
            conversation_id=uuid4(),
            node_state=DataCompilationState.init_empty(),
            orchestrator_state=orchestrator_state,
            read_only_messages_history=[ChatMessage(role="user", content="compile it")],
        )
    )

    payload = result.new_node_state.payload
    assert result.status == "PENDING"
    assert result.action == "NEEDS_INPUT"
    assert payload.phase == "ACTION_REQUIRED"
    assert payload.validation_status == "FAIL"
    assert payload.compiled_dataset_id is not None
    assert payload.compiled_dataset_id in data_repo.dataframes
    assert any(
        "observed data type is NUMERIC" in issue.message for issue in payload.validation_issues
    )
    assert orchestrator_state.get("causal_spec") is None
    assert orchestrator_state.get("data_transformation_plan") is None
    assert orchestrator_state.get("is_validated") is False


def test_data_compilation_node_retry_transform_then_confirm_publishes_outputs() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM(
        json_outputs=[
            _causal_spec_payload(),
            _bad_transform_draft(),
            {
                "action": "retry_transform",
                "assistant_message": "I will revise the transformation plan.",
                "repair_request": "Use a numeric preset for isex.",
            },
            _good_transform_draft(),
            {
                "assistant_message": "Review the repaired compiled setup.",
            },
            {
                "action": "confirm",
                "assistant_message": "Confirmed compiled setup.",
            },
        ]
    )
    data_repo = _InMemoryDataRepo(dataframes={dataset_id: dataframe.copy()})
    node = DataCompilationNode(
        data_repo=data_repo,
        llm=llm,
        tools_factory=_tool_factory(data_manipulation_tool=_FakeDataManipulationTool()),
    )
    orchestrator_state = _build_orchestrator_state(
        dataset_id=dataset_id,
        dataset_summary=dataset_summary,
    )
    first_result = node.run(
        request=NodeRequest(
            user_id=uuid4(),
            conversation_id=uuid4(),
            node_state=DataCompilationState.init_empty(),
            orchestrator_state=orchestrator_state,
            read_only_messages_history=[ChatMessage(role="user", content="compile it")],
        )
    )
    assert first_result.new_node_state.payload.phase == "ACTION_REQUIRED"

    second_result = node.run(
        request=NodeRequest(
            user_id=uuid4(),
            conversation_id=uuid4(),
            node_state=first_result.new_node_state,
            orchestrator_state=orchestrator_state,
            read_only_messages_history=[
                ChatMessage(role="assistant", content="Validation failed."),
                ChatMessage(role="user", content="Use numeric encoding for isex."),
            ],
        )
    )

    second_payload = second_result.new_node_state.payload
    assert second_result.status == "PENDING"
    assert second_result.action == "NEEDS_INPUT"
    assert second_payload.phase == "REVIEW_READY"
    assert second_payload.transformation_plan is not None
    assert second_payload.validation_status in {"PASS", "WARN"}

    third_result = node.run(
        request=NodeRequest(
            user_id=uuid4(),
            conversation_id=uuid4(),
            node_state=second_result.new_node_state,
            orchestrator_state=orchestrator_state,
            read_only_messages_history=[
                ChatMessage(role="assistant", content=second_payload.assistant_message or ""),
                ChatMessage(role="user", content="confirm"),
            ],
        )
    )

    assert third_result.status == "DONE"
    assert third_result.action == "NONE"
    assert third_result.new_node_state.payload.phase == "CONFIRMED"
    assert orchestrator_state.get("working_dataset_id") == second_payload.compiled_dataset_id
    assert orchestrator_state.get("causal_spec") is not None
    assert orchestrator_state.get("data_transformation_plan") is not None
    assert orchestrator_state.get("is_validated") is True
