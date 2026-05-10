from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pandas as pd

from python.domain.models.validation import ValidationIssueModel
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig
from python.domain.workflows.node import NodeRequest
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.data_compilation.data_compilation_cleaning import (
    CleaningResult,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_node import (
    DataCompilationNode,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_state import (
    DataCompilationState,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_transformation import (
    ColumnTransformationSuggestion,
    ColumnTransformationSuggestionList,
    TransformationResult,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_valiation import (
    DataCompilationValidationResult,
)
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_state import (
    DataManupulationState,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)
from python.implementation.workflows.ochestrator.causal_ochestrator_state import (
    CausalOchestratorState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.encoding.encoding_plan_tool import (
    EncodingPlanTool,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import CausalSpecDraft
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)


def _dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "treatment": "drug" if index % 2 == 0 else "control",
                "outcome": "event" if index % 3 == 0 else "non_event",
                "age": 30 + index,
                "isex": 1 if index % 2 == 0 else 2,
            }
            for index in range(20)
        ]
    )


def _summary(dataframe: pd.DataFrame):
    return DatasetProfilingTool().extract_dataset_summary(
        dataframe,
        max_categories=20,
        sample_distinct=20,
        compute_quantiles=False,
        strict=True,
    )


def _causal_draft() -> CausalSpecDraft:
    return CausalSpecDraft(
        treatment_column="treatment",
        outcome_column="outcome",
        covariates=["age"],
        effect_modifiers=["isex"],
    )


def _causal_spec() -> CausalSpec:
    return CausalSpec.model_validate(
        {
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
            "experiment_type": "OBSERVATIONAL",
            "id_col": "auto_id",
        }
    )


def _transform_plan() -> TransformPlan:
    return TransformPlan.model_validate(
        {
            "columns": [
                {
                    "column": "age",
                    "role": "covariate",
                    "encoding": {"preset": "num_standard"},
                },
                {
                    "column": "isex",
                    "role": "effect_modifier",
                    "encoding": {"preset": "num_standard"},
                },
            ]
        }
    )


def _transformation_suggestions() -> ColumnTransformationSuggestionList:
    return ColumnTransformationSuggestionList(
        suggestions=[
            ColumnTransformationSuggestion(
                column="age",
                role="covariate",
                preferred_type="NUMERIC",
                preferred_type_reason="Age is numeric.",
            ),
            ColumnTransformationSuggestion(
                column="isex",
                role="effect_modifier",
                preferred_type="CATEGORICAL",
                preferred_type_reason="Sex codes are easier to inspect as labels.",
            ),
        ]
    )


def _transformation_result() -> TransformationResult:
    return TransformationResult(
        transformation_plan=_transform_plan(),
        transformation_suggestions=_transformation_suggestions(),
    )


def _cleaning_result(dataframe: pd.DataFrame, *, suffix: str = "") -> CleaningResult:
    return CleaningResult(
        causal=_causal_spec(),
        pd_cleaned=dataframe.copy(),
        cleaned_data_summary=_summary(dataframe),
        summary_str=f"Cleaning summary{suffix}: rows and columns updated.",
        cleaning_notes=(f"note{suffix}",) if suffix else (),
        effective_draft=_causal_draft(),
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
    get_csv_data_calls: list[UUID] = field(default_factory=list)
    save_csv_data_calls: list[UUID] = field(default_factory=list)

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
        self.get_csv_data_calls.append(dataset_id)
        dataframe = self.dataframes[dataset_id].iloc[start:]
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
        self.save_csv_data_calls.append(dataset_id)
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
    NAME: str = DataManipulationTool.NAME

    def get_tool_info(self) -> str:
        return "Fake data manipulation tool for tests."


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


def _tool_factory() -> _FakeToolFactory:
    return _FakeToolFactory(
        tool_by_name={
            DatasetProfilingTool.NAME: DatasetProfilingTool(),
            DataManipulationTool.NAME: _FakeDataManipulationTool(),
            EncodingPlanTool.NAME: EncodingPlanTool(),
        }
    )


def _orchestrator_state(dataset_id: UUID, dataset_summary: Any) -> CausalOchestratorState:
    state = CausalOchestratorState.init_empty()
    state.set(
        DataManupulationState.NAME,
        {
            "working_dataset_id": dataset_id,
            "latest_dataset_summary": dataset_summary,
        },
    )
    state.set(ProtocolDiscussionState.NAME, {"causal_spec_draft": _causal_draft()})
    return state


def _node(data_repo: _InMemoryDataRepo, llm: _FakeLLM) -> DataCompilationNode:
    return DataCompilationNode(data_repo=data_repo, llm=llm, tools_factory=_tool_factory())


def test_node_publishes_preview_dataset_without_freezing_stage3() -> None:
    dataframe = _dataframe()
    source_dataset_id = uuid4()
    data_repo = _InMemoryDataRepo(dataframes={source_dataset_id: dataframe})
    llm = _FakeLLM(json_outputs=[{"assistant_message": "Review the compiled preview."}])
    state = _orchestrator_state(source_dataset_id, _summary(dataframe))

    with (
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.clean",
            return_value=_cleaning_result(dataframe),
        ) as clean_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            return_value=_transformation_result(),
        ) as transform_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.validate_data_compilation",
            return_value=DataCompilationValidationResult([], None),
        ),
    ):
        result = _node(data_repo, llm).run(
            request=NodeRequest(
                user_id=uuid4(),
                conversation_id=uuid4(),
                node_state=DataCompilationState.init_empty(),
                orchestrator_state=state,
                read_only_messages_history=[ChatMessage(role="user", content="compile")],
            )
        )

    payload = result.new_node_state.payload
    assert result.status == "PENDING"
    assert result.action == "NEEDS_INPUT"
    assert payload.phase == "REVIEW_READY"
    assert payload.compiled_dataset_id is not None
    assert state.get("working_dataset_id") == payload.compiled_dataset_id
    assert state.get("causal_spec") is None
    assert state.get("data_transformation_plan") is None
    assert state.get("working_dataset_frozen") is False
    assert state.get("is_validated") is False
    assert data_repo.save_csv_data_calls == [payload.compiled_dataset_id]
    assert clean_mock.call_count == 1
    assert transform_mock.call_count == 1


def test_node_confirm_freezes_existing_preview_without_duplicate_dataset_id() -> None:
    dataframe = _dataframe()
    source_dataset_id = uuid4()
    data_repo = _InMemoryDataRepo(dataframes={source_dataset_id: dataframe})
    llm = _FakeLLM(
        json_outputs=[
            {"assistant_message": "Review the compiled preview."},
            {"action": "confirm", "assistant_message": "Confirmed."},
        ]
    )
    state = _orchestrator_state(source_dataset_id, _summary(dataframe))
    node = _node(data_repo, llm)

    with (
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.clean",
            return_value=_cleaning_result(dataframe),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            return_value=_transformation_result(),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.validate_data_compilation",
            return_value=DataCompilationValidationResult([], None),
        ),
    ):
        first = node.run(
            request=NodeRequest(
                user_id=uuid4(),
                conversation_id=uuid4(),
                node_state=DataCompilationState.init_empty(),
                orchestrator_state=state,
                read_only_messages_history=[ChatMessage(role="user", content="compile")],
            )
        )
        dataset_ids_after_preview = state.get("working_dataset_ids")
        second = node.run(
            request=NodeRequest(
                user_id=uuid4(),
                conversation_id=uuid4(),
                node_state=first.new_node_state,
                orchestrator_state=state,
                read_only_messages_history=[ChatMessage(role="user", content="confirm")],
            )
        )

    assert second.status == "DONE"
    assert second.new_node_state.payload.phase == "CONFIRMED"
    assert state.get("working_dataset_ids") == dataset_ids_after_preview
    assert state.get("working_dataset_id") == first.new_node_state.payload.compiled_dataset_id
    assert state.get("causal_spec") is not None
    assert state.get("data_transformation_plan") is not None
    assert state.get("working_dataset_frozen") is True
    assert state.get("is_validated") is True


def test_node_reject_reverts_preview_dataset() -> None:
    dataframe = _dataframe()
    source_dataset_id = uuid4()
    source_summary = _summary(dataframe)
    data_repo = _InMemoryDataRepo(dataframes={source_dataset_id: dataframe})
    llm = _FakeLLM(
        json_outputs=[
            {"assistant_message": "Review the compiled preview."},
            {
                "action": "reject",
                "assistant_message": "Rejected; return to the previous dataset.",
            },
        ]
    )
    state = _orchestrator_state(source_dataset_id, source_summary)
    node = _node(data_repo, llm)

    with (
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.clean",
            return_value=_cleaning_result(dataframe),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            return_value=_transformation_result(),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.validate_data_compilation",
            return_value=DataCompilationValidationResult([], None),
        ),
    ):
        first = node.run(
            request=NodeRequest(
                user_id=uuid4(),
                conversation_id=uuid4(),
                node_state=DataCompilationState.init_empty(),
                orchestrator_state=state,
                read_only_messages_history=[ChatMessage(role="user", content="compile")],
            )
        )
        preview_dataset_id = first.new_node_state.payload.compiled_dataset_id
        second = node.run(
            request=NodeRequest(
                user_id=uuid4(),
                conversation_id=uuid4(),
                node_state=first.new_node_state,
                orchestrator_state=state,
                read_only_messages_history=[ChatMessage(role="user", content="reject")],
            )
        )

    assert second.status == "ABORTED"
    assert second.new_node_state.payload.phase == "FAILED"
    assert second.new_node_state.payload.hard_failure is False
    assert state.get("working_dataset_id") == source_dataset_id
    assert state.get("latest_dataset_summary").model_dump(mode="json") == source_summary.model_dump(
        mode="json"
    )
    assert preview_dataset_id not in state.get("working_dataset_ids")
    assert state.get("working_dataset_frozen") is False


def test_node_validation_failure_runs_one_full_retry_before_preview_publish() -> None:
    dataframe = _dataframe()
    source_dataset_id = uuid4()
    data_repo = _InMemoryDataRepo(dataframes={source_dataset_id: dataframe})
    llm = _FakeLLM(json_outputs=[{"assistant_message": "Review repaired preview."}])
    state = _orchestrator_state(source_dataset_id, _summary(dataframe))
    issue = ValidationIssueModel(
        severity="FAIL",
        message="Transform preset is incompatible with the observed coding.",
        fix_hint="Use a compatible effect-modifier encoding.",
    )

    with (
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.clean",
            side_effect=[_cleaning_result(dataframe), _cleaning_result(dataframe, suffix=" retry")],
        ) as clean_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            side_effect=[_transformation_result(), _transformation_result()],
        ) as transform_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.validate_data_compilation",
            side_effect=[
                DataCompilationValidationResult(
                    validation_errors=[issue],
                    user_suggestion_message=(
                        "Validation found repairable transformation or encoding issues.\n\n"
                        "Repairable validation errors:\n"
                        "- Transform preset is incompatible with the observed coding.\n"
                        "  What to fix: Use a compatible effect-modifier encoding."
                    ),
                ),
                DataCompilationValidationResult([], None),
            ],
        ) as validate_mock,
    ):
        result = _node(data_repo, llm).run(
            request=NodeRequest(
                user_id=uuid4(),
                conversation_id=uuid4(),
                node_state=DataCompilationState.init_empty(),
                orchestrator_state=state,
                read_only_messages_history=[ChatMessage(role="user", content="compile")],
            )
        )

    assert result.status == "PENDING"
    assert result.new_node_state.payload.phase == "REVIEW_READY"
    assert result.new_node_state.payload.validation_retry_count == 1
    assert clean_mock.call_count == 2
    assert clean_mock.call_args_list[1].kwargs["revised_instructions"]
    assert transform_mock.call_count == 2
    assert validate_mock.call_count == 2
    assert len(data_repo.save_csv_data_calls) == 1
    assert state.get("working_dataset_id") == result.new_node_state.payload.compiled_dataset_id


def test_node_nonrepairable_validation_failure_does_not_publish_preview() -> None:
    dataframe = _dataframe()
    source_dataset_id = uuid4()
    data_repo = _InMemoryDataRepo(dataframes={source_dataset_id: dataframe})
    llm = _FakeLLM()
    state = _orchestrator_state(source_dataset_id, _summary(dataframe))
    issue = ValidationIssueModel(
        severity="FAIL",
        message="The selected treatment column is missing.",
        fix_hint="Revise the treatment column in the causal draft.",
    )

    with (
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.clean",
            return_value=_cleaning_result(dataframe),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            return_value=_transformation_result(),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.validate_data_compilation",
            return_value=DataCompilationValidationResult([issue], None),
        ),
    ):
        result = _node(data_repo, llm).run(
            request=NodeRequest(
                user_id=uuid4(),
                conversation_id=uuid4(),
                node_state=DataCompilationState.init_empty(),
                orchestrator_state=state,
                read_only_messages_history=[ChatMessage(role="user", content="compile")],
            )
        )

    assert result.status == "ABORTED"
    assert result.new_node_state.payload.phase == "FAILED"
    assert result.new_node_state.payload.hard_failure is True
    assert data_repo.save_csv_data_calls == []
    assert state.get("working_dataset_id") == source_dataset_id
