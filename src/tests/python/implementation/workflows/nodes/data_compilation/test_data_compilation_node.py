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
from python.implementation.workflows.ochestrator.writable_ochestrator_state import (
    WritableOchestratorState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    CausalSpecDraft,
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
            "experiment_type": "RCT",
        }
    )


def _causal_draft() -> CausalSpecDraft:
    return CausalSpecDraft.model_validate(
        {
            "treatment_column": "treatment",
            "outcome_column": "outcome",
            "covariates": ["age"],
            "effect_modifiers": ["isex"],
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
            "causal_spec_draft": _causal_draft(),
        },
    )
    return state


def _cleaning_result(dataframe: pd.DataFrame) -> CleaningResult:
    return CleaningResult(
        cleaned_data_summary=_build_summary(dataframe),
        pd_cleaned=dataframe.copy(),
        causal=_causal_spec(),
    )


def test_data_compilation_node_auto_retries_full_compile_when_transform_requires_dataset_changes() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM(
        json_outputs=[{"assistant_message": "Review the automatically repaired compiled setup."}]
    )
    data_repo = _InMemoryDataRepo(dataframes={dataset_id: dataframe.copy()})
    node = DataCompilationNode(data_repo=data_repo, llm=llm, tools_factory=_tool_factory())
    orchestrator_state = _build_orchestrator_state(
        dataset_id=dataset_id,
        dataset_summary=dataset_summary,
    )

    with (
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.cleaning",
            side_effect=[_cleaning_result(dataframe), _cleaning_result(dataframe)],
        ) as cleaning_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            side_effect=[
                TransformationResult(
                    transformation_plan=None,
                    required_dataset_changes=(
                        "Column 'isex' must be recoded into a grounded categorical "
                        "representation before transformation."
                    ),
                ),
                TransformationResult(
                    transformation_plan=_transform_plan(),
                    required_dataset_changes=None,
                ),
            ],
        ) as transform_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.validate_data_compilation",
            return_value=DataCompilationValidationResult(
                validation_errors=[],
                user_suggestion_message=None,
            ),
        ),
    ):
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
    assert payload.phase == "REVIEW_READY"
    assert payload.transformation_retry_count == 1
    assert payload.compiled_dataset_id is not None
    assert payload.compiled_dataset_id in data_repo.dataframes
    assert orchestrator_state.get("working_dataset_id") == payload.compiled_dataset_id
    assert orchestrator_state.get("latest_dataset_summary") == payload.compiled_dataset_summary
    assert (
        orchestrator_state.get("causal_spec_draft").model_dump(mode="json")
        == _causal_draft().model_dump(mode="json")
    )
    assert orchestrator_state.get("causal_spec") is None
    assert orchestrator_state.get("data_transformation_plan") is None
    assert orchestrator_state.get("working_dataset_frozen") is False
    assert orchestrator_state.get("is_validated") is False
    assert cleaning_mock.call_count == 2
    assert transform_mock.call_count == 2
    second_cleaning_call = cleaning_mock.call_args_list[1]
    assert "isex" in second_cleaning_call.kwargs["cleaning_instructions"]


def test_data_compilation_node_auto_retries_validation_on_cleaned_dataset() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM(
        json_outputs=[{"assistant_message": "Review the repaired compiled setup."}]
    )
    data_repo = _InMemoryDataRepo(dataframes={dataset_id: dataframe.copy()})
    node = DataCompilationNode(data_repo=data_repo, llm=llm, tools_factory=_tool_factory())
    orchestrator_state = _build_orchestrator_state(
        dataset_id=dataset_id,
        dataset_summary=dataset_summary,
    )
    repairable_issue = ValidationIssueModel(
        severity="FAIL",
        message="Transform preset is incompatible with the observed numeric coding.",
        fix_hint="Use a grounded numeric encoding for isex.",
    )

    with (
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.cleaning",
            return_value=_cleaning_result(dataframe),
        ) as cleaning_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            side_effect=[
                TransformationResult(
                    transformation_plan=_transform_plan(),
                    required_dataset_changes=None,
                ),
                TransformationResult(
                    transformation_plan=_transform_plan(),
                    required_dataset_changes=None,
                ),
            ],
        ) as transform_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.validate_data_compilation",
            side_effect=[
                DataCompilationValidationResult(
                    validation_errors=[repairable_issue],
                    user_suggestion_message="Validation found repairable transformation or encoding issues.",
                ),
                DataCompilationValidationResult(
                    validation_errors=[],
                    user_suggestion_message=None,
                ),
            ],
        ) as validate_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.compile_causal_spec_from_cleaned_summary",
            return_value=_causal_spec(),
        ) as compile_retry_mock,
    ):
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
    assert payload.phase == "REVIEW_READY"
    assert payload.validation_retry_count == 1
    assert orchestrator_state.get("working_dataset_id") == payload.compiled_dataset_id
    assert orchestrator_state.get("latest_dataset_summary") == payload.compiled_dataset_summary
    assert orchestrator_state.get("causal_spec") is None
    assert orchestrator_state.get("data_transformation_plan") is None
    assert orchestrator_state.get("working_dataset_frozen") is False
    assert orchestrator_state.get("is_validated") is False
    assert cleaning_mock.call_count == 1
    assert transform_mock.call_count == 2
    assert validate_mock.call_count == 2
    assert compile_retry_mock.call_count == 1
    assert len(data_repo.dataframes) == 2


def test_data_compilation_node_aborts_on_hard_validation_failure() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM()
    data_repo = _InMemoryDataRepo(dataframes={dataset_id: dataframe.copy()})
    node = DataCompilationNode(data_repo=data_repo, llm=llm, tools_factory=_tool_factory())
    orchestrator_state = _build_orchestrator_state(
        dataset_id=dataset_id,
        dataset_summary=dataset_summary,
    )
    hard_issue = ValidationIssueModel(
        severity="FAIL",
        message="Dataframe is missing columns referenced by the causal spec.",
        fix_hint="Restore the missing treatment, outcome, or adjustment columns before retrying.",
    )

    with (
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.cleaning",
            return_value=_cleaning_result(dataframe),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            return_value=TransformationResult(
                transformation_plan=_transform_plan(),
                required_dataset_changes=None,
            ),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.validate_data_compilation",
            return_value=DataCompilationValidationResult(
                validation_errors=[hard_issue],
                user_suggestion_message=None,
            ),
        ),
    ):
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
    assert result.status == "ABORTED"
    assert result.action == "NONE"
    assert payload.phase == "FAILED"
    assert payload.validation_retry_count == 0
    assert payload.compiled_dataset_id is not None


def test_data_compilation_node_review_confirm_publishes_outputs() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM(
        json_outputs=[
            {"assistant_message": "Detailed clinician review."},
            {"action": "confirm", "assistant_message": "Confirmed compiled setup."},
        ]
    )
    data_repo = _InMemoryDataRepo(dataframes={dataset_id: dataframe.copy()})
    node = DataCompilationNode(data_repo=data_repo, llm=llm, tools_factory=_tool_factory())
    orchestrator_state = _build_orchestrator_state(
        dataset_id=dataset_id,
        dataset_summary=dataset_summary,
    )

    with (
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.cleaning",
            return_value=_cleaning_result(dataframe),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            return_value=TransformationResult(
                transformation_plan=_transform_plan(),
                required_dataset_changes=None,
            ),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.validate_data_compilation",
            return_value=DataCompilationValidationResult(
                validation_errors=[],
                user_suggestion_message=None,
            ),
        ),
    ):
        first_result = node.run(
            request=NodeRequest(
                user_id=uuid4(),
                conversation_id=uuid4(),
                node_state=DataCompilationState.init_empty(),
                orchestrator_state=orchestrator_state,
                read_only_messages_history=[ChatMessage(role="user", content="compile it")],
            )
        )

        first_payload = first_result.new_node_state.payload
        second_result = node.run(
            request=NodeRequest(
                user_id=uuid4(),
                conversation_id=uuid4(),
                node_state=first_result.new_node_state,
                orchestrator_state=orchestrator_state,
                read_only_messages_history=[
                    ChatMessage(role="assistant", content=first_payload.assistant_message or ""),
                    ChatMessage(role="user", content="confirm"),
                ],
            )
        )

    assert first_result.status == "PENDING"
    assert first_result.action == "NEEDS_INPUT"
    assert first_payload.phase == "REVIEW_READY"
    assert orchestrator_state.get("working_dataset_id") == first_payload.compiled_dataset_id
    assert orchestrator_state.get("latest_dataset_summary") == first_payload.compiled_dataset_summary
    assert orchestrator_state.get("causal_spec") is None
    assert orchestrator_state.get("data_transformation_plan") is None
    assert orchestrator_state.get("working_dataset_frozen") is False
    assert orchestrator_state.get("is_validated") is False
    assert second_result.status == "DONE"
    assert second_result.action == "NONE"
    assert second_result.new_node_state.payload.phase == "CONFIRMED"
    assert orchestrator_state.get("working_dataset_id") == first_payload.compiled_dataset_id
    assert orchestrator_state.get("causal_spec") is not None
    assert orchestrator_state.get("data_transformation_plan") is not None
    assert orchestrator_state.get("working_dataset_frozen") is True
    assert orchestrator_state.get("is_validated") is True
    assert orchestrator_state.get("causal_spec_draft").model_dump(mode="json") == _causal_draft().model_dump(mode="json")


def test_data_compilation_node_review_revise_keeps_only_preaccept_dataset_refresh() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM(
        json_outputs=[
            {"assistant_message": "Detailed clinician review."},
            {"action": "revise", "assistant_message": "Please revise this compiled setup."},
        ]
    )
    data_repo = _InMemoryDataRepo(dataframes={dataset_id: dataframe.copy()})
    node = DataCompilationNode(data_repo=data_repo, llm=llm, tools_factory=_tool_factory())
    orchestrator_state = _build_orchestrator_state(
        dataset_id=dataset_id,
        dataset_summary=dataset_summary,
    )

    with (
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.cleaning",
            return_value=_cleaning_result(dataframe),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            return_value=TransformationResult(
                transformation_plan=_transform_plan(),
                required_dataset_changes=None,
            ),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.validate_data_compilation",
            return_value=DataCompilationValidationResult(
                validation_errors=[],
                user_suggestion_message=None,
            ),
        ),
    ):
        first_result = node.run(
            request=NodeRequest(
                user_id=uuid4(),
                conversation_id=uuid4(),
                node_state=DataCompilationState.init_empty(),
                orchestrator_state=orchestrator_state,
                read_only_messages_history=[ChatMessage(role="user", content="compile it")],
            )
        )
        first_payload = first_result.new_node_state.payload
        second_result = node.run(
            request=NodeRequest(
                user_id=uuid4(),
                conversation_id=uuid4(),
                node_state=first_result.new_node_state,
                orchestrator_state=orchestrator_state,
                read_only_messages_history=[
                    ChatMessage(role="assistant", content=first_payload.assistant_message or ""),
                    ChatMessage(role="user", content="revise it"),
                ],
            )
        )

    assert first_result.status == "PENDING"
    assert first_payload.phase == "REVIEW_READY"
    assert second_result.status == "ABORTED"
    assert second_result.new_node_state.payload.phase == "FAILED"
    assert orchestrator_state.get("working_dataset_id") == first_payload.compiled_dataset_id
    assert orchestrator_state.get("latest_dataset_summary") == first_payload.compiled_dataset_summary
    assert orchestrator_state.get("causal_spec") is None
    assert orchestrator_state.get("data_transformation_plan") is None
    assert orchestrator_state.get("working_dataset_frozen") is False
    assert orchestrator_state.get("is_validated") is False
