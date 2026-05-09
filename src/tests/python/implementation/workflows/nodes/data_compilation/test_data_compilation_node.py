from __future__ import annotations

import json
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
    MissingnessDecision,
    MissingnessDecisionList,
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
            "id_col": "__rowid__",
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


def _transformation_suggestions() -> ColumnTransformationSuggestionList:
    return ColumnTransformationSuggestionList(
        suggestions=[
            ColumnTransformationSuggestion(
                column="age",
                role="covariate",
                preferred_type="NUMERIC",
                preferred_type_reason="Age is already stored as numeric values.",
            ),
            ColumnTransformationSuggestion(
                column="isex",
                role="effect_modifier",
                preferred_type="CATEGORICAL",
                preferred_type_reason="The numeric codes would be clearer as explicit category labels.",
            ),
        ]
    )


def _transformation_result(
    *,
    transformation_plan: TransformPlan | None = None,
    transformation_suggestions: ColumnTransformationSuggestionList | None = None,
) -> TransformationResult:
    suggestions = transformation_suggestions
    if transformation_plan is not None and suggestions is None:
        suggestions = _transformation_suggestions()
    return TransformationResult(
        transformation_plan=transformation_plan,
        transformation_suggestions=suggestions,
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


def _build_orchestrator_state(
    *,
    dataset_id: UUID,
    dataset_summary: Any,
    protocol_cleaning_instructions: str | None = "Normalize only grounded values.",
) -> CausalOchestratorState:
    state = CausalOchestratorState.init_empty()
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
            "protocol_cleaning_instructions": protocol_cleaning_instructions,
            "causal_spec_draft": _causal_draft(),
        },
    )
    return state


def _cleaning_result(
    dataframe: pd.DataFrame,
    *,
    cleaning_notes: tuple[str, ...] = (),
) -> CleaningResult:
    return CleaningResult(
        cleaned_data_summary=_build_summary(dataframe),
        pd_cleaned=dataframe.copy(),
        causal=_causal_spec(),
        missingness_decisions=_missingness_decisions(),
        cleaning_notes=cleaning_notes,
    )


def _missingness_decisions() -> MissingnessDecisionList:
    return MissingnessDecisionList(
        decisions=[
            MissingnessDecision(
                column="treatment",
                role="treatment",
                missing_count_before=0,
                resolution="none_needed",
                reason="Treatment is already complete.",
                instruction="No missingness action is required.",
                missing_count_after=0,
            ),
            MissingnessDecision(
                column="outcome",
                role="outcome",
                missing_count_before=0,
                resolution="none_needed",
                reason="Outcome is already complete.",
                instruction="No missingness action is required.",
                missing_count_after=0,
            ),
            MissingnessDecision(
                column="age",
                role="covariate",
                missing_count_before=0,
                resolution="none_needed",
                reason="Age is already complete.",
                instruction="No missingness action is required.",
                missing_count_after=0,
            ),
            MissingnessDecision(
                column="isex",
                role="effect_modifier",
                missing_count_before=0,
                resolution="none_needed",
                reason="Effect modifier is already complete.",
                instruction="No missingness action is required.",
                missing_count_after=0,
            ),
        ]
    )


def test_data_compilation_node_saves_transformation_suggestions_without_cleaning_retry() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM(
        json_outputs=[{"assistant_message": "Review the compiled setup with type recommendations."}]
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
        ) as cleaning_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            return_value=_transformation_result(transformation_plan=_transform_plan()),
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
    assert payload.compiled_dataset_id is not None
    assert payload.compiled_dataset_id in data_repo.dataframes
    assert payload.missingness_decisions is not None
    assert payload.transformation_suggestions is not None
    assert len(payload.transformation_suggestions.suggestions) == 2
    assert "preferred future raw type is CATEGORICAL" in " ".join(payload.compilation_warnings)
    assert not hasattr(payload, "transformation_retry_count")
    assert orchestrator_state.get("working_dataset_id") == dataset_id
    assert orchestrator_state.get("latest_dataset_summary") == dataset_summary
    assert (
        orchestrator_state.get("causal_spec_draft").model_dump(mode="json")
        == _causal_draft().model_dump(mode="json")
    )
    assert orchestrator_state.get("causal_spec") is None
    assert orchestrator_state.get("data_transformation_plan") is None
    assert orchestrator_state.get("working_dataset_frozen") is False
    assert orchestrator_state.get("is_validated") is False
    assert cleaning_mock.call_count == 1
    assert transform_mock.call_count == 1
    review_payload = json.loads(str(llm.generate_json_calls[0]["user_prompt"]))
    assert review_payload["missingness_decisions"]["decisions"][2]["column"] == "age"
    assert review_payload["transformation_suggestions"]["suggestions"][1]["preferred_type"] == (
        "CATEGORICAL"
    )


def test_data_compilation_node_reports_default_cleaning_when_instructions_are_absent() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM(
        json_outputs=[{"assistant_message": "Review the compiled setup."}]
    )
    data_repo = _InMemoryDataRepo(dataframes={dataset_id: dataframe.copy()})
    node = DataCompilationNode(data_repo=data_repo, llm=llm, tools_factory=_tool_factory())
    orchestrator_state = _build_orchestrator_state(
        dataset_id=dataset_id,
        dataset_summary=dataset_summary,
        protocol_cleaning_instructions=None,
    )

    with (
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.cleaning",
            return_value=_cleaning_result(dataframe),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            return_value=_transformation_result(transformation_plan=_transform_plan()),
        ),
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

    actions = result.new_node_state.payload.compilation_actions
    assert any(
        "No explicit protocol cleaning instructions were provided" in action
        for action in actions
    )


def test_data_compilation_node_reports_cleaning_contradiction_notes() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM(json_outputs=[{"assistant_message": "Review the compiled setup."}])
    data_repo = _InMemoryDataRepo(dataframes={dataset_id: dataframe.copy()})
    node = DataCompilationNode(data_repo=data_repo, llm=llm, tools_factory=_tool_factory())
    orchestrator_state = _build_orchestrator_state(
        dataset_id=dataset_id,
        dataset_summary=dataset_summary,
    )
    contradiction_note = (
        "Cleaning decision (missingness): Protocol discussion contradicted the cleaning "
        "instructions on treatment missingness, so the conservative protocol-safe "
        "interpretation dropped rows with missing treatment."
    )

    with (
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.cleaning",
            return_value=_cleaning_result(
                dataframe,
                cleaning_notes=(contradiction_note,),
            ),
        ),
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            return_value=_transformation_result(transformation_plan=_transform_plan()),
        ),
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

    assert contradiction_note in result.new_node_state.payload.compilation_actions


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
                _transformation_result(transformation_plan=_transform_plan()),
                _transformation_result(transformation_plan=_transform_plan()),
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
    assert orchestrator_state.get("working_dataset_id") == dataset_id
    assert orchestrator_state.get("latest_dataset_summary") == dataset_summary
    assert orchestrator_state.get("causal_spec") is None
    assert orchestrator_state.get("data_transformation_plan") is None
    assert orchestrator_state.get("working_dataset_frozen") is False
    assert orchestrator_state.get("is_validated") is False
    assert cleaning_mock.call_count == 1
    assert transform_mock.call_count == 2
    assert validate_mock.call_count == 2
    assert compile_retry_mock.call_count == 1
    assert len(data_repo.dataframes) == 2
    assert payload.transformation_suggestions is not None


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
            return_value=_transformation_result(transformation_plan=_transform_plan()),
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
    assert payload.hard_failure is True
    assert payload.validation_retry_count == 0
    assert payload.compiled_dataset_id is not None
    assert payload.system_message == "DATA_COMPILATION_HARD_FAILED"


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
            return_value=_transformation_result(transformation_plan=_transform_plan()),
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
        assert first_result.status == "PENDING"
        assert first_result.action == "NEEDS_INPUT"
        assert first_payload.phase == "REVIEW_READY"
        assert orchestrator_state.get("working_dataset_id") == dataset_id
        assert orchestrator_state.get("latest_dataset_summary") == dataset_summary
        assert orchestrator_state.get("causal_spec") is None
        assert orchestrator_state.get("data_transformation_plan") is None
        assert orchestrator_state.get("working_dataset_frozen") is False
        assert orchestrator_state.get("is_validated") is False

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

    assert second_result.status == "DONE"
    assert second_result.action == "NONE"
    assert second_result.new_node_state.payload.phase == "CONFIRMED"
    assert orchestrator_state.get("working_dataset_id") == first_payload.compiled_dataset_id
    assert orchestrator_state.get("causal_spec") is not None
    assert orchestrator_state.get("data_transformation_plan") is not None
    assert orchestrator_state.get("working_dataset_frozen") is True
    assert orchestrator_state.get("is_validated") is True
    assert orchestrator_state.get("causal_spec_draft").model_dump(mode="json") == _causal_draft().model_dump(mode="json")


def test_data_compilation_node_review_reject_aborts_and_leaves_upstream_state_unchanged() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM(
        json_outputs=[
            {"assistant_message": "Detailed clinician review."},
            {
                "action": "reject",
                "assistant_message": "I do not accept this setup. Please send me back.",
            },
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
            return_value=_transformation_result(transformation_plan=_transform_plan()),
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
                    ChatMessage(role="user", content="I do not accept this, take me back."),
                ],
            )
        )

    assert first_result.status == "PENDING"
    assert first_payload.phase == "REVIEW_READY"
    assert second_result.status == "ABORTED"
    assert second_result.new_node_state.payload.phase == "FAILED"
    assert second_result.new_node_state.payload.hard_failure is False
    assert second_result.new_node_state.payload.system_message == (
        "DATA_COMPILATION_REVISION_REQUESTED"
    )
    assert orchestrator_state.get("working_dataset_id") == dataset_id
    assert orchestrator_state.get("latest_dataset_summary") == dataset_summary
    assert orchestrator_state.get("causal_spec") is None
    assert orchestrator_state.get("data_transformation_plan") is None
    assert orchestrator_state.get("working_dataset_frozen") is False
    assert orchestrator_state.get("is_validated") is False


def test_data_compilation_node_review_question_reuses_cached_payload_without_recompile() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM(
        json_outputs=[
            {"assistant_message": "Detailed clinician review."},
            {
                "action": "answer_query",
                "assistant_message": "I can answer from the cached compiled payload.",
            },
            {"assistant_message": "The cleaned dataset is cached; no recompilation is needed."},
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
        ) as cleaning_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            return_value=_transformation_result(transformation_plan=_transform_plan()),
        ) as transform_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.validate_data_compilation",
            return_value=DataCompilationValidationResult(
                validation_errors=[],
                user_suggestion_message=None,
            ),
        ) as validate_mock,
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
                    ChatMessage(role="user", content="What changed in the compiled data?"),
                ],
            )
        )

    assert first_result.status == "PENDING"
    assert first_payload.phase == "REVIEW_READY"
    assert second_result.status == "PENDING"
    assert second_result.action == "NEEDS_INPUT"
    assert second_result.new_node_state.payload.phase == "REVIEW_READY"
    assert (
        second_result.new_node_state.payload.assistant_message
        == "The cleaned dataset is cached; no recompilation is needed."
    )
    assert len(llm.generate_json_calls) == 3
    decision_payload = json.loads(str(llm.generate_json_calls[1]["user_prompt"]))
    assert decision_payload["missingness_decisions"]["decisions"][0]["column"] == "treatment"
    answer_payload = json.loads(str(llm.generate_json_calls[2]["user_prompt"]))
    assert answer_payload["latest_user_message"] == "What changed in the compiled data?"
    assert cleaning_mock.call_count == 1
    assert transform_mock.call_count == 1
    assert validate_mock.call_count == 1
    assert data_repo.get_csv_data_calls == [dataset_id]
    assert orchestrator_state.get("working_dataset_id") == dataset_id
    assert orchestrator_state.get("latest_dataset_summary") == dataset_summary


def test_data_compilation_node_review_recompile_uses_original_source_dataset() -> None:
    dataframe = _build_dataframe()
    updated_dataframe = dataframe.copy()
    updated_dataframe["age"] = updated_dataframe["age"] + 5
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM(
        json_outputs=[
            {"assistant_message": "Detailed clinician review."},
            {
                "action": "recompile",
                "assistant_message": "I will recompile from the original dataset.",
                "recompile_request": "Reclean age without changing columns or roles.",
            },
            {"assistant_message": "Recompiled clinician review."},
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
            side_effect=[_cleaning_result(dataframe), _cleaning_result(updated_dataframe)],
        ) as cleaning_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            side_effect=[
                _transformation_result(transformation_plan=_transform_plan()),
                _transformation_result(transformation_plan=_transform_plan()),
            ],
        ) as transform_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.validate_data_compilation",
            side_effect=[
                DataCompilationValidationResult(
                    validation_errors=[],
                    user_suggestion_message=None,
                ),
                DataCompilationValidationResult(
                    validation_errors=[],
                    user_suggestion_message=None,
                ),
            ],
        ) as validate_mock,
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
                    ChatMessage(
                        role="user",
                        content="Please reclean age from the original dataset before I accept.",
                    ),
                ],
            )
        )

    assert first_payload.phase == "REVIEW_READY"
    assert second_result.status == "PENDING"
    assert second_result.action == "NEEDS_INPUT"
    assert second_result.new_node_state.payload.phase == "REVIEW_READY"
    assert second_result.new_node_state.payload.compiled_dataset_id != first_payload.compiled_dataset_id
    assert second_result.new_node_state.payload.assistant_message == "Recompiled clinician review."
    assert cleaning_mock.call_count == 2
    assert cleaning_mock.call_args_list[1].kwargs["review_recompile_request"] == (
        "Reclean age without changing columns or roles."
    )
    assert transform_mock.call_count == 2
    assert validate_mock.call_count == 2
    assert data_repo.get_csv_data_calls == [dataset_id, dataset_id]
    assert second_result.new_node_state.payload.missingness_decisions is not None
    assert any(
        "Applied a review-time recompilation request on the original working dataset"
        in action
        for action in second_result.new_node_state.payload.compilation_actions
    )


def test_data_compilation_node_recompiles_when_upstream_protocol_changes() -> None:
    dataframe = _build_dataframe()
    dataset_summary = _build_summary(dataframe)
    dataset_id = uuid4()
    llm = _FakeLLM(
        json_outputs=[
            {"assistant_message": "Detailed clinician review."},
            {"assistant_message": "Recompiled clinician review."},
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
        ) as cleaning_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.transform",
            return_value=_transformation_result(transformation_plan=_transform_plan()),
        ) as transform_mock,
        patch(
            "python.implementation.workflows.nodes.data_compilation.data_compilation_node.validate_data_compilation",
            return_value=DataCompilationValidationResult(
                validation_errors=[],
                user_suggestion_message=None,
            ),
        ) as validate_mock,
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

        orchestrator_state.set(
            ProtocolDiscussionState.NAME,
            {
                "protocol_discussion": "Updated confirmed protocol discussion.",
                "protocol_cleaning_instructions": "Normalize only grounded values.",
                "causal_spec_draft": _causal_draft(),
            },
        )

        second_result = node.run(
            request=NodeRequest(
                user_id=uuid4(),
                conversation_id=uuid4(),
                node_state=first_result.new_node_state,
                orchestrator_state=orchestrator_state,
                read_only_messages_history=[ChatMessage(role="user", content="compile it again")],
            )
        )

    assert first_result.new_node_state.payload.phase == "REVIEW_READY"
    assert second_result.status == "PENDING"
    assert second_result.action == "NEEDS_INPUT"
    assert second_result.new_node_state.payload.phase == "REVIEW_READY"
    assert "The active dataset or confirmed protocol changed" in (
        second_result.new_node_state.payload.assistant_message or ""
    )
    assert cleaning_mock.call_count == 2
    assert transform_mock.call_count == 2
    assert validate_mock.call_count == 2
    assert data_repo.get_csv_data_calls == [dataset_id, dataset_id]
