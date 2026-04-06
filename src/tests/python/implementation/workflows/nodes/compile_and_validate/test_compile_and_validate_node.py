from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
import pytest

from python.domain.models.errors import StateDependencyError
from python.domain.repo.data_repo import DataRepo
from python.domain.service.llm_service import ChatMessage, LLMConfig
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_deps import (
    CompileAndValidateDeps,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_node import (
    CompileAndValidateNode,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_prompts import (
    get_compile_and_validate_node_info,
    get_compile_causal_spec_prompt,
    get_compile_review_decision_prompt,
    get_compile_transformation_plan_prompt,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_state import (
    CompileAndValidatePayloadModel,
    CompileAndValidateState,
)
from python.implementation.workflows.nodes.dataset.dataset_state import (
    DatasetIterationModel,
    DatasetPayloadModel,
    DatasetState,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionPayloadModel,
    ProtocolDiscussionState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import (
    TransformPlan,
)
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
    rows: list[dict[str, object]] = []
    for index in range(48):
        rows.append(
            {
                "treatment": "drug" if index % 2 == 0 else "control",
                "outcome": float(index) / 10.0,
                "age": 40 + index,
                "sex": "F" if index % 3 == 0 else "M",
            }
        )
    return pd.DataFrame(rows)


def _build_summary(df: pd.DataFrame) -> DatasetSummaryModel:
    return DatasetProfilingTool().extract_dataset_summary(
        df,
        max_categories=10,
        sample_distinct=10,
        compute_quantiles=False,
        strict=True,
    )


def _dataset_state(*, dataset_id: UUID, summary: DatasetSummaryModel) -> DatasetState:
    return DatasetState(
        DatasetPayloadModel(
            dataset_iterations=[DatasetIterationModel(dataset_id=dataset_id)],
            latest_summary=summary,
        )
    )


def _protocol_state(
    *,
    dataset_id: UUID,
    summary: DatasetSummaryModel,
    discussion: str = "Confirmed protocol discussion",
    phase: str = "CONFIRMED",
) -> ProtocolDiscussionState:
    return ProtocolDiscussionState(
        ProtocolDiscussionPayloadModel(
            dataset_id=dataset_id,
            dataset_summary=summary,
            discussion=discussion,
            phase=phase,  # pyright: ignore[reportArgumentType]
            assistant_message="Confirmed discussion",
        )
    )


@dataclass
class _FakeLLM:
    json_outputs: list[object] = field(default_factory=list)
    generate_json_calls: list[dict[str, object]] = field(default_factory=list)

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
        try:
            return self.tool_by_name[name]
        except KeyError as exc:
            raise KeyError(name) from exc


def _tool_factory() -> _FakeToolFactory:
    return _FakeToolFactory(
        tool_by_name={
            CausalSpecsTool.NAME: CausalSpecsTool(),
            EncodingPlanTool.NAME: EncodingPlanTool(),
            ValidationBackdoorTool.NAME: ValidationBackdoorTool(),
        }
    )


def test_compile_and_validate_prompts_and_info_have_expected_scope() -> None:
    assert "causal specification" in get_compile_and_validate_node_info().lower()
    assert "dataset summary is authoritative" in get_compile_causal_spec_prompt().lower()
    assert "build the plan only for covariates and effect modifiers" in get_compile_transformation_plan_prompt().lower()
    assert "full meaning of the user reply" in get_compile_review_decision_prompt().lower()


def test_compile_and_validate_state_roundtrip_and_statuses() -> None:
    state = CompileAndValidateState.init_empty()
    assert state.status() == "PENDING"

    failed = CompileAndValidateState(
        CompileAndValidatePayloadModel(
            phase="FAILED",
            assistant_message="Blocked",
            system_message="Technical issue",
            error_message="boom",
        )
    )
    assert failed.status() == "ABORTED"
    assert failed.error() is not None

    restored = CompileAndValidateState.from_json_dict(failed.to_json_dict())
    assert restored.payload.model_dump(mode="json") == failed.payload.model_dump(mode="json")


def test_compile_and_validate_deps_require_confirmed_protocol_and_allow_cleaned_dataset_revision() -> None:
    df = _build_dataframe()
    summary = _build_summary(df)
    dataset_id = uuid4()

    deps = CompileAndValidateDeps.from_loaded(
        {
            DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary),
            ProtocolDiscussionState.NAME: _protocol_state(dataset_id=dataset_id, summary=summary),
        }
    )
    assert deps.dataset_id == dataset_id
    assert deps.protocol_discussion == "Confirmed protocol discussion"

    with pytest.raises(StateDependencyError):
        CompileAndValidateDeps.from_loaded(
            {
                DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary),
                ProtocolDiscussionState.NAME: _protocol_state(
                    dataset_id=dataset_id,
                    summary=summary,
                    phase="DISCUSSING",
                ),
            }
        )

    cleaned_dataset_id = uuid4()
    cleaned_deps = CompileAndValidateDeps.from_loaded(
        {
            DatasetState.NAME: _dataset_state(dataset_id=cleaned_dataset_id, summary=summary),
            ProtocolDiscussionState.NAME: _protocol_state(
                dataset_id=dataset_id,
                summary=summary,
            ),
        }
    )
    assert cleaned_deps.dataset_id == cleaned_dataset_id
    assert cleaned_deps.protocol_discussion == "Confirmed protocol discussion"


def test_compile_and_validate_node_compiles_and_waits_for_confirmation() -> None:
    df = _build_dataframe()
    summary = _build_summary(df)
    dataset_id = uuid4()
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
        ]
    )
    node = CompileAndValidateNode(
        llm=llm,
        data_repo=_FakeDataRepo(dataframe=df),
        tool_factory=_tool_factory(),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={
            DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary),
            ProtocolDiscussionState.NAME: _protocol_state(dataset_id=dataset_id, summary=summary),
        },
        messages_history=[ChatMessage(role="user", content="Yes, confirm the protocol discussion.")],
        state=CompileAndValidateState.init_empty(),
    )

    assert isinstance(result, CompileAndValidateState)
    assert result.payload.phase == "REVIEW_READY"
    assert result.status() == "PENDING"
    assert result.payload.compiled_causal_spec is not None
    assert result.payload.transformation_plan is not None
    assert result.payload.inference_ready_causal_spec is not None
    assert result.payload.system_message is None
    assert result.payload.assistant_message is not None
    assert "please confirm this compiled setup" in result.payload.assistant_message.lower()
    assert "age: num_standard" in result.payload.assistant_message
    assert len(llm.generate_json_calls) == 2


def test_compile_and_validate_node_confirmed_review_marks_done() -> None:
    df = _build_dataframe()
    summary = _build_summary(df)
    dataset_id = uuid4()
    compiled_state = CompileAndValidateState(
        CompileAndValidatePayloadModel(
            phase="REVIEW_READY",
            assistant_message="Please confirm this compiled setup.",
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
            ),
            inference_ready_causal_spec=None,
        )
    )
    node = CompileAndValidateNode(
        llm=_FakeLLM(
            json_outputs=[
                {
                    "action": "confirm",
                    "assistant_message": (
                        "The compiled causal specification, transformation plan, and validation "
                        "review are now confirmed. We can proceed with this setup."
                    ),
                }
            ]
        ),
        data_repo=_FakeDataRepo(dataframe=df),
        tool_factory=_tool_factory(),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={
            DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary),
            ProtocolDiscussionState.NAME: _protocol_state(dataset_id=dataset_id, summary=summary),
        },
        messages_history=[ChatMessage(role="user", content="Yes, I confirm this compiled setup.")],
        state=compiled_state,
    )

    assert result.payload.phase == "CONFIRMED"
    assert result.status() == "DONE"
    assert "now confirmed" in (result.payload.assistant_message or "")


def test_compile_and_validate_node_rejection_aborts_review() -> None:
    df = _build_dataframe()
    summary = _build_summary(df)
    dataset_id = uuid4()
    state = CompileAndValidateState(
        CompileAndValidatePayloadModel(
            phase="REVIEW_READY",
            assistant_message="Please confirm this compiled setup.",
        )
    )
    node = CompileAndValidateNode(
        llm=_FakeLLM(
            json_outputs=[
                {
                    "action": "revise",
                    "assistant_message": (
                        "The compiled protocol review was not confirmed. Please go back and revise "
                        "the protocol or dataset assumptions before we continue."
                    ),
                }
            ]
        ),
        data_repo=_FakeDataRepo(dataframe=df),
        tool_factory=_tool_factory(),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={
            DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary),
            ProtocolDiscussionState.NAME: _protocol_state(dataset_id=dataset_id, summary=summary),
        },
        messages_history=[ChatMessage(role="user", content="No, change the covariates.")],
        state=state,
    )

    assert result.payload.phase == "FAILED"
    assert result.status() == "ABORTED"
    assert result.payload.system_message is not None
    assert result.payload.system_message.startswith("COMPILE_AND_VALIDATE_BLOCKED")


def test_compile_and_validate_node_unclear_review_reply_stays_pending() -> None:
    df = _build_dataframe()
    summary = _build_summary(df)
    dataset_id = uuid4()
    state = CompileAndValidateState(
        CompileAndValidatePayloadModel(
            phase="REVIEW_READY",
            assistant_message="Please confirm this compiled setup.",
        )
    )
    node = CompileAndValidateNode(
        llm=_FakeLLM(
            json_outputs=[
                {
                    "action": "clarify",
                    "assistant_message": "Please tell me whether you want to confirm this setup or which part should change.",
                }
            ]
        ),
        data_repo=_FakeDataRepo(dataframe=df),
        tool_factory=_tool_factory(),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={
            DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary),
            ProtocolDiscussionState.NAME: _protocol_state(dataset_id=dataset_id, summary=summary),
        },
        messages_history=[ChatMessage(role="user", content="Hmm, maybe.")],
        state=state,
    )

    assert result.payload.phase == "REVIEW_READY"
    assert result.status() == "PENDING"
    assert result.payload.assistant_message == (
        "Please tell me whether you want to confirm this setup or which part should change."
    )


def test_compile_and_validate_node_spec_compile_failure_aborts() -> None:
    df = _build_dataframe()
    summary = _build_summary(df)
    dataset_id = uuid4()
    llm = _FakeLLM(json_outputs=[RuntimeError("spec compile exploded")])
    node = CompileAndValidateNode(
        llm=llm,
        data_repo=_FakeDataRepo(dataframe=df),
        tool_factory=_tool_factory(),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={
            DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary),
            ProtocolDiscussionState.NAME: _protocol_state(dataset_id=dataset_id, summary=summary),
        },
        messages_history=[ChatMessage(role="user", content="Proceed.")],
        state=CompileAndValidateState.init_empty(),
    )

    assert result.payload.phase == "FAILED"
    assert result.status() == "ABORTED"
    assert result.payload.error_message is not None
    assert "causal specification compilation failed" in result.payload.error_message


def test_compile_and_validate_node_validation_failures_abort() -> None:
    df = _build_dataframe()
    df.loc[0, "treatment"] = None
    summary = _build_summary(df)
    dataset_id = uuid4()
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
        ]
    )
    node = CompileAndValidateNode(
        llm=llm,
        data_repo=_FakeDataRepo(dataframe=df),
        tool_factory=_tool_factory(),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={
            DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary),
            ProtocolDiscussionState.NAME: _protocol_state(dataset_id=dataset_id, summary=summary),
        },
        messages_history=[ChatMessage(role="user", content="Proceed.")],
        state=CompileAndValidateState.init_empty(),
    )

    assert result.payload.phase == "FAILED"
    assert result.status() == "ABORTED"
    assert any(issue.severity == "FAIL" for issue in result.payload.validation_issues)
    assert "blocking issues" in (result.payload.assistant_message or "").lower()


def test_compile_and_validate_node_extra_columns_in_cleaned_dataset_abort() -> None:
    df = _build_dataframe()
    df["patient_id"] = [f"id-{i}" for i in range(len(df))]
    summary = _build_summary(df)
    dataset_id = uuid4()
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
        ]
    )
    node = CompileAndValidateNode(
        llm=llm,
        data_repo=_FakeDataRepo(dataframe=df),
        tool_factory=_tool_factory(),
    )

    result = node.run(
        user_id=uuid4(),
        conversation_id=uuid4(),
        previous_state_dependencies={
            DatasetState.NAME: _dataset_state(dataset_id=dataset_id, summary=summary),
            ProtocolDiscussionState.NAME: _protocol_state(dataset_id=dataset_id, summary=summary),
        },
        messages_history=[ChatMessage(role="user", content="Proceed.")],
        state=CompileAndValidateState.init_empty(),
    )

    assert result.payload.phase == "FAILED"
    assert result.status() == "ABORTED"
    assert result.payload.system_message is not None
    assert "extra_columns=['patient_id']" in result.payload.system_message
    assert result.payload.assistant_message is not None
    assert "Extra columns currently present: patient_id" in result.payload.assistant_message
    assert "tell me exactly which columns they are" in result.payload.assistant_message
    assert any(
        issue.message == "Cleaned dataset contains columns outside the confirmed protocol scope."
        for issue in result.payload.validation_issues
    )
    offending_issue = next(
        issue
        for issue in result.payload.validation_issues
        if issue.message == "Cleaned dataset contains columns outside the confirmed protocol scope."
    )
    assert offending_issue.evidence["extra_columns"] == ["patient_id"]
