from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any
from uuid import uuid4

from python.domain.service.llm_service import ChatMessage, LLMConfig, LLMResponse
from python.domain.workflows.node import NodeRequest
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import (
    ProtocolDiscussionCausalDraftResult,
    ProtocolDiscussionNode,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import (
    ProtocolDiscussionState,
)
from python.implementation.workflows.tools.common.model.data_summary import (
    CategoricalColumnProfileModel,
    CategoricalSummaryModel,
    CategoryCountModel,
    DatasetSummaryModel,
    NumericColumnProfileModel,
    NumericSummaryModel,
)


def _categorical_profile(
    name: str,
    *,
    distinct_count: int = 2,
    n_missing: int = 0,
) -> CategoricalColumnProfileModel:
    return CategoricalColumnProfileModel(
        name=name,
        dtype="object",
        n_rows=10,
        n_missing=n_missing,
        missing_rate=n_missing / 10,
        distinct_count=distinct_count,
        inferred_kind="CATEGORICAL",
        summary=CategoricalSummaryModel(
            top_categories=[
                CategoryCountModel(value=f"value_{index}", count=1)
                for index in range(distinct_count)
            ],
            other_count=0,
        ),
    )


def _numeric_profile(
    name: str,
    *,
    distinct_count: int = 5,
    n_missing: int = 0,
) -> NumericColumnProfileModel:
    return NumericColumnProfileModel(
        name=name,
        dtype="float64",
        n_rows=10,
        n_missing=n_missing,
        missing_rate=n_missing / 10,
        distinct_count=distinct_count,
        inferred_kind="NUMERIC",
        summary=NumericSummaryModel(min=0, max=10, mean=5, std=1),
    )


def _summary(*profiles: Any) -> DatasetSummaryModel:
    return DatasetSummaryModel(n_rows=10, profiles=list(profiles))


def _install_fake_pandas(monkeypatch: Any) -> None:
    pandas_module = ModuleType("pandas")
    pandas_module.DataFrame = object
    monkeypatch.setitem(sys.modules, "pandas", pandas_module)


class _FakeOrchestratorState:
    def __init__(self, *, dataset_summary: DatasetSummaryModel) -> None:
        self.dataset_id = uuid4()
        self.dataset_summary = dataset_summary
        self.set_calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, key: str) -> Any:
        return {
            "working_dataset_id": self.dataset_id,
            "latest_dataset_summary": self.dataset_summary,
        }.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.set_calls.append((key, value))


@dataclass
class _FakeLLM:
    json_outputs: list[Any] = field(default_factory=list)
    text_outputs: list[str] = field(default_factory=list)
    generate_json_calls: list[dict[str, Any]] = field(default_factory=list)
    generate_calls: list[dict[str, Any]] = field(default_factory=list)

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
        output = self.json_outputs.pop(0)
        if isinstance(output, dict):
            return schema.model_validate(output)
        return output

    def generate(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        config: LLMConfig,
        history: list[ChatMessage] | None,
    ) -> LLMResponse:
        self.generate_calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "config": config,
                "history": history,
            }
        )
        return LLMResponse(content=self.text_outputs.pop(0))


def _request(
    *,
    dataset_summary: DatasetSummaryModel,
    llm: _FakeLLM,
    node: ProtocolDiscussionNode | None = None,
    messages: list[ChatMessage] | None = None,
) -> tuple[ProtocolDiscussionNode, NodeRequest, _FakeOrchestratorState]:
    orchestrator_state = _FakeOrchestratorState(dataset_summary=dataset_summary)
    request = NodeRequest(
        user_id=uuid4(),
        conversation_id=uuid4(),
        node_state=ProtocolDiscussionState.init_empty(),
        orchestrator_state=orchestrator_state,  # type: ignore[arg-type]
        read_only_messages_history=(
            messages
            if messages is not None
            else [ChatMessage(role="user", content="Use RXASP as treatment and DIED as outcome.")]
        ),
    )
    return node or ProtocolDiscussionNode(llm=llm), request, orchestrator_state


def _draft_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "id_col": "auto_id",
        "treatment_column": "RXASP",
        "outcome_column": "DIED",
        "negative_control_outcome": None,
        "covariates": ["AGE"],
        "effect_modifiers": ["SEX"],
        "target_population": "all rows",
        "study_type": "OBSERVATIONAL",
        "time_zero": "baseline treatment decision",
    }
    payload.update(overrides)
    return payload


def test_protocol_discussion_starts_with_welcome_without_llm_call() -> None:
    llm = _FakeLLM()
    summary = _summary(_categorical_profile("RXASP"), _categorical_profile("DIED"))
    node, request, orchestrator_state = _request(
        dataset_summary=summary,
        llm=llm,
        messages=[],
    )

    result = node.run(request=request)

    assert result.status == "PENDING"
    assert result.action == "NEEDS_INPUT"
    assert "Welcome" in result.response_messages[0].content
    assert llm.generate_json_calls == []
    assert llm.generate_calls == []
    assert orchestrator_state.set_calls == []


def test_protocol_discussion_updates_string_and_uses_plain_text_response() -> None:
    llm = _FakeLLM(
        json_outputs=[
            {"status": "DISCUSSING"}
        ],
        text_outputs=[
            "PROTOCOL DISCUSSION\nQ1: Treatment\nA: RXASP\nSource: user",
            "I captured RXASP. What outcome should we use?",
        ],
    )
    summary = _summary(_categorical_profile("RXASP"), _categorical_profile("DIED"))
    messages = [
        ChatMessage(role="user", content=f"message {index}")
        for index in range(6)
    ]
    node, request, orchestrator_state = _request(
        dataset_summary=summary,
        llm=llm,
        messages=messages,
    )

    result = node.run(request=request)
    compile_payload = json.loads(llm.generate_calls[0]["user_prompt"])
    status_payload = json.loads(llm.generate_json_calls[0]["user_prompt"])
    response_payload = json.loads(llm.generate_calls[1]["user_prompt"])

    assert result.status == "PENDING"
    assert result.action == "NEEDS_INPUT"
    assert result.new_node_state.payload.protocol_discussion.startswith("PROTOCOL DISCUSSION")
    assert result.new_node_state.payload.status == "DISCUSSING"
    assert len(compile_payload["recent_messages"]) == 5
    assert status_payload["protocol_discussion"].startswith("PROTOCOL DISCUSSION")
    assert "protocol_discussion" in response_payload
    assert len(llm.generate_json_calls) == 1
    assert len(llm.generate_calls) == 2
    assert orchestrator_state.set_calls == []


def test_ready_valid_binary_outcome_writes_causal_spec_draft(monkeypatch: Any) -> None:
    _install_fake_pandas(monkeypatch)
    llm = _FakeLLM(
        json_outputs=[
            {"status": "READY"},
            _draft_payload(negative_control_outcome="NEGCTRL"),
        ],
        text_outputs=["PROTOCOL DISCUSSION", "The protocol is ready."],
    )
    summary = _summary(
        _categorical_profile("RXASP"),
        _categorical_profile("DIED"),
        _numeric_profile("AGE"),
        _categorical_profile("SEX"),
        _categorical_profile("NEGCTRL"),
    )
    node, request, orchestrator_state = _request(dataset_summary=summary, llm=llm)

    result = node.run(request=request)

    assert result.status == "DONE"
    assert result.action == "NONE"
    assert len(orchestrator_state.set_calls) == 1
    _, payload = orchestrator_state.set_calls[0]
    assert list(payload.keys()) == ["causal_spec_draft"]
    assert payload["causal_spec_draft"].treatment_column == "RXASP"
    assert payload["causal_spec_draft"].outcome_column == "DIED"
    assert payload["causal_spec_draft"].negative_control_outcome == "NEGCTRL"


def test_causal_draft_accepts_continuous_numeric_outcome(monkeypatch: Any) -> None:
    _install_fake_pandas(monkeypatch)
    llm = _FakeLLM(json_outputs=[_draft_payload(outcome_column="SCORE")])
    summary = _summary(
        _categorical_profile("RXASP"),
        _numeric_profile("SCORE", distinct_count=8),
        _numeric_profile("AGE"),
        _categorical_profile("SEX"),
    )

    result = ProtocolDiscussionNode(llm=llm).protocol_discussion_causal_draft(
        protocol_discussion="PROTOCOL DISCUSSION",
        dataset_summary=summary,
    )

    assert result.validation_issues == []
    assert result.draft.outcome_column == "SCORE"


def test_ready_invalid_treatment_cardinality_blocks_and_suggests_update_dataset(
    monkeypatch: Any,
) -> None:
    _install_fake_pandas(monkeypatch)
    llm = _FakeLLM(
        json_outputs=[
            {"status": "READY"},
            _draft_payload(),
        ],
        text_outputs=[
            "PROTOCOL DISCUSSION",
            "update dataset and create a cleaned binary treatment column from RXASP",
        ],
    )
    summary = _summary(
        _categorical_profile("RXASP", distinct_count=3),
        _categorical_profile("DIED"),
        _numeric_profile("AGE"),
        _categorical_profile("SEX"),
    )
    node, request, orchestrator_state = _request(dataset_summary=summary, llm=llm)

    result = node.run(request=request)

    assert result.status == "PENDING"
    assert result.action == "NEEDS_INPUT"
    assert result.new_node_state.payload.status == "DISCUSSING"
    assert result.response_messages[0].content.startswith("update dataset")
    assert orchestrator_state.set_calls == []
    suggestion_payload = json.loads(llm.generate_calls[1]["user_prompt"])
    assert suggestion_payload["validation_issues"][0]["role"] == "treatment"


def test_invalid_categorical_outcome_cardinality_is_validation_issue(
    monkeypatch: Any,
) -> None:
    _install_fake_pandas(monkeypatch)
    llm = _FakeLLM(json_outputs=[_draft_payload(outcome_column="DIED")])
    summary = _summary(
        _categorical_profile("RXASP"),
        _categorical_profile("DIED", distinct_count=4),
        _numeric_profile("AGE"),
        _categorical_profile("SEX"),
    )

    result = ProtocolDiscussionNode(llm=llm).protocol_discussion_causal_draft(
        protocol_discussion="PROTOCOL DISCUSSION",
        dataset_summary=summary,
    )

    assert result.draft is not None
    assert any(issue["role"] == "outcome" for issue in result.validation_issues)


def test_negative_control_and_missingness_are_validation_issues(monkeypatch: Any) -> None:
    _install_fake_pandas(monkeypatch)
    llm = _FakeLLM(
        json_outputs=[
            _draft_payload(
                id_col="PATIENT_ID",
                negative_control_outcome="NEGCTRL",
            )
        ]
    )
    summary = _summary(
        _categorical_profile("RXASP", n_missing=1),
        _categorical_profile("DIED", n_missing=2),
        _numeric_profile("AGE"),
        _categorical_profile("SEX"),
        _categorical_profile("PATIENT_ID", distinct_count=10, n_missing=1),
        _categorical_profile("NEGCTRL", distinct_count=3, n_missing=1),
    )

    result = ProtocolDiscussionNode(llm=llm).protocol_discussion_causal_draft(
        protocol_discussion="PROTOCOL DISCUSSION",
        dataset_summary=summary,
    )

    roles = [issue["role"] for issue in result.validation_issues]
    assert "treatment" in roles
    assert "outcome" in roles
    assert "id" in roles
    assert "negative_control_outcome" in roles
    assert all(
        suggestion.startswith("update dataset")
        for issue in result.validation_issues
        for suggestion in issue["suggestions"]
    )


def test_auto_id_skips_id_existence_and_missingness_validation(monkeypatch: Any) -> None:
    _install_fake_pandas(monkeypatch)
    llm = _FakeLLM(json_outputs=[_draft_payload(id_col="auto_id")])
    summary = _summary(
        _categorical_profile("RXASP"),
        _categorical_profile("DIED"),
        _numeric_profile("AGE"),
        _categorical_profile("SEX"),
    )

    result = ProtocolDiscussionNode(llm=llm).protocol_discussion_causal_draft(
        protocol_discussion="PROTOCOL DISCUSSION",
        dataset_summary=summary,
    )

    assert result.validation_issues == []
