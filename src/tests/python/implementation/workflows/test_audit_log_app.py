from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from python.domain.models.models import ChatMessage
from python.domain.repo.workflow_state_repo import Conversation
from python.domain.workflows.ochestrator_state import OchestratorState
from python.implementation.workflows.audit_log_app import AuditLogApp
from python.implementation.workflows.dataflow_app import DataflowArtifactResponse


@dataclass
class _FakeOrchestratorState(OchestratorState):
    payload: dict[str, Any]
    current_node: str = "CAUSAL_INFERENCE"

    def name(self) -> str:
        return "FAKE_OCHESTRATOR"

    def get_update_counter(self) -> int:
        return 0

    def set_update_counter(self, value: int) -> None:
        del value

    def get(self, key: str) -> Any:
        return self.payload[key]

    def set(self, key: str, value: dict[str, Any]) -> None:
        del key
        self.payload.update(value)

    def get_current_node_name(self) -> str:
        return self.current_node

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
        dataset_ids = self.payload.get("working_dataset_ids") or []
        dataset_id = UUID(str(dataset_ids[-1])) if dataset_ids else None
        return dataset_id, bool(self.payload.get("working_dataset_frozen"))

    def get_ochestration_prompt(self) -> str:
        return ""

    def to_json_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> _FakeOrchestratorState:
        return cls(payload=dict(payload))

    @classmethod
    def init_empty(cls) -> _FakeOrchestratorState:
        return cls(payload={})


@dataclass
class _FakeWorkflowRepo:
    conversation: Conversation
    orchestrator_state: OchestratorState
    messages: list[ChatMessage]
    history_limits: list[int | None] = field(default_factory=list)

    def get_conversations(self, *, user_id: UUID) -> list[Conversation]:
        del user_id
        return [self.conversation]

    def load_ochestrator_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> OchestratorState | None:
        del user_id, conversation_id
        return self.orchestrator_state

    def load_message_history(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        limit: int | None = 20,
    ) -> list[ChatMessage]:
        del user_id, conversation_id
        self.history_limits.append(limit)
        return list(self.messages)


@dataclass
class _FakeDataflowApp:
    graph_payloads: dict[UUID, dict[str, Any]]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def get_artifact(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
        artifact_id: UUID,
        artifact_kind: str,
        artifact_format: str,
    ) -> DataflowArtifactResponse:
        self.calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "conversation_type": conversation_type,
                "artifact_id": artifact_id,
                "artifact_kind": artifact_kind,
                "artifact_format": artifact_format,
            }
        )
        payload = self.graph_payloads[artifact_id]
        return DataflowArtifactResponse(
            id=artifact_id,
            kind="data",
            format="json",
            mime="application/json",
            content=json.dumps(payload).encode("utf-8"),
        )


def test_audit_log_html_uses_full_history_escapes_text_and_renders_graph() -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    dataset_id = uuid4()
    graph_id = uuid4()
    csv_id = uuid4()
    conversation = Conversation(
        conversation_id=conversation_id,
        conversation_type="causal",
        name="Audit <Case>",
        last_updated_at_utc=1712345678.123,
    )
    repo = _FakeWorkflowRepo(
        conversation=conversation,
        orchestrator_state=_FakeOrchestratorState(
            payload={
                "working_dataset_ids": [str(dataset_id)],
                "latest_dataset_summary": {"n_rows": 4},
                "protocol_discussion": "Protocol <b>text</b>",
                "causal_spec": {"treatment": "drug"},
                "data_transformation_plan": {"columns": []},
                "selected_model": "econml.dml.CausalForestDML",
                "selection_reasoning": "Best fit",
                "trained_model_id": str(uuid4()),
                "training_warnings": ["warn <x>"],
                "training_spec": {
                    "fit": {"backend": "fake"},
                    "causal_spec": {"duplicated": True},
                    "transformation_plan": {"duplicated": True},
                },
                "training_error_message": None,
                "working_dataset_frozen": True,
            }
        ),
        messages=[
            ChatMessage(
                role="user",
                content="<script>alert(1)</script>",
                created_at_utc=1712345600.0,
            ),
            ChatMessage(
                role="assistant",
                content="# Here is a graph\n\n- Graph follows",
                created_at_utc=1712345700.0,
                artifact_refs=[
                    {
                        "id": graph_id,
                        "kind": "data",
                        "format": "json",
                        "artifact_meta": {"kind": "chart_spec", "title": "ATE graph"},
                    },
                    {
                        "id": csv_id,
                        "kind": "data",
                        "format": "csv",
                    },
                ],
            ),
        ],
    )
    graph_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "bar",
        "data": {"values": [{"x": "a", "y": 1}]},
        "encoding": {
            "x": {"field": "x", "type": "nominal"},
            "y": {"field": "y", "type": "quantitative"},
        },
    }
    dataflow = _FakeDataflowApp(graph_payloads={graph_id: graph_spec})
    app = AuditLogApp(repo=repo, dataflow=dataflow)  # type: ignore[arg-type]

    html = app.render_html(
        user_id=user_id,
        conversation_id=conversation_id,
        conversation_type="causal",
    )

    assert repo.history_limits == [None]
    assert "Audit &lt;Case&gt;" in html
    assert "Audit Summary" in html
    assert "Dataset Lineage" in html
    assert "Workflow Stage Truth" in html
    assert "Current Orchestration State" in html
    assert "Stage Evidence" in html
    assert "Message Timeline" in html
    assert "Chat Transcript" in html
    assert html.index("Message Timeline") < html.index("Workflow Stage Truth")
    assert "Appendix" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert '<h3 class="chat-heading chat-heading-1">Here is a graph</h3>' in html
    assert "<li>Graph follows</li>" in html
    assert "Protocol &lt;b&gt;text&lt;/b&gt;" in html
    assert "Model Training" in html
    assert "Recorded" in html
    assert "Warning" in html
    assert "Open CSV" in html
    assert f"/artifacts/{dataset_id}?artifact_kind=data&amp;artifact_format=csv" in html
    assert f"/artifacts/{csv_id}?artifact_kind=data&amp;artifact_format=csv" in html
    assert f"/artifacts/{graph_id}" not in html
    assert "https://cdn.jsdelivr.net/npm/vega-lite@5" in html
    assert "auditGraphSpecs" in html
    assert "auditPrepareGraphSpec" in html
    assert "auditApplyHorizontalComposition" in html
    assert "auditShouldTransposeCategoricalXAxis" in html
    assert "AUDIT_GRAPH_MIN_PLOT_WIDTH = 760" in html
    assert "fit-x" in html
    assert "continuousWidth: 1040" in html
    assert ".audit { max-width: 1680px;" in html
    assert ".artifacts { display: grid; gap: 14px; width: 100%; max-width: none;" in html
    assert ".graph { min-height: 420px; min-width: 760px; }" in html
    assert ".graph svg, .graph canvas { display: block; max-width: none; }" in html
    assert '<div class="graph-card-header"><h3>ATE graph</h3></div>' in html
    assert (
        '<div class="graph-viewport"><div id="audit-graph-2-1" class="graph"></div></div>' in html
    )
    assert "ATE graph" in html
    training_section = html.split("<h3>Model Training</h3>", maxsplit=1)[1].split(
        "<h3>Causal Inference</h3>", maxsplit=1
    )[0]
    assert "Fit Logs" in training_section
    assert "fake" in training_section
    assert "duplicated" not in training_section
    assert "causal_spec" not in training_section
    assert "transformation_plan" not in training_section
    assert dataflow.calls == [
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "conversation_type": "causal",
            "artifact_id": graph_id,
            "artifact_kind": "data",
            "artifact_format": "json",
        }
    ]


def test_audit_log_html_handles_missing_graph_without_failing_report() -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    graph_id = uuid4()
    conversation = Conversation(
        conversation_id=conversation_id,
        conversation_type="causal",
        name=None,
        last_updated_at_utc=1712345678.123,
    )
    repo = _FakeWorkflowRepo(
        conversation=conversation,
        orchestrator_state=_FakeOrchestratorState(payload={"working_dataset_ids": []}),
        messages=[
            ChatMessage(
                role="assistant",
                content="graph",
                artifact_refs=[
                    {
                        "id": graph_id,
                        "kind": "data",
                        "format": "json",
                        "artifact_meta": {"kind": "chart_spec"},
                    }
                ],
            )
        ],
    )
    dataflow = _FakeDataflowApp(graph_payloads={})
    app = AuditLogApp(repo=repo, dataflow=dataflow)  # type: ignore[arg-type]

    html = app.render_html(
        user_id=user_id,
        conversation_id=conversation_id,
        conversation_type="causal",
    )

    assert "Conversation Audit Log" in html
    assert "Dataset Lineage" in html
    assert "Stage Evidence" in html
    assert "Missing" in html
    assert "Not recorded" in html
    assert "Graph artifact could not be rendered" not in html
    assert "Graph Error" not in html
    assert "KeyError" in html or str(graph_id) in html
