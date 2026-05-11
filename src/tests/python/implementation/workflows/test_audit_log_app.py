from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

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
    graph_payloads: dict[UUID, Any]
    artifact_payloads: dict[tuple[UUID, str, str], DataflowArtifactResponse] = field(
        default_factory=dict
    )
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
        key = (artifact_id, artifact_kind, artifact_format)
        if key in self.artifact_payloads:
            return self.artifact_payloads[key]
        payload = self.graph_payloads[artifact_id]
        return DataflowArtifactResponse(
            id=artifact_id,
            kind=artifact_kind,  # type: ignore[arg-type]
            format=artifact_format,  # type: ignore[arg-type]
            mime="application/json",
            content=json.dumps(payload).encode("utf-8"),
        )


def _render_html_for_graph_payload(payload: Any) -> str:
    user_id = uuid4()
    conversation_id = uuid4()
    graph_id = uuid4()
    conversation = Conversation(
        conversation_id=conversation_id,
        conversation_type="causal",
        name="Graph audit",
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
    dataflow = _FakeDataflowApp(graph_payloads={graph_id: payload})
    app = AuditLogApp(repo=repo, dataflow=dataflow)  # type: ignore[arg-type]
    return app.render_html(
        user_id=user_id,
        conversation_id=conversation_id,
        conversation_type="causal",
    )


def _audit_graph_specs(html: str) -> dict[str, dict[str, Any]]:
    match = re.search(r"const auditGraphSpecs = (.*?);\n", html)
    assert match is not None
    return json.loads(match.group(1))


def _simple_spec(*, title: str | None = None, mark: str = "bar") -> dict[str, Any]:
    spec: dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": mark,
        "data": {"values": [{"x": "a", "y": 1}]},
        "encoding": {
            "x": {"field": "x", "type": "nominal"},
            "y": {"field": "y", "type": "quantitative"},
        },
    }
    if title is not None:
        spec["title"] = title
    return spec


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
    assert "const titleDefaults = {anchor: 'start', color: '#0f766e'" in html
    assert "AUDIT_GRAPH_MIN_PLOT_WIDTH = 760" in html
    assert "fit-x" in html
    assert "continuousWidth: 1040" in html
    assert ".audit { max-width: 1680px;" in html
    assert ".artifacts { display: grid; gap: 14px; width: 100%; max-width: none;" in html
    assert ".graph { min-height: 420px; min-width: 760px; }" in html
    assert ".graph-card-header h3 { overflow-wrap: anywhere; color: var(--teal); }" in html
    assert ".graph .vega-embed .vega-actions a { color: var(--teal) !important; }" in html
    assert (
        ".graph .vega-embed summary svg path { fill: currentColor !important; stroke: currentColor !important; }"
        in html
    )
    assert ".graph svg, .graph canvas { display: block; max-width: none; }" in html
    assert '<div class="graph-card-header"><h3>ATE graph</h3></div>' in html
    assert (
        '<div class="graph-viewport"><div id="audit-graph-2-1-1" class="graph"></div></div>' in html
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


def test_audit_log_zip_packages_data_artifacts_and_rewrites_links() -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    dataset_id = uuid4()
    graph_id = uuid4()
    all_row_cate_id = uuid4()
    refutation_id = uuid4()
    refutation_vectors_id = uuid4()
    trained_model_id = uuid4()
    conversation = Conversation(
        conversation_id=conversation_id,
        conversation_type="causal",
        name="Packaged audit",
        last_updated_at_utc=1712345678.123,
    )
    repo = _FakeWorkflowRepo(
        conversation=conversation,
        orchestrator_state=_FakeOrchestratorState(
            payload={
                "working_dataset_ids": [str(dataset_id)],
                "working_dataset_frozen": True,
                "trained_model_id": str(trained_model_id),
                "all_row_cate_dataset_id": str(all_row_cate_id),
                "all_row_cate_summary": {"row_count": 2},
                "negative_control_refutation_artifact_id": str(refutation_id),
                "negative_control_refutation_vectors_dataset_id": str(refutation_vectors_id),
                "negative_control_refutation_summary": {"status": "COMPLETED"},
            }
        ),
        messages=[
            ChatMessage(
                role="assistant",
                content="packaged outputs",
                artifact_refs=[
                    {
                        "id": graph_id,
                        "kind": "data",
                        "format": "json",
                        "artifact_meta": {"kind": "chart_spec", "title": "Embedded chart"},
                    },
                    {
                        "id": all_row_cate_id,
                        "kind": "data",
                        "format": "csv",
                        "artifact_meta": {"title": "All row CATE"},
                    },
                ],
            )
        ],
    )
    dataflow = _FakeDataflowApp(
        graph_payloads={graph_id: _simple_spec(title="Embedded chart")},
        artifact_payloads={
            (dataset_id, "data", "csv"): DataflowArtifactResponse(
                id=dataset_id,
                kind="data",
                format="csv",
                mime="text/csv",
                content=b"id,outcome\n1,10\n",
            ),
            (all_row_cate_id, "data", "csv"): DataflowArtifactResponse(
                id=all_row_cate_id,
                kind="data",
                format="csv",
                mime="text/csv",
                content=b"id,cate\n1,0.2\n2,0.3\n",
            ),
            (refutation_id, "data", "json"): DataflowArtifactResponse(
                id=refutation_id,
                kind="data",
                format="json",
                mime="application/json",
                content=b'{"status":"COMPLETED"}',
            ),
            (refutation_vectors_id, "data", "csv"): DataflowArtifactResponse(
                id=refutation_vectors_id,
                kind="data",
                format="csv",
                mime="text/csv",
                content=b"id,primary_cate,negative_control_cate\n1,0.2,0.01\n",
            ),
        },
    )
    app = AuditLogApp(repo=repo, dataflow=dataflow)  # type: ignore[arg-type]

    zip_bytes = app.render_zip(
        user_id=user_id,
        conversation_id=conversation_id,
        conversation_type="causal",
    )

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = archive.namelist()
        html = archive.read("audit-log.html").decode("utf-8")
        assert archive.read(f"artifacts/data-{dataset_id}.csv") == b"id,outcome\n1,10\n"
        assert (
            archive.read(f"artifacts/data-{all_row_cate_id}.csv")
            == b"id,cate\n1,0.2\n2,0.3\n"
        )
        assert archive.read(f"artifacts/data-{refutation_id}.json") == b'{"status":"COMPLETED"}'
        assert (
            archive.read(f"artifacts/data-{refutation_vectors_id}.csv")
            == b"id,primary_cate,negative_control_cate\n1,0.2,0.01\n"
        )

    assert names == [
        "audit-log.html",
        f"artifacts/data-{dataset_id}.csv",
        f"artifacts/data-{all_row_cate_id}.csv",
        f"artifacts/data-{refutation_id}.json",
        f"artifacts/data-{refutation_vectors_id}.csv",
    ]
    assert f"artifacts/data-{dataset_id}.csv" in html
    assert f"artifacts/data-{all_row_cate_id}.csv" in html
    assert f"artifacts/data-{refutation_id}.json" in html
    assert f"artifacts/data-{refutation_vectors_id}.csv" in html
    assert f"/artifacts/{dataset_id}?artifact_kind=data" not in html
    assert f"/artifacts/{all_row_cate_id}?artifact_kind=data" not in html
    assert f"artifacts/data-{graph_id}.json" not in html
    assert "auditGraphSpecs" in html
    assert "https://cdn.jsdelivr.net/npm/vega-lite@5" in html
    assert str(trained_model_id) in html
    assert (
        "The trained model object is not included in this audit export. "
        "It can be exported separately on request."
    ) in html
    assert all(str(trained_model_id) not in name for name in names)

    fetched_ids = [call["artifact_id"] for call in dataflow.calls]
    assert fetched_ids.count(graph_id) == 1
    assert fetched_ids.count(all_row_cate_id) == 1
    assert trained_model_id not in fetched_ids


def test_audit_log_zip_fails_when_packaged_artifact_is_missing() -> None:
    user_id = uuid4()
    conversation_id = uuid4()
    dataset_id = uuid4()
    conversation = Conversation(
        conversation_id=conversation_id,
        conversation_type="causal",
        name="Incomplete package",
        last_updated_at_utc=1712345678.123,
    )
    repo = _FakeWorkflowRepo(
        conversation=conversation,
        orchestrator_state=_FakeOrchestratorState(
            payload={"working_dataset_ids": [str(dataset_id)]}
        ),
        messages=[],
    )
    dataflow = _FakeDataflowApp(graph_payloads={})
    app = AuditLogApp(repo=repo, dataflow=dataflow)  # type: ignore[arg-type]

    with pytest.raises(KeyError):
        app.render_zip(
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_type="causal",
        )


def test_audit_log_html_renders_raw_list_of_chart_specs_as_separate_cards() -> None:
    html = _render_html_for_graph_payload([_simple_spec(), _simple_spec(mark="point")])

    graph_specs = _audit_graph_specs(html)

    assert sorted(graph_specs) == ["audit-graph-1-1-1", "audit-graph-1-1-2"]
    assert '<div class="graph-card-header"><h3>Graph 1</h3></div>' in html
    assert '<div class="graph-card-header"><h3>Graph 2</h3></div>' in html
    assert graph_specs["audit-graph-1-1-1"]["mark"] == "bar"
    assert graph_specs["audit-graph-1-1-2"]["mark"] == "point"
    assert '<div class="graph-card graph-error">' not in html


def test_audit_log_html_renders_wrapped_chart_specs_with_wrapper_titles() -> None:
    html = _render_html_for_graph_payload(
        {
            "charts": [
                {"title": "Age distribution", "spec": _simple_spec()},
                {"title": "Outcome points", "spec": _simple_spec(mark="point")},
            ]
        }
    )

    graph_specs = _audit_graph_specs(html)

    assert sorted(graph_specs) == ["audit-graph-1-1-1", "audit-graph-1-1-2"]
    assert '<div class="graph-card-header"><h3>Age distribution</h3></div>' in html
    assert '<div class="graph-card-header"><h3>Outcome points</h3></div>' in html
    assert graph_specs["audit-graph-1-1-1"]["mark"] == "bar"
    assert graph_specs["audit-graph-1-1-2"]["mark"] == "point"


def test_audit_log_html_keeps_faceted_chart_as_single_renderable_graph() -> None:
    payload = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Forest Plot of Subgroup Treatment Effects (CATE)",
        "facet": {"row": {"field": "subgroup_category", "type": "nominal"}},
        "spec": {
            "layer": [
                {
                    "mark": {"type": "rule"},
                    "encoding": {
                        "x": {"field": "mean_cate_lower", "type": "quantitative"},
                        "x2": {"field": "mean_cate_upper"},
                        "y": {"field": "subgroup_value", "type": "nominal"},
                    },
                },
                {
                    "mark": {"type": "point", "filled": True},
                    "encoding": {
                        "x": {"field": "mean_cate", "type": "quantitative"},
                        "y": {"field": "subgroup_value", "type": "nominal"},
                    },
                },
            ]
        },
        "data": {
            "values": [
                {
                    "subgroup_category": "Overall",
                    "subgroup_value": "All Patients",
                    "mean_cate_lower": -0.15,
                    "mean_cate_upper": 0.22,
                    "mean_cate": 0.03,
                }
            ]
        },
    }

    html = _render_html_for_graph_payload(payload)
    graph_specs = _audit_graph_specs(html)

    assert sorted(graph_specs) == ["audit-graph-1-1-1"]
    assert (
        '<div class="graph-card-header"><h3>Forest Plot of Subgroup Treatment Effects (CATE)</h3></div>'
        in html
    )
    assert graph_specs["audit-graph-1-1-1"]["facet"] == payload["facet"]
    assert graph_specs["audit-graph-1-1-1"]["spec"] == payload["spec"]
    assert '<div class="graph-card graph-error">' not in html
    assert "function auditUsesCompositeLayout" in html
    assert "auditAutosizeType(prepared.autosize).startsWith('fit')" in html
    assert "auditApplyAutosize(prepared)" in html


def test_audit_log_html_keeps_valid_sibling_when_bundled_chart_spec_is_invalid() -> None:
    html = _render_html_for_graph_payload(
        [_simple_spec(title="Valid graph"), {"title": "Broken graph", "encoding": {}}]
    )

    graph_specs = _audit_graph_specs(html)

    assert sorted(graph_specs) == ["audit-graph-1-1-1"]
    assert '<div class="graph-card-header"><h3>Valid graph</h3></div>' in html
    assert '<div class="graph-card graph-error"><h3>Broken graph</h3>' in html
    assert "must define a Vega-Lite visual grammar" in html


def test_audit_log_html_splits_top_level_vconcat_and_inherits_root_spec_fields() -> None:
    payload = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": [{"x": "a", "y": 1}]},
        "config": {"view": {"stroke": None}},
        "vconcat": [
            {
                "title": "First panel",
                "mark": "bar",
                "encoding": {
                    "x": {"field": "x", "type": "nominal"},
                    "y": {"field": "y", "type": "quantitative"},
                },
            },
            {
                "mark": "point",
                "encoding": {
                    "x": {"field": "x", "type": "nominal"},
                    "y": {"field": "y", "type": "quantitative"},
                },
            },
        ],
    }

    html = _render_html_for_graph_payload(payload)
    graph_specs = _audit_graph_specs(html)

    assert sorted(graph_specs) == ["audit-graph-1-1-1", "audit-graph-1-1-2"]
    assert '<div class="graph-card-header"><h3>First panel</h3></div>' in html
    assert '<div class="graph-card-header"><h3>Graph 2</h3></div>' in html
    assert "vconcat" not in graph_specs["audit-graph-1-1-1"]
    assert "vconcat" not in graph_specs["audit-graph-1-1-2"]
    assert graph_specs["audit-graph-1-1-1"]["data"] == payload["data"]
    assert graph_specs["audit-graph-1-1-2"]["data"] == payload["data"]
    assert graph_specs["audit-graph-1-1-1"]["config"] == payload["config"]
    assert graph_specs["audit-graph-1-1-2"]["config"] == payload["config"]


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
