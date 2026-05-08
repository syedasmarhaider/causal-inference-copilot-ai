from __future__ import annotations

import html
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode
from uuid import UUID

from python.domain.models.errors import ConversationNotFoundError
from python.domain.models.models import ArtifactRef, ChatMessage
from python.domain.repo.workflow_state_repo import Conversation, ConversationType, WorkflowStateRepo
from python.domain.workflows.ochestrator_state import OchestratorState
from python.implementation.workflows.dataflow_app import DataflowApp
from python.implementation.workflows.ochestrator.causal_ochestrator_state import (
    CausalOchestratorState,
)


@dataclass(frozen=True)
class AuditGraph:
    element_id: str
    title: str
    spec: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class AuditArtifact:
    label: str
    href: str | None
    graph: AuditGraph | None = None


@dataclass(frozen=True)
class AuditHtml:
    html: str


@dataclass(frozen=True)
class AuditMessage:
    role: str
    content: str
    created_at_utc: float
    artifacts: list[AuditArtifact] = field(default_factory=list)


@dataclass(frozen=True)
class AuditReport:
    conversation: Conversation
    generated_at_utc: float
    current_stage_name: str
    completed_and_current_nodes: list[str]
    current_companion_nodes: list[str]
    forward_nodes_after_current: list[str]
    current_dataset_id: UUID | None
    is_dataset_frozen: bool
    orchestrator_state_name: str
    orchestrator_update_counter: int | None
    orchestrator_payload: dict[str, Any]
    messages: list[AuditMessage]


class AuditLogApp:
    def __init__(
        self,
        *,
        repo: WorkflowStateRepo,
        dataflow: DataflowApp,
    ) -> None:
        self._repo = repo
        self._dataflow = dataflow

    def render_html(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: ConversationType,
    ) -> str:
        conversation = self._get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
        )
        orchestrator_state = self._repo.load_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if orchestrator_state is None:
            orchestrator_state = (
                CausalOchestratorState.init_empty()
                if conversation_type == "causal"
                else self._empty_data_orchestrator_state()
            )

        messages = self._repo.load_message_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=None,
        )
        current_stage_name = orchestrator_state.get_current_node_name()
        current_dataset_id, is_dataset_frozen = (
            orchestrator_state.get_working_dataset_id_and_frozen_status()
        )
        report = AuditReport(
            conversation=conversation,
            generated_at_utc=datetime.now(UTC).timestamp(),
            current_stage_name=current_stage_name,
            completed_and_current_nodes=_safe_state_list(
                lambda: orchestrator_state.get_completed_and_last_pending_nodes()
            ),
            current_companion_nodes=_safe_state_list(
                lambda: orchestrator_state.get_current_node_companion_names(current_stage_name)
            ),
            forward_nodes_after_current=_safe_state_list(
                lambda: orchestrator_state.get_forward_states_after_node(current_stage_name)
            ),
            current_dataset_id=current_dataset_id,
            is_dataset_frozen=is_dataset_frozen,
            orchestrator_state_name=orchestrator_state.name(),
            orchestrator_update_counter=_safe_state_int(
                lambda: orchestrator_state.get_update_counter()
            ),
            orchestrator_payload=orchestrator_state.to_json_dict(),
            messages=self._build_messages(
                user_id=user_id,
                conversation=conversation,
                messages=messages,
            ),
        )
        return AuditLogHtmlRenderer().render(report)

    def _get_conversation(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: ConversationType,
    ) -> Conversation:
        for conversation in self._repo.get_conversations(user_id=user_id):
            if (
                conversation.conversation_id == conversation_id
                and conversation.conversation_type == conversation_type
            ):
                return conversation
        raise ConversationNotFoundError(user_id=user_id, conversation_id=conversation_id)

    @staticmethod
    def _empty_data_orchestrator_state() -> OchestratorState:
        from python.implementation.workflows.ochestrator.data_ochestrator_state import (
            DataOchestratorState,
        )

        return DataOchestratorState.init_empty()

    def _build_messages(
        self,
        *,
        user_id: UUID,
        conversation: Conversation,
        messages: Sequence[ChatMessage],
    ) -> list[AuditMessage]:
        audit_messages: list[AuditMessage] = []
        graph_counter = 0

        for msg_index, message in enumerate(messages, start=1):
            artifacts: list[AuditArtifact] = []
            for artifact_ref in message.artifact_refs or ():
                if _is_graph_artifact_ref(artifact_ref):
                    graph_counter += 1
                    artifacts.append(
                        AuditArtifact(
                            label=_artifact_label(artifact_ref, default=f"Graph {graph_counter}"),
                            href=None,
                            graph=self._load_graph(
                                user_id=user_id,
                                conversation=conversation,
                                artifact_ref=artifact_ref,
                                element_id=f"audit-graph-{msg_index}-{graph_counter}",
                            ),
                        )
                    )
                    continue

                artifacts.append(
                    AuditArtifact(
                        label=_artifact_label(artifact_ref, default="Artifact"),
                        href=_artifact_href(
                            conversation_id=conversation.conversation_id,
                            conversation_type=conversation.conversation_type,
                            artifact_ref=artifact_ref,
                        ),
                    )
                )

            audit_messages.append(
                AuditMessage(
                    role=message.role,
                    content=message.content,
                    created_at_utc=message.created_at_utc,
                    artifacts=artifacts,
                )
            )

        return audit_messages

    def _load_graph(
        self,
        *,
        user_id: UUID,
        conversation: Conversation,
        artifact_ref: ArtifactRef,
        element_id: str,
    ) -> AuditGraph:
        title = _artifact_label(artifact_ref, default="Graph")
        try:
            artifact = self._dataflow.get_artifact(
                user_id=user_id,
                conversation_id=conversation.conversation_id,
                conversation_type=conversation.conversation_type,
                artifact_id=artifact_ref["id"],
                artifact_kind=artifact_ref["kind"],
                artifact_format=artifact_ref["format"],
            )
            payload = json.loads(artifact.content.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Graph artifact JSON must be an object.")
            return AuditGraph(element_id=element_id, title=title, spec=payload)
        except Exception as exc:
            return AuditGraph(element_id=element_id, title=title, error=str(exc))


class AuditLogHtmlRenderer:
    def render(self, report: AuditReport) -> str:
        graph_specs = self._graph_specs(report)
        return "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f"<title>{_e(_title(report))}</title>",
                self._style(),
                '<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>',
                '<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>',
                '<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>',
                "</head>",
                "<body>",
                '<main class="audit">',
                self._header(report),
                self.render_metric_strip(self.summarize_report(report)),
                self._dataset_lineage(report),
                self._stage_truth(report),
                self._orchestration_truth(report),
                self._stage_evidence(report),
                self._messages(report),
                self._appendix(report),
                "</main>",
                self._graph_script(graph_specs),
                "</body>",
                "</html>",
            ]
        )

    @staticmethod
    def _graph_specs(report: AuditReport) -> dict[str, dict[str, Any]]:
        return {
            graph.element_id: graph.spec
            for message in report.messages
            for artifact in message.artifacts
            if artifact.graph is not None and (graph := artifact.graph).spec is not None
        }

    def _header(self, report: AuditReport) -> str:
        conversation = report.conversation
        current_dataset_link = (
            _dataset_csv_link(
                conversation_id=conversation.conversation_id,
                conversation_type=conversation.conversation_type,
                dataset_id=report.current_dataset_id,
            )
            if report.current_dataset_id is not None
            else "None"
        )
        rows = [
            ("Conversation Name", conversation.name or ""),
            ("Conversation Type", conversation.conversation_type),
            ("Conversation ID", str(conversation.conversation_id)),
            ("Generated At", _format_ts(report.generated_at_utc)),
            ("Current Stage", report.current_stage_name),
            ("Current Dataset", current_dataset_link),
            ("Dataset Frozen", str(report.is_dataset_frozen)),
        ]
        return (
            '<section class="hero">'
            "<h1>Conversation Audit Log</h1>"
            '<p class="hero-subtitle">Exported workflow evidence for review, traceability, and handoff.</p>'
            f"<dl>{''.join(_definition_row(k, v, raw_value=k == 'Current Dataset') for k, v in rows)}</dl>"
            "</section>"
        )

    def summarize_report(self, report: AuditReport) -> dict[str, str]:
        payload = report.orchestrator_payload
        dataset_ids = _coerce_uuid_list(report.orchestrator_payload.get("working_dataset_ids"))
        graph_count = sum(
            1
            for message in report.messages
            for artifact in message.artifacts
            if artifact.graph is not None
        )
        data_artifact_count = sum(
            1
            for message in report.messages
            for artifact in message.artifacts
            if artifact.href is not None
        )
        return {
            "Messages": str(len(report.messages)),
            "Dataset Versions": str(len(dataset_ids)),
            "Graphs": str(graph_count),
            "Linked Data Artifacts": str(data_artifact_count),
            "Current Dataset": (
                str(report.current_dataset_id) if report.current_dataset_id else "None"
            ),
            "Selected Model": str(payload.get("selected_model") or "Not recorded"),
            "Trained Model": str(payload.get("trained_model_id") or "Not recorded"),
        }

    def render_metric_strip(self, metrics: dict[str, str]) -> str:
        items = [
            '<div class="metric">'
            f"<span>{_e(label)}</span>"
            f"<strong>{_e(value)}</strong>"
            "</div>"
            for label, value in metrics.items()
        ]
        return (
            f'<section><h2>Audit Summary</h2><div class="metrics">{"".join(items)}</div></section>'
        )

    def _dataset_lineage(self, report: AuditReport) -> str:
        dataset_ids = _coerce_uuid_list(report.orchestrator_payload.get("working_dataset_ids"))
        if not dataset_ids:
            body = '<p class="empty">No dataset versions were recorded.</p>'
        else:
            rows = []
            for index, dataset_id in enumerate(dataset_ids, start=1):
                label = _dataset_version_label(
                    index=index,
                    total=len(dataset_ids),
                    dataset_id=dataset_id,
                    current_dataset_id=report.current_dataset_id,
                )
                link = _dataset_csv_link(
                    conversation_id=report.conversation.conversation_id,
                    conversation_type=report.conversation.conversation_type,
                    dataset_id=dataset_id,
                )
                rows.append(
                    "<tr>"
                    f"<td>{index}</td>"
                    f"<td>{_e(label)}</td>"
                    f"<td><code>{_e(str(dataset_id))}</code></td>"
                    f"<td>{link}</td>"
                    "</tr>"
                )
            body = (
                "<table>"
                "<thead><tr><th>Version</th><th>Label</th><th>Dataset ID</th><th>CSV</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody>"
                "</table>"
            )
        return f"<section><h2>Dataset Lineage</h2>{body}</section>"

    def _stage_truth(self, report: AuditReport) -> str:
        nodes = _canonical_stage_nodes(
            report.conversation.conversation_type,
            current_node=report.current_stage_name,
            recorded_nodes=report.completed_and_current_nodes,
        )
        current_key = _normalize_state_key(report.current_stage_name)
        current_index = next(
            (
                index
                for index, node in enumerate(nodes)
                if _normalize_state_key(node) == current_key
            ),
            None,
        )

        steps = []
        for index, node in enumerate(nodes):
            if current_index is None:
                status = (
                    "current"
                    if _normalize_state_key(node) == current_key
                    else "recorded" if node in report.completed_and_current_nodes else "pending"
                )
            elif index < current_index:
                status = "complete"
            elif index == current_index:
                status = "current"
            else:
                status = "pending"
            steps.append(self._stage_truth_step(index=index, node=node, status=status))

        return (
            '<section class="section-block">'
            '<p class="section-kicker">Stage truth</p>'
            "<h2>Workflow Stage Truth</h2>"
            '<div class="stage-truth">'
            f"{''.join(steps)}"
            "</div>"
            "</section>"
        )

    @staticmethod
    def _stage_truth_step(*, index: int, node: str, status: str) -> str:
        return (
            f'<article class="truth-step truth-step-{_attr(status)}">'
            '<div class="truth-step-index">'
            f"{index + 1}"
            "</div>"
            '<div class="truth-step-copy">'
            f"<strong>{_e(_friendly_stage_label(node))}</strong>"
            f"<code>{_e(node)}</code>"
            "</div>"
            f'<span class="truth-status truth-status-{_attr(status)}">'
            f"{_e(status.title())}"
            "</span>"
            "</article>"
        )

    def _orchestration_truth(self, report: AuditReport) -> str:
        facts = [
            ("State Source", report.orchestrator_state_name),
            ("Current Node", report.current_stage_name),
            (
                "Update Counter",
                (
                    str(report.orchestrator_update_counter)
                    if report.orchestrator_update_counter is not None
                    else "Not recorded"
                ),
            ),
            ("Recorded Path", report.completed_and_current_nodes),
            ("Companion Nodes", report.current_companion_nodes),
            ("Forward Nodes After Current", report.forward_nodes_after_current),
        ]
        cards = [
            '<div class="truth-card">'
            f"<span>{_e(label)}</span>"
            f"<strong>{self._truth_value(value)}</strong>"
            "</div>"
            for label, value in facts
        ]
        return (
            '<section class="section-block">'
            '<p class="section-kicker">Orchestration truth</p>'
            "<h2>Current Orchestration State</h2>"
            f'<div class="truth-grid">{"".join(cards)}</div>'
            "</section>"
        )

    @staticmethod
    def _truth_value(value: Any) -> str:
        if isinstance(value, list):
            if not value:
                return '<span class="muted">None</span>'
            return " ".join(f'<code class="truth-code">{_e(str(item))}</code>' for item in value)
        return _e(str(value))

    def _messages(self, report: AuditReport) -> str:
        if not report.messages:
            return (
                '<section class="section-block">'
                '<p class="section-kicker">Message Timeline</p>'
                "<h2>Chat Transcript</h2>"
                '<p class="empty">No messages recorded.</p>'
                "</section>"
            )

        rendered = []
        for message in report.messages:
            artifact_html = self._message_artifacts(message)
            role_key = _role_key(message.role)
            role_label = _role_label(message.role)
            rendered.append(
                f'<article class="chat-message chat-message-{_attr(role_key)}">'
                f'<div class="chat-avatar chat-avatar-{_attr(role_key)}">'
                f"{_e(_role_avatar(message.role))}"
                "</div>"
                '<div class="chat-body">'
                '<div class="chat-meta">'
                f"<span>{_e(role_label)}</span>"
                f"<time>{_e(_format_ts(message.created_at_utc))}</time>"
                "</div>"
                '<div class="chat-bubble">'
                f"{_render_chat_markdown(message.content)}"
                "</div>"
                f"{artifact_html}"
                "</div>"
                "</article>"
            )
        return (
            '<section class="section-block">'
            '<p class="section-kicker">Message Timeline</p>'
            "<h2>Chat Transcript</h2>"
            f'<div class="chat-thread">{"".join(rendered)}</div>'
            "</section>"
        )

    def _message_artifacts(self, message: AuditMessage) -> str:
        if not message.artifacts:
            return ""
        graph_items: list[str] = []
        link_items: list[str] = []
        for artifact in message.artifacts:
            if artifact.graph is not None:
                graph = artifact.graph
                if graph.spec is None:
                    graph_items.append(
                        '<div class="graph-card graph-error">'
                        f"<h3>{_e(graph.title)}</h3>"
                        f"<p>{_e(graph.error or 'Graph artifact could not be rendered.')}</p>"
                        "</div>"
                    )
                else:
                    graph_items.append(
                        '<div class="graph-card">'
                        '<div class="graph-card-header">'
                        f"<h3>{_e(graph.title)}</h3>"
                        "</div>"
                        '<div class="graph-viewport">'
                        f'<div id="{_e(graph.element_id)}" class="graph"></div>'
                        "</div>"
                        "</div>"
                    )
                continue

            if artifact.href is not None:
                link_items.append(
                    f'<a class="artifact-chip" href="{_attr(artifact.href)}">'
                    f"{_e(artifact.label)}"
                    "</a>"
                )
        if not graph_items and not link_items:
            return ""
        links_html = (
            f'<div class="artifact-chips">{"".join(link_items)}</div>' if link_items else ""
        )
        return f'<div class="artifacts">{links_html}{"".join(graph_items)}</div>'

    def _stage_evidence(self, report: AuditReport) -> str:
        payload = report.orchestrator_payload
        cards = [
            self.render_stage_card(
                "Dataset Stage",
                {
                    "Current Dataset": (
                        str(report.current_dataset_id)
                        if report.current_dataset_id is not None
                        else None
                    ),
                    "Dataset Versions": payload.get("working_dataset_ids"),
                    "Latest Dataset Summary": payload.get("latest_dataset_summary"),
                },
            ),
        ]
        if report.conversation.conversation_type == "causal":
            cards.extend(
                [
                    self.render_stage_card(
                        "Protocol Discussion",
                        {
                            "Final Protocol": payload.get("protocol_discussion"),
                            "Cleaning Instructions": payload.get("protocol_cleaning_instructions"),
                            "Causal Spec Draft": payload.get("causal_spec_draft"),
                        },
                    ),
                    self.render_stage_card(
                        "Data Compilation",
                        {
                            "Compiled Causal Spec": payload.get("causal_spec"),
                            "Transformation Plan": payload.get("data_transformation_plan"),
                            "Validation Issues": payload.get("validation_issues"),
                            "Is Validated": payload.get("is_validated"),
                            "Working Dataset Frozen": payload.get("working_dataset_frozen"),
                        },
                    ),
                    self.render_stage_card(
                        "Model Selection",
                        {
                            "Selected Model": payload.get("selected_model"),
                            "Selection Reasoning": payload.get("selection_reasoning"),
                        },
                    ),
                    self.render_stage_card(
                        "Model Training",
                        {
                            "Trained Model ID": payload.get("trained_model_id"),
                            "Training Warnings": payload.get("training_warnings"),
                            "Fit Logs": _training_fit_logs(payload.get("training_spec")),
                            "All Row CATE Dataset": _dataset_link_or_value(
                                conversation_id=report.conversation.conversation_id,
                                conversation_type=report.conversation.conversation_type,
                                value=payload.get("all_row_cate_dataset_id"),
                            ),
                            "All Row CATE Summary": payload.get("all_row_cate_summary"),
                            "Negative Control Refutation Artifact": payload.get(
                                "negative_control_refutation_artifact_id"
                            ),
                            "Negative Control Refutation Vectors Dataset": _dataset_link_or_value(
                                conversation_id=report.conversation.conversation_id,
                                conversation_type=report.conversation.conversation_type,
                                value=payload.get("negative_control_refutation_vectors_dataset_id"),
                            ),
                            "Negative Control Refutation Summary": payload.get(
                                "negative_control_refutation_summary"
                            ),
                            "Training Error": payload.get("training_error_message"),
                        },
                    ),
                    self.render_stage_card(
                        "Causal Inference",
                        {
                            "Evidence Location": (
                                "Inference outputs are shown in the message timeline "
                                "through assistant messages and rendered graph artifacts."
                            )
                        },
                    ),
                ]
            )
        return f'<section><h2>Stage Evidence</h2><div class="stage-grid">{"".join(cards)}</div></section>'

    def render_stage_card(self, title: str, values: dict[str, Any]) -> str:
        status = _stage_status(values)
        status_badge = self.render_status_badge(status)
        parts: list[str] = []
        for label, value in values.items():
            parts.append(self._stage_field(label, value))
        return (
            '<article class="stage-card">'
            '<div class="stage-card-header">'
            f"<h3>{_e(title)}</h3>"
            f"{status_badge}"
            "</div>"
            f"{''.join(parts)}"
            "</article>"
        )

    @staticmethod
    def render_status_badge(status: str) -> str:
        return f'<span class="status status-{_attr(status.lower())}">{_e(status)}</span>'

    def _stage_field(self, label: str, value: Any) -> str:
        return (
            '<div class="stage-field">'
            f"<h4>{_e(label)}</h4>"
            f"{self._render_field_value(label, value)}"
            "</div>"
        )

    def _render_field_value(self, label: str, value: Any) -> str:
        if _is_missing(value):
            return '<p class="empty">Not recorded.</p>'
        if isinstance(value, AuditHtml):
            return f"<p>{value.html}</p>"
        if isinstance(value, (str, int, float, bool)):
            return f"<p>{_e(str(value))}</p>"
        return self.render_json_details(label, value)

    def render_json_details(self, label: str, value: Any, *, open_details: bool = False) -> str:
        open_attr = " open" if open_details else ""
        return (
            f"<details{open_attr}>"
            f"<summary>{_e(label)}</summary>"
            f"<pre>{_json_pre(value)}</pre>"
            "</details>"
        )

    def _appendix(self, report: AuditReport) -> str:
        state_name = (
            '<div class="appendix-meta">'
            "<span>State Name</span>"
            f"<strong>{_e(report.orchestrator_state_name)}</strong>"
            "</div>"
        )
        return (
            "<section>"
            "<h2>Appendix</h2>"
            f"{state_name}"
            f"{self.render_json_details('Raw Orchestration State', report.orchestrator_payload)}"
            "</section>"
        )

    def _graph_script(self, graph_specs: dict[str, dict[str, Any]]) -> str:
        safe_json = _safe_script_json(graph_specs)
        return (
            "<script>"
            f"const auditGraphSpecs = {safe_json};"
            "function auditClone(value) {"
            "return typeof structuredClone === 'function' ? structuredClone(value) : JSON.parse(JSON.stringify(value));"
            "}"
            "function auditPlainObject(value) {"
            "return value && typeof value === 'object' && !Array.isArray(value);"
            "}"
            "function auditLooksLikeVegaLite(spec) {"
            "const schema = String(spec && spec.$schema || '');"
            "return schema.includes('vega-lite') || 'mark' in spec || 'encoding' in spec || 'layer' in spec || 'facet' in spec || 'hconcat' in spec || 'vconcat' in spec || 'repeat' in spec;"
            "}"
            "function auditPrepareGraphSpec(spec) {"
            "if (!auditPlainObject(spec)) return spec;"
            "const prepared = auditClone(spec);"
            "if (!auditLooksLikeVegaLite(prepared)) return prepared;"
            "if (!prepared.autosize) prepared.autosize = {type: 'fit-x', contains: 'padding'};"
            "if (!('width' in prepared) || (typeof prepared.width === 'number' && prepared.width < 640)) prepared.width = 'container';"
            "if (!('height' in prepared) || (typeof prepared.height === 'number' && prepared.height < 320)) prepared.height = 380;"
            "prepared.config = Object.assign({}, prepared.config || {});"
            "prepared.config.view = Object.assign({stroke: null, continuousWidth: 1040, continuousHeight: 380}, prepared.config.view || {});"
            "prepared.config.axis = Object.assign({labelColor: '#475569', titleColor: '#334155', gridColor: '#e7edf3', labelFontSize: 12, titleFontSize: 12}, prepared.config.axis || {});"
            "prepared.config.legend = Object.assign({labelColor: '#475569', titleColor: '#334155', labelFontSize: 12, titleFontSize: 12}, prepared.config.legend || {});"
            "return prepared;"
            "}"
            "for (const [id, spec] of Object.entries(auditGraphSpecs)) {"
            "const target = document.getElementById(id);"
            "if (!target) continue;"
            "vegaEmbed(target, auditPrepareGraphSpec(spec), {actions: {export: true, source: false, compiled: false, editor: false}, renderer: 'svg'}).catch((error) => {"
            "target.classList.add('graph-error');"
            "target.textContent = `Graph render failed: ${error.message || error}`;"
            "});"
            "}"
            "</script>"
        )

    @staticmethod
    def _style() -> str:
        return """
<style>
:root { color-scheme: light; --border: #d8dee4; --border-soft: #e7edf3; --muted: #64748b; --soft: #f8fafc; --soft-2: #eef7f4; --ink: #0f172a; --ink-2: #334155; --primary: #2563eb; --teal: #0f766e; --ok: #15803d; --warn: #a16207; --error: #b91c1c; }
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: linear-gradient(180deg, #f8fbfd 0%, #ffffff 220px); }
.audit { max-width: 1440px; margin: 0 auto; padding: 36px 32px 72px; }
.hero { border: 1px solid var(--border-soft); border-radius: 8px; padding: 24px; background: #fff; box-shadow: 0 18px 44px rgba(15, 23, 42, 0.07); }
h1 { font-size: 30px; line-height: 1.12; margin: 0 0 8px; letter-spacing: 0; }
.hero-subtitle { color: var(--muted); margin: 0 0 20px; max-width: 720px; }
h2 { font-size: 20px; margin: 0 0 14px; letter-spacing: 0; }
h3 { font-size: 15px; margin: 0; color: var(--ink); }
h4 { font-size: 12px; margin: 0 0 6px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
.section-block, section { margin-top: 28px; }
.section-kicker { margin: 0 0 6px; color: var(--teal); font-size: 11px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
dl { display: grid; grid-template-columns: 180px 1fr; gap: 8px 18px; margin: 0; }
dt { color: var(--muted); font-weight: 700; }
dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
table { width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 8px; }
th, td { text-align: left; padding: 10px 11px; border: 1px solid var(--border-soft); vertical-align: top; }
th { background: var(--soft); color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { margin: 0; padding: 12px; overflow: auto; background: var(--soft); border: 1px solid var(--border-soft); border-radius: 8px; max-height: 520px; }
a { color: var(--primary); text-decoration: none; }
a:hover { text-decoration: underline; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.metric, .truth-card, .stage-card { border: 1px solid var(--border-soft); border-radius: 8px; padding: 14px; background: #fff; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04); }
.metric span, .truth-card span { display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.metric strong, .truth-card strong { display: block; margin-top: 6px; overflow-wrap: anywhere; }
.truth-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }
.truth-code { display: inline-block; margin: 4px 4px 0 0; padding: 4px 6px; border-radius: 6px; background: var(--soft); color: var(--ink-2); font-size: 12px; }
.stage-truth { display: grid; gap: 10px; }
.truth-step { display: grid; grid-template-columns: 36px 1fr auto; gap: 12px; align-items: center; border: 1px solid var(--border-soft); border-radius: 8px; padding: 12px; background: #fff; }
.truth-step-index { display: flex; width: 28px; height: 28px; align-items: center; justify-content: center; border-radius: 999px; background: var(--soft); color: var(--muted); font-size: 12px; font-weight: 800; }
.truth-step-copy { display: grid; gap: 2px; min-width: 0; }
.truth-step-copy code { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
.truth-status, .status { border-radius: 999px; padding: 3px 9px; font-size: 12px; font-weight: 800; border: 1px solid var(--border-soft); white-space: nowrap; }
.truth-step-complete .truth-step-index, .truth-status-complete, .status-recorded { color: var(--ok); background: #dcfce7; border-color: #bbf7d0; }
.truth-step-current .truth-step-index, .truth-status-current { color: #1d4ed8; background: #dbeafe; border-color: #bfdbfe; }
.truth-status-pending, .truth-status-recorded, .status-missing { color: var(--muted); background: var(--soft); }
.status-warning { color: var(--warn); background: #fef9c3; border-color: #fde68a; }
.status-error { color: var(--error); background: #fee2e2; border-color: #fecaca; }
.stage-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }
.stage-card-header { display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 12px; }
.chat-thread { display: grid; gap: 20px; padding: 22px; border: 1px solid var(--border-soft); border-radius: 8px; background: linear-gradient(180deg, #f8fafc, #ffffff); }
.chat-message { display: flex; align-items: flex-start; gap: 12px; width: min(100%, 1180px); }
.chat-message-assistant { width: 100%; }
.chat-message-user { margin-left: auto; flex-direction: row-reverse; }
.chat-message-system { max-width: 760px; margin: 0 auto; }
.chat-avatar { display: flex; width: 36px; height: 36px; flex: 0 0 36px; align-items: center; justify-content: center; border-radius: 8px; color: #fff; font-size: 11px; font-weight: 800; background: linear-gradient(145deg, var(--primary), #0f172a); box-shadow: 0 8px 20px rgba(15, 23, 42, 0.14); }
.chat-avatar-user { background: linear-gradient(145deg, #0f766e, #134e4a); }
.chat-avatar-system { background: #64748b; }
.chat-body { min-width: 0; max-width: min(100%, 900px); }
.chat-message-assistant .chat-body { width: 100%; max-width: none; }
.chat-message-user .chat-body { display: flex; flex-direction: column; align-items: flex-end; }
.chat-meta { display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; margin: 0 0 5px; color: var(--muted); font-size: 11px; }
.chat-meta span { font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
.chat-bubble { max-width: 780px; overflow: hidden; border: 1px solid var(--border-soft); border-radius: 8px; padding: 13px 15px; background: #fff; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05); overflow-wrap: anywhere; }
.chat-message-assistant .chat-bubble { max-width: 820px; }
.chat-message-user .chat-bubble { background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 64%, #0f766e 100%); color: #fff; border-color: transparent; box-shadow: 0 12px 26px rgba(29, 78, 216, 0.22); }
.chat-message-system .chat-bubble { background: var(--soft); color: var(--ink-2); }
.chat-bubble p, .chat-bubble ul, .chat-bubble ol, .chat-bubble blockquote { margin: 0 0 10px; }
.chat-bubble p:last-child, .chat-bubble ul:last-child, .chat-bubble ol:last-child, .chat-bubble pre:last-child { margin-bottom: 0; }
.chat-bubble ul, .chat-bubble ol { padding-left: 22px; }
.chat-bubble li { margin: 4px 0; }
.chat-heading { margin: 2px 0 8px; line-height: 1.25; }
.chat-heading-1 { font-size: 17px; }
.chat-heading-2 { font-size: 15px; }
.chat-heading-3 { font-size: 14px; color: var(--ink-2); }
.chat-message-user .chat-heading-3 { color: rgba(255,255,255,.88); }
.chat-bubble blockquote { border-left: 3px solid rgba(37, 99, 235, .28); padding: 3px 0 3px 10px; color: var(--ink-2); }
.chat-message-user blockquote { border-left-color: rgba(255,255,255,.42); color: rgba(255,255,255,.88); }
.chat-bubble code { border-radius: 6px; padding: 2px 5px; background: rgba(15, 23, 42, .06); font-size: 12px; }
.chat-message-user code { background: rgba(255,255,255,.18); color: #fff; }
.chat-bubble pre { margin: 10px 0; white-space: pre; }
.artifacts { width: min(100%, 1120px); margin-top: 14px; }
.chat-message-assistant .artifacts { width: min(100%, 1180px); }
.artifact-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.artifact-chip { display: inline-flex; align-items: center; border: 1px solid var(--border-soft); border-radius: 999px; background: #fff; padding: 6px 10px; font-size: 12px; font-weight: 800; }
.graph-card { width: 100%; margin-top: 16px; overflow: hidden; border: 1px solid #dbe5ef; border-radius: 8px; background: #fff; box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08); }
.graph-card-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 13px 16px; border-bottom: 1px solid var(--border-soft); background: linear-gradient(180deg, #ffffff, #f8fafc); }
.graph-card-header h3 { overflow-wrap: anywhere; }
.graph-viewport { width: 100%; overflow-x: auto; padding: 18px; background: #fff; }
.graph { min-height: 380px; min-width: 360px; }
.graph > .vega-embed { width: 100%; }
.graph svg, .graph canvas { max-width: 100%; }
.graph-error { color: var(--error); }
.stage-field { margin: 12px 0; }
.stage-field p { margin: 0; overflow-wrap: anywhere; }
.appendix-meta { display: grid; grid-template-columns: 180px 1fr; gap: 12px; margin-bottom: 12px; }
.appendix-meta span, .muted { color: var(--muted); font-weight: 600; }
details { margin-top: 6px; }
summary { cursor: pointer; color: var(--primary); font-weight: 700; margin-bottom: 8px; }
.empty { color: var(--muted); margin: 0; }
@media (max-width: 720px) { .audit { padding: 20px 14px 44px; } .hero { padding: 18px; } dl, .appendix-meta { grid-template-columns: 1fr; } .truth-step { grid-template-columns: 32px 1fr; } .truth-status { grid-column: 2; justify-self: start; } .chat-thread { padding: 12px; } .chat-message, .chat-message-user { width: 100%; max-width: 100%; } .chat-avatar { display: none; } .chat-body, .chat-message-assistant .chat-body { max-width: 100%; } .chat-bubble, .chat-message-assistant .chat-bubble { max-width: 100%; } .artifacts, .chat-message-assistant .artifacts { width: 100%; } .graph-viewport { padding: 10px; } .graph { min-height: 300px; } }
@media print { body { background: #fff; } .audit { max-width: none; padding: 16px; } .hero, .metric, .truth-card, .stage-card, .chat-bubble, .graph-card { box-shadow: none; } pre { max-height: none; } a { color: inherit; } }
</style>
""".strip()


def _title(report: AuditReport) -> str:
    name = report.conversation.name or str(report.conversation.conversation_id)
    return f"Audit Log - {name}"


def _definition_row(key: str, value: str, *, raw_value: bool = False) -> str:
    rendered_value = value if raw_value else _e(value)
    return f"<dt>{_e(key)}</dt><dd>{rendered_value}</dd>"


def _format_ts(value: float) -> str:
    return datetime.fromtimestamp(value, tz=UTC).isoformat()


def _dataset_version_label(
    *,
    index: int,
    total: int,
    dataset_id: UUID,
    current_dataset_id: UUID | None,
) -> str:
    is_current = current_dataset_id is not None and dataset_id == current_dataset_id
    if total == 1:
        return "Initial / Current" if is_current else "Initial"
    if index == 1:
        return "Initial"
    if is_current or index == total:
        return "Current"
    return f"Intermediate {index}"


_CAUSAL_STAGE_NODES = [
    "DATA_MANUPULATION",
    "PROTOCOL_DISCUSSION",
    "DATA_COMPILATION",
    "MODEL_SELECTION",
    "MODEL_TRAIN",
    "CAUSAL_INFERENCE",
]
_DATA_STAGE_NODES = ["DATA_MANUPULATION"]
_STAGE_LABELS = {
    "DATA_MANUPULATION": "Dataset",
    "PROTOCOL_DISCUSSION": "Protocol discussion",
    "DATA_COMPILATION": "Data compilation",
    "MODEL_SELECTION": "Model selection",
    "MODEL_TRAIN": "Model training",
    "CAUSAL_INFERENCE": "Causal inference",
}


def _safe_state_list(producer: Callable[[], Sequence[Any]]) -> list[str]:
    try:
        return [str(item) for item in producer()]
    except Exception:
        return []


def _safe_state_int(producer: Callable[[], int]) -> int | None:
    try:
        return int(producer())
    except Exception:
        return None


def _canonical_stage_nodes(
    conversation_type: ConversationType,
    *,
    current_node: str,
    recorded_nodes: Sequence[str],
) -> list[str]:
    base = list(_CAUSAL_STAGE_NODES if conversation_type == "causal" else _DATA_STAGE_NODES)
    normalized = {_normalize_state_key(node) for node in base}
    for node in [*recorded_nodes, current_node]:
        key = _normalize_state_key(node)
        if key and key not in normalized:
            base.append(str(node))
            normalized.add(key)
    return base


def _normalize_state_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").upper()


def _friendly_stage_label(value: str) -> str:
    key = _normalize_state_key(value)
    if key in _STAGE_LABELS:
        return _STAGE_LABELS[key]
    if not key:
        return "Workflow step"
    return " ".join(part.capitalize() for part in key.split("_"))


def _role_key(value: str) -> str:
    normalized = _normalize_state_key(value).lower()
    if normalized in {"assistant", "ai", "agent"}:
        return "assistant"
    if normalized == "user":
        return "user"
    return "system"


def _role_label(value: str) -> str:
    key = _role_key(value)
    if key == "assistant":
        return "Agent"
    if key == "user":
        return "User"
    return str(value or "System").title()


def _role_avatar(value: str) -> str:
    key = _role_key(value)
    if key == "assistant":
        return "AI"
    if key == "user":
        return "You"
    return "Sys"


def _render_chat_markdown(content: str) -> str:
    lines = str(content or "").splitlines()
    html_parts: list[str] = []
    paragraph_lines: list[str] = []
    quote_lines: list[str] = []
    list_tag: str | None = None
    list_items: list[str] = []
    code_lines: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        if paragraph_lines:
            text = " ".join(line.strip() for line in paragraph_lines).strip()
            if text:
                html_parts.append(f"<p>{_render_inline_markdown(text)}</p>")
            paragraph_lines.clear()

    def flush_quote() -> None:
        if quote_lines:
            body = " ".join(line.strip() for line in quote_lines).strip()
            if body:
                html_parts.append(
                    f"<blockquote><p>{_render_inline_markdown(body)}</p></blockquote>"
                )
            quote_lines.clear()

    def flush_list() -> None:
        nonlocal list_tag
        if list_tag and list_items:
            items = "".join(f"<li>{_render_inline_markdown(item)}</li>" for item in list_items)
            html_parts.append(f"<{list_tag}>{items}</{list_tag}>")
        list_tag = None
        list_items.clear()

    def flush_open_blocks() -> None:
        flush_paragraph()
        flush_quote()
        flush_list()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                html_parts.append(f"<pre><code>{_e(chr(10).join(code_lines))}</code></pre>")
                code_lines.clear()
                in_code = False
            else:
                flush_open_blocks()
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            flush_open_blocks()
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            flush_open_blocks()
            level = len(heading.group(1))
            html_parts.append(
                f'<h3 class="chat-heading chat-heading-{level}">'
                f"{_render_inline_markdown(heading.group(2).strip())}"
                "</h3>"
            )
            continue

        unordered = re.match(r"^[-*+]\s+(.+)$", stripped)
        ordered = re.match(r"^\d+[\.)]\s+(.+)$", stripped)
        if unordered or ordered:
            flush_paragraph()
            flush_quote()
            next_tag = "ul" if unordered else "ol"
            if list_tag and list_tag != next_tag:
                flush_list()
            list_tag = next_tag
            list_items.append((unordered or ordered).group(1).strip())
            continue

        quote = re.match(r"^>\s?(.*)$", stripped)
        if quote:
            flush_paragraph()
            flush_list()
            quote_lines.append(quote.group(1))
            continue

        flush_quote()
        flush_list()
        paragraph_lines.append(line)

    if in_code:
        html_parts.append(f"<pre><code>{_e(chr(10).join(code_lines))}</code></pre>")
    flush_open_blocks()

    return "".join(html_parts) or '<p class="empty">No message content.</p>'


def _render_inline_markdown(text: str) -> str:
    rendered = _e(text)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", rendered)
    return rendered


def _dataset_link_or_value(
    *,
    conversation_id: UUID,
    conversation_type: ConversationType,
    value: Any,
) -> AuditHtml | None:
    if _is_missing(value):
        return None
    try:
        dataset_id = value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError):
        return AuditHtml(_e(str(value)))
    return AuditHtml(
        f"<code>{_e(str(dataset_id))}</code> "
        f"{_dataset_csv_link(conversation_id=conversation_id, conversation_type=conversation_type, dataset_id=dataset_id)}"
    )


def _training_fit_logs(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    fit = value.get("fit")
    return fit if isinstance(fit, dict) else None


def _stage_status(values: dict[str, Any]) -> str:
    normalized = {key.lower(): value for key, value in values.items()}
    error_value = normalized.get("training error")
    if not _is_missing(error_value):
        return "Error"
    warnings = normalized.get("training warnings")
    validation_issues = normalized.get("validation issues")
    if not _is_missing(warnings) or not _is_missing(validation_issues):
        return "Warning"
    if any(_is_status_present(value) for value in values.values()):
        return "Recorded"
    return "Missing"


def _is_status_present(value: Any) -> bool:
    return not _is_missing(value) and value is not False


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _json_pre(value: Any) -> str:
    return _e(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _render_value(value: Any) -> str:
    if _is_missing(value):
        return '<p class="empty">Not recorded.</p>'
    if isinstance(value, (str, int, float, bool)):
        return f"<p>{_e(str(value))}</p>"
    return f"<pre>{_json_pre(value)}</pre>"


def _is_graph_artifact_ref(artifact_ref: ArtifactRef) -> bool:
    meta = artifact_ref.get("artifact_meta") or {}
    return artifact_ref.get("kind") == "graph" or meta.get("kind") == "chart_spec"


def _artifact_label(artifact_ref: ArtifactRef, *, default: str) -> str:
    meta = artifact_ref.get("artifact_meta") or {}
    title = meta.get("title")
    if title:
        return str(title)
    kind = meta.get("kind")
    if kind:
        return str(kind).replace("_", " ").title()
    return f"{default} ({artifact_ref['format']})"


def _artifact_href(
    *,
    conversation_id: UUID,
    conversation_type: ConversationType,
    artifact_ref: ArtifactRef,
) -> str:
    query = urlencode(
        {
            "artifact_kind": artifact_ref["kind"],
            "artifact_format": artifact_ref["format"],
        }
    )
    return (
        f"/v1/conversations/{conversation_id}/types/{conversation_type}"
        f"/artifacts/{artifact_ref['id']}?{query}"
    )


def _dataset_csv_link(
    *,
    conversation_id: UUID,
    conversation_type: ConversationType,
    dataset_id: UUID,
) -> str:
    artifact_ref: ArtifactRef = {
        "id": dataset_id,
        "kind": "data",
        "format": "csv",
    }
    href = _artifact_href(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        artifact_ref=artifact_ref,
    )
    return f'<a href="{_attr(href)}">Open CSV</a>'


def _coerce_uuid_list(value: Any) -> list[UUID]:
    if not isinstance(value, list):
        return []
    ids: list[UUID] = []
    for item in value:
        try:
            ids.append(item if isinstance(item, UUID) else UUID(str(item)))
        except (TypeError, ValueError):
            continue
    return ids


def _safe_script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _e(value: str) -> str:
    return html.escape(value, quote=False)


def _attr(value: str) -> str:
    return html.escape(value, quote=True)


__all__ = ["AuditLogApp", "AuditLogHtmlRenderer"]
