from __future__ import annotations

import html
import json
from collections.abc import Sequence
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
    current_dataset_id: UUID | None
    is_dataset_frozen: bool
    orchestrator_state_name: str
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
        current_dataset_id, is_dataset_frozen = (
            orchestrator_state.get_working_dataset_id_and_frozen_status()
        )
        report = AuditReport(
            conversation=conversation,
            generated_at_utc=datetime.now(UTC).timestamp(),
            current_stage_name=orchestrator_state.get_current_node_name(),
            current_dataset_id=current_dataset_id,
            is_dataset_frozen=is_dataset_frozen,
            orchestrator_state_name=orchestrator_state.name(),
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
        graphs = [
            artifact.graph
            for message in report.messages
            for artifact in message.artifacts
            if artifact.graph is not None and artifact.graph.spec is not None
        ]
        graph_specs = {
            graph.element_id: graph.spec
            for graph in graphs
            if graph.spec is not None
        }
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
                self._dataset_history(report),
                self._messages(report),
                self._stage_logs(report),
                "</main>",
                self._graph_script(graph_specs),
                "</body>",
                "</html>",
            ]
        )

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
            ("Conversation ID", str(conversation.conversation_id)),
            ("Conversation Type", conversation.conversation_type),
            ("Conversation Name", conversation.name or ""),
            ("Generated At", _format_ts(report.generated_at_utc)),
            ("Current Stage", report.current_stage_name),
            ("Current Dataset", current_dataset_link),
            ("Dataset Frozen", str(report.is_dataset_frozen)),
        ]
        return (
            "<section class=\"hero\">"
            "<h1>Conversation Audit Log</h1>"
            f"<dl>{''.join(_definition_row(k, v, raw_value=k == 'Current Dataset') for k, v in rows)}</dl>"
            "</section>"
        )

    def _dataset_history(self, report: AuditReport) -> str:
        dataset_ids = _coerce_uuid_list(report.orchestrator_payload.get("working_dataset_ids"))
        if not dataset_ids:
            body = '<p class="empty">No dataset versions were recorded.</p>'
        else:
            rows = []
            for index, dataset_id in enumerate(dataset_ids, start=1):
                link = _dataset_csv_link(
                    conversation_id=report.conversation.conversation_id,
                    conversation_type=report.conversation.conversation_type,
                    dataset_id=dataset_id,
                )
                rows.append(
                    "<tr>"
                    f"<td>{index}</td>"
                    f"<td><code>{_e(str(dataset_id))}</code></td>"
                    f"<td>{link}</td>"
                    "</tr>"
                )
            body = (
                "<table>"
                "<thead><tr><th>Version</th><th>Dataset ID</th><th>CSV</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody>"
                "</table>"
            )
        return f'<section><h2>Dataset History</h2>{body}</section>'

    def _messages(self, report: AuditReport) -> str:
        if not report.messages:
            return '<section><h2>Message Timeline</h2><p class="empty">No messages recorded.</p></section>'

        rendered = []
        for message in report.messages:
            artifact_html = self._message_artifacts(message)
            rendered.append(
                '<article class="message">'
                '<div class="message-meta">'
                f'<span class="role role-{_e(message.role)}">{_e(message.role)}</span>'
                f"<time>{_e(_format_ts(message.created_at_utc))}</time>"
                "</div>"
                f'<p class="message-content">{_e(message.content)}</p>'
                f"{artifact_html}"
                "</article>"
            )
        return f'<section><h2>Message Timeline</h2>{"".join(rendered)}</section>'

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
                        f"<h3>{_e(graph.title)}</h3>"
                        f'<div id="{_e(graph.element_id)}" class="graph"></div>'
                        "</div>"
                    )
                continue

            if artifact.href is not None:
                link_items.append(
                    f'<li><a href="{_attr(artifact.href)}">{_e(artifact.label)}</a></li>'
                )
        if not graph_items and not link_items:
            return ""
        links_html = f"<ul>{''.join(link_items)}</ul>" if link_items else ""
        return f'<div class="artifacts">{links_html}{"".join(graph_items)}</div>'

    def _stage_logs(self, report: AuditReport) -> str:
        payload = report.orchestrator_payload
        sections = [
            self._stage_section(
                "Dataset Stage",
                {
                    "Working Dataset IDs": payload.get("working_dataset_ids"),
                    "Latest Dataset Summary": payload.get("latest_dataset_summary"),
                },
            )
        ]
        if report.conversation.conversation_type == "causal":
            sections.extend(
                [
                    self._stage_section(
                        "Protocol Discussion",
                        {
                            "Protocol Discussion": payload.get("protocol_discussion"),
                            "Cleaning Instructions": payload.get(
                                "protocol_cleaning_instructions"
                            ),
                            "Causal Spec Draft": payload.get("causal_spec_draft"),
                        },
                    ),
                    self._stage_section(
                        "Data Compilation",
                        {
                            "Compiled Causal Spec": payload.get("causal_spec"),
                            "Transformation Plan": payload.get("data_transformation_plan"),
                            "Validation Issues": payload.get("validation_issues"),
                            "Is Validated": payload.get("is_validated"),
                            "Working Dataset Frozen": payload.get("working_dataset_frozen"),
                        },
                    ),
                    self._stage_section(
                        "Model Selection",
                        {
                            "Selected Model": payload.get("selected_model"),
                            "Selection Reasoning": payload.get("selection_reasoning"),
                        },
                    ),
                    self._stage_section(
                        "Model Train",
                        {
                            "Trained Model ID": payload.get("trained_model_id"),
                            "Training Warnings": payload.get("training_warnings"),
                            "Training Spec": payload.get("training_spec"),
                            "Training Error": payload.get("training_error_message"),
                        },
                    ),
                    '<section><h2>Causal Inference</h2>'
                    '<p class="empty">Inference outputs are shown in the message timeline through rendered graph artifacts.</p>'
                    "</section>",
                ]
            )
        sections.append(
            self._stage_section(
                "Raw Orchestration State",
                {
                    "State Name": report.orchestrator_state_name,
                    "Payload": payload,
                },
            )
        )
        return "".join(sections)

    def _stage_section(self, title: str, values: dict[str, Any]) -> str:
        parts: list[str] = []
        for label, value in values.items():
            parts.append(
                '<div class="stage-field">'
                f"<h3>{_e(label)}</h3>"
                f"{_render_value(value)}"
                "</div>"
            )
        return f'<section><h2>{_e(title)}</h2>{"".join(parts)}</section>'

    def _graph_script(self, graph_specs: dict[str, dict[str, Any]]) -> str:
        safe_json = _safe_script_json(graph_specs)
        return (
            "<script>"
            f"const auditGraphSpecs = {safe_json};"
            "for (const [id, spec] of Object.entries(auditGraphSpecs)) {"
            "const target = document.getElementById(id);"
            "if (!target) continue;"
            "vegaEmbed(target, spec, {actions: false}).catch((error) => {"
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
:root { color-scheme: light; --border: #d8dee4; --muted: #57606a; --bg: #f6f8fa; }
body { margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #24292f; background: #fff; }
.audit { max-width: 1100px; margin: 0 auto; padding: 32px 24px 56px; }
.hero { border-bottom: 1px solid var(--border); padding-bottom: 20px; }
h1 { font-size: 28px; margin: 0 0 16px; }
h2 { font-size: 20px; margin: 32px 0 14px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
h3 { font-size: 14px; margin: 0 0 8px; color: #24292f; }
dl { display: grid; grid-template-columns: 180px 1fr; gap: 8px 18px; margin: 0; }
dt { color: var(--muted); font-weight: 600; }
dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 10px; border: 1px solid var(--border); vertical-align: top; }
th { background: var(--bg); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { margin: 0; padding: 12px; overflow: auto; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; max-height: 520px; }
a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
.message { border: 1px solid var(--border); border-radius: 6px; padding: 14px; margin: 12px 0; }
.message-meta { display: flex; gap: 10px; align-items: center; color: var(--muted); margin-bottom: 8px; }
.role { text-transform: uppercase; font-size: 12px; font-weight: 700; letter-spacing: .04em; }
.role-user { color: #8250df; }
.role-assistant { color: #1a7f37; }
.role-system { color: #57606a; }
.message-content { white-space: pre-wrap; margin: 0; }
.artifacts { margin-top: 12px; }
.graph-card { margin-top: 12px; border-top: 1px solid var(--border); padding-top: 12px; }
.graph { min-height: 220px; overflow-x: auto; }
.graph-error { color: #cf222e; }
.stage-field { margin: 14px 0; }
.empty { color: var(--muted); margin: 0; }
@media print { .audit { max-width: none; padding: 16px; } pre { max-height: none; } a { color: inherit; } }
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


def _render_value(value: Any) -> str:
    if value is None or value == "" or value == []:
        return '<p class="empty">Not recorded.</p>'
    if isinstance(value, (str, int, float, bool)):
        return f"<p>{_e(str(value))}</p>"
    return f"<pre>{_e(json.dumps(value, ensure_ascii=False, indent=2, default=str))}</pre>"


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
