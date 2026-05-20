from __future__ import annotations

import io
import zipfile
from collections.abc import Generator
from types import SimpleNamespace
from uuid import UUID, uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from python.adapters.api.app import app
from python.adapters.api.dependencies import (
    get_audit_log_app,
    get_authenticated_user,
    get_dataflow_app,
    get_workflow_app,
)
from python.domain.models.errors import ValidationError
from python.domain.models.models import ChatMessage
from python.domain.repo.workflow_state_repo import Conversation
from python.domain.service.auth_service import AuthenticatedUser
from python.implementation.workflows.dataflow_app import DataflowArtifactResponse


class _StubWorkflowApp:
    def __init__(self) -> None:
        self.list_conversations_calls: list[UUID] = []
        self.create_conversation_calls: list[dict[str, object]] = []
        self.current_info_calls: list[dict[str, object]] = []
        self.handle_calls: list[dict[str, object | None]] = []
        self.revert_calls: list[dict[str, object]] = []

        self.list_conversations_result: list[Conversation] = []
        self.create_conversation_result = Conversation(
            conversation_id=uuid4(),
            conversation_type="causal",
            name="Created conversation",
            last_updated_at_utc=1712345678.123,
        )
        self.current_info_result = SimpleNamespace(
            messages=(),
            states=[],
            current_data_id=None,
            is_dataset_frozen=None,
        )
        self.handle_result = SimpleNamespace(
            messages=(),
            action="NONE",
            current_stage_name="NOOP_DONE",
            current_stage_status="DONE",
            current_data_id=None,
            is_dataset_frozen=None,
        )
        self.revert_result = self.current_info_result

    def list_conversations(self, user_id: UUID) -> list[Conversation]:
        self.list_conversations_calls.append(user_id)
        return self.list_conversations_result

    def create_conversation(
        self,
        user_id: UUID,
        conversation_type: str,
        conversation_name: str | None = None,
    ) -> Conversation:
        self.create_conversation_calls.append(
            {
                "user_id": user_id,
                "conversation_type": conversation_type,
                "conversation_name": conversation_name,
            }
        )
        return self.create_conversation_result

    def get_current_conversation_info(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
    ) -> object:
        self.current_info_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "conversation_type": conversation_type,
            }
        )
        return self.current_info_result

    def handle(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
        user_message: str | None,
    ) -> object:
        self.handle_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "conversation_type": conversation_type,
                "user_message": user_message,
            }
        )
        return self.handle_result

    def revert_to_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
        state_name: str,
    ) -> object:
        self.revert_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "conversation_type": conversation_type,
                "state_name": state_name,
            }
        )
        return self.revert_result


class _StubDataflowApp:
    def __init__(self) -> None:
        self.upload_calls: list[dict[str, object]] = []
        self.dataset_calls: list[dict[str, object]] = []
        self.diff_calls: list[dict[str, object]] = []
        self.artifact_calls: list[dict[str, object]] = []

        self.upload_result = uuid4()
        self.dataset_result = pd.DataFrame(
            [
                {"patient_id": "P001", "age": 41},
                {"patient_id": "P002", "age": 44},
                {"patient_id": "P003", "age": 52},
            ]
        )
        self.diff_result = SimpleNamespace(
            previous_dataset_id=uuid4(),
            current_dataset_id=uuid4(),
            diff={
                "identity_mode": "position",
                "key_columns": [],
                "schema_diff": {
                    "columns_added": ["bmi"],
                    "columns_removed": [],
                    "column_type_changes": [],
                },
                "row_changes": [
                    {
                        "row_ref": {
                            "mode": "position",
                            "key": None,
                            "position": 1,
                        },
                        "op": "updated",
                        "cell_changes": [
                            {
                                "column": "age",
                                "op": "modified",
                                "old_value": 44,
                                "new_value": 45,
                            }
                        ],
                    }
                ],
                "summary": {
                    "old_row_count": 2,
                    "new_row_count": 2,
                    "inserted_rows": 0,
                    "deleted_rows": 0,
                    "updated_rows": 1,
                    "total_changed_rows": 1,
                    "total_changed_cells": 1,
                },
            },
        )
        self.artifact_result = DataflowArtifactResponse(
            id=uuid4(),
            kind="graph",
            format="json",
            mime="application/json",
            content=b'{"ok":true}',
        )

    def upload_csv_data(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
        csv_bytes: bytes,
    ) -> UUID:
        self.upload_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "conversation_type": conversation_type,
                "csv_bytes": csv_bytes,
            }
        )
        return self.upload_result

    def get_csv_data(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
        dataset_id: UUID,
        start: int = 0,
        limit: int | None = None,
    ) -> pd.DataFrame:
        self.dataset_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "conversation_type": conversation_type,
                "dataset_id": dataset_id,
                "start": start,
                "limit": limit,
            }
        )
        frame = self.dataset_result.iloc[start:].copy()
        if limit is None:
            return frame
        return frame.head(limit).copy()

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
        self.artifact_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "conversation_type": conversation_type,
                "artifact_id": artifact_id,
                "artifact_kind": artifact_kind,
                "artifact_format": artifact_format,
            }
        )
        return self.artifact_result

    def get_working_dataset_diff(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
        key_columns: list[str] | None,
    ) -> object:
        self.diff_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "conversation_type": conversation_type,
                "key_columns": key_columns,
            }
        )
        return self.diff_result


class _StubAuditLogApp:
    def __init__(self) -> None:
        self.render_calls: list[dict[str, object]] = []
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "audit-log.html",
                (
                    "<!doctype html><html><body>"
                    "<h1>Conversation Audit Log</h1>"
                    "<p>Escaped &lt;message&gt;</p>"
                    "</body></html>"
                ),
            )
        self.render_result = buffer.getvalue()

    def render_zip(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
    ) -> bytes:
        self.render_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "conversation_type": conversation_type,
            }
        )
        return self.render_result


@pytest.fixture
def api_client() -> (
    Generator[tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser], None, None]
):
    workflow = _StubWorkflowApp()
    dataflow = _StubDataflowApp()
    audit_log = _StubAuditLogApp()
    workflow.audit_log = audit_log  # type: ignore[attr-defined]
    user = AuthenticatedUser(
        uid=uuid4(),
        email="tester@example.com",
        email_verified=True,
        claims={"role": "tester"},
    )

    app.dependency_overrides[get_workflow_app] = lambda: workflow
    app.dependency_overrides[get_dataflow_app] = lambda: dataflow
    app.dependency_overrides[get_audit_log_app] = lambda: audit_log
    app.dependency_overrides[get_authenticated_user] = lambda: user
    app.openapi_schema = None
    with TestClient(app) as client:
        yield client, workflow, dataflow, user
    app.dependency_overrides.clear()
    app.openapi_schema = None


def test_healthz_returns_ok(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, _, _ = api_client

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_list_conversations_returns_conversation_resources(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, user = api_client
    first_id = uuid4()
    second_id = uuid4()
    workflow.list_conversations_result = [
        Conversation(
            conversation_id=first_id,
            conversation_type="causal",
            name="Causal review",
            last_updated_at_utc=1712345678.123,
        ),
        Conversation(
            conversation_id=second_id,
            conversation_type="data",
            name=None,
            last_updated_at_utc=1712000000.0,
        ),
    ]

    response = client.get("/v1/conversations")

    assert response.status_code == 200
    assert response.json() == [
        {
            "conversation_id": str(first_id),
            "conversation_type": "causal",
            "conversation_name": "Causal review",
            "last_updated_at_utc": 1712345678.123,
        },
        {
            "conversation_id": str(second_id),
            "conversation_type": "data",
            "conversation_name": None,
            "last_updated_at_utc": 1712000000.0,
        },
    ]
    assert workflow.list_conversations_calls == [user.uid]


def test_create_conversation_returns_created_resource(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, user = api_client
    workflow.create_conversation_result = Conversation(
        conversation_id=uuid4(),
        conversation_type="causal",
        name="Hypertension cohort review",
        last_updated_at_utc=1712345678.123,
    )

    response = client.post(
        "/v1/conversations",
        json={
            "conversation_type": "causal",
            "conversation_name": "Hypertension cohort review",
        },
    )

    assert response.status_code == 201
    assert response.json() == {
        "conversation_id": str(workflow.create_conversation_result.conversation_id),
        "conversation_type": "causal",
        "conversation_name": "Hypertension cohort review",
        "last_updated_at_utc": 1712345678.123,
    }
    assert workflow.create_conversation_calls == [
        {
            "user_id": user.uid,
            "conversation_type": "causal",
            "conversation_name": "Hypertension cohort review",
        }
    ]


def test_get_conversation_returns_snapshot(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, dataflow, user = api_client
    conversation_id = uuid4()
    dataset_id = uuid4()
    artifact_id = uuid4()
    workflow.current_info_result = SimpleNamespace(
        messages=(
            ChatMessage(
                role="assistant",
                content="Snapshot message",
                created_at_utc=1712345678.123,
                artifact_refs=[
                    {
                        "id": artifact_id,
                        "kind": "graph",
                        "format": "json",
                        "artifact_meta": {"title": "Snapshot plot"},
                    }
                ],
            ),
        ),
        states=["DATASET", "DATA_COMPILATION"],
        current_data_id=dataset_id,
        is_dataset_frozen=True,
    )

    response = client.get(f"/v1/conversations/{conversation_id}/types/causal")

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "conversation_type": "causal",
        "messages": [
            {
                "role": "assistant",
                "content": "Snapshot message",
                "created_at_utc": 1712345678.123,
                "id": None,
                "artifact_refs": [
                    {
                        "id": str(artifact_id),
                        "kind": "graph",
                        "format": "json",
                        "artifact_meta": {"title": "Snapshot plot"},
                    }
                ],
            }
        ],
        "states": ["DATASET", "DATA_COMPILATION"],
        "working_dataset": {
            "dataset_id": str(dataset_id),
            "is_frozen": True,
        },
    }
    assert workflow.current_info_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "conversation_type": "causal",
        }
    ]
    assert dataflow.upload_calls == []
    assert dataflow.artifact_calls == []


def test_get_audit_log_returns_zip(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, dataflow, user = api_client
    conversation_id = uuid4()
    audit_log = workflow.audit_log  # type: ignore[attr-defined]

    response = client.get(f"/v1/conversations/{conversation_id}/types/causal/audit-log")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert (
        response.headers["content-disposition"]
        == f'attachment; filename="audit-log-{conversation_id}.zip"'
    )
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        html = archive.read("audit-log.html").decode("utf-8")
    assert "Conversation Audit Log" in html
    assert "Escaped &lt;message&gt;" in html
    assert audit_log.render_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "conversation_type": "causal",
        }
    ]
    assert dataflow.artifact_calls == []


def test_create_conversation_message_returns_workflow_result(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, dataflow, user = api_client
    conversation_id = uuid4()
    dataset_id = uuid4()
    artifact_id = uuid4()
    workflow.handle_result = SimpleNamespace(
        messages=(
            ChatMessage(
                role="assistant",
                content="I summarized the dataset.",
                created_at_utc=1712345678.456,
                artifact_refs=[
                    {
                        "id": artifact_id,
                        "kind": "graph",
                        "format": "json",
                        "artifact_meta": {"title": "Summary plot"},
                    }
                ],
            ),
        ),
        action="NEEDS_INPUT",
        current_stage_name="DATA_COMPILATION",
        current_stage_status="PENDING",
        current_data_id=dataset_id,
        is_dataset_frozen=False,
    )

    response = client.post(
        f"/v1/conversations/{conversation_id}/types/data/messages",
        json={"user_text": "Summarize the data"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "conversation_type": "data",
        "messages": [
            {
                "role": "assistant",
                "content": "I summarized the dataset.",
                "created_at_utc": 1712345678.456,
                "id": None,
                "artifact_refs": [
                    {
                        "id": str(artifact_id),
                        "kind": "graph",
                        "format": "json",
                        "artifact_meta": {"title": "Summary plot"},
                    }
                ],
            }
        ],
        "action": "NEEDS_INPUT",
        "current_stage_name": "DATA_COMPILATION",
        "current_stage_status": "PENDING",
        "working_dataset": {
            "dataset_id": str(dataset_id),
            "is_frozen": False,
        },
    }
    assert workflow.handle_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "conversation_type": "data",
            "user_message": "Summarize the data",
        }
    ]
    assert dataflow.upload_calls == []
    assert dataflow.artifact_calls == []


def test_create_conversation_message_strips_blank_text_to_none(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, user = api_client
    conversation_id = uuid4()

    response = client.post(
        f"/v1/conversations/{conversation_id}/types/causal/messages",
        json={"user_text": "   "},
    )

    assert response.status_code == 200
    assert workflow.handle_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "conversation_type": "causal",
            "user_message": None,
        }
    ]


def test_create_state_reversion_returns_updated_snapshot(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, user = api_client
    conversation_id = uuid4()
    dataset_id = uuid4()
    workflow.revert_result = SimpleNamespace(
        messages=(
            ChatMessage(
                role="system",
                content="Reverted to MODEL_SELECTION",
                created_at_utc=1712345678.789,
            ),
        ),
        states=["DATASET", "MODEL_SELECTION"],
        current_data_id=dataset_id,
        is_dataset_frozen=False,
    )

    response = client.post(
        f"/v1/conversations/{conversation_id}/types/causal/state-reversions",
        json={"state_name": "MODEL_SELECTION"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "conversation_type": "causal",
        "messages": [
            {
                "role": "system",
                "content": "Reverted to MODEL_SELECTION",
                "created_at_utc": 1712345678.789,
                "id": None,
                "artifact_refs": None,
            }
        ],
        "states": ["DATASET", "MODEL_SELECTION"],
        "working_dataset": {
            "dataset_id": str(dataset_id),
            "is_frozen": False,
        },
    }
    assert workflow.revert_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "conversation_type": "causal",
            "state_name": "MODEL_SELECTION",
        }
    ]


def test_upload_dataset_success_uses_scoped_path(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, user = api_client
    conversation_id = uuid4()
    csv_bytes = b"col1,col2\n1,2\n"

    response = client.post(
        f"/v1/conversations/{conversation_id}/types/data/datasets",
        files={"file": ("dataset.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 201
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "conversation_type": "data",
        "dataset_id": str(dataflow.upload_result),
    }
    assert dataflow.upload_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "conversation_type": "data",
            "csv_bytes": csv_bytes,
        }
    ]


def test_upload_dataset_rejects_non_csv_files(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, _ = api_client
    conversation_id = uuid4()

    response = client.post(
        f"/v1/conversations/{conversation_id}/types/causal/datasets",
        files={"file": ("dataset.txt", b"not,csv\n", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV uploads are supported."
    assert dataflow.upload_calls == []


def test_upload_dataset_maps_value_error_to_400(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, _ = api_client
    conversation_id = uuid4()

    def _raise_value_error(**_: object) -> UUID:
        raise ValueError("Uploaded file is not a valid CSV: parse error")

    dataflow.upload_csv_data = _raise_value_error  # type: ignore[method-assign]

    response = client.post(
        f"/v1/conversations/{conversation_id}/types/causal/datasets",
        files={"file": ("dataset.csv", b"bad", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid CSV: parse error"


def test_get_dataset_returns_paginated_rows(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, user = api_client
    conversation_id = uuid4()
    dataset_id = uuid4()

    response = client.get(
        f"/v1/conversations/{conversation_id}/types/data/datasets/{dataset_id}",
        params={"start": 1, "limit": 1},
    )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "conversation_type": "data",
        "dataset_id": str(dataset_id),
        "start": 1,
        "limit": 1,
        "row_count": 1,
        "columns": ["patient_id", "age"],
        "rows": [{"patient_id": "P002", "age": 44}],
    }
    assert dataflow.dataset_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "conversation_type": "data",
            "dataset_id": dataset_id,
            "start": 1,
            "limit": 1,
        }
    ]


def test_create_dataset_diff_returns_structured_response_without_request_body(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, user = api_client
    conversation_id = uuid4()

    response = client.post(
        f"/v1/conversations/{conversation_id}/types/data/dataset-diffs",
    )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "conversation_type": "data",
        "previous_dataset_id": str(dataflow.diff_result.previous_dataset_id),
        "current_dataset_id": str(dataflow.diff_result.current_dataset_id),
        "diff": dataflow.diff_result.diff,
    }
    assert dataflow.diff_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "conversation_type": "data",
            "key_columns": None,
        }
    ]


def test_get_artifact_returns_csv_attachment(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, user = api_client
    conversation_id = uuid4()
    artifact_id = uuid4()
    dataflow.artifact_result = DataflowArtifactResponse(
        id=artifact_id,
        kind="data",
        format="csv",
        mime="text/csv",
        content=b"a,b\n1,2\n",
    )

    response = client.get(
        f"/v1/conversations/{conversation_id}/types/data/artifacts/{artifact_id}",
        params={"artifact_kind": "data", "artifact_format": "csv"},
    )

    assert response.status_code == 200
    assert response.content == b"a,b\n1,2\n"
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"] == f'attachment; filename="{artifact_id}.csv"'
    assert dataflow.artifact_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "conversation_type": "data",
            "artifact_id": artifact_id,
            "artifact_kind": "data",
            "artifact_format": "csv",
        }
    ]


def test_get_artifact_maps_validation_error_to_422(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, _ = api_client
    conversation_id = uuid4()
    artifact_id = uuid4()

    def _raise_validation_error(**_: object) -> DataflowArtifactResponse:
        raise ValidationError(
            field="artifact_format",
            reason="Graph artifacts must be in JSON format",
        )

    dataflow.get_artifact = _raise_validation_error  # type: ignore[method-assign]

    response = client.get(
        f"/v1/conversations/{conversation_id}/types/causal/artifacts/{artifact_id}",
        params={"artifact_kind": "graph", "artifact_format": "csv"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "validation_failed",
        "message": "Validation failed",
        "detail": "Validation error for 'artifact_format': Graph artifacts must be in JSON format",
    }


def test_openapi_mentions_scoped_paths_and_enums() -> None:
    app.openapi_schema = None
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Agent API"
    assert schema["info"]["summary"] == "Authenticated API for workflow interactions."

    create_schema = schema["paths"]["/v1/conversations"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    create_schema_ref = create_schema["$ref"].split("/")[-1]
    assert schema["components"]["schemas"][create_schema_ref]["properties"]["conversation_type"][
        "enum"
    ] == ["causal", "data"]
    conversation_name_schema = schema["components"]["schemas"][create_schema_ref]["properties"][
        "conversation_name"
    ]
    assert conversation_name_schema["anyOf"][0]["maxLength"] == 100
    assert conversation_name_schema["anyOf"][0]["minLength"] == 1

    conversation_summary_schema = schema["components"]["schemas"]["ConversationSummaryResponse"]
    assert "conversation_name" in conversation_summary_schema["properties"]
    assert "last_updated_at_utc" in conversation_summary_schema["properties"]

    message_operation = schema["paths"][
        "/v1/conversations/{conversation_id}/types/{conversation_type}/messages"
    ]["post"]
    assert "revert_data_changes" in message_operation["description"]

    scoped_get_operation = schema["paths"][
        "/v1/conversations/{conversation_id}/types/{conversation_type}"
    ]["get"]
    scoped_parameters = {param["name"]: param for param in scoped_get_operation["parameters"]}
    assert scoped_parameters["conversation_type"]["schema"]["enum"] == ["causal", "data"]

    artifact_operation = schema["paths"][
        "/v1/conversations/{conversation_id}/types/{conversation_type}/artifacts/{artifact_id}"
    ]["get"]
    artifact_parameters = {param["name"]: param for param in artifact_operation["parameters"]}
    assert artifact_parameters["artifact_kind"]["schema"]["enum"] == ["graph", "data"]
    assert artifact_parameters["artifact_format"]["schema"]["enum"] == ["json", "csv"]

    audit_operation = schema["paths"][
        "/v1/conversations/{conversation_id}/types/{conversation_type}/audit-log"
    ]["get"]
    assert "application/zip" in audit_operation["responses"]["200"]["content"]
    assert "Trained model objects are not included" in audit_operation["description"]

    diff_operation = schema["paths"][
        "/v1/conversations/{conversation_id}/types/{conversation_type}/dataset-diffs"
    ]["post"]
    assert "key_columns" in diff_operation["description"]
    assert "Inserted/deleted rows are counted in `summary`" in diff_operation["description"]
    assert "row_changes` may be truncated" in diff_operation["description"]
    diff_response_ref = diff_operation["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"
    ].split("/")[-1]
    diff_response_schema = schema["components"]["schemas"][diff_response_ref]
    assert "previous_dataset_id" in diff_response_schema["properties"]
    assert "current_dataset_id" in diff_response_schema["properties"]

    dataset_operation = schema["paths"][
        "/v1/conversations/{conversation_id}/types/{conversation_type}/datasets/{dataset_id}"
    ]["get"]
    dataset_parameters = {param["name"]: param for param in dataset_operation["parameters"]}
    assert dataset_parameters["dataset_id"]["schema"]["format"] == "uuid"
    assert dataset_parameters["start"]["schema"]["minimum"] == 0
    assert dataset_parameters["limit"]["schema"]["anyOf"][0]["minimum"] == 0
    dataset_response_ref = dataset_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"].split("/")[-1]
    dataset_response_schema = schema["components"]["schemas"][dataset_response_ref]
    assert "columns" in dataset_response_schema["properties"]
    assert "rows" in dataset_response_schema["properties"]
    assert "start" in dataset_response_schema["properties"]
    assert "limit" in dataset_response_schema["properties"]

    dataframe_diff_schema = schema["components"]["schemas"]["DataFrameDiff"]
    assert "schema_diff" in dataframe_diff_schema["properties"]
    assert "row_changes" in dataframe_diff_schema["properties"]
    assert "summary" in dataframe_diff_schema["properties"]
