from __future__ import annotations

from collections.abc import Generator
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from python.adapters.api.app import app
from python.adapters.api.dependencies import (
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
        self.create_conversation_result = uuid4()
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

    def create_conversation(self, user_id: UUID, conversation_type: str) -> UUID:
        self.create_conversation_calls.append(
            {
                "user_id": user_id,
                "conversation_type": conversation_type,
            }
        )
        return self.create_conversation_result

    def get_current_workflow_info(
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
        self.artifact_calls: list[dict[str, object]] = []

        self.upload_result = uuid4()
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


@pytest.fixture
def api_client() -> (
    Generator[tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser], None, None]
):
    workflow = _StubWorkflowApp()
    dataflow = _StubDataflowApp()
    user = AuthenticatedUser(
        uid=uuid4(),
        email="tester@example.com",
        email_verified=True,
        claims={"role": "tester"},
    )

    app.dependency_overrides[get_workflow_app] = lambda: workflow
    app.dependency_overrides[get_dataflow_app] = lambda: dataflow
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
        Conversation(conversation_id=first_id, conversation_type="causal"),
        Conversation(conversation_id=second_id, conversation_type="data"),
    ]

    response = client.get("/v1/conversations")

    assert response.status_code == 200
    assert response.json() == [
        {
            "conversation_id": str(first_id),
            "conversation_type": "causal",
        },
        {
            "conversation_id": str(second_id),
            "conversation_type": "data",
        },
    ]
    assert workflow.list_conversations_calls == [user.uid]


def test_create_conversation_returns_created_resource(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, user = api_client

    response = client.post(
        "/v1/conversations",
        json={"conversation_type": "causal"},
    )

    assert response.status_code == 201
    assert response.json() == {
        "conversation_id": str(workflow.create_conversation_result),
        "conversation_type": "causal",
    }
    assert workflow.create_conversation_calls == [
        {
            "user_id": user.uid,
            "conversation_type": "causal",
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
        messages=(ChatMessage(role="system", content="Reverted to MODEL_SELECTION"),),
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

    create_schema = schema["paths"]["/v1/conversations"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    create_schema_ref = create_schema["$ref"].split("/")[-1]
    assert schema["components"]["schemas"][create_schema_ref]["properties"]["conversation_type"][
        "enum"
    ] == ["causal", "data"]

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
