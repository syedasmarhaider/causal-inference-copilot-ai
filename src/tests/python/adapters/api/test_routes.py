from __future__ import annotations

"""Integration tests for the API route layer.

Strategy
--------
All domain services (``WorkflowApp``, ``DataflowApp``) are replaced with
hand-rolled stubs injected via FastAPI's ``dependency_overrides`` mechanism.
Authentication is bypassed so every test runs as a fixed ``AuthenticatedUser``.

Key architectural invariant under test
---------------------------------------
Since the workflow redesign, ``WorkflowResponse`` embeds dataset state directly
(``current_data_id``, ``is_dataset_frozen``).  The route layer therefore must
**not** call ``dataflow.get_current_working_dataset_info()`` for the ``invoke``
or ``lateststate`` endpoints — those stubs intentionally omit that method to
catch any regression.
"""

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
from python.domain.service.auth_service import AuthenticatedUser
from python.implementation.workflows.dataflow_app import DataflowArtifactResponse


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubWorkflowApp:
    """Minimal WorkflowApp stand-in that records every call made to it.

    ``handle_result`` and ``latest_state_result`` are ``SimpleNamespace`` objects
    whose fields mirror ``WorkflowResponse``, including the new
    ``current_data_id`` and ``is_dataset_frozen`` fields introduced in the
    workflow redesign.
    """

    def __init__(self) -> None:
        self.user_checks: list[tuple[UUID, UUID]] = []
        self.create_conversation_calls: list[UUID] = []
        self.handle_calls: list[dict[str, object | None]] = []
        self.get_last_state_calls: list[tuple[UUID, UUID]] = []
        self.revert_calls: list[dict[str, object]] = []

        self.create_conversation_result = uuid4()
        self.handle_result = SimpleNamespace(
            messages=(),
            action="NONE",
            current_stage_name="NOOP_DONE",
            current_stage_status="DONE",
            current_data_id=None,
            is_dataset_frozen=None,
        )
        # Latest-state defaults to the same shape as handle_result.
        self.latest_state_result: object | None = self.handle_result

    def raise_if_userid_not_relates_to_conversation_id(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> None:
        self.user_checks.append((user_id, conversation_id))

    def create_conversation(self, *, user_id: UUID) -> UUID:
        self.create_conversation_calls.append(user_id)
        return self.create_conversation_result

    def handle(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        user_message: str | None,
    ) -> object:
        self.handle_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "user_message": user_message,
            }
        )
        return self.handle_result

    def get_last_conversation_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> object | None:
        self.get_last_state_calls.append((user_id, conversation_id))
        return self.latest_state_result

    def revert_to_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        state_name: str,
    ) -> None:
        self.revert_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "state_name": state_name,
            }
        )


class _StubDataflowApp:
    """Minimal DataflowApp stand-in used only by upload and artifact routes.

    Note: ``get_current_working_dataset_info`` is intentionally absent.
    After the workflow redesign the route layer derives dataset info from
    ``WorkflowResponse`` directly.  If a route ever calls the removed method,
    Python will raise ``AttributeError``, immediately surfacing the regression.
    """

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
        csv_bytes: bytes,
    ) -> UUID:
        self.upload_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "csv_bytes": csv_bytes,
            }
        )
        return self.upload_result

    def get_artifact(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        artifact_kind: str,
        artifact_format: str,
    ) -> DataflowArtifactResponse:
        self.artifact_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "artifact_id": artifact_id,
                "artifact_kind": artifact_kind,
                "artifact_format": artifact_format,
            }
        )
        return self.artifact_result


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client() -> (
    Generator[tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser], None, None]
):
    """Yield a TestClient wired to stub services and a fixed authenticated user."""
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


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


def test_healthz_returns_ok(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, _, _ = api_client
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


# ---------------------------------------------------------------------------
# Conversations — create
# ---------------------------------------------------------------------------


def test_create_conversation_returns_ids(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, user = api_client

    response = client.post("/v1/conversations")

    assert response.status_code == 200
    assert response.json() == {
        "user_id": str(user.uid),
        "conversation_id": str(workflow.create_conversation_result),
    }
    assert workflow.create_conversation_calls == [user.uid]


# ---------------------------------------------------------------------------
# Conversations — invoke
# ---------------------------------------------------------------------------


def test_invoke_returns_messages_action_stage_and_latest_dataset_info(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    """The route derives latest_working_dataset from WorkflowResponse — no dataflow call."""
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
        current_stage_name="DATASET",
        current_stage_status="PENDING",
        current_data_id=dataset_id,
        is_dataset_frozen=False,
    )

    response = client.post(
        f"/v1/conversations/{conversation_id}/invoke",
        json={"user_text": "Summarize the data"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "user_id": str(user.uid),
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
        "current_stage_name": "DATASET",
        "current_stage_status": "PENDING",
        "latest_working_dataset": {
            "dataset_id": str(dataset_id),
            "is_freezed": False,
        },
    }
    assert workflow.handle_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "user_message": "Summarize the data",
        }
    ]
    assert workflow.user_checks == [(user.uid, conversation_id)]
    # DataflowApp must NOT be called — dataset info comes from WorkflowResponse.
    assert dataflow.artifact_calls == []
    assert dataflow.upload_calls == []


def test_invoke_returns_null_dataset_when_no_data_uploaded(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    """latest_working_dataset is null when current_data_id is None in the workflow response."""
    client, workflow, _, _ = api_client
    conversation_id = uuid4()

    workflow.handle_result = SimpleNamespace(
        messages=(),
        action="NEEDS_DATA",
        current_stage_name="DATASET",
        current_stage_status="PENDING",
        current_data_id=None,
        is_dataset_frozen=None,
    )

    response = client.post(
        f"/v1/conversations/{conversation_id}/invoke",
        json={"user_text": "start"},
    )

    assert response.status_code == 200
    assert response.json()["latest_working_dataset"] is None


def test_invoke_strips_blank_message_to_none(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, user = api_client
    conversation_id = uuid4()

    workflow.handle_result = SimpleNamespace(
        messages=(),
        action="NONE",
        current_stage_name="NOOP_DONE",
        current_stage_status="DONE",
        current_data_id=None,
        is_dataset_frozen=None,
    )

    response = client.post(
        f"/v1/conversations/{conversation_id}/invoke",
        json={"user_text": "   "},
    )

    assert response.status_code == 200
    assert workflow.handle_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "user_message": None,
        }
    ]


# ---------------------------------------------------------------------------
# Conversations — lateststate
# ---------------------------------------------------------------------------


def test_lateststate_returns_404_when_workflow_has_no_active_state(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, user = api_client
    conversation_id = uuid4()
    workflow.latest_state_result = None

    response = client.get(f"/v1/conversations/{conversation_id}/lateststate")

    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "conversation_not_found"
    assert payload["message"] == "Conversation not found"
    assert str(user.uid) in payload["detail"]
    assert str(conversation_id) in payload["detail"]


def test_lateststate_returns_response_shape(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    """Dataset info is derived from WorkflowResponse — no DataflowApp call."""
    client, workflow, dataflow, user = api_client
    conversation_id = uuid4()
    dataset_id = uuid4()

    workflow.latest_state_result = SimpleNamespace(
        messages=(ChatMessage(role="assistant", content="Current state message"),),
        action="NEEDS_DATA",
        current_stage_name="DATASET",
        current_stage_status="PENDING",
        current_data_id=dataset_id,
        is_dataset_frozen=True,
    )

    response = client.get(f"/v1/conversations/{conversation_id}/lateststate")

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == str(conversation_id)
    assert payload["user_id"] == str(user.uid)
    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": "Current state message",
            "id": None,
            "artifact_refs": None,
        }
    ]
    assert payload["action"] == "NEEDS_DATA"
    assert payload["current_stage_name"] == "DATASET"
    assert payload["current_stage_status"] == "PENDING"
    assert payload["latest_working_dataset"] == {
        "dataset_id": str(dataset_id),
        "is_freezed": True,
    }
    # DataflowApp must NOT be called — dataset info comes from WorkflowResponse.
    assert dataflow.artifact_calls == []
    assert dataflow.upload_calls == []


def test_lateststate_returns_null_dataset_when_no_data_uploaded(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, _ = api_client
    conversation_id = uuid4()

    workflow.latest_state_result = SimpleNamespace(
        messages=(),
        action="NEEDS_DATA",
        current_stage_name="DATASET",
        current_stage_status="PENDING",
        current_data_id=None,
        is_dataset_frozen=None,
    )

    response = client.get(f"/v1/conversations/{conversation_id}/lateststate")

    assert response.status_code == 200
    assert response.json()["latest_working_dataset"] is None


# ---------------------------------------------------------------------------
# Datasets — upload
# ---------------------------------------------------------------------------


def test_upload_dataset_csv_success_uses_dataflow_app(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, user = api_client
    conversation_id = uuid4()
    csv_bytes = b"col1,col2\n1,2\n"

    response = client.post(
        f"/v1/conversations/{conversation_id}/datasets",
        files={"file": ("dataset.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": str(user.uid),
        "conversation_id": str(conversation_id),
        "dataset_id": str(dataflow.upload_result),
    }
    assert dataflow.upload_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "csv_bytes": csv_bytes,
        }
    ]


def test_upload_dataset_csv_rejects_non_csv_file(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, _ = api_client
    conversation_id = uuid4()

    response = client.post(
        f"/v1/conversations/{conversation_id}/datasets",
        files={"file": ("dataset.txt", b"not,csv\n", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV uploads are supported."
    assert dataflow.upload_calls == []


def test_upload_dataset_csv_rejects_empty_payload(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, _ = api_client
    conversation_id = uuid4()

    response = client.post(
        f"/v1/conversations/{conversation_id}/datasets",
        files={"file": ("dataset.csv", b"", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."
    assert dataflow.upload_calls == []


def test_upload_dataset_csv_maps_value_error_to_400(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, _ = api_client
    conversation_id = uuid4()

    def _raise_value_error(**_: object) -> UUID:
        raise ValueError("Uploaded file is not a valid CSV: parse error")

    dataflow.upload_csv_data = _raise_value_error  # type: ignore[method-assign]

    response = client.post(
        f"/v1/conversations/{conversation_id}/datasets",
        files={"file": ("dataset.csv", b"bad", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid CSV: parse error"


# ---------------------------------------------------------------------------
# Artifacts — download
# ---------------------------------------------------------------------------


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
        f"/v1/conversations/{conversation_id}/artifacts/{artifact_id}",
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
            "artifact_id": artifact_id,
            "artifact_kind": "data",
            "artifact_format": "csv",
        }
    ]


def test_get_artifact_returns_json_inline(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, _ = api_client
    conversation_id = uuid4()
    artifact_id = uuid4()
    dataflow.artifact_result = DataflowArtifactResponse(
        id=artifact_id,
        kind="graph",
        format="json",
        mime="application/json",
        content=b'{"spec":1}',
    )

    response = client.get(
        f"/v1/conversations/{conversation_id}/artifacts/{artifact_id}",
        params={"artifact_kind": "graph", "artifact_format": "json"},
    )

    assert response.status_code == 200
    assert response.content == b'{"spec":1}'
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["content-disposition"] == "inline"


def test_get_artifact_maps_validation_error_to_422(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, _ = api_client
    conversation_id = uuid4()
    artifact_id = uuid4()

    def _raise_validation_error(**_: object) -> DataflowArtifactResponse:
        raise ValidationError(
            field="artifact_format", reason="Graph artifacts must be in JSON format"
        )

    dataflow.get_artifact = _raise_validation_error  # type: ignore[method-assign]

    response = client.get(
        f"/v1/conversations/{conversation_id}/artifacts/{artifact_id}",
        params={"artifact_kind": "graph", "artifact_format": "csv"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "validation_failed",
        "message": "Validation failed",
        "detail": "Validation error for 'artifact_format': Graph artifacts must be in JSON format",
    }


# ---------------------------------------------------------------------------
# Conversations — revert
# ---------------------------------------------------------------------------


def test_revert_to_state_requires_non_empty_state_name(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, _ = api_client
    conversation_id = uuid4()

    response = client.post(
        f"/v1/conversations/{conversation_id}/revert",
        json={"state_name": "   "},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "state_name must be a non-empty string"
    assert workflow.revert_calls == []


def test_revert_to_state_calls_workflow_app(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, user = api_client
    conversation_id = uuid4()

    response = client.post(
        f"/v1/conversations/{conversation_id}/revert",
        json={"state_name": "MODEL_SELECTION"},
    )

    assert response.status_code == 200
    assert response.content == b"null"
    assert workflow.revert_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "state_name": "MODEL_SELECTION",
        }
    ]
    assert workflow.user_checks == [(user.uid, conversation_id)]


# ---------------------------------------------------------------------------
# OpenAPI schema
# ---------------------------------------------------------------------------


def test_openapi_mentions_revert_message_and_artifact_enums() -> None:
    app.openapi_schema = None
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    invoke_description = schema["paths"]["/v1/conversations/{conversation_id}/invoke"]["post"][
        "description"
    ]
    assert "revert_data_changes" in invoke_description

    artifact_operation = schema["paths"][
        "/v1/conversations/{conversation_id}/artifacts/{artifact_id}"
    ]["get"]
    parameters = {param["name"]: param for param in artifact_operation["parameters"]}
    assert parameters["artifact_kind"]["schema"]["enum"] == ["graph", "data"]
    assert parameters["artifact_format"]["schema"]["enum"] == ["json", "csv"]
    assert "graph -> json" in artifact_operation["description"]
    assert "data -> json|csv" in artifact_operation["description"]

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
        current_stage_name="DATASET",
        current_stage_status="PENDING",
    )
    dataflow.latest_info_result = WorkingDatasetInfo(dataset_id=dataset_id, is_freezed=False)

    response = client.post(
        f"/v1/conversations/{conversation_id}/invoke",
        json={"user_text": "Summarize the data"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": str(conversation_id),
        "user_id": str(user.uid),
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
        "current_stage_name": "DATASET",
        "current_stage_status": "PENDING",
        "latest_working_dataset": {
            "dataset_id": str(dataset_id),
            "is_freezed": False,
        },
    }
    assert workflow.handle_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "user_message": "Summarize the data",
        }
    ]
    assert workflow.user_checks == [(user.uid, conversation_id)]
    assert dataflow.latest_info_calls == [(user.uid, conversation_id)]


def test_invoke_strips_blank_message_to_none(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, user = api_client
    conversation_id = uuid4()
    workflow.handle_result = SimpleNamespace(
        messages=(),
        action="NONE",
        current_stage_name="NOOP_DONE",
        current_stage_status="DONE",
    )

    response = client.post(
        f"/v1/conversations/{conversation_id}/invoke",
        json={"user_text": "   "},
    )

    assert response.status_code == 200
    assert workflow.handle_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "user_message": None,
        }
    ]


def test_lateststate_returns_404_when_workflow_has_no_active_state(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, user = api_client
    conversation_id = uuid4()
    workflow.latest_state_result = None

    response = client.get(f"/v1/conversations/{conversation_id}/lateststate")

    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "conversation_not_found"
    assert payload["message"] == "Conversation not found"
    assert str(user.uid) in payload["detail"]
    assert str(conversation_id) in payload["detail"]


def test_lateststate_returns_response_shape(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, dataflow, user = api_client
    conversation_id = uuid4()
    workflow.latest_state_result = SimpleNamespace(
        messages=(ChatMessage(role="assistant", content="Current state message"),),
        action="NEEDS_DATA",
        current_stage_name="DATASET",
        current_stage_status="PENDING",
    )
    dataflow.latest_info_result = WorkingDatasetInfo(dataset_id=uuid4(), is_freezed=True)

    response = client.get(f"/v1/conversations/{conversation_id}/lateststate")

    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == str(conversation_id)
    assert payload["user_id"] == str(user.uid)
    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": "Current state message",
            "id": None,
            "artifact_refs": None,
        }
    ]
    assert payload["action"] == "NEEDS_DATA"
    assert payload["current_stage_name"] == "DATASET"
    assert payload["current_stage_status"] == "PENDING"
    assert payload["latest_working_dataset"]["is_freezed"] is True


def test_upload_dataset_csv_success_uses_dataflow_app(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, user = api_client
    conversation_id = uuid4()
    csv_bytes = b"col1,col2\n1,2\n"

    response = client.post(
        f"/v1/conversations/{conversation_id}/datasets",
        files={"file": ("dataset.csv", csv_bytes, "text/csv")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": str(user.uid),
        "conversation_id": str(conversation_id),
        "dataset_id": str(dataflow.upload_result),
    }
    assert dataflow.upload_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "csv_bytes": csv_bytes,
        }
    ]


def test_upload_dataset_csv_rejects_non_csv_file(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, _ = api_client
    conversation_id = uuid4()

    response = client.post(
        f"/v1/conversations/{conversation_id}/datasets",
        files={"file": ("dataset.txt", b"not,csv\n", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV uploads are supported."
    assert dataflow.upload_calls == []


def test_upload_dataset_csv_rejects_empty_payload(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, _ = api_client
    conversation_id = uuid4()

    response = client.post(
        f"/v1/conversations/{conversation_id}/datasets",
        files={"file": ("dataset.csv", b"", "text/csv")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."
    assert dataflow.upload_calls == []


def test_upload_dataset_csv_maps_value_error_to_400(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, _ = api_client
    conversation_id = uuid4()

    def _raise_value_error(**_: object) -> UUID:
        raise ValueError("Uploaded file is not a valid CSV: parse error")

    dataflow.upload_csv_data = _raise_value_error  # type: ignore[method-assign]

    response = client.post(
        f"/v1/conversations/{conversation_id}/datasets",
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
        f"/v1/conversations/{conversation_id}/artifacts/{artifact_id}",
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
            "artifact_id": artifact_id,
            "artifact_kind": "data",
            "artifact_format": "csv",
        }
    ]


def test_get_artifact_returns_json_inline(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, _ = api_client
    conversation_id = uuid4()
    artifact_id = uuid4()
    dataflow.artifact_result = DataflowArtifactResponse(
        id=artifact_id,
        kind="graph",
        format="json",
        mime="application/json",
        content=b'{"spec":1}',
    )

    response = client.get(
        f"/v1/conversations/{conversation_id}/artifacts/{artifact_id}",
        params={"artifact_kind": "graph", "artifact_format": "json"},
    )

    assert response.status_code == 200
    assert response.content == b'{"spec":1}'
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["content-disposition"] == "inline"


def test_get_artifact_maps_validation_error_to_422(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, _, dataflow, _ = api_client
    conversation_id = uuid4()
    artifact_id = uuid4()

    def _raise_validation_error(**_: object) -> DataflowArtifactResponse:
        raise ValidationError(
            field="artifact_format", reason="Graph artifacts must be in JSON format"
        )

    dataflow.get_artifact = _raise_validation_error  # type: ignore[method-assign]

    response = client.get(
        f"/v1/conversations/{conversation_id}/artifacts/{artifact_id}",
        params={"artifact_kind": "graph", "artifact_format": "csv"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "validation_failed",
        "message": "Validation failed",
        "detail": "Validation error for 'artifact_format': Graph artifacts must be in JSON format",
    }


def test_revert_to_state_requires_non_empty_state_name(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, _ = api_client
    conversation_id = uuid4()

    response = client.post(
        f"/v1/conversations/{conversation_id}/revert",
        json={"state_name": "   "},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "state_name must be a non-empty string"
    assert workflow.revert_calls == []


def test_revert_to_state_calls_workflow_app(
    api_client: tuple[TestClient, _StubWorkflowApp, _StubDataflowApp, AuthenticatedUser],
) -> None:
    client, workflow, _, user = api_client
    conversation_id = uuid4()

    response = client.post(
        f"/v1/conversations/{conversation_id}/revert",
        json={"state_name": "MODEL_SELECTION"},
    )

    assert response.status_code == 200
    assert response.content == b"null"
    assert workflow.revert_calls == [
        {
            "user_id": user.uid,
            "conversation_id": conversation_id,
            "state_name": "MODEL_SELECTION",
        }
    ]
    assert workflow.user_checks == [(user.uid, conversation_id)]


def test_openapi_mentions_revert_message_and_artifact_enums() -> None:
    app.openapi_schema = None
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    invoke_description = schema["paths"]["/v1/conversations/{conversation_id}/invoke"]["post"][
        "description"
    ]
    assert "revert_data_changes" in invoke_description

    artifact_operation = schema["paths"][
        "/v1/conversations/{conversation_id}/artifacts/{artifact_id}"
    ]["get"]
    parameters = {param["name"]: param for param in artifact_operation["parameters"]}
    assert parameters["artifact_kind"]["schema"]["enum"] == ["graph", "data"]
    assert parameters["artifact_format"]["schema"]["enum"] == ["json", "csv"]
    assert "graph -> json" in artifact_operation["description"]
    assert "data -> json|csv" in artifact_operation["description"]
