from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from python.adapters.api import app as app_module
from python.domain.service.auth_service import AuthenticatedUser
from python.implementation.service.firebsae_auth_service import InvalidTokenError
from python.implementation.workflows.workflow_app import (
    ArtifactResponse,
    WorkflowRequest,
    WorkflowResponse,
)


def _auth_header(token: str = "valid-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class FakeAuthService:
    def __init__(
        self,
        *,
        user: AuthenticatedUser | None = None,
        error: Exception | None = None,
    ) -> None:
        self._user = user
        self._error = error
        self.tokens: list[str] = []

    def verify_token_and_get_user(self, token: str) -> AuthenticatedUser:
        self.tokens.append(token)
        if self._error is not None:
            raise self._error
        if self._user is None:
            raise AssertionError("FakeAuthService requires a user when no error is configured")
        return self._user


@dataclass
class FakeWorkflow:
    dataset_id: UUID = field(default_factory=uuid4)
    artifact_response: ArtifactResponse = field(
        default_factory=lambda: ArtifactResponse(mime="image/png", content=b"artifact-bytes")
    )
    workflow_response: WorkflowResponse = field(
        default_factory=lambda: WorkflowResponse(
            node_message="node-message",
            needs_input=False,
            needs_data=False,
            current_stage="LOAD_DATASET",
            current_stage_status="PENDING",
            artifact_ids=["artifact-1"],
        )
    )
    created_conversations: list[tuple[UUID, UUID]] = field(default_factory=list)
    uploaded_datasets: list[tuple[UUID, UUID, bytes]] = field(default_factory=list)
    artifact_requests: list[tuple[UUID, UUID, UUID]] = field(default_factory=list)
    handled_requests: list[WorkflowRequest] = field(default_factory=list)

    def create_conversation(self, *, user_id: UUID, conversation_id: UUID) -> None:
        self.created_conversations.append((user_id, conversation_id))

    def upload_csv_data(self, *, user_id: UUID, conversation_id: UUID, csv_bytes: bytes) -> UUID:
        self.uploaded_datasets.append((user_id, conversation_id, csv_bytes))
        return self.dataset_id

    def get_artifact(self, *, user_id: UUID, conversation_id: UUID, artifact_id: UUID) -> ArtifactResponse:
        self.artifact_requests.append((user_id, conversation_id, artifact_id))
        return self.artifact_response

    def handle(self, req: WorkflowRequest) -> WorkflowResponse:
        self.handled_requests.append(req)
        return self.workflow_response


@pytest.fixture(autouse=True)
def _clear_overrides() -> None:
    app_module.app.dependency_overrides.clear()
    if hasattr(app_module.get_workflow_app, "cache_clear"):
        app_module.get_workflow_app.cache_clear()
    if hasattr(app_module.get_auth_service, "cache_clear"):
        app_module.get_auth_service.cache_clear()
    yield
    app_module.app.dependency_overrides.clear()
    if hasattr(app_module.get_workflow_app, "cache_clear"):
        app_module.get_workflow_app.cache_clear()
    if hasattr(app_module.get_auth_service, "cache_clear"):
        app_module.get_auth_service.cache_clear()


def _make_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    workflow: FakeWorkflow | None = None,
    auth_service: FakeAuthService | None = None,
) -> tuple[TestClient, FakeWorkflow]:
    fake_workflow = workflow or FakeWorkflow()
    app_module.app.dependency_overrides[app_module.get_workflow_app] = lambda: fake_workflow
    if auth_service is not None:
        monkeypatch.setattr(app_module, "get_auth_service", lambda: auth_service)
    client = TestClient(app_module.app)
    return client, fake_workflow


def test_healthz_is_public() -> None:
    client = TestClient(app_module.app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_protected_routes_require_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)

    response = client.post("/v1/conversations", json={})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"] == "Missing Authorization header."


def test_protected_routes_require_bearer_scheme(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)

    response = client.post(
        "/v1/conversations",
        headers={"Authorization": "Basic abc"},
        json={},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"] == "Authorization header must use the Bearer scheme."


def test_protected_routes_require_non_empty_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _make_client(monkeypatch)

    response = client.post(
        "/v1/conversations",
        headers={"Authorization": "Bearer   "},
        json={},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"] == "Bearer token is missing."


def test_invalid_token_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    auth_service = FakeAuthService(error=InvalidTokenError("bad token"))
    client, _ = _make_client(monkeypatch, auth_service=auth_service)

    response = client.post("/v1/conversations", headers=_auth_header(), json={})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"] == "Invalid or expired bearer token."


def test_create_conversation_uses_authenticated_user(monkeypatch: pytest.MonkeyPatch) -> None:
    user = AuthenticatedUser(uid=uuid4(), email="user@example.com", email_verified=False, claims={})
    auth_service = FakeAuthService(user=user)
    client, workflow = _make_client(monkeypatch, auth_service=auth_service)

    response = client.post("/v1/conversations", headers=_auth_header(), json={})

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == str(user.uid)
    assert len(workflow.created_conversations) == 1
    created_user_id, created_conversation_id = workflow.created_conversations[0]
    assert created_user_id == user.uid
    assert created_conversation_id == UUID(body["conversation_id"])
    assert auth_service.tokens == ["valid-token"]


def test_invoke_rejects_stale_user_id_field(monkeypatch: pytest.MonkeyPatch) -> None:
    user = AuthenticatedUser(uid=uuid4(), email=None, email_verified=False, claims={})
    auth_service = FakeAuthService(user=user)
    conversation_id = uuid4()
    client, workflow = _make_client(monkeypatch, auth_service=auth_service)

    response = client.post(
        f"/v1/conversations/{conversation_id}/invoke",
        headers=_auth_header(),
        json={"user_id": str(uuid4()), "user_text": "hello"},
    )

    assert response.status_code == 422
    assert workflow.handled_requests == []


def test_invoke_uses_authenticated_user(monkeypatch: pytest.MonkeyPatch) -> None:
    user = AuthenticatedUser(uid=uuid4(), email=None, email_verified=False, claims={})
    auth_service = FakeAuthService(user=user)
    conversation_id = uuid4()
    client, workflow = _make_client(monkeypatch, auth_service=auth_service)

    response = client.post(
        f"/v1/conversations/{conversation_id}/invoke",
        headers=_auth_header(),
        json={"user_text": "hello"},
    )

    assert response.status_code == 200
    assert len(workflow.handled_requests) == 1
    request = workflow.handled_requests[0]
    assert request.user_id == user.uid
    assert request.conversation_id == conversation_id
    assert request.user_message == "hello"
    assert response.json()["user_id"] == str(user.uid)


def test_upload_dataset_uses_authenticated_user(monkeypatch: pytest.MonkeyPatch) -> None:
    user = AuthenticatedUser(uid=uuid4(), email=None, email_verified=False, claims={})
    auth_service = FakeAuthService(user=user)
    conversation_id = uuid4()
    workflow = FakeWorkflow(dataset_id=uuid4())
    client, workflow = _make_client(monkeypatch, auth_service=auth_service, workflow=workflow)

    response = client.post(
        f"/v1/conversations/{conversation_id}/datasets",
        headers=_auth_header(),
        files={"file": ("dataset.csv", b"a,b\n1,2\n", "text/csv")},
    )

    assert response.status_code == 200
    assert workflow.uploaded_datasets == [(user.uid, conversation_id, b"a,b\n1,2\n")]
    assert response.json()["user_id"] == str(user.uid)
    assert response.json()["dataset_id"] == str(workflow.dataset_id)


def test_get_artifact_uses_authenticated_user(monkeypatch: pytest.MonkeyPatch) -> None:
    user = AuthenticatedUser(uid=uuid4(), email=None, email_verified=False, claims={})
    auth_service = FakeAuthService(user=user)
    conversation_id = uuid4()
    artifact_id = uuid4()
    client, workflow = _make_client(monkeypatch, auth_service=auth_service)

    response = client.get(
        f"/v1/conversations/{conversation_id}/artifacts/{artifact_id}",
        headers=_auth_header(),
    )

    assert response.status_code == 200
    assert workflow.artifact_requests == [(user.uid, conversation_id, artifact_id)]
    assert response.content == workflow.artifact_response.content
    assert response.headers["content-type"].startswith(workflow.artifact_response.mime)
