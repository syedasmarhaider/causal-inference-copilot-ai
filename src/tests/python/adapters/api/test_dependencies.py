from __future__ import annotations

import asyncio
import base64
import json
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from python.adapters.api import dependencies
from python.domain.service.auth_service import (
    AuthenticatedUser,
    AuthServiceError,
    InvalidTokenError,
)
from python.implementation.service.local_token_auth_service import LocalTokenAuthService


def _credentials(token: str = "token") -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _jwt_like(payload: dict[str, object]) -> str:
    def _encode(value: dict[str, object]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_encode({'alg': 'none', 'typ': 'JWT'})}.{_encode(payload)}."


def test_get_workflow_and_dataflow_apps_share_cached_make_apps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    workflow_app = object()
    dataflow_app = object()
    audit_log_app = object()

    dependencies.get_apps.cache_clear()
    dependencies.get_workflow_app.cache_clear()
    dependencies.get_dataflow_app.cache_clear()
    dependencies.get_audit_log_app.cache_clear()

    def _make_apps():
        calls.append("make_apps")
        return workflow_app, dataflow_app, audit_log_app

    monkeypatch.setattr(dependencies, "make_apps", _make_apps)

    assert dependencies.get_workflow_app() is workflow_app
    assert dependencies.get_dataflow_app() is dataflow_app
    assert dependencies.get_audit_log_app() is audit_log_app
    assert calls == ["make_apps"]

    dependencies.get_apps.cache_clear()
    dependencies.get_workflow_app.cache_clear()
    dependencies.get_dataflow_app.cache_clear()
    dependencies.get_audit_log_app.cache_clear()


def test_get_auth_service_uses_local_token_auth() -> None:
    dependencies.get_auth_service.cache_clear()
    user_id = uuid4()
    token = _jwt_like({"id": str(user_id), "email": "local@example.test"})

    auth_service = dependencies.get_auth_service()

    assert isinstance(auth_service, LocalTokenAuthService)
    user = auth_service.verify_token_and_get_user(token)
    assert user.uid == user_id
    assert user.email == "local@example.test"
    assert user.claims["auth_provider"] == "local"
    assert user.claims["local_subject_claim"] == "id"

    with pytest.raises(InvalidTokenError):
        auth_service.verify_token_and_get_user("other-token")

    dependencies.get_auth_service.cache_clear()


def test_get_auth_service_accepts_uppercase_id_claim() -> None:
    dependencies.get_auth_service.cache_clear()
    user_id = uuid4()
    token = _jwt_like({"ID": str(user_id), "email_verified": False})

    auth_service = dependencies.get_auth_service()
    user = auth_service.verify_token_and_get_user(token)

    assert user.uid == user_id
    assert user.email is None
    assert user.email_verified is False
    assert user.claims["local_subject_claim"] == "ID"

    dependencies.get_auth_service.cache_clear()


def test_get_authenticated_user_returns_verified_user(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_user = AuthenticatedUser(
        uid=uuid4(),
        email="user@example.com",
        email_verified=True,
        claims={"role": "tester"},
    )

    class _AuthService:
        def verify_token_and_get_user(self, token: str) -> AuthenticatedUser:
            assert token == "valid-token"
            return expected_user

    monkeypatch.setattr(dependencies, "get_auth_service", lambda: _AuthService())

    user = asyncio.run(
        dependencies.get_authenticated_user(
            credentials=_credentials("valid-token"),
            authorization="Bearer valid-token",
        )
    )

    assert user == expected_user


def test_get_authenticated_user_accepts_local_jwt_like_token() -> None:
    dependencies.get_auth_service.cache_clear()
    user_id = uuid4()
    token = _jwt_like({"uuid": str(user_id)})

    user = asyncio.run(
        dependencies.get_authenticated_user(
            credentials=_credentials(token),
            authorization=f"Bearer {token}",
        )
    )

    assert user.uid == user_id
    assert user.claims["local_subject_claim"] == "uuid"

    dependencies.get_auth_service.cache_clear()


@pytest.mark.parametrize(
    ("authorization", "credentials", "detail"),
    [
        (None, _credentials(), "Missing Authorization header."),
        ("Basic abc", _credentials(), "Authorization header must use the Bearer scheme."),
        ("Bearer   ", _credentials(""), "Bearer token is missing."),
        ("Bearer valid-token", None, "Invalid or expired bearer token."),
    ],
)
def test_get_authenticated_user_rejects_invalid_headers(
    authorization: str | None,
    credentials: HTTPAuthorizationCredentials | None,
    detail: str,
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            dependencies.get_authenticated_user(
                credentials=credentials,
                authorization=authorization,
            )
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == detail


def test_get_authenticated_user_maps_invalid_token_to_401(monkeypatch: pytest.MonkeyPatch) -> None:
    class _AuthService:
        def verify_token_and_get_user(self, token: str) -> AuthenticatedUser:
            raise InvalidTokenError("bad token")

    monkeypatch.setattr(dependencies, "get_auth_service", lambda: _AuthService())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            dependencies.get_authenticated_user(
                credentials=_credentials("bad-token"),
                authorization="Bearer bad-token",
            )
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid or expired bearer token."


def test_get_authenticated_user_maps_auth_service_error_to_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AuthService:
        def verify_token_and_get_user(self, token: str) -> AuthenticatedUser:
            raise AuthServiceError("service unavailable")

    monkeypatch.setattr(dependencies, "get_auth_service", lambda: _AuthService())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            dependencies.get_authenticated_user(
                credentials=_credentials("any-token"),
                authorization="Bearer any-token",
            )
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "authentication service unavailable"


def test_get_authenticated_user_maps_unexpected_error_to_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AuthService:
        def verify_token_and_get_user(self, token: str) -> AuthenticatedUser:
            raise RuntimeError("boom")

    monkeypatch.setattr(dependencies, "get_auth_service", lambda: _AuthService())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            dependencies.get_authenticated_user(
                credentials=_credentials("any-token"),
                authorization="Bearer any-token",
            )
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "authentication service unavailable"
