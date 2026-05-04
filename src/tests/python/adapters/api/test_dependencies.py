from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from python.adapters.api import dependencies
from python.domain.service.auth_service import AuthenticatedUser
from python.implementation.service.firebsae_auth_service import (
    AuthServiceError,
    InvalidTokenError,
)


def _credentials(token: str = "token") -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


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
