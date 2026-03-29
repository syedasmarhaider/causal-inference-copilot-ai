from __future__ import annotations

from uuid import uuid5

import pytest

import python.implementation.service.firebsae_auth_service as firebase_auth_module
from python.implementation.service.firebsae_auth_service import (
    FirebaseAuthService,
    InvalidTokenError,
)


def test_verify_token_returns_authenticated_user(monkeypatch: pytest.MonkeyPatch) -> None:
    app = object()
    captured: dict[str, object] = {}
    decoded = {
        "uid": "firebase-user-1",
        "email": "user@example.com",
        "email_verified": 1,
        "role": "admin",
    }

    def fake_verify_id_token(token: str, *, app: object, check_revoked: bool) -> dict[str, object]:
        captured["token"] = token
        captured["app"] = app
        captured["check_revoked"] = check_revoked
        return decoded

    monkeypatch.setattr(firebase_auth_module.auth, "verify_id_token", fake_verify_id_token)

    service = FirebaseAuthService(app=app)
    user = service.verify_token_and_get_user("  firebase-token  ")

    assert captured == {
        "token": "firebase-token",
        "app": app,
        "check_revoked": True,
    }
    assert user.uid == uuid5(firebase_auth_module._FIREBASE_USER_ID_NAMESPACE, "firebase-user-1")
    assert user.email == "user@example.com"
    assert user.email_verified is True
    assert dict(user.claims) == decoded

    decoded["role"] = "viewer"
    assert dict(user.claims)["role"] == "admin"


def test_verify_token_rejects_empty_input() -> None:
    service = FirebaseAuthService(app=object())

    with pytest.raises(ValueError, match=r"non-empty string"):
        service.verify_token_and_get_user("   ")


def test_verify_token_wraps_firebase_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_verify_id_token(token: str, *, app: object, check_revoked: bool) -> dict[str, object]:
        raise RuntimeError("firebase failure")

    monkeypatch.setattr(firebase_auth_module.auth, "verify_id_token", fake_verify_id_token)
    service = FirebaseAuthService(app=object())

    with pytest.raises(InvalidTokenError, match=r"failed to verify Firebase token"):
        service.verify_token_and_get_user("token")


@pytest.mark.parametrize("uid", [None, "", "   ", 123])
def test_verify_token_requires_uid(monkeypatch: pytest.MonkeyPatch, uid: object) -> None:
    def fake_verify_id_token(token: str, *, app: object, check_revoked: bool) -> dict[str, object]:
        return {
            "uid": uid,
            "email": "user@example.com",
        }

    monkeypatch.setattr(firebase_auth_module.auth, "verify_id_token", fake_verify_id_token)
    service = FirebaseAuthService(app=object())

    with pytest.raises(InvalidTokenError, match=r"missing uid"):
        service.verify_token_and_get_user("token")


def test_verify_token_non_string_email_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_verify_id_token(token: str, *, app: object, check_revoked: bool) -> dict[str, object]:
        return {
            "uid": "firebase-user-2",
            "email": 42,
            "email_verified": False,
        }

    monkeypatch.setattr(firebase_auth_module.auth, "verify_id_token", fake_verify_id_token)
    service = FirebaseAuthService(app=object())

    user = service.verify_token_and_get_user("token")

    assert user.email is None
    assert user.email_verified is False


def test_get_firebase_auth_default_app_returns_existing_app(monkeypatch: pytest.MonkeyPatch) -> None:
    existing_app = object()

    monkeypatch.setattr(firebase_auth_module.firebase_admin, "get_app", lambda: existing_app)

    app = FirebaseAuthService.get_firebase_auth_default_app()

    assert app is existing_app


def test_get_firebase_auth_default_app_requires_project_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_app() -> object:
        raise ValueError("no default app")

    monkeypatch.setattr(firebase_auth_module.firebase_admin, "get_app", fake_get_app)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT_ID", raising=False)
    monkeypatch.setenv("FIREBASE_DATABASE_URL", "https://db.example")

    with pytest.raises(ValueError, match=r"GOOGLE_CLOUD_PROJECT_ID"):
        FirebaseAuthService.get_firebase_auth_default_app()


def test_get_firebase_auth_default_app_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_app() -> object:
        raise ValueError("no default app")

    monkeypatch.setattr(firebase_auth_module.firebase_admin, "get_app", fake_get_app)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_ID", "proj-1")
    monkeypatch.delenv("FIREBASE_DATABASE_URL", raising=False)

    with pytest.raises(ValueError, match=r"FIREBASE_DATABASE_URL"):
        FirebaseAuthService.get_firebase_auth_default_app()


def test_get_firebase_auth_default_app_initializes_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_app() -> object:
        raise ValueError("no default app")

    captured: dict[str, object] = {}
    expected_app = object()
    expected_cred = object()

    def fake_initialize_app(cred: object, options: dict[str, str] | None = None) -> object:
        captured["cred"] = cred
        captured["options"] = options
        return expected_app

    monkeypatch.setattr(firebase_auth_module.firebase_admin, "get_app", fake_get_app)
    monkeypatch.setattr(firebase_auth_module.firebase_admin, "initialize_app", fake_initialize_app)
    monkeypatch.setattr(
        firebase_auth_module.credentials,
        "ApplicationDefault",
        lambda: expected_cred,
    )
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT_ID", "proj-1")
    monkeypatch.setenv("FIREBASE_DATABASE_URL", "https://db.example")

    app = FirebaseAuthService.get_firebase_auth_default_app()

    assert app is expected_app
    assert captured == {
        "cred": expected_cred,
        "options": {
            "projectId": "proj-1",
            "databaseURL": "https://db.example",
        },
    }
