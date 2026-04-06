from __future__ import annotations

import os
from uuid import UUID, uuid5

import firebase_admin
from firebase_admin import auth, credentials

from python.domain.service.auth_service import AuthenticatedUser, AuthService
from python.implementation.service.logging.default_logging import get_logger

_FIREBASE_USER_ID_NAMESPACE = UUID("2d5c4b6d-7f6b-4d8e-9a2d-1f5e9d9d8c11")
log = get_logger(__name__)


class AuthServiceError(Exception):
    pass


class InvalidTokenError(AuthServiceError):
    pass


class FirebaseAuthService(AuthService):
    def __init__(self, *, app: firebase_admin.App) -> None:
        self._app = app

    def verify_token_and_get_user(self, token: str) -> AuthenticatedUser:
        normalized_token = token.strip()
        if not normalized_token:
            raise ValueError("token must be a non-empty string")

        try:
            decoded = auth.verify_id_token(
                normalized_token,
                app=self._app,
                check_revoked=True,
            )
        except Exception as exc:
            log.warning("Failed to verify Firebase token", error=exc)
            raise InvalidTokenError("failed to verify Firebase token") from exc
        
        raw_uid = decoded.get("uid")
        if not isinstance(raw_uid, str) or not raw_uid.strip():
            raise InvalidTokenError("verified Firebase token is missing uid")

        email = decoded.get("email")
        if email is not None and not isinstance(email, str):
            email = None

   
        return AuthenticatedUser(
            uid=uuid5(_FIREBASE_USER_ID_NAMESPACE, raw_uid),
            email=email,
            email_verified=bool(decoded.get("email_verified", False)),
            claims=dict(decoded),
        )

    @staticmethod
    def get_firebase_auth_default_app() -> firebase_admin.App:
        try:
            return firebase_admin.get_app()
        except ValueError:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "").strip()
            if not project_id:
                raise ValueError(
                    "GOOGLE_CLOUD_PROJECT_ID environment variable must be set for FirebaseAuthService"
                ) from None
            database_url = os.getenv("FIREBASE_DATABASE_URL", "").strip()
            if not database_url:
                raise ValueError(
                    "FIREBASE_DATABASE_URL environment variable must be set for FirebaseAuthService"
                ) from None

            options: dict[str, str] = {}
            options["projectId"] = project_id
            options["databaseURL"] = database_url

            return firebase_admin.initialize_app(
                credentials.ApplicationDefault(),
                options or None,
            )
