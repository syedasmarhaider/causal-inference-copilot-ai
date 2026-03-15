from __future__ import annotations

from firebase_admin import auth
import firebase_admin

from python.domain.service.auth_service import AuthService, AuthenticatedUser

class AuthServiceError(Exception):
    """Base authentication service error."""


class InvalidTokenError(AuthServiceError):
    """Raised when the provided token is invalid or cannot be verified."""


class FirebaseAuthService(AuthService):
    def __init__(
        self,
        *,
        app: firebase_admin.App
    ) -> None:
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
            raise InvalidTokenError("failed to verify Firebase token") from exc

        uid = decoded.get("uid")
        if not isinstance(uid, str) or not uid:
            raise InvalidTokenError("verified Firebase token is missing uid")

        email = decoded.get("email")
        if email is not None and not isinstance(email, str):
            email = None

        return AuthenticatedUser(
            uid=uid,
            email=email,
            email_verified=bool(decoded.get("email_verified", False)),
            claims=dict(decoded),
        )
        
    @staticmethod
    def get_firebase_auth_default_app() -> firebase_admin.App:        
        try:
            return firebase_admin.get_app()
        except ValueError:
            return firebase_admin.initialize_app()