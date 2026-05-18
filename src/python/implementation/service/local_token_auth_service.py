from __future__ import annotations

import os
from uuid import UUID, uuid5

from python.domain.service.auth_service import AuthenticatedUser, AuthService, InvalidTokenError

_LOCAL_USER_ID_NAMESPACE = UUID("d6ba21a4-85d9-4bb4-a8f8-7fc181ca4e3e")


class LocalTokenAuthService(AuthService):
    """Dev-only auth service that validates a configured local bearer token."""

    def __init__(self, *, id_token: str) -> None:
        normalized = id_token.strip()
        if not normalized:
            raise ValueError("ID_TOKEN environment variable must be set")
        self._id_token = normalized
        self._user_id = uuid5(_LOCAL_USER_ID_NAMESPACE, normalized)

    @classmethod
    def from_env(cls) -> LocalTokenAuthService:
        return cls(id_token=os.getenv("ID_TOKEN", ""))

    def verify_token_and_get_user(self, token: str) -> AuthenticatedUser:
        normalized = token.strip()
        if not normalized:
            raise ValueError("token must be a non-empty string")
        if normalized != self._id_token:
            raise InvalidTokenError("local bearer token does not match ID_TOKEN")

        return AuthenticatedUser(
            uid=self._user_id,
            email="local@example.test",
            email_verified=True,
            claims={
                "auth_provider": "local",
                "local_dev": True,
                "uid": str(self._user_id),
            },
        )
