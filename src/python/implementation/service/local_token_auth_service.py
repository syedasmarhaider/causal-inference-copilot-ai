from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from typing import Any
from uuid import UUID

from python.domain.service.auth_service import AuthenticatedUser, AuthService, InvalidTokenError

_UUID_CLAIM_KEYS = ("id", "uuid", "user_id", "uid", "sub")


class LocalTokenAuthService(AuthService):
    """Dev-only auth service that trusts a UUID-bearing local bearer token."""

    @classmethod
    def from_env(cls) -> LocalTokenAuthService:
        return cls()

    def verify_token_and_get_user(self, token: str) -> AuthenticatedUser:
        normalized = token.strip()
        if not normalized:
            raise ValueError("token must be a non-empty string")

        payload = _decode_jwt_like_payload(normalized)
        user_id, subject_claim = _extract_user_id(normalized, payload)
        email = payload.get("email") if payload is not None else None
        if not isinstance(email, str):
            email = None
        email_verified_claim = payload.get("email_verified") if payload is not None else None
        email_verified = email_verified_claim if isinstance(email_verified_claim, bool) else True

        return AuthenticatedUser(
            uid=user_id,
            email=email,
            email_verified=email_verified,
            claims={
                **dict(payload or {}),
                "auth_provider": "local",
                "local_dev": True,
                "local_subject_claim": subject_claim,
                "uid": str(user_id),
            },
        )


def _decode_jwt_like_payload(token: str) -> Mapping[str, Any] | None:
    parts = token.split(".")
    if len(parts) == 1:
        return None
    if len(parts) < 2 or not parts[1]:
        raise InvalidTokenError("local bearer token payload is missing")

    payload_segment = parts[1]
    padded_payload = payload_segment + "=" * (-len(payload_segment) % 4)
    try:
        decoded_payload = base64.urlsafe_b64decode(padded_payload.encode("ascii"))
        payload = json.loads(decoded_payload.decode("utf-8"))
    except (binascii.Error, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidTokenError("local bearer token payload is not valid JWT JSON") from exc

    if not isinstance(payload, dict):
        raise InvalidTokenError("local bearer token payload must be a JSON object")
    return payload


def _extract_user_id(
    token: str,
    payload: Mapping[str, Any] | None,
) -> tuple[UUID, str]:
    if payload is None:
        try:
            return UUID(token), "token"
        except ValueError as exc:
            raise InvalidTokenError("local bearer token must be a UUID or JWT-like token") from exc

    for claim in _UUID_CLAIM_KEYS:
        if claim not in payload:
            continue
        candidate = payload[claim]
        if not isinstance(candidate, str):
            raise InvalidTokenError(f"local bearer token claim `{claim}` must be a UUID string")
        try:
            return UUID(candidate), claim
        except ValueError as exc:
            raise InvalidTokenError(f"local bearer token claim `{claim}` must be a UUID") from exc

    raise InvalidTokenError("local bearer token payload must include a UUID identity claim")
