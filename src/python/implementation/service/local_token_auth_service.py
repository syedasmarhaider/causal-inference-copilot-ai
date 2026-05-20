from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid5

from python.domain.service.auth_service import AuthenticatedUser, AuthService, InvalidTokenError

_UUID_CLAIM_KEYS = ("id", "ID", "uuid", "user_id", "uid", "sub")
_OPAQUE_TOKEN_NAMESPACE = UUID("0a8d8840-1d84-46ab-b7e5-9dfaf72e8b42")


@dataclass(frozen=True)
class DecodedLocalBearerToken:
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class LocalUserIdentity:
    uid: UUID
    subject_claim: str
    email: str | None
    email_verified: bool
    claims: Mapping[str, Any]


class LocalTokenAuthService(AuthService):
    """Dev-only auth service for unsigned local JWT-like or opaque bearer tokens."""

    @classmethod
    def from_env(cls) -> LocalTokenAuthService:
        return cls()

    def verify_token_and_get_user(self, token: str) -> AuthenticatedUser:
        decoded_token = decode_local_bearer_token(token)
        identity = validate_local_token_identity(decoded_token)
        return build_authenticated_user(identity)


def decode_local_bearer_token(token: str) -> DecodedLocalBearerToken:
    normalized = token.strip()
    if not normalized:
        raise ValueError("token must be a non-empty string")

    parts = normalized.split(".")
    if len(parts) != 3:
        return _decode_raw_uuid_token(normalized)

    _, payload_segment, _ = parts
    if not payload_segment:
        raise InvalidTokenError("local bearer token payload is missing")

    payload = _decode_json_segment(payload_segment)
    if not isinstance(payload, dict):
        raise InvalidTokenError("local bearer token payload must be a JSON object")
    return DecodedLocalBearerToken(payload=payload)


def _decode_raw_uuid_token(token: str) -> DecodedLocalBearerToken:
    try:
        user_id = UUID(token)
    except ValueError:
        user_id = uuid5(_OPAQUE_TOKEN_NAMESPACE, token)
        return DecodedLocalBearerToken(
            payload={
                "id": str(user_id),
                "local_token_kind": "opaque",
            }
        )

    return DecodedLocalBearerToken(payload={"id": str(user_id), "local_token_kind": "uuid"})


def _decode_json_segment(segment: str) -> Any:
    padded_segment = segment + "=" * (-len(segment) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded_segment.encode("ascii"))
        return json.loads(decoded.decode("utf-8"))
    except (binascii.Error, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidTokenError("local bearer token payload is not valid JWT JSON") from exc


def validate_local_token_identity(decoded_token: DecodedLocalBearerToken) -> LocalUserIdentity:
    user_id, subject_claim = _extract_user_id(decoded_token.payload)
    email = _optional_string_claim(decoded_token.payload, "email")
    email_verified_claim = decoded_token.payload.get("email_verified")
    email_verified = email_verified_claim if isinstance(email_verified_claim, bool) else True
    return LocalUserIdentity(
        uid=user_id,
        subject_claim=subject_claim,
        email=email,
        email_verified=email_verified,
        claims=decoded_token.payload,
    )


def _extract_user_id(payload: Mapping[str, Any]) -> tuple[UUID, str]:
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


def _optional_string_claim(payload: Mapping[str, Any], claim: str) -> str | None:
    value = payload.get(claim)
    return value if isinstance(value, str) else None


def build_authenticated_user(identity: LocalUserIdentity) -> AuthenticatedUser:
    return AuthenticatedUser(
        uid=identity.uid,
        email=identity.email,
        email_verified=identity.email_verified,
        claims={
            **dict(identity.claims),
            "auth_provider": "local",
            "local_dev": True,
            "local_subject_claim": identity.subject_claim,
            "uid": str(identity.uid),
        },
    )
