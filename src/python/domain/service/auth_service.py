from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID


class AuthServiceError(Exception):
    pass


class InvalidTokenError(AuthServiceError):
    pass


@dataclass(frozen=True)
class AuthenticatedUser:
    uid: UUID
    email: str | None
    email_verified: bool
    claims: Mapping[str, Any]


class AuthService(ABC):
    @abstractmethod
    def verify_token_and_get_user(self, token: str) -> AuthenticatedUser:
        raise NotImplementedError
