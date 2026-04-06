from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID


class ModelNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class ModelRecord:
    model_id: UUID
    model: Any
    metadata: dict[str, Any]


class ModelsRepo(ABC):
    """
    Persistent storage for fitted models keyed by (user_id, conversation_id, model_id).
    """

    @abstractmethod
    def save_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
        model: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Persist a model artifact and optional metadata.
        Must be idempotent for the same (user_id, conversation_id, model_id).
        """

    @abstractmethod
    def load_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> ModelRecord | None:
        """
        Load a model artifact and metadata.
        Returns None if not found.
        """

    @abstractmethod
    def model_exists(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> bool:  # pyright: ignore[reportReturnType]
        """
        True iff the model artifact exists.
        """

    @abstractmethod
    def delete_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> None:
        """
        Best-effort delete. No-op if missing.
        """
