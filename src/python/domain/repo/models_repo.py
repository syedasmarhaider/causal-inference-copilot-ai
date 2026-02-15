from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol
from uuid import UUID


class ModelNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class ModelRecord:
    model_id: UUID
    model: Any
    metadata: Dict[str, Any]


class ModelsRepo(Protocol):
    """
    Persistent storage for fitted models keyed by (user_id, conversation_id, model_id).
    """

    def save_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
        model: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        Persist a model artifact and optional metadata.
        Must be idempotent for the same (user_id, conversation_id, model_id).
        """

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

    def model_exists(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> bool: # pyright: ignore[reportReturnType]
        """
        True iff the model artifact exists.
        """

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
