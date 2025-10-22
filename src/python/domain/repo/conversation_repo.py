from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Union
from uuid import UUID

# ======================
# Scopes (for Facts only)
# ======================

@dataclass(frozen=True)
class GlobalScope:
    kind: Literal["global"] = "global"

@dataclass(frozen=True)
class LocalScope:
    conversation_id: UUID
    kind: Literal["local"] = "local"

Scope = Union[GlobalScope, LocalScope]

def global_scope() -> GlobalScope:
    return GlobalScope()

def local_scope(conversation_id: UUID) -> LocalScope:
    return LocalScope(conversation_id=conversation_id)


# ======================
# Domain models
# ======================

@dataclass(frozen=True)
class Conversation:
    """
    A conversation is keyed by (user_id, conversation_id).
    NOTE: Conversations do NOT have scope.
    """
    user_id: UUID
    conversation_id: UUID
    value: str                          # conversation text / summary blob can be json/xml/...
    title: str | None                # human-friendly title
    created_at: datetime
    updated_at: datetime

@dataclass(frozen=True)
class Fact:
    """
    A fact is a key-value attributed to a user and a Scope (global or local-to-conversation).
    For LocalScope, the scope carries conversation_id.
    """
    user_id: UUID
    key: str
    value: str
    scope: Scope
    created_at: datetime
    updated_at: datetime


# ======================
# Repository interface
# ======================

class ConversationRepo(ABC):
    # -------- Conversations (no scope) --------

    @abstractmethod
    def upsert_append_conversation(self, *, conv: Conversation) -> Conversation:
        """
        Insert or update a conversation matched by (user_id, conversation_id).
        On update, append/merge `value` (implementation-defined; commonly newline-append)
        and update `updated_at`. Return the persisted row.
        """

    @abstractmethod
    def get_conversation(self, *, user_id: UUID, conversation_id: UUID) -> Conversation | None:
        """
        Fetch a conversation by (user_id, conversation_id). Return None if not found.
        """

    @abstractmethod
    def delete_conversation(self, *, user_id: UUID, conversation_id: UUID) -> int:
        """
        Delete a conversation by (user_id, conversation_id). Return rows affected (0 or 1).
        """

    @abstractmethod
    def list_conversations(
        self,
        *,
        user_id: UUID,
        limit: int | None = None,
        offset: int | None = None,
        sort: Literal["updated_desc", "updated_asc", "created_desc", "created_asc"] = "updated_desc",
    ) -> list[Conversation]:
        """
        List conversations for a user with deterministic ordering and optional pagination.
        Default sort: most-recently-updated first.
        """

    @abstractmethod
    def set_conversation_title(self, *, user_id: UUID, conversation_id: UUID, title: str | None) -> Conversation:
        """
        Set (or clear if None) the conversation title and bump `updated_at`. Return updated row.
        """

    # -------- Facts (scoped) --------

    @abstractmethod
    def upsert_fact(self, *, fact: Fact) -> Fact:
        """
        Insert-or-update a fact matched by (user_id, key, scope).
        On update, overwrite value and bump `updated_at`. Return persisted row.
        """

    @abstractmethod
    def get_fact(self, *, user_id: UUID, key: str, scope: Scope) -> Fact | None:
        """
        Fetch a single fact by (user_id, key, scope). Return None if not found.
        """

    @abstractmethod
    def list_facts(
        self,
        *,
        user_id: UUID,
        scope: Scope,
        key_prefix: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        sort: Literal["key_asc_created_asc", "key_asc_created_desc"] = "key_asc_created_asc",
    ) -> list[Fact]:
        """
        List facts for (user_id, scope), optionally filtered by key prefix and paginated.
        Ordering is deterministic.
        """

    @abstractmethod
    def delete_fact(self, *, user_id: UUID, key: str, scope: Scope) -> int:
        """
        Delete a fact by (user_id, key, scope). Return rows affected (0 or 1).
        """
