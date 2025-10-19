from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Union, Literal, Optional, List
from uuid import UUID

# --- Scopes ---

@dataclass(frozen=True)
class GlobalScope:
    kind: Literal["global"] = "global"

@dataclass(frozen=True)
class LocalScope:
    conversation_id: UUID
    kind: Literal["local"] = "local"

Scope = Union[GlobalScope, LocalScope]


# --- Domain models (conversation_id is carried by scope when local) ---

@dataclass(frozen=True)
class Conversation:
    user_id: UUID
    value: str                # conversation text / summary blob
    scope: Scope              # GlobalScope | LocalScope
    created_at: datetime
    updated_at: datetime

@dataclass(frozen=True)
class Fact:
    user_id: UUID
    key: str
    value: str
    scope: Scope              # GlobalScope | LocalScope (ties to a conversation when local)
    created_at: datetime
    updated_at: datetime


# --- Utilities (optional) ---

def is_global(scope: Scope) -> bool:
    return isinstance(scope, GlobalScope)

def local_scope(conversation_id: UUID) -> LocalScope:
    return LocalScope(conversation_id=conversation_id)

def global_scope() -> GlobalScope:
    return GlobalScope()


# --- Repository interface (LLM memory) ---

# TODO: no tnx for now but later would add to make it more robust

class ConversationRepo(ABC):
    # Conversations
    @abstractmethod
    def upsert_append_conversation(self, *, conv: Conversation) -> Conversation:
        """
        Insert-or-update a conversation matched by (user_id, scope):
          - GLOBAL: (user_id, kind='global')
          - LOCAL : (user_id, kind='local', conversation_id)
        Implementations may append/merge `value` on upsert.
        Returns the persisted row.
        """
        raise NotImplementedError

    @abstractmethod
    def get_conversation(self, *, user_id: UUID, scope: Scope) -> Optional[Conversation]:
        """
        Fetch a single conversation by (user_id, scope).
        Returns None if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_conversation(self, *, user_id: UUID, scope: Scope) -> int:
        """
        Delete conversation(s) matched by (user_id, scope).
        For GLOBAL, at most one row should exist per user.
        Returns rows affected.
        """
        raise NotImplementedError

    # Facts
    @abstractmethod
    def upsert_fact(self, *, fact: Fact) -> Fact:
        """
        Insert-or-update a fact matched by (user_id, key, scope).
        Returns the persisted row.
        """
        raise NotImplementedError

    @abstractmethod
    def get_fact(self, *, user_id: UUID, key: str, scope: Scope) -> Optional[Fact]:
        """
        Fetch a single fact by (user_id, key, scope).
        Returns None if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def list_facts(
        self,
        *,
        user_id: UUID,
        scope: Scope,
        key_prefix: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Fact]:
        """
        List facts for (user_id, scope), optionally filtered by key prefix and paginated.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_fact(self, *, user_id: UUID, key: str, scope: Scope) -> int:
        """
        Delete a fact by (user_id, key, scope). Returns rows affected.
        """
        raise NotImplementedError
