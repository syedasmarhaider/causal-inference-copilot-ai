from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime
from threading import RLock
from typing import Any, cast
from uuid import UUID

from python.domain.repo.conversation_repo import (
    Conversation,
    ConversationRepo,
    Fact,
    GlobalScope,
    LocalScope,
    Scope,
)

# -----------------------------
# Helpers: scope & serialization
# -----------------------------

def _scope_key(scope: Scope) -> tuple[str, UUID | None]:
    if isinstance(scope, LocalScope):
        return ("local", scope.conversation_id)
    return ("global", None)

def _dt_to_str(dt: datetime) -> str:
    return dt.isoformat()

def _dt_from_str(s: str) -> datetime:
    return datetime.fromisoformat(s)

def _scope_to_dict(scope: Scope) -> dict[str, str]:
    if isinstance(scope, LocalScope):
        return {"kind": "local", "conversation_id": str(scope.conversation_id)}
    return {"kind": "global"}

def _scope_from_dict(d: Mapping[str, Any]) -> Scope:
    kind = cast(str | None, d.get("kind"))
    if kind == "local":
        cid_raw = d.get("conversation_id")
        if not isinstance(cid_raw, str):
            raise TypeError("scope.conversation_id must be a string when kind='local'")
        return LocalScope(conversation_id=UUID(cid_raw))
    return GlobalScope()

def _conv_to_dict(c: Conversation) -> dict[str, Any]:
    return {
        "user_id": str(c.user_id),
        "conversation_id": str(c.conversation_id),
        "value": c.value,
        "title": c.title,
        "created_at": _dt_to_str(c.created_at),
        "updated_at": _dt_to_str(c.updated_at),
    }

def _conv_from_dict(d: Mapping[str, Any]) -> Conversation:
    user_id_str = cast(str, d["user_id"])
    conv_id_str = cast(str, d["conversation_id"])
    value = cast(str, d["value"])
    title = cast(str | None, d.get("title"))
    created_at_str = cast(str, d["created_at"])
    updated_at_str = cast(str, d["updated_at"])
    return Conversation(
        user_id=UUID(user_id_str),
        conversation_id=UUID(conv_id_str),
        value=value,
        title=title,
        created_at=_dt_from_str(created_at_str),
        updated_at=_dt_from_str(updated_at_str),
    )

def _fact_to_dict(f: Fact) -> dict[str, Any]:
    return {
        "user_id": str(f.user_id),
        "key": f.key,
        "value": f.value,
        "scope": _scope_to_dict(f.scope),
        "created_at": _dt_to_str(f.created_at),
        "updated_at": _dt_to_str(f.updated_at),
    }

def _fact_from_dict(d: Mapping[str, Any]) -> Fact:
    user_id_str = cast(str, d["user_id"])
    key = cast(str, d["key"])
    value = cast(str, d["value"])
    scope_dict = cast(Mapping[str, Any], d["scope"])
    created_at_str = cast(str, d["created_at"])
    updated_at_str = cast(str, d["updated_at"])
    return Fact(
        user_id=UUID(user_id_str),
        key=key,
        value=value,
        scope=_scope_from_dict(scope_dict),
        created_at=_dt_from_str(created_at_str),
        updated_at=_dt_from_str(updated_at_str),
    )

# -----------------------------
# Dumb file-backed repository
# -----------------------------

class FileConversationRepo(ConversationRepo):
    """
    DUMB file-backed repo: insert/replace exactly what you pass.
    Writes a compact JSON snapshot on each mutation (atomic rename).
    Thread-safe within a single process (RLock).
    """

    def __init__(self, path: str) -> None:
        self._path: str = path
        self._tmp: str = path + ".tmp"
        self._lock = RLock()
        self._convs: dict[tuple[UUID, UUID], Conversation] = {}
        self._facts: dict[tuple[UUID, str, str, UUID | None], Fact] = {}
        self._load()

    # --- persistence ---

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)

        self._convs.clear()
        self._facts.clear()

        for d in data.get("conversations", []):
            c = _conv_from_dict(d)
            self._convs[(c.user_id, c.conversation_id)] = c

        for d in data.get("facts", []):
            fa = _fact_from_dict(d)
            kind, cid = _scope_key(fa.scope)
            self._facts[(fa.user_id, fa.key, kind, cid)] = fa

    def _flush(self) -> None:
        data = {
            "conversations": [_conv_to_dict(c) for c in self._convs.values()],
            "facts": [_fact_to_dict(f) for f in self._facts.values()],
        }
        with open(self._tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(self._tmp, self._path)

    # -------- Conversations --------

    def upsert_append_conversation(self, *, conv: Conversation) -> Conversation:
        # DUMB = upsert/replace as-is
        with self._lock:
            self._convs[(conv.user_id, conv.conversation_id)] = conv
            self._flush()
            return conv

    def get_conversation(self, *, user_id: UUID, conversation_id: UUID) -> Conversation | None:
        with self._lock:
            return self._convs.get((user_id, conversation_id))

    def delete_conversation(self, *, user_id: UUID, conversation_id: UUID) -> int:
        with self._lock:
            k = (user_id, conversation_id)
            if k in self._convs:
                del self._convs[k]
                self._flush()
                return 1
            return 0

    def list_conversations(
        self,
        *,
        user_id: UUID,
        limit: int | None = None,
        offset: int | None = None,
        sort: str = "updated_desc",  # accepted but ignored (dumb repo)
    ) -> list[Conversation]:
        with self._lock:
            items = [c for (uid, _), c in self._convs.items() if uid == user_id]
            if offset:
                items = items[offset:]
            if limit is not None:
                items = items[:limit]
            return list(items)

    def set_conversation_title(self, *, user_id: UUID, conversation_id: UUID, title: str | None) -> Conversation:
        # DUMB: replace the object with identical fields except title; no timestamp changes
        with self._lock:
            k = (user_id, conversation_id)
            existing = self._convs.get(k)
            if existing is None:
                raise KeyError("conversation not found")
            new_conv = Conversation(
                user_id=existing.user_id,
                conversation_id=existing.conversation_id,
                value=existing.value,
                title=title,
                created_at=existing.created_at,
                updated_at=existing.updated_at,
            )
            self._convs[k] = new_conv
            self._flush()
            return new_conv

    # -------- Facts --------

    def upsert_fact(self, *, fact: Fact) -> Fact:
        with self._lock:
            kind, cid = _scope_key(fact.scope)
            self._facts[(fact.user_id, fact.key, kind, cid)] = fact
            self._flush()
            return fact

    def get_fact(self, *, user_id: UUID, key: str, scope: Scope) -> Fact | None:
        with self._lock:
            kind, cid = _scope_key(scope)
            return self._facts.get((user_id, key, kind, cid))

    def list_facts(
        self,
        *,
        user_id: UUID,
        scope: Scope,
        key_prefix: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        sort: str = "key_asc_created_asc",  # accepted but ignored (dumb repo)
    ) -> list[Fact]:
        with self._lock:
            kind, cid = _scope_key(scope)
            items = [
                f for (uid, k, knd, scid), f in self._facts.items()
                if uid == user_id and knd == kind and scid == cid
                and (key_prefix is None or k.startswith(key_prefix))
            ]
            if offset:
                items = items[offset:]
            if limit is not None:
                items = items[:limit]
            return list(items)

    def delete_fact(self, *, user_id: UUID, key: str, scope: Scope) -> int:
        with self._lock:
            kind, cid = _scope_key(scope)
            k = (user_id, key, kind, cid)
            if k in self._facts:
                del self._facts[k]
                self._flush()
                return 1
            return 0
