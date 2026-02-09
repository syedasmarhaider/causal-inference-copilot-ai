from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import UUID


class ConversationLocks:
    """
    Prevent concurrent drains on the same (user_id, conversation_id).
    Without this, overlapping REST/WS calls can interleave workflow steps.
    """

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._locks: dict[tuple[UUID, UUID], asyncio.Lock] = {}

    async def _get_lock(self, user_id: UUID, conversation_id: UUID) -> asyncio.Lock:
        key = (user_id, conversation_id)
        async with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    @asynccontextmanager
    async def lock(self, user_id: UUID, conversation_id: UUID) -> AsyncIterator[None]:
        lock = await self._get_lock(user_id, conversation_id)
        async with lock:
            yield
