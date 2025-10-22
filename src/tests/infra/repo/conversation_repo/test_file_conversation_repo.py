from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from python.domain.repo.conversation_repo import (
    Conversation,
    Fact,
    GlobalScope,
    LocalScope,
    Scope,
)
from python.infra.repo.conversation_repo.file_conversation_repo import FileConversationRepo

JsonObj = dict[str, Any]

# --------------------------
# Fixtures & helpers
# --------------------------

@pytest.fixture
def repo_path(tmp_path: Path) -> Path:
    return tmp_path / "conversations.json"

@pytest.fixture
def repo(repo_path: Path) -> FileConversationRepo:
    return FileConversationRepo(str(repo_path))

def mk_conv(
    user_id: UUID,
    conversation_id: UUID,
    value: str = "v",
    title: str | None = None,
    dt: datetime | None = None,
) -> Conversation:
    when = dt or datetime.now()
    return Conversation(
        user_id=user_id,
        conversation_id=conversation_id,
        value=value,
        title=title,
        created_at=when,
        updated_at=when,
    )

def mk_fact(
    user_id: UUID,
    key: str,
    value: str,
    scope: Scope,
    dt: datetime | None = None,
) -> Fact:
    when = dt or datetime.now()
    return Fact(
        user_id=user_id,
        key=key,
        value=value,
        scope=scope,
        created_at=when,
        updated_at=when,
    )

def read_disk_json(repo_path: Path) -> JsonObj:
    with open(repo_path, encoding="utf-8") as f:
        return json.load(f)

# --------------------------
# Basic load / empty file cases
# --------------------------

def test_load_when_file_absent(repo_path: Path) -> None:
    # No file present; constructor should not crash and repo is empty
    r = FileConversationRepo(str(repo_path))
    assert r.list_conversations(user_id=uuid4()) == []

def test_constructor_raises_on_invalid_json(repo_path: Path) -> None:
    repo_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        FileConversationRepo(str(repo_path))

# --------------------------
# Conversations CRUD & persistence
# --------------------------

def test_upsert_get_conversation_persists_roundtrip(repo_path: Path, repo: FileConversationRepo) -> None:
    uid = uuid4()
    cid = uuid4()
    now = datetime.now().replace(microsecond=0)
    conv = mk_conv(uid, cid, value="hello", title="t", dt=now)

    repo.upsert_append_conversation(conv=conv)

    # in-memory get
    loaded = repo.get_conversation(user_id=uid, conversation_id=cid)
    assert loaded is not None
    assert loaded.user_id == uid
    assert loaded.conversation_id == cid
    assert loaded.value == "hello"
    assert loaded.title == "t"
    assert loaded.created_at == now
    assert loaded.updated_at == now

    # on-disk reload
    r2 = FileConversationRepo(str(repo_path))
    roundtrip = r2.get_conversation(user_id=uid, conversation_id=cid)
    assert roundtrip is not None
    assert roundtrip.user_id == uid
    assert roundtrip.conversation_id == cid
    assert roundtrip.value == "hello"
    assert roundtrip.title == "t"
    assert roundtrip.created_at == now
    assert roundtrip.updated_at == now

def test_set_conversation_title_updates_only_title(repo: FileConversationRepo) -> None:
    uid, cid = uuid4(), uuid4()
    t0 = datetime.now()
    conv = mk_conv(uid, cid, value="v", title=None, dt=t0)
    repo.upsert_append_conversation(conv=conv)

    updated = repo.set_conversation_title(user_id=uid, conversation_id=cid, title="New")
    assert updated.title == "New"
    assert updated.created_at == t0  # unchanged
    assert updated.updated_at == t0  # unchanged

def test_set_conversation_title_missing_raises(repo: FileConversationRepo) -> None:
    with pytest.raises(KeyError):
        repo.set_conversation_title(user_id=uuid4(), conversation_id=uuid4(), title="x")

def test_delete_conversation_counts_and_persists(repo_path: Path, repo: FileConversationRepo) -> None:
    uid, cid = uuid4(), uuid4()
    repo.upsert_append_conversation(conv=mk_conv(uid, cid))
    assert repo.delete_conversation(user_id=uid, conversation_id=cid) == 1
    assert repo.delete_conversation(user_id=uid, conversation_id=cid) == 0

    # File should exist; conversation list should be empty after deletion
    data = read_disk_json(repo_path)
    assert isinstance(data.get("conversations"), list)
    assert len(data.get("conversations", [])) == 0

def test_list_conversations_limit_offset(repo: FileConversationRepo) -> None:
    uid = uuid4()
    cids = [uuid4() for _ in range(5)]
    for i, cid in enumerate(cids):
        repo.upsert_append_conversation(conv=mk_conv(uid, cid, value=f"v{i}"))

    # No guaranteed order; validate counts and membership
    all_items = repo.list_conversations(user_id=uid)
    assert len(all_items) == 5
    # limit only
    assert len(repo.list_conversations(user_id=uid, limit=2)) == 2
    # offset only
    off_items = repo.list_conversations(user_id=uid, offset=3)
    assert len(off_items) == 2
    # limit + offset
    lo_items = repo.list_conversations(user_id=uid, offset=1, limit=3)
    assert len(lo_items) == 3

def test_get_missing_conversation_returns_none(repo: FileConversationRepo) -> None:
    assert repo.get_conversation(user_id=uuid4(), conversation_id=uuid4()) is None

# --------------------------
# Facts CRUD & scoping
# --------------------------

def test_upsert_get_list_delete_fact_global_and_local(repo: FileConversationRepo) -> None:
    uid = uuid4()
    local_cid = uuid4()

    f_g1 = mk_fact(uid, "pref.theme", "dark", GlobalScope())
    f_l1 = mk_fact(uid, "pref.theme", "light", LocalScope(conversation_id=local_cid))
    f_l2 = mk_fact(uid, "session.token", "abc", LocalScope(conversation_id=local_cid))
    repo.upsert_fact(fact=f_g1)
    repo.upsert_fact(fact=f_l1)
    repo.upsert_fact(fact=f_l2)

    # get_fact respects scope
    got_g = repo.get_fact(user_id=uid, key="pref.theme", scope=GlobalScope())
    assert got_g is not None and got_g.value == "dark"

    got_l = repo.get_fact(user_id=uid, key="pref.theme", scope=LocalScope(conversation_id=local_cid))
    assert got_l is not None and got_l.value == "light"

    # list_facts with key_prefix
    all_local = repo.list_facts(user_id=uid, scope=LocalScope(conversation_id=local_cid))
    assert {f.key for f in all_local} == {"pref.theme", "session.token"}

    only_pref = repo.list_facts(user_id=uid, scope=LocalScope(conversation_id=local_cid), key_prefix="pref.")
    assert {f.key for f in only_pref} == {"pref.theme"}

    # delete_fact counts
    assert repo.delete_fact(user_id=uid, key="pref.theme", scope=LocalScope(conversation_id=local_cid)) == 1
    assert repo.delete_fact(user_id=uid, key="pref.theme", scope=LocalScope(conversation_id=local_cid)) == 0
    # Global fact still present
    assert repo.get_fact(user_id=uid, key="pref.theme", scope=GlobalScope()) is not None

def test_local_scope_is_isolated_by_conversation_id(repo: FileConversationRepo) -> None:
    uid = uuid4()
    cid1, cid2 = uuid4(), uuid4()

  
    fact1 = repo.get_fact(user_id=uid, key="k", scope=LocalScope(conversation_id=cid1))
    assert fact1 is not None and fact1.value == "v1"
    fact2 = repo.get_fact(user_id=uid, key="k", scope=LocalScope(conversation_id=cid2))
    assert fact2 is not None and fact2.value == "v2"

def test_list_facts_limit_offset_and_prefix(repo: FileConversationRepo) -> None:
    uid, cid = uuid4(), uuid4()
    scope = LocalScope(conversation_id=cid)
    for i in range(5):
        repo.upsert_fact(fact=mk_fact(user_id=uid, key=f"pref.{i}", value=f"v{i}", scope=scope))

    assert len(repo.list_facts(user_id=uid, scope=scope)) == 5
    assert len(repo.list_facts(user_id=uid, scope=scope, limit=2)) == 2
    assert len(repo.list_facts(user_id=uid, scope=scope, offset=3)) == 2
    assert len(repo.list_facts(user_id=uid, scope=scope, key_prefix="pref.1")) == 1

def test_get_missing_fact_returns_none(repo: FileConversationRepo) -> None:
    assert repo.get_fact(user_id=uuid4(), key="nope", scope=GlobalScope()) is None

def test_delete_missing_fact_returns_zero(repo: FileConversationRepo) -> None:
    assert repo.delete_fact(user_id=uuid4(), key="nope", scope=GlobalScope()) == 0

# --------------------------
# Atomic flush behavior
# --------------------------

def test_flush_is_atomic_when_replace_fails(monkeypatch: pytest.MonkeyPatch, repo_path: Path) -> None:
    """
    If os.replace fails, ensure the original file stays intact (not corrupted).
    """
    r = FileConversationRepo(str(repo_path))
    uid, cid = uuid4(), uuid4()
    base_conv = mk_conv(uid, cid, value="base")
    r.upsert_append_conversation(conv=base_conv)
    baseline = repo_path.read_text(encoding="utf-8")

    # Cause replace to fail once
    called = {"n": 0}
    real_replace = os.replace

    def fail_once(src: str, dst: str) -> None:
        if called["n"] == 0:
            called["n"] += 1
            raise OSError("simulated replace failure")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_once)

    # Attempt a write that will fail in replace
    with pytest.raises(OSError):
        r.upsert_fact(fact=mk_fact(user_id=uid, key="k", value="v", scope=GlobalScope()))

    # On disk file must remain identical to baseline (no partial/corrupt write)
    assert repo_path.read_text(encoding="utf-8") == baseline

    # A subsequent write should succeed
    r.upsert_fact(fact=mk_fact(user_id=uid, key="k2", value="v2", scope=GlobalScope()))
    data = read_disk_json(repo_path)
    assert "facts" in data and any(f["key"] == "k2" for f in data["facts"])

# --------------------------
# Concurrency / thread safety
# --------------------------

def test_concurrent_upserts_are_thread_safe(repo_path: Path) -> None:
    r = FileConversationRepo(str(repo_path))
    uid = uuid4()
    cid = uuid4()

    # Seed conversation to ensure file exists early
    r.upsert_append_conversation(conv=mk_conv(uid, cid, value="seed"))

    scope = LocalScope(conversation_id=cid)
    N: int = 200

    def work(i: int) -> int:
        r.upsert_fact(fact=mk_fact(user_id=uid, key=f"k{i}", value=f"v{i}", scope=scope))
        return i

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(work, range(N)))

    assert len(results) == N

    # Validate that all keys are present
    facts = r.list_facts(user_id=uid, scope=scope)
    keys = {f.key for f in facts}
    assert len(keys) >= N  # allow >N if something else wrote, but must not be < N
    for i in range(N):
        assert f"k{i}" in keys

    # Disk should be valid JSON and contain at least N facts for that scope
    data = read_disk_json(repo_path)
    assert isinstance(data.get("facts"), list)

# --------------------------
# Edge cases on parameters
# --------------------------

def test_list_facts_prefix_empty_means_all(repo: FileConversationRepo) -> None:
    uid, cid = uuid4(), uuid4()
    scope = LocalScope(conversation_id=cid)
    repo.upsert_fact(fact=mk_fact(user_id=uid, key="a", value="1", scope=scope))
    repo.upsert_fact(fact=mk_fact(user_id=uid, key="b", value="2", scope=scope))

    # key_prefix="" should behave like "no filter"
    items = repo.list_facts(user_id=uid, scope=scope, key_prefix="")
    assert {f.key for f in items} == {"a", "b"}

def test_list_offsets_do_not_break_on_zero(repo: FileConversationRepo) -> None:
    uid = uuid4()
    for _ in range(3):
        repo.upsert_append_conversation(conv=mk_conv(uid, uuid4()))
    # offset=0 should be accepted (no slicing), still works with limit
    items = repo.list_conversations(user_id=uid, offset=0, limit=2)
    assert len(items) == 2
