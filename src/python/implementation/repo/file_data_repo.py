# src/python/implementation/repo/data_repo/file_data_repo.py
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Dict, Optional, cast
from uuid import UUID, uuid4
import json

import pandas as _pd  # real import

from domain.repo.data_repo import DataRepo

# Treat pandas as Any so member access doesn't become "Unknown"
pd = cast(Any, _pd)

JSONDict = Dict[str, Any]
DatasetsIndex = Dict[str, Dict[str, str]]        # dataset_id -> {"path": "..."}
ConversationsIndex = Dict[str, str]              # conversation_id -> dataset_id


class FileDataRepo(DataRepo):
    """
    File-backed implementation of DataRepo.

    - Stores an index of dataset_id -> dataset_path in datasets_index.json
    - Stores conversation_id -> dataset_id in conversation_index.json

    Intended for local / dev use; safe enough for your LangGraph prototype.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

        self._datasets_index_path = self._root / "datasets_index.json"
        self._conversations_index_path = self._root / "conversation_index.json"

        # Ensure index files exist
        for p in (self._datasets_index_path, self._conversations_index_path):
            if not p.exists():
                p.write_text("{}", encoding="utf-8")

    # ---------- helpers ----------

    def _load_json(self, path: Path) -> JSONDict:
        try:
            raw = path.read_text(encoding="utf-8")
            if not raw.strip():
                return {}
            data = json.loads(raw)
            if isinstance(data, dict):
                return data # pyright: ignore[reportUnknownVariableType]
            return {}
        except Exception:
            # If file is corrupted, treat as empty but don't crash the app
            return {}

    def _load_datasets_index(self) -> DatasetsIndex:
        raw = self._load_json(self._datasets_index_path)
        result: DatasetsIndex = {}
        for k, v in raw.items():
            if not isinstance(k, str): # pyright: ignore[reportUnnecessaryIsInstance]
                continue
            if not isinstance(v, dict):
                continue
            path_val = v.get("path") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            if not isinstance(path_val, str):
                continue
            result[k] = {"path": path_val}
        return result

    def _load_conversations_index(self) -> ConversationsIndex:
        raw = self._load_json(self._conversations_index_path)
        result: ConversationsIndex = {}
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, str): # pyright: ignore[reportUnnecessaryIsInstance]
                result[k] = v
        return result

    def _save_datasets_index(self, data: DatasetsIndex) -> None:
        self._save_json(self._datasets_index_path, data)

    def _save_conversations_index(self, data: ConversationsIndex) -> None:
        self._save_json(self._conversations_index_path, data)

    def _save_json(self, path: Path, data: JSONDict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    # ---------- DataRepo implementation ----------

    def register_csv_dataset(
        self, *, conversation_id: UUID, dataset_path: str
    ) -> UUID:
        datasets = self._load_datasets_index()
        conversations = self._load_conversations_index()

        conv_key = str(conversation_id)
        dataset_id: UUID

        existing_id_val = conversations.get(conv_key)
        if isinstance(existing_id_val, str) and existing_id_val in datasets:
            # Reuse existing dataset_id for this conversation
            dataset_id = UUID(existing_id_val)
        else:
            dataset_id = uuid4()

        datasets[str(dataset_id)] = {"path": dataset_path}
        conversations[conv_key] = str(dataset_id)

        self._save_datasets_index(datasets)
        self._save_conversations_index(conversations)

        return dataset_id

    def get_last_dataset(
        self, *, conversation_id: UUID
    ) -> tuple[Optional[UUID], Optional[str]]:
        datasets = self._load_datasets_index()
        conversations = self._load_conversations_index()

        conv_key = str(conversation_id)
        ds_id_val = conversations.get(conv_key)
        if not isinstance(ds_id_val, str):
            return None, None

        entry = datasets.get(ds_id_val)
        if not isinstance(entry, dict):
            return None, None

        path_val = entry.get("path")
        if not isinstance(path_val, str):
            return None, None

        try:
            ds_id = UUID(ds_id_val)
        except (ValueError, TypeError):
            return None, None

        return ds_id, path_val

    def _resolve_path(self, id: UUID) -> Path:
        datasets = self._load_datasets_index()
        entry = datasets.get(str(id))
        if not isinstance(entry, dict):
            raise KeyError(f"No dataset registered for id={id}")

        path_val = entry.get("path")
        if not isinstance(path_val, str):
            raise KeyError(f"Dataset {id} has no valid 'path' entry")

        return Path(path_val)

    def get_csv_data(self, id: UUID, limit: int | None = None) -> _pd.DataFrame:
        path = self._resolve_path(id)
        df = pd.read_csv(path)
        if limit is not None and limit >= 0:
            df = df.head(limit)
        return cast(_pd.DataFrame, df)

    def get_csv_data_iteratively(
        self, id: UUID, chunk_size: int
    ) -> Iterator[_pd.DataFrame]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        path = self._resolve_path(id)
        reader = pd.read_csv(path, chunksize=chunk_size)
        return cast(Iterator[_pd.DataFrame], reader)
