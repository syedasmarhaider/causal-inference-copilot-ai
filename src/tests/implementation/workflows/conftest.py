from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
from uuid import UUID

import pandas as pd

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelRecord, ModelsRepo


class InMemoryDataRepo(DataRepo):
    """
    Implements DataRepo ABC.
    Stores frames in-memory keyed by (user_id, conversation_id, dataset_id).
    save_csv_data also writes a CSV to base_dir so the Path contract is satisfied.
    """

    def __init__(self, *, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._dfs: Dict[Tuple[UUID, UUID, UUID], pd.DataFrame] = {}

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        key = (user_id, conversation_id, dataset_id)
        if key not in self._dfs:
            raise FileNotFoundError(f"dataset_id={dataset_id} not found for user={user_id} conv={conversation_id}")
        df = self._dfs[key]
        out = df.head(limit) if limit is not None else df
        return out.copy(deep=True)

    def save_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        df: pd.DataFrame,
        *,
        overwrite: bool = True,
        include_index: bool = False,
    ) -> Path:
        key = (user_id, conversation_id, dataset_id)
        self._dfs[key] = df.copy(deep=True)

        out_dir = self._base_dir / "data" / str(user_id) / str(conversation_id) / str(dataset_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "data.csv"

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {out_path}")

        df.to_csv(out_path, index=include_index)
        return out_path


class InMemoryModelsRepo(ModelsRepo):
    """
    Implements ModelsRepo Protocol.
    Idempotent save: overwrites same key.
    """

    def __init__(self) -> None:
        self._store: Dict[Tuple[UUID, UUID, UUID], ModelRecord] = {}
        self.save_calls: list[Tuple[UUID, UUID, UUID]] = []
        self.load_calls: list[Tuple[UUID, UUID, UUID]] = []
        self.delete_calls: list[Tuple[UUID, UUID, UUID]] = []
        self.exists_calls: list[Tuple[UUID, UUID, UUID]] = []

    def save_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
        model: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.save_calls.append((user_id, conversation_id, model_id))
        meta_dict: Dict[str, Any] = dict(metadata) if metadata is not None else {}
        self._store[(user_id, conversation_id, model_id)] = ModelRecord(
            model_id=model_id,
            model=model,
            metadata=meta_dict,
        )

    def load_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> ModelRecord | None:
        self.load_calls.append((user_id, conversation_id, model_id))
        return self._store.get((user_id, conversation_id, model_id))

    def model_exists(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> bool:
        self.exists_calls.append((user_id, conversation_id, model_id))
        return (user_id, conversation_id, model_id) in self._store

    def delete_model(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        model_id: UUID,
    ) -> None:
        self.delete_calls.append((user_id, conversation_id, model_id))
        self._store.pop((user_id, conversation_id, model_id), None)