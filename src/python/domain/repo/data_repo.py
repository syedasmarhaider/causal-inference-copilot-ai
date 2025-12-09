# src/python/domain/repo/data_repo.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Optional
from uuid import UUID

import pandas as pd


class DataRepo(ABC):
    """
    Abstract repository for datasets.

    Responsibilities:
    - Register a CSV dataset (path on disk) under a stable UUID.
    - Load that dataset as a pandas DataFrame (full or in chunks).
    - Remember the *last* dataset used in a conversation, so the
      workflow can resume without the user re-supplying the path.
    """

    # -------- registration / lookup --------

    @abstractmethod
    def register_csv_dataset(
        self, *, conversation_id: UUID, dataset_path: str
    ) -> UUID:
        """
        Register or update a CSV dataset for this conversation.

        Returns a dataset_id (UUID) that can be used later with get_csv_data.
        Implementations are free to reuse an existing UUID if the same
        path is registered again for the same conversation.
        """

    @abstractmethod
    def get_last_dataset(
        self, *, conversation_id: UUID
    ) -> tuple[Optional[UUID], Optional[str]]:
        """
        Return (dataset_id, dataset_path) last associated with this
        conversation, or (None, None) if nothing stored yet.
        """

    # -------- data access --------

    @abstractmethod
    def get_csv_data(self, id: UUID, limit: int | None = None) -> pd.DataFrame:
        """
        Retrieve data for a dataset_id as a pandas DataFrame.

        :param id: Dataset UUID.
        :param limit: Optional row limit (head).
        """

    @abstractmethod
    def get_csv_data_iteratively(
        self, id: UUID, chunk_size: int
    ) -> Iterator[pd.DataFrame]:
        """
        Retrieve data for a dataset_id in chunks.

        :param id: Dataset UUID.
        :param chunk_size: Number of rows per chunk.
        :return: Iterator yielding DataFrames.
        """
