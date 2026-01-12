# src/python/domain/repo/data_repo.py
from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

import pandas as pd


class DataRepo(ABC):
    @abstractmethod
    def get_csv_data(self, user_id: UUID, conversation_id: UUID, dataset_id: UUID, limit: int | None = None) -> pd.DataFrame:
        """
        Retrieve data for a dataset_id as a pandas DataFrame.

        :param user_id: User UUID.
        :param conversation_id: Conversation UUID.
        :param limit: Optional row limit (head).
        """
