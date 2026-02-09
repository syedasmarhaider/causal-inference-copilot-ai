from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from uuid import UUID

import pandas as pd


class DataRepo(ABC):
    @abstractmethod
    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve data for a dataset_id as a pandas DataFrame.

        :param user_id: User UUID.
        :param conversation_id: Conversation UUID.
        :param dataset_id: Dataset UUID.
        :param limit: Optional row limit (head).
        """

    @abstractmethod
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
        """
        Persist data for a dataset_id to durable storage.

        Expected storage layout (file-backed impl):
          ./data/<user_id>/<conversation_id>/<dataset_id>/data.csv

        :param user_id: User UUID.
        :param conversation_id: Conversation UUID.
        :param dataset_id: Dataset UUID.
        :param df: DataFrame to persist.
        :param overwrite: If False, raise if the target already exists.
        :param include_index: If True, write the DataFrame index into the CSV.
        :return: Path/URI to the persisted CSV (Path for file-backed repos).
        """
