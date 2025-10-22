from abc import ABC, abstractmethod
from collections.abc import Iterator

import pandas as pd


class DataRepo(ABC):
    @abstractmethod
    def get_csv_data(self, id: str, limit: int | None = None) -> pd.DataFrame:
        """
        Retrieves data from a CSV source.

        :param id: A string identifier for the data source.
        :param limit: An optional integer to limit the number of columns returned.
        :return: A pandas DataFrame containing the CSV data.
        """

    @abstractmethod
    def get_csv_data_iteratively(self, id: str, chunk_size: int) -> Iterator[pd.DataFrame]:
        """
        Retrieves data from a CSV source in iterative chunks.

        :param id: A string identifier for the data source.
        :param chunk_size: The number of rows per chunk.
        :return: An iterator that yields pandas DataFrames.
        """
