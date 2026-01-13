# src/python/implementation/repo/file_data_repo.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import UUID

import pandas as pd

from python.domain.repo.data_repo import DataRepo


# Edit this later when you wire real dataset storage/registration.
DEFAULT_DATASET_PATH: Final[Path] = Path(
    "./data/486f4975-6cd9-4261-a122-e6b0fc46462d/data.csv"
).resolve()


@dataclass(frozen=True)
class FileDataRepo(DataRepo):
    """
    Temporary implementation:
      - ignores user_id / conversation_id / dataset_id
      - always reads DEFAULT_DATASET_PATH
    """

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        csv_path = DEFAULT_DATASET_PATH

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at DEFAULT_DATASET_PATH: {csv_path}")
        if not csv_path.is_file():
            raise FileNotFoundError(f"DEFAULT_DATASET_PATH is not a file: {csv_path}")

        nrows: int | None
        if limit is None:
            nrows = None
        else:
            if limit <= 0:
                raise ValueError(f"limit must be a positive int or None, got: {limit!r}")
            nrows = limit

        try:
            return pd.read_csv( # pyright: ignore[reportUnknownMemberType]
                csv_path,
                nrows=nrows,
                low_memory=False,
            )
        except Exception as e:
            raise ValueError(f"Failed to read CSV at {csv_path}: {e}") from e
