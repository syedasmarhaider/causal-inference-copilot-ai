# src/python/implementation/repo/file_data_repo.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import UUID, uuid4

import pandas as pd

from python.domain.repo.data_repo import DataRepo


# Root folder for file-backed datasets.
DATA_ROOT: Final[Path] = Path("./data").resolve()
CSV_FILENAME: Final[str] = "data.csv"

# Edit this later when you wire real dataset storage/registration.
DEFAULT_DATASET_PATH: Final[Path] = Path(
    "./data/486f4975-6cd9-4261-a122-e6b0fc46462d/data.csv"
).resolve()


@dataclass(frozen=True)
class FileDataRepo(DataRepo):
    """
    Temporary implementation:
      - Reads: prefers ./data/<user>/<conversation>/<dataset>/data.csv if present,
               otherwise falls back to DEFAULT_DATASET_PATH.
      - Writes: always writes to ./data/<user>/<conversation>/<dataset>/data.csv
    """

    def _dataset_csv_path(self, user_id: UUID, conversation_id: UUID, dataset_id: UUID) -> Path:
        # UUIDs are safe path segments; still keep construction centralized.
        return (DATA_ROOT / str(user_id) / str(conversation_id) / str(dataset_id) / CSV_FILENAME).resolve()

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        # Prefer structured location; fallback keeps your current behavior working.
        candidate_path = self._dataset_csv_path(user_id, conversation_id, dataset_id)
        csv_path = candidate_path if candidate_path.exists() else DEFAULT_DATASET_PATH

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at path: {csv_path}")
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV path is not a file: {csv_path}")

        nrows: int | None
        if limit is None:
            nrows = None
        else:
            if limit <= 0:
                raise ValueError(f"limit must be a positive int or None, got: {limit!r}")
            nrows = limit

        try:
            return pd.read_csv(  # pyright: ignore[reportUnknownMemberType]
                csv_path,
                nrows=nrows,
                low_memory=False,
            )
        except Exception as e:
            raise ValueError(f"Failed to read CSV at {csv_path}: {e}") from e

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
        target = self._dataset_csv_path(user_id, conversation_id, dataset_id)

        # Ensure the resolved target is still under DATA_ROOT (belt-and-suspenders safety).
        try:
            target.relative_to(DATA_ROOT)
        except ValueError as e:
            raise ValueError(f"Resolved target path escapes DATA_ROOT: {target}") from e

        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing CSV at: {target}")

        tmp = target.with_suffix(target.suffix + f".tmp-{uuid4().hex}")
        try:
            df.to_csv(tmp, index=include_index)  # pyright: ignore[reportUnknownMemberType]
            # Atomic on same filesystem; replaces if exists.
            tmp.replace(target)
            return target
        except Exception as e:
            # Best-effort cleanup.
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise ValueError(f"Failed to write CSV at {target}: {e}") from e
