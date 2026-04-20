from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import pytest

from python.implementation.repo.local_data_repo import LocalFileDataRepo


def _ids() -> tuple[UUID, UUID, UUID]:
    return uuid4(), uuid4(), uuid4()


def test_get_csv_data_applies_start_and_limit(tmp_path: Path) -> None:
    user_id, conversation_id, dataset_id = _ids()
    repo = LocalFileDataRepo(tmp_path)
    repo.save_csv_data(
        user_id,
        conversation_id,
        dataset_id,
        pd.DataFrame([{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]),
    )

    frame = repo.get_csv_data(
        user_id,
        conversation_id,
        dataset_id,
        start=1,
        limit=1,
    )

    assert frame.to_dict(orient="records") == [{"a": 3, "b": 4}]


def test_get_csv_data_validates_pagination(tmp_path: Path) -> None:
    user_id, conversation_id, dataset_id = _ids()
    repo = LocalFileDataRepo(tmp_path)

    with pytest.raises(ValueError, match=r"start must be >= 0"):
        repo.get_csv_data(user_id, conversation_id, dataset_id, start=-1)

    with pytest.raises(ValueError, match=r"limit must be >= 0"):
        repo.get_csv_data(user_id, conversation_id, dataset_id, limit=-1)
