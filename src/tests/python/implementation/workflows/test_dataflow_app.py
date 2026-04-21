from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from python.implementation.workflows.dataflow_app import DataflowApp


@dataclass
class _FakeWorkflowStateRepo:
    conversation_checks: list[dict[str, Any]] = field(default_factory=list)

    def is_conversation_id_for_user_id_exists(
        self,
        *,
        user_id: UUID,
        conversation: Any,
    ) -> bool:
        self.conversation_checks.append(
            {
                "user_id": user_id,
                "conversation": conversation,
            }
        )
        return True


@dataclass
class _FakeDataRepo:
    dataframe: pd.DataFrame
    get_csv_calls: list[dict[str, Any]] = field(default_factory=list)

    def get_csv_data(
        self,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        start: int = 0,
        limit: int | None = None,
    ) -> pd.DataFrame:
        self.get_csv_calls.append(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "dataset_id": dataset_id,
                "start": start,
                "limit": limit,
            }
        )
        return self.dataframe.copy()


def test_get_csv_data_forwards_start_and_limit() -> None:
    repo = _FakeWorkflowStateRepo()
    data_repo = _FakeDataRepo(dataframe=pd.DataFrame([{"a": 1}, {"a": 2}]))
    app = DataflowApp(repo=repo, data_repo=data_repo)  # type: ignore[arg-type]
    user_id = uuid4()
    conversation_id = uuid4()
    dataset_id = uuid4()

    result = app.get_csv_data(
        user_id=user_id,
        conversation_id=conversation_id,
        conversation_type="data",
        dataset_id=dataset_id,
        start=5,
        limit=10,
    )

    assert data_repo.get_csv_calls == [
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "dataset_id": dataset_id,
            "start": 5,
            "limit": 10,
        }
    ]
    assert result.equals(data_repo.dataframe)
