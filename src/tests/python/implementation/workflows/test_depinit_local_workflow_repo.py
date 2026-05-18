from __future__ import annotations

from typing import Any

from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.implementation.repo.local_json_workflow_state_repo import (
    LocalJsonWorkflowStateRepo,
)
from python.implementation.workflows import depinit


def test_make_workflow_state_repo_uses_local_json_repo_when_local_files_enabled(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        depinit,
        "_LOCAL_WORKFLOW_STATE_DB_PATH",
        tmp_path / "workflow_state.json",
    )

    repo = depinit._make_workflow_state_repo(
        state_classes_by_name={},
        use_local_files=True,
    )

    assert isinstance(repo, LocalJsonWorkflowStateRepo)


def test_make_workflow_state_repo_uses_firebase_when_local_files_disabled(
    monkeypatch,
) -> None:
    class _FakeFirebaseRepo(WorkflowStateRepo):
        @staticmethod
        def get_default_firebase_database_app() -> object:
            return object()

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def save_conversation(self, **kwargs: Any) -> None:
            raise NotImplementedError

        def get_conversations(self, **kwargs: Any):
            raise NotImplementedError

        def is_conversation_id_for_user_id_exists(self, **kwargs: Any) -> bool:
            raise NotImplementedError

        def load_ochestrator_state(self, **kwargs: Any):
            raise NotImplementedError

        def store_ochestrator_state(self, **kwargs: Any) -> None:
            raise NotImplementedError

        def load_state(self, **kwargs: Any):
            raise NotImplementedError

        def store_state(self, **kwargs: Any) -> None:
            raise NotImplementedError

        def delete_state(self, **kwargs: Any) -> None:
            raise NotImplementedError

        def append_message(self, **kwargs: Any) -> None:
            raise NotImplementedError

        def append_messages(self, **kwargs: Any) -> None:
            raise NotImplementedError

        def load_message_history(self, **kwargs: Any):
            raise NotImplementedError

        def clear_message_history(self, **kwargs: Any) -> None:
            raise NotImplementedError

    monkeypatch.setattr(depinit, "FirebaseRealtimeWorkflowStateRepo", _FakeFirebaseRepo)

    repo = depinit._make_workflow_state_repo(
        state_classes_by_name={},
        use_local_files=False,
    )

    assert isinstance(repo, _FakeFirebaseRepo)
