from __future__ import annotations

from python.implementation.repo.local_data_repo import LocalFileDataRepo
from python.implementation.repo.local_json_workflow_state_repo import (
    LocalJsonWorkflowStateRepo,
)
from python.implementation.repo.local_models_repo import LocalFileModelsRepo
from python.implementation.workflows import depinit


def test_make_workflow_state_repo_uses_local_json_repo(
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
    )

    assert isinstance(repo, LocalJsonWorkflowStateRepo)


def test_make_data_repo_uses_local_file_repo(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(depinit, "_LOCAL_DATA_ROOT", tmp_path / "data")

    repo = depinit._make_data_repo()

    assert isinstance(repo, LocalFileDataRepo)


def test_make_models_repo_uses_local_file_repo(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(depinit, "_LOCAL_MODELS_ROOT", tmp_path / "models")

    repo = depinit._make_models_repo()

    assert isinstance(repo, LocalFileModelsRepo)
