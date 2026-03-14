from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Literal, Mapping, Optional, Type

import firebase_admin
from firebase_admin import credentials

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.route import Router
from python.domain.workflows.state import State

from python.implementation.repo.file_data_repo import FileDataRepo
from python.implementation.repo.models_repo import FileSystemModelsRepo
from python.implementation.repo.json_file_workflow_state_repo import (
    JsonFileRepoConfig,
    JsonFileWorkflowStateRepo,
)

from python.implementation.service.llms.llm_service_factory import LLMServiceSettings, make_llm_service

from python.implementation.workflows.router.llm_assisted_router import (
    LLMAssistedRouterRouter,
    build_state_classes_by_name,
    init_all_nodoes_with_name_as_key,
)

from python.implementation.workflows.tools.tools_factory import DefaultToolFactory
from python.implementation.workflows.workflow_app import WorkflowApp

@dataclass(frozen=True)
class FirebaseRealtimeRepoSettings:
    database_url: Optional[str] = None


@dataclass(frozen=True)
class WorkflowSettings:
    """
    Composition-root settings for initializing the workflow runtime.
    """
    workflow_state_dir: Path = Path("./data/workflow_state")
    models_root_dir: Path = Path("./models")
    workflow_repo_backend: Literal["json", "firebase_rtdb"] = "json"

    llm: LLMServiceSettings = field(default_factory=LLMServiceSettings)

    history_limit: int = 30

    # JSON repo robustness knobs (optional)
    json_repo: JsonFileRepoConfig = field(default_factory=JsonFileRepoConfig)
    firebase_repo: FirebaseRealtimeRepoSettings = field(default_factory=FirebaseRealtimeRepoSettings)


def make_workflow_app(settings: WorkflowSettings) -> WorkflowApp:
    # 1) LLM
    llm: LLMService = make_llm_service(settings.llm)

    # 2) Repos
    data_repo: DataRepo = FileDataRepo() 
    models_repo: ModelsRepo = FileSystemModelsRepo(root_dir=settings.models_root_dir) 

    state_classes_by_name = build_state_classes_by_name()

    workflow_repo = _make_workflow_state_repo(
        settings=settings,
        state_classes_by_name=state_classes_by_name,
    )

    # 3) Router (LLM-assisted)
    router: Router = LLMAssistedRouterRouter(
        llm=llm,
    )

    # 4) Nodes registry (keyed by node.name == State.NAME)
    nodes_by_state_name: dict[str, Node] = init_all_nodoes_with_name_as_key(
        llm=llm,
        data_repo=data_repo,
        models_repo=models_repo,
    )


    # 6) Workflow app
    return WorkflowApp(
        repo=workflow_repo,
        data_repo=data_repo,
        router=router,
        nodes_by_state_name=nodes_by_state_name,
        state_classes_by_name=state_classes_by_name,
        tool_factory=DefaultToolFactory(data_repo=data_repo, models_repo=models_repo),
        history_limit=settings.history_limit,
    )


def _make_workflow_state_repo(
    *,
    settings: WorkflowSettings,
    state_classes_by_name: Mapping[str, Type[State]],
) -> WorkflowStateRepo:
    if settings.workflow_repo_backend == "json":
        return JsonFileWorkflowStateRepo(
            base_dir=settings.workflow_state_dir,
            state_classes_by_name=state_classes_by_name,
            config=settings.json_repo,
        )

    if settings.workflow_repo_backend == "firebase_rtdb":
        return _make_firebase_workflow_state_repo(
            settings=settings.firebase_repo,
            state_classes_by_name=state_classes_by_name,
        )

    raise ValueError(f"Unsupported workflow repo backend: {settings.workflow_repo_backend}")


def _make_firebase_workflow_state_repo(
    *,
    settings: FirebaseRealtimeRepoSettings,
    state_classes_by_name: Mapping[str, Type[State]],
) -> WorkflowStateRepo:
    database_url = os.getenv("FIREBASE_DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError(
            "FIREBASE_DATABASE_URL must be configured when workflow_repo_backend='firebase_rtdb'"
        )

    try:
        app = firebase_admin.get_app()
    except ValueError:
        app = firebase_admin.initialize_app(
            credentials.ApplicationDefault(),
            {"databaseURL": database_url},
        )

    from python.implementation.repo.firebase_realtime_workflow_state_repo import (
        FirebaseRealtimeWorkflowStateRepo,
    )

    return FirebaseRealtimeWorkflowStateRepo(
        app=app,
        root_path="/workflows",
        state_classes_by_name=state_classes_by_name,
    )
