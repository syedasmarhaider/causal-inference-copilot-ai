from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.route import Router

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

from python.implementation.workflows.utils.utils import DEFAULT_MODEL_GEMNI

@dataclass(frozen=True)
class WorkflowSettings:
    """
    Composition-root settings for initializing the workflow runtime.
    """
    workflow_state_dir: Path = Path("./data/workflow_state")
    models_root_dir: Path = Path("./models")

    llm: LLMServiceSettings = LLMServiceSettings()
    
    history_limit: int = 30
    
    # JSON repo robustness knobs (optional)
    json_repo: JsonFileRepoConfig = JsonFileRepoConfig()


def make_workflow_app(settings: WorkflowSettings) -> WorkflowApp:
    # 1) LLM
    llm: LLMService = make_llm_service(settings.llm)

    # 2) Repos
    data_repo: DataRepo = FileDataRepo() 
    models_repo: ModelsRepo = FileSystemModelsRepo(root_dir=settings.models_root_dir) 

    state_classes_by_name = build_state_classes_by_name()

    workflow_repo: WorkflowStateRepo = JsonFileWorkflowStateRepo(
        base_dir=settings.workflow_state_dir,
        state_classes_by_name=state_classes_by_name,
        config=settings.json_repo,
    )

    # 3) Router (LLM-assisted)
    router: Router = LLMAssistedRouterRouter(
        llm=llm,
        model_name=settings.llm.model or DEFAULT_MODEL_GEMNI,
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
        router=router,
        nodes_by_state_name=nodes_by_state_name,
        state_classes_by_name=state_classes_by_name,
        tool_factory=DefaultToolFactory(),
        history_limit=settings.history_limit,
    )