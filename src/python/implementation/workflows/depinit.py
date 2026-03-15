from __future__ import annotations

from dataclasses import dataclass, field
from typing import  Mapping, Type


from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.route import Router
from python.domain.workflows.state import State
from python.implementation.repo.firebase_realtime_workflow_state_repo import FirebaseRealtimeWorkflowStateRepo
from python.implementation.repo.google_cloud_storage_data_repo import GoogleCloudStorageDataRepo
from python.implementation.repo.google_cloud_storage_model_repo import GoogleCloudStorageModelsRepo
from python.implementation.service.llms.llm_service_factory import LLMServiceSettings, make_llm_service

from python.implementation.workflows.router.llm_assisted_router import (
    LLMAssistedRouterRouter,
    build_state_classes_by_name,
    init_all_nodoes_with_name_as_key,
)

from python.implementation.workflows.tools.tools_factory import DefaultToolFactory
from python.implementation.workflows.workflow_app import WorkflowApp

@dataclass(frozen=True)
class WorkflowSettings:
    llm: LLMServiceSettings = field(default_factory=LLMServiceSettings)
    history_limit: int = 30


def make_workflow_app(settings: WorkflowSettings) -> WorkflowApp:
    # 1) LLM
    llm: LLMService = make_llm_service(settings.llm)

    # 2) Repos
    data_repo: DataRepo = GoogleCloudStorageDataRepo(bucket_name="your-bucket-name")
    models_repo: ModelsRepo = GoogleCloudStorageModelsRepo(bucket_name="your-bucket-name")  

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
    app =  FirebaseRealtimeWorkflowStateRepo.get_default_firebase_database_app()
    return FirebaseRealtimeWorkflowStateRepo(
        app=app,
        state_classes_by_name=state_classes_by_name,
    )
   




