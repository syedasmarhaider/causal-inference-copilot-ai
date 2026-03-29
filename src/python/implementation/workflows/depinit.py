from __future__ import annotations

from collections.abc import Mapping

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import LLMService
from python.domain.workflows.node import Node
from python.domain.workflows.route import Router
from python.domain.workflows.state import State
from python.implementation.repo.firebase_realtime_workflow_state_repo import (
    FirebaseRealtimeWorkflowStateRepo,
)
from python.implementation.repo.google_cloud_storage_data_repo import GoogleCloudStorageDataRepo
from python.implementation.repo.google_cloud_storage_model_repo import GoogleCloudStorageModelsRepo
from python.implementation.service.llms.llm_service_factory import (
    LLMServiceSettings,
    make_llm_service,
)
from python.implementation.workflows.router.llm_assisted_router import (
    LLMAssistedRouterRouter,
    build_state_classes_by_name,
    init_all_nodoes_with_name_as_key,
)
from python.implementation.workflows.tools.tools_factory import DefaultToolFactory
from python.implementation.workflows.workflow_app import WorkflowApp


def make_workflow_app() -> WorkflowApp:
    llm: LLMService = make_llm_service(settings=LLMServiceSettings())
    data_repo: DataRepo = _make_data_repo()
    models_repo: ModelsRepo = _make_models_repo()

    state_classes_by_name = build_state_classes_by_name()

    workflow_repo = _make_workflow_state_repo(
        state_classes_by_name=state_classes_by_name,
    )
    
    router: Router = LLMAssistedRouterRouter(
        llm=llm,
    )
    
    nodes_by_state_name: dict[str, Node] = init_all_nodoes_with_name_as_key(
        llm=llm,
        data_repo=data_repo,
        models_repo=models_repo,
    )

    return WorkflowApp(
        repo=workflow_repo,
        data_repo=data_repo,
        router=router,
        nodes_by_state_name=nodes_by_state_name,
        state_classes_by_name=state_classes_by_name,
        tool_factory=DefaultToolFactory(data_repo=data_repo, models_repo=models_repo),
    )


def _make_workflow_state_repo(
    *,
    state_classes_by_name: Mapping[str, type[State]],
) -> WorkflowStateRepo:
    app =  FirebaseRealtimeWorkflowStateRepo.get_default_firebase_database_app()
    return FirebaseRealtimeWorkflowStateRepo(
        app=app,
        state_classes_by_name=state_classes_by_name,
    )


def _make_models_repo() -> ModelsRepo:
    return GoogleCloudStorageModelsRepo(GoogleCloudStorageModelsRepo.get_default_bucket())   

def _make_data_repo() -> DataRepo:
    return GoogleCloudStorageDataRepo(GoogleCloudStorageDataRepo.get_default_bucket())   
   




