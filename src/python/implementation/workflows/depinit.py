from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from python.domain.repo.analytics_repo import AnalyticsRepo
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.domain.service.llm_service import LLMService
from python.domain.workflows.node_state import NodeState
from python.implementation.repo.duckdb_working_analytics_repo import DuckDBAnalyticsRepo
from python.implementation.repo.firebase_realtime_workflow_state_repo import (
    FirebaseRealtimeWorkflowStateRepo,
)
from python.implementation.repo.google_cloud_storage_data_repo import GoogleCloudStorageDataRepo
from python.implementation.repo.google_cloud_storage_model_repo import GoogleCloudStorageModelsRepo
from python.implementation.repo.local_data_repo import LocalFileDataRepo
from python.implementation.repo.local_models_repo import LocalFileModelsRepo
from python.implementation.service.llms.llm_service_factory import (
    LLMServiceSettings,
    make_llm_service,
)
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.dataflow_app import DataflowApp
from python.implementation.workflows.ochestrator.data_ochestrator_state import DataOchestratorState
from python.implementation.workflows.ochestrator.ochestraotor import (
    Ochestrator,
    build_state_classes_by_name,
)

from python.implementation.workflows.ochestrator.causal_ochestrator_state import CausalOchestratorState
from python.implementation.workflows.workflow_app import WorkflowApp

log = get_logger(__name__, component="workflow_depinit", log_type="dependency_bootstrap")

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LOCAL_STORAGE_ROOT = _PROJECT_ROOT / ".local_storage"
_LOCAL_MODELS_ROOT = _LOCAL_STORAGE_ROOT / "models"
_LOCAL_DATA_ROOT = _LOCAL_STORAGE_ROOT / "data"


def make_apps(*, use_local_files: bool = False) -> tuple[WorkflowApp, DataflowApp]:
    log.info("building workflow app dependencies")
    llm: LLMService = make_llm_service(settings=LLMServiceSettings())
    data_repo: DataRepo = _make_data_repo(use_local_files=use_local_files)
    models_repo: ModelsRepo = _make_models_repo(use_local_files=use_local_files)
    analytics_repo: AnalyticsRepo = _make_analytics_repo()

    state_classes_by_name = build_state_classes_by_name()
    workflow_repo = _make_workflow_state_repo(
        state_classes_by_name=state_classes_by_name,
    )

    ochestrator = Ochestrator(
        workflow_repo=workflow_repo,
        llm=llm,
        data_repo=data_repo,
        models_repo=models_repo,
        analytics_repo=analytics_repo,
    )
    dataflow_app = DataflowApp(
        repo=workflow_repo,
        data_repo=data_repo,
    )

    log.info("workflow app dependencies created", use_local_files=use_local_files)
    return (
        WorkflowApp(
            repo=workflow_repo,
            ochestrator=ochestrator,
        ),
        dataflow_app,
    )


def make_dataflow_app(*, use_local_files: bool = False) -> DataflowApp:
    log.info("building dataflow app dependencies")
    state_classes_by_name = build_state_classes_by_name()
    workflow_repo = _make_workflow_state_repo(
        state_classes_by_name=state_classes_by_name,
    )
    return DataflowApp(
        repo=workflow_repo,
        data_repo=_make_data_repo(use_local_files=use_local_files),
    )


def _make_workflow_state_repo(
    *,
    state_classes_by_name: Mapping[str, type[NodeState]],
) -> WorkflowStateRepo:
    app = FirebaseRealtimeWorkflowStateRepo.get_default_firebase_database_app()
    return FirebaseRealtimeWorkflowStateRepo(
        app=app,
        state_classes_by_name=state_classes_by_name,
        ochestrator_state_classes_by_name={CausalOchestratorState.NAME: CausalOchestratorState,
                                           DataOchestratorState.NAME: DataOchestratorState
                                                                                       },  # TODO: make this dynamic when we have more orch states
        
    )


def _make_models_repo(*, use_local_files: bool) -> ModelsRepo:
    if use_local_files:
        log.info("using local file models repo", root_dir=str(_LOCAL_MODELS_ROOT))
        return LocalFileModelsRepo(root_dir=_LOCAL_MODELS_ROOT)
    return GoogleCloudStorageModelsRepo(GoogleCloudStorageModelsRepo.get_default_bucket())


def _make_data_repo(*, use_local_files: bool) -> DataRepo:
    if use_local_files:
        log.info("using local file data repo", root_dir=str(_LOCAL_DATA_ROOT))
        return LocalFileDataRepo(root_dir=_LOCAL_DATA_ROOT)
    return GoogleCloudStorageDataRepo(GoogleCloudStorageDataRepo.get_default_bucket())


def _make_analytics_repo() -> AnalyticsRepo:
    return DuckDBAnalyticsRepo()
