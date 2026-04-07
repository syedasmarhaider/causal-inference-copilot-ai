from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast
from uuid import UUID

import pandas as pd

from python.domain.models.errors import (
    ArtifactNotFoundError,
    ConversationNotFoundError,
    ValidationError,
)
from python.domain.models.models import ArtifactFormat, ArtifactKind, WorkingDatasetInfo
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.workflow_state_repo import WorkflowStateRepo
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.dataset.dataset_state import DatasetState
from python.implementation.workflows.ochestrator.ochestrator_global_state import OchestratorWritableGlobalState

# TODO: add distributed tnx or locks later


@dataclass(frozen=True)
class DataflowArtifactResponse:
    id: UUID
    kind: ArtifactKind
    format: ArtifactFormat
    mime: str
    content: bytes


class DataflowApp:
    def __init__(
        self,
        *,
        repo: WorkflowStateRepo,
        data_repo: DataRepo,
    ) -> None:
        self._repo = repo
        self._data_repo = data_repo
        self._log = get_app_logger(
            __name__,
            component=self.__class__.__name__,
            log_type="workflow_service",
        )

    def raise_if_userid_not_relates_to_conversation_id(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> None:
        if not self._repo.is_conversation_id_for_user_id_exists(
            user_id=user_id,
            conversation_id=conversation_id,
        ):
            raise ConversationNotFoundError(user_id=user_id, conversation_id=conversation_id)

    def get_csv_data(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        dataset_id: UUID,
        limit: int | None = None,
    ) -> pd.DataFrame:
        self.raise_if_userid_not_relates_to_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        return self._data_repo.get_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
            limit=limit,
        )

    def get_all_working_dataset_ids(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> Sequence[UUID]:
        self.raise_if_userid_not_relates_to_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        dataset_state = self._load_dataset_state(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if dataset_state is None:
            return ()
        return tuple(iteration.dataset_id for iteration in dataset_state.payload.dataset_iterations)

    def get_current_working_dataset_info(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> WorkingDatasetInfo | None:
        self.raise_if_userid_not_relates_to_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        dataset_state = self._load_dataset_state(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if dataset_state is None:
            return None
        return dataset_state.get_working_dataset_info()

    def upload_csv_data(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        csv_bytes: bytes,
    ) -> UUID:
        self.raise_if_userid_not_relates_to_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        try:
            df = pd.read_csv(
                io.BytesIO(csv_bytes), low_memory=False
            )  # pyright: ignore[reportUnknownMemberType]
        except Exception as exc:
            self._log.info(
                "csv upload rejected due to invalid payload",
                user_id=user_id,
                conversation_id=conversation_id,
                csv_size_bytes=len(csv_bytes),
            )
            raise ValueError(f"Uploaded file is not a valid CSV: {exc}") from exc

        ochestrator_state = self._repo.load_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not ochestrator_state:
            ochestrator_state = OchestratorWritableGlobalState.init_empty()
        
        ochestrator_state = cast(OchestratorWritableGlobalState, ochestrator_state)    

        if ochestrator_state.needs_node_name() != DatasetState.NAME:
            self._log.info(
                "csv upload rejected because conversation is not at dataset state",
                user_id=user_id,
                conversation_id=conversation_id,
                active_state_name=ochestrator_state.needs_node_name(),
                required_state_name=DatasetState.NAME,
            )
            raise ConversationNotFoundError(user_id=user_id, conversation_id=conversation_id)

        dataset_id = DatasetState.INIT_DATA_ID
        self._data_repo.save_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
            df=df,
            overwrite=True,
        )
            
        self._log.info(
            "csv dataset uploaded",
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
            rows_count=len(df),
            columns_count=len(df.columns),
        )
        return dataset_id

    def get_artifact(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        artifact_id: UUID,
        artifact_kind: ArtifactKind,
        artifact_format: ArtifactFormat,
    ) -> DataflowArtifactResponse:
        self.raise_if_userid_not_relates_to_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if artifact_kind == "graph" and artifact_format != "json":
            raise ValidationError(
                field="artifact_format", reason="Graph artifacts must be in JSON format"
            )

        try:
            mime: str
            content: bytes
            if artifact_format == "csv":
                dataframe = self._data_repo.get_csv_data(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataset_id=artifact_id,
                )
                mime = "text/csv"
                content = dataframe.to_csv(index=False).encode("utf-8")
            elif artifact_format == "json":
                json_data = self._data_repo.get_json_data(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataset_id=artifact_id,
                )
                mime = "application/json"
                content = json_data.encode("utf-8")
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError(artifact_id=artifact_id) from exc

        self._log.debug(
            "dataflow artifact fetched",
            user_id=user_id,
            conversation_id=conversation_id,
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            artifact_format=artifact_format,
            artifact_mime=mime,
            content_size_bytes=len(content),
        )
        return DataflowArtifactResponse(
            id=artifact_id,
            kind=artifact_kind,
            format=artifact_format,
            mime=mime,
            content=content,
        )

    def _load_dataset_state(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> DatasetState | None:
        state = self._repo.load_state(
            user_id=user_id,
            conversation_id=conversation_id,
            state_name=DatasetState.NAME,
        )
        if state is None:
            return None
        if not isinstance(state, DatasetState):
            raise TypeError(f"Expected DatasetState, got {type(state).__name__}")
        return state


__all__ = [
    "DataflowApp",
    "DataflowArtifactResponse",
]
