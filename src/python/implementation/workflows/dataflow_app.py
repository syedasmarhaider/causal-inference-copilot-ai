from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
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
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_node import DataManupulationNode
from python.implementation.workflows.ochestrator.writable_ochestrator_state import WritableOchestratorState

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
        ochestrator_state = self._repo.load_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if ochestrator_state is None:
            return []
        dataset_ids = ochestrator_state.get("working_dataset_ids")
        if dataset_ids is None:
            return []
        return dataset_ids

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
        ochestrator_state = self._repo.load_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if ochestrator_state is None:
            return None
        dataset_ids = ochestrator_state.get("working_dataset_ids")
        if not dataset_ids:
            return None
        current_dataset_id = dataset_ids[-1]
        is_frozen = ochestrator_state.get("working_dataset_frozen")
        if is_frozen is None:
            is_frozen = False
        return WorkingDatasetInfo(dataset_id=current_dataset_id, is_freezed=is_frozen)

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
            ochestrator_state = WritableOchestratorState.init_empty()
        
        if not isinstance(ochestrator_state, WritableOchestratorState):
            raise ConversationNotFoundError(user_id=user_id, conversation_id=conversation_id)

        if ochestrator_state.get_current_node_name() != DataManupulationNode.NAME:
            raise ValidationError(
                field="conversation_id",
                reason="Cannot upload dataset while in the middle of a manipulation stage. Please finish or cancel the current stage before uploading a new dataset.",
            )

        dataset_id = WritableOchestratorState.INIT_DATA_ID
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
            if artifact_kind == "graph":
                json_data = self._data_repo.get_json_data(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataset_id=artifact_id,
                )
                mime = "application/json"
                content = json_data.encode("utf-8")
            elif artifact_format == "csv":
                dataframe = self._data_repo.get_csv_data(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    dataset_id=artifact_id,
                )
                mime = "text/csv"
                content = dataframe.to_csv(index=False).encode("utf-8")
            else:
                try:
                    json_data = self._data_repo.get_json_data(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        dataset_id=artifact_id,
                    )
                    mime = "application/json"
                    content = json_data.encode("utf-8")
                except FileNotFoundError:
                    dataframe = self._data_repo.get_csv_data(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        dataset_id=artifact_id,
                    )
                    payload: dict[str, Any] = {
                        "columns": [str(column) for column in dataframe.columns],
                        "row_count": int(len(dataframe)),
                        "rows": dataframe.to_dict(orient="records"),
                    }
                    mime = "application/json"
                    content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except FileNotFoundError as exc:
            self._log.warning(
                "artifact fetch failed: not found",
                user_id=user_id,
                conversation_id=conversation_id,
                artifact_id=artifact_id,
                artifact_kind=artifact_kind,
                artifact_format=artifact_format,
            )
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
