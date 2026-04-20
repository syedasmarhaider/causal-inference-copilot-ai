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
from python.domain.models.models import ArtifactFormat, ArtifactKind, utc_now
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.workflow_state_repo import Conversation, WorkflowStateRepo
from python.implementation.service.logging.default_logging import get_app_logger
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_node import DataManupulationNode
from python.implementation.workflows.ochestrator.causal_ochestrator_state import CausalOchestratorState
from python.implementation.workflows.ochestrator.data_ochestrator_state import DataOchestratorState
from python.implementation.workflows.utils.diff_util import DataFrameDiff, diff_dataframes

# TODO: add distributed tnx or locks later

@dataclass(frozen=True)
class DataflowArtifactResponse:
    id: UUID
    kind: ArtifactKind
    format: ArtifactFormat
    mime: str
    content: bytes


@dataclass(frozen=True)
class DataflowDatasetDiffResponse:
    previous_dataset_id: UUID
    current_dataset_id: UUID
    diff: DataFrameDiff


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
        
    def _raise_if_userid_not_relates_to_conversation_id(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
    ) -> None:
        if conversation_type not in ["causal", "data"]:
            raise ValidationError("conversation_type", f"Invalid conversation type: {conversation_type}")
        
        conversation = Conversation(conversation_id=conversation_id, conversation_type=conversation_type, last_updated_at_utc=utc_now())
        if not self._repo.is_conversation_id_for_user_id_exists(
            user_id=user_id,
            conversation=conversation,
        ):
            raise ConversationNotFoundError(user_id=user_id, conversation_id=conversation.conversation_id)

    def get_csv_data(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
        dataset_id: UUID,
        start: int = 0,
        limit: int | None = None,
    ) -> pd.DataFrame:
        self._raise_if_userid_not_relates_to_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
        )
        return self._data_repo.get_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
            start=start,
            limit=limit,
        )

    def get_working_dataset_diff(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
        key_columns: Sequence[str] | None = None,
    ) -> DataflowDatasetDiffResponse:
        self._raise_if_userid_not_relates_to_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
        )

        working_dataset_ids = self._get_working_dataset_ids(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if len(working_dataset_ids) < 2:
            raise ValidationError(
                field="working_dataset_ids",
                reason=(
                    "At least two working dataset versions are required to calculate a diff."
                ),
            )

        previous_dataset_id = working_dataset_ids[-2]
        current_dataset_id = working_dataset_ids[-1]
        older_df = self._data_repo.get_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=previous_dataset_id,
        )
        newer_df = self._data_repo.get_csv_data(
            user_id=user_id,
            conversation_id=conversation_id,
            dataset_id=current_dataset_id,
        )

        try:
            diff = diff_dataframes(
                older_df=older_df,
                newer_df=newer_df,
                key_columns=self._normalize_key_columns(key_columns),
            )
        except ValueError as exc:
            raise ValidationError(field="key_columns", reason=str(exc)) from exc

        self._log.debug(
            "working dataset diff calculated",
            user_id=user_id,
            conversation_id=conversation_id,
            previous_dataset_id=previous_dataset_id,
            current_dataset_id=current_dataset_id,
            key_columns=list(diff.key_columns),
            changed_rows=diff.summary.total_changed_rows,
            changed_cells=diff.summary.total_changed_cells,
        )
        return DataflowDatasetDiffResponse(
            previous_dataset_id=previous_dataset_id,
            current_dataset_id=current_dataset_id,
            diff=diff,
        )

    def upload_csv_data(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        conversation_type: str,
        csv_bytes: bytes,
    ) -> UUID:
        self._raise_if_userid_not_relates_to_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
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
        conversation_is_data = conversation_type == "data"
        if not ochestrator_state:
            if conversation_is_data:
                ochestrator_state = DataOchestratorState.init_empty()
            else:
                ochestrator_state = CausalOchestratorState.init_empty()    
        
        if ochestrator_state.get_current_node_name() != DataManupulationNode.NAME:
            raise ValidationError(
                field="conversation_id",
                reason="Cannot upload dataset while in the middle of a manipulation stage. Please finish or cancel the current stage before uploading a new dataset.",
            )

        dataset_id = DataOchestratorState.INIT_DATA_ID if conversation_is_data else CausalOchestratorState.INIT_DATA_ID
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
        conversation_type: str,
        artifact_id: UUID,
        artifact_kind: ArtifactKind,
        artifact_format: ArtifactFormat,
    ) -> DataflowArtifactResponse:
        self._raise_if_userid_not_relates_to_conversation_id(
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
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

    def _get_working_dataset_ids(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> list[UUID]:
        ochestrator_state = self._repo.load_ochestrator_state(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if ochestrator_state is None:
            return []

        try:
            dataset_ids_raw = ochestrator_state.get("working_dataset_ids") or []
        except KeyError:
            return []

        if not isinstance(dataset_ids_raw, list):
            raise ValidationError(
                field="working_dataset_ids",
                reason="Stored working dataset history is invalid.",
            )

        dataset_ids: list[UUID] = []
        for item in dataset_ids_raw:
            try:
                dataset_ids.append(item if isinstance(item, UUID) else UUID(str(item)))
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    field="working_dataset_ids",
                    reason=f"Stored working dataset history contains an invalid dataset id: {item!r}",
                ) from exc
        return dataset_ids

    @staticmethod
    def _normalize_key_columns(key_columns: Sequence[str] | None) -> list[str]:
        normalized_key_columns: list[str] = []
        seen: set[str] = set()

        for raw_column in key_columns or ():
            column = raw_column.strip()
            if not column:
                raise ValidationError(
                    field="key_columns",
                    reason="Key columns cannot contain empty values.",
                )
            if column in seen:
                raise ValidationError(
                    field="key_columns",
                    reason=f"Duplicate key column provided: {column}",
                )
            seen.add(column)
            normalized_key_columns.append(column)

        return normalized_key_columns
