from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.models.errors import NodeExecutionError
from python.domain.workflows.state import State, StateMessage, Status
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_deps import (
    CleanProtocolDeps,
)
from python.implementation.workflows.tools.causal.causal_spec import CausalSpec
from python.implementation.workflows.tools.data_processing.data_processing_tool import (
    SQLStatements,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetSummaryModel,
)
from python.implementation.workflows.utils.utils import uuid_from_any


class CleanDataDiffModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    rows_delta: int
    cols_delta: int
    added_columns: list[str] = Field(default_factory=list)
    removed_columns: list[str] = Field(default_factory=list)


class SQLHistoryItemModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    iteration_index: int
    source_dataset_id: UUID
    output_dataset_id: UUID
    sql_request: SQLStatements

    @field_validator("source_dataset_id", "output_dataset_id", mode="before")
    @classmethod
    def _parse_uuid(cls, v: Any) -> UUID:
        out = uuid_from_any(v)
        if out is None:
            raise TypeError("source_dataset_id/output_dataset_id must be UUID")
        return out


class CleanIterationRecordModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    iteration_index: int
    source_dataset_id: UUID
    output_dataset_id: UUID
    diff: CleanDataDiffModel
    summary: DatasetSummaryModel

    @field_validator("source_dataset_id", "output_dataset_id", mode="before")
    @classmethod
    def _parse_uuid(cls, v: Any) -> UUID:
        out = uuid_from_any(v)
        if out is None:
            raise TypeError("source_dataset_id/output_dataset_id must be UUID")
        return out


class CausalSpecHistoryItemModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    iteration_index: int
    causal_spec: CausalSpec


class CleanProtocolPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    clean_dataset_id: UUID | None = None
    cleaning_error: str | None = None
    user_message: str | None = None
    summary: DatasetSummaryModel | None = None
    user_acceptance: bool | None = None
    graph_picture_ids: Sequence[UUID] | None = None

    iteration_index: int = 0
    latest_diff: CleanDataDiffModel | None = None
    compiled_causal_spec: CausalSpec | None = None
    sql_history: list[SQLHistoryItemModel] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]
    iteration_history: list[CleanIterationRecordModel] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]
    causal_spec_history: list[CausalSpecHistoryItemModel] = Field(default_factory=list) # pyright: ignore[reportUnknownVariableType]

    @field_validator("cleaning_error", "user_message", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        raise TypeError("cleaning_error/user_message must be str|null")

    @field_validator("clean_dataset_id", mode="before")
    @classmethod
    def _parse_uuid(cls, v: Any) -> UUID | None:
        return uuid_from_any(v)


class CleanProtocolState(State):
    NAME: ClassVar[str] = "CLEAN_PROTOCOL"

    def __init__(self, payload: CleanProtocolPayloadModel) -> None:
        self.payload = payload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        if self.payload.cleaning_error is not None:
            return "ABORTED"
        if (
            self.payload.user_acceptance is True
            and self.payload.clean_dataset_id is not None
            and self.payload.summary is not None
            and self.payload.compiled_causal_spec is not None
        ):
            return "DONE"
        return "PENDING"

    @property
    def message(self) -> StateMessage:
        if self.payload.user_message is None:
            raise ValueError(
                "CleanProtocolState message is required but missing. "
                "State must have user message in node context."
            )
        artifacts = (
            [str(id_) for id_ in self.payload.graph_picture_ids]
            if self.payload.graph_picture_ids
            else []
        )
        action = "NONE" if self.status in ("ABORTED", "DONE") else "NEEDS_INPUT"
        return StateMessage(txt_message=self.payload.user_message, artifact_ids=artifacts, action=action)

    @property
    def error(self) -> NodeExecutionError | None:
        if self.payload.cleaning_error is not None:
            return NodeExecutionError(state_name=self.NAME, error=self.payload.cleaning_error)
        return None
    
    def pre_required_states_names(self) -> Sequence[str]:
        return CleanProtocolDeps.pre_required_states_names()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> CleanProtocolState:
        model = CleanProtocolPayloadModel.model_validate(payload)
        return cls(model)

    @classmethod
    def init_empty(cls) -> CleanProtocolState:
        return cls(CleanProtocolPayloadModel())
