from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from python.domain.workflows.state import ACTION, State, StateMessage, Status
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import DatasetSummaryModel
from python.implementation.workflows.utils.utils import  uuid_from_any

class LoadDatasetPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: Optional[UUID] = None
    summary: Optional[DatasetSummaryModel] = None
    load_error: Optional[str] = None
    graph_picture_ids: Optional[Sequence[UUID]] = None
    # TODO: solve this problem later of prerun
    user_message: Optional[str] = "Not run yet"

    @field_validator("load_error", "user_message", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return None

    @field_validator("id", mode="before")
    @classmethod
    def _parse_uuid(cls, v: Any) -> Optional[UUID]:
        # accept UUID | str | None
        return uuid_from_any(v)


class LoadDatasetState(State):
    NAME: ClassVar[str] = "LOAD_DATASET"

    def __init__(self, payload: LoadDatasetPayloadModel) -> None:
        self.payload = payload

    # ---- required by State ABC ----
    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        if self.error is not None:
            return "ABORTED"
        if self.payload.summary is not None:
            return "DONE"
        return "PENDING"

    @property
    def message(self) -> StateMessage:
        if self.payload.user_message is None:
            raise ValueError("LoadDatasetState message is required but missing. State must have user message. Dont call this property if this is not runned in the node context where user_message is guaranteed to be set.")
        artifact_ids : Sequence[str] = []
        if self.payload.graph_picture_ids:
            artifact_ids.extend(str(id) for id in self.payload.graph_picture_ids)
        return StateMessage(txt_message=self.payload.user_message, artifact_ids=artifact_ids)

    @property
    def error(self) -> Optional[str]:
        return self.payload.load_error

    @property
    def needs_action(self) -> ACTION:
        return "NEEDS_INPUT" if self.error is not None else "NONE"

    def pre_required_states_names(self) -> Sequence[str]:
        return ()

    def to_json_dict(self) -> Dict[str, Any]:
       return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "LoadDatasetState":
        model = LoadDatasetPayloadModel.model_validate(payload)
        return cls(model)
    
    @classmethod
    def init_empty(cls) -> "LoadDatasetState":
        return cls(LoadDatasetPayloadModel())
