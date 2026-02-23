from __future__ import annotations

import json
from typing import Any, ClassVar, Dict, List, Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_deps import TransformProtocolDeps
from python.implementation.workflows.tools.data.data_profiling_tool import DatasetSummaryModel
from python.implementation.workflows.utils.validation import (
    NonEmptyStr,
    ValidationIssueModel,
)
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_specs import (
    TransformedProtocolSpec,
)


class TransformProtocolPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    
    transform_protocol_plan: Optional[TransformedProtocolSpec] = None

    transformed_dataset_id: Optional[UUID] = None
    transformed_spec: Optional[TransformedProtocolSpec] = None

    transformation_issues: List[ValidationIssueModel] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
   
    cleaned_dataset_id: Optional[UUID] = None
    cleaned_dataset_summary: Optional[DatasetSummaryModel] = None
    cleaned_dataset_validation_issues : List[ValidationIssueModel] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    user_message: Optional[NonEmptyStr] = None


class TransformProtocolState(State):
    NAME: ClassVar[str] = "TRANSFORM_PROTOCOL"

    def __init__(self, payload: TransformProtocolPayloadModel) -> None:
        self.payload = payload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        if self.payload.transformation_issues and any(i.severity == "FAIL" for i in self.payload.transformation_issues):
            return "ABORTED"
        if (
            self.payload.transformed_dataset_id
            and self.payload.cleaned_dataset_id
            and self.payload.cleaned_dataset_summary
            and self.payload.transform_protocol_plan
            and self.payload.transformed_spec is not None
        ):
            return "DONE"
        return "PENDING"

    @property
    def message(self) -> str:
        if self.payload.user_message is None:
            raise ValueError("TransformProtocolState message is required but missing. State must have user message. Dont call this property if this is not runned in the node context where user_message is guaranteed to be set.")
        return self.payload.user_message

    @property
    def error(self) -> Optional[str]:
        return json.dumps([i.model_dump(mode="json") for i in self.payload.transformation_issues], ensure_ascii=False, sort_keys=True, separators=(",", ":")) if self.payload.transformation_issues else None

    @property
    def needs_action(self) -> ACTION:
        return "NONE"

    def pre_required_states_names(self) -> Sequence[str]:
        return TransformProtocolDeps.pre_required_states_names()

    def to_json_dict(self) -> Dict[str, Any]:
        # payload-only
        return self.payload.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "TransformProtocolState":
        model = TransformProtocolPayloadModel.model_validate(payload)
        return cls(model)
    
    @classmethod
    def init_empty(cls) -> "TransformProtocolState":
        return cls(TransformProtocolPayloadModel())