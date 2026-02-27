from __future__ import annotations

import json
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.compile_protocol.protocol_specs import ProtocolSpec
from python.implementation.workflows.nodes.transform_protocol.transform_protcol_plan import TransformPlanModel
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_deps import TransformProtocolDeps
from python.implementation.workflows.tools.data.data_profiling_tool import DatasetSummaryModel
from python.implementation.workflows.utils.validation import (
    NonEmptyStr,
    ValidationIssueModel,
)

MaxAttempt = 4
TransformStage = Literal["PLAN", "APPLY", "VALIDATE", "DONE","NEGOTIATE"]

class TransformProtocolPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    
    needs_user_input: bool = False
    stage: Optional[TransformStage] = None 
    attempt: Optional[int] = None 
    repair_context_json : Optional[str] = None
    transform_protocol_plan: Optional[TransformPlanModel] = None
    transformation_mapping : Optional[Mapping[str, List[str]]] = None

    protoctol_spec: Optional[ProtocolSpec] = None
    transformed_dataset_id: Optional[UUID] = None
    transformed_spec: Optional[ProtocolSpec] = None

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
        if self.payload.transformation_issues and any(i.severity == "FAIL" for i in self.payload.transformation_issues) and self.payload.attempt and self.payload.attempt >= MaxAttempt:
            return "ABORTED"
        if (
            self.payload.transformed_dataset_id
            and self.payload.cleaned_dataset_id
            and self.payload.cleaned_dataset_summary
            and self.payload.transform_protocol_plan
            and self.payload.transformation_mapping
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
        return "NEEDS_INPUT" if self.payload.needs_user_input else "NONE"

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