from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.nodes.validate_cleaned_protocol.validate_cleaned_protocol_state import ValidateCleanProtocolState
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

    error: Optional[NonEmptyStr] = None

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
        if self.payload.error:
            return "ABORTED"
        if (
            self.payload.transformed_dataset_id
            and self.payload.cleaned_dataset_id
            and self.payload.cleaned_dataset_summary
            and self.payload.transformed_spec is not None
        ):
            return "DONE"
        raise ValueError("TransformProtocolState status is indeterminate due to missing fields or FAIL validation status")

    @property
    def message(self) -> Optional[str]:
        return self.payload.user_message

    @property
    def error(self) -> Optional[str]:
        return self.payload.error

    @property
    def needs_action(self) -> ACTION:
        return "NONE"

    def required_states_keys(self) -> Sequence[str]:
        return [CleanProtocolState.NAME, CompileProtocolState.NAME,ValidateCleanProtocolState.NAME]

    def to_json_dict(self) -> Dict[str, Any]:
        # payload-only
        return self.payload.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "TransformProtocolState":
        model = TransformProtocolPayloadModel.model_validate(payload)
        return cls(model)