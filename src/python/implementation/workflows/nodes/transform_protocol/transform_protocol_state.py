from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, computed_field

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.clean_protocol.clean_protocol_state import CleanProtocolState
from python.implementation.workflows.nodes.compile_protocol.compile_protocol_state import CompileProtocolState
from python.implementation.workflows.utils.validation import (
    NonEmptyStr,
    ValidationIssueModel,
    ValidationStatus,
)
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_specs import (
    TransformedProtocolSpec,
)


class TransformProtocolPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    error: Optional[NonEmptyStr] = None

    transformed_dataset_id: Optional[NonEmptyStr] = None
    transformed_spec: Optional[TransformedProtocolSpec] = None

    issues: List[ValidationIssueModel] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]

    transformed_dataset_summary: Dict[str, Any] = Field(default_factory=dict)
    user_message: Optional[NonEmptyStr] = None

    @computed_field  # type: ignore[misc]
    @property
    def validation_status(self) -> ValidationStatus:
        has_fail = any(i.severity == "FAIL" for i in self.issues)
        has_warn = any(i.severity == "WARN" for i in self.issues)
        if has_fail:
            return "FAIL"
        if has_warn:
            return "WARN"
        return "PASS"


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
            and self.payload.transformed_spec is not None
            and self.payload.validation_status != "FAIL"
        ):
            return "DONE"
        return "PENDING"

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
        return [CleanProtocolState.NAME, CompileProtocolState.NAME]

    def to_json_dict(self) -> Dict[str, Any]:
        # payload-only
        return self.payload.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "TransformProtocolState":
        model = TransformProtocolPayloadModel.model_validate(payload)
        return cls(model)