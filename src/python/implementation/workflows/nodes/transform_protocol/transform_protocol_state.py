from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.utils.validation import (
    NonEmptyStr,
    ValidationIssueModel,
    ValidationStatus,
)
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_specs import (
    TransformedProtocolSpec,
)


class ValidationPayloadModel(BaseModel):
    """
    Static validator output payload.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    issues: List[ValidationIssueModel] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    status: ValidationStatus = "PASS"

    @model_validator(mode="after")
    def _compute_status(self) -> "ValidationPayloadModel":
        has_fail = any(i.severity == "FAIL" for i in self.issues)
        has_warn = any(i.severity == "WARN" for i in self.issues)
        if has_fail:
            self.status = "FAIL"
        elif has_warn:
            self.status = "WARN"
        else:
            self.status = "PASS"
        return self


class TransformProtocolStatePayloadModel(BaseModel):
    """
    Payload-only state content. No status/needs_action/message stored here;
    they are derived by the State wrapper.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    error: Optional[NonEmptyStr] = None

    transformed_dataset_id: Optional[NonEmptyStr] = None
    transformed_spec: Optional[TransformedProtocolSpec] = None

    validation: ValidationPayloadModel = Field(default_factory=ValidationPayloadModel)
    transformed_dataset_summary: Dict[str, Any] = Field(default_factory=dict)

    # optional human-facing message (typically produced by LLM)
    user_message: Optional[NonEmptyStr] = None


class TransformProtocolState(State):
    """
    Constructible wrapper over payload.
    State interface fields are derived deterministically from payload.
    """

    def __init__(self, payload: TransformProtocolStatePayloadModel) -> None:
        self._payload = payload

    @property
    def payload(self) -> TransformProtocolStatePayloadModel:
        return self._payload

    @property
    def status(self) -> Status:
        # Deterministic stage status rules:
        # - error => ABORTED
        # - otherwise, if we have produced the required outputs => DONE
        # - else => PENDING
        if self._payload.error:
            return "ABORTED"

        if (
            self._payload.transformed_dataset_id
            and self._payload.transformed_spec is not None
            and self._payload.validation.status != "FAIL"
        ):
            return "DONE"

        return "PENDING"

    @property
    def message(self) -> Optional[str]:
        # Prefer user_message if present; otherwise None.
        return self._payload.user_message

    @property
    def error(self) -> Optional[str]:
        return self._payload.error

    @property
    def needs_action(self) -> ACTION:
        # Transform stage is automatic. If you later want human intervention when FAIL,
        # set NEEDS_INPUT on validation FAIL and return that here.
        if self._payload.validation.status == "FAIL":
            return "NEEDS_INPUT"
        return "NONE"

    def required_states_keys(self) -> Sequence[str]:
        return ("dataset_state", "compile_protocol_state")

    def to_json_dict(self) -> Dict[str, Any]:
        # Store ONLY payload in JSON.
        return self._payload.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "TransformProtocolState":
        model = TransformProtocolStatePayloadModel.model_validate(payload)
        return cls(model)