from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from python.domain.workflows.state import ACTION, State, Status
from python.implementation.workflows.nodes.transform_protocol.transform_protocol_specs import (
    TransformedProtocolSpec,
)
from python.implementation.workflows.utils.validation import NonEmptyStr, ValidationStatus
from python.implementation.workflows.utils.utils import uuid_from_any


# =============================================================================
# Payload
# =============================================================================

class InferenceReadyProtocolPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    error: Optional[NonEmptyStr] = None
    user_message: Optional[NonEmptyStr] = None

    user_confirmed: bool = False

    # NOTE: keep as string to be JSON-friendly across persistence layers
    confirmed_dataset_id: Optional[NonEmptyStr] = None
    transformed_dataset_id: Optional[NonEmptyStr] = None

    confirmed_spec: Optional[TransformedProtocolSpec] = None

    clean_dataset_summary: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("error", "user_message", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return None

    @field_validator("confirmed_dataset_id", "transformed_dataset_id", mode="before")
    @classmethod
    def _parse_id_as_str(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        # accept UUID | str | other; normalize via uuid_from_any if possible
        try:
            u = uuid_from_any(v)
            if u is not None:
                return str(u)
        except Exception:
            pass
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        return str(v)

    @computed_field  # type: ignore[misc]
    @property
    def validation_status(self) -> ValidationStatus:
        """
        Simple gate:
        - FAIL if error exists
        - PASS if user_confirmed and all required confirmed fields are present
        - WARN otherwise (incomplete but not errored)
        """
        if self.error:
            return "FAIL"

        if self.user_confirmed:
            if self.confirmed_dataset_id and self.transformed_dataset_id and self.confirmed_spec is not None:
                return "PASS"
            return "WARN"

        return "WARN"


# =============================================================================
# State
# =============================================================================

class InferenceReadyProtocolState(State):
    """
    State that represents a user confirmation checkpoint before running inference.

    DONE:
      - user_confirmed == True
      - confirmed_dataset_id != None
      - transformed_dataset_id != None
      - confirmed_spec != None
      - error == None

    PENDING:
      - otherwise (and error == None)

    ABORTED:
      - error != None
    """
    NAME: ClassVar[str] = "INFERENCE_READY_PROTOCOL"

    def __init__(self, payload: InferenceReadyProtocolPayloadModel) -> None:
        self.payload = payload

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def status(self) -> Status:
        if self.payload.error:
            return "ABORTED"

        if (
            self.payload.user_confirmed
            and self.payload.confirmed_dataset_id is not None
            and self.payload.transformed_dataset_id is not None
            and self.payload.confirmed_spec is not None
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
        # If not confirmed yet, the UI typically needs to ask for confirmation.
        if self.payload.error is not None:
            return "NEEDS_INPUT"
        if not self.payload.user_confirmed:
            return "NEEDS_INPUT"
        return "NONE"

    def required_states_keys(self) -> Sequence[str]:
        # Update these if you want hard dependencies for the router.
        # Typically you'd require TransformProtocolState + CompileProtocolState (or similar).
        return ()

    def to_json_dict(self) -> Dict[str, Any]:
        return self.payload.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "InferenceReadyProtocolState":
        model = InferenceReadyProtocolPayloadModel.model_validate(payload)
        return cls(model)