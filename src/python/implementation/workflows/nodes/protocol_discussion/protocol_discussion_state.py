from __future__ import annotations

from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.workflows.node_state import NodeState
from python.implementation.workflows.utils.utils import uuid_from_any

ProtocolDiscussionPhase = Literal["DISCUSSING", "CONFIRMED"]
StudyType = Literal["RCT", "OBSERVATIONAL"]


class ProtocolCausalDraftModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    treatment_column: str | None = None
    outcome_column: str | None = None
    covariates: list[str] = Field(default_factory=list)
    effect_modifiers: list[str] = Field(default_factory=list)
    target_population: str | None = None
    study_type: StudyType | None = None
    negative_control_outcome: str | None = None
    time_zero: str | None = None

    @field_validator(
        "treatment_column",
        "outcome_column",
        "target_population",
        "negative_control_outcome",
        "time_zero",
        mode="before",
    )
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        raise TypeError("draft text fields must be str|null")

    @field_validator("study_type", mode="before")
    @classmethod
    def _normalize_study_type(cls, value: Any) -> StudyType | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("study_type must be str|null")
        normalized = value.strip().upper()
        if normalized in {"OBSERVATIONAL", "OBSERVATION", "OBS"}:
            return "OBSERVATIONAL"
        if normalized == "RCT":
            return "RCT"
        raise ValueError("study_type must be RCT or OBSERVATIONAL")

    @field_validator("covariates", "effect_modifiers", mode="before")
    @classmethod
    def _normalize_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            values = [part.strip() for part in value.split(",")]
        elif isinstance(value, (list, tuple)):
            values = [str(item).strip() for item in value]
        else:
            raise TypeError("draft list fields must be list[str]|str|null")

        deduped: list[str] = []
        for item in values:
            if item and item not in deduped:
                deduped.append(item)
        return deduped


class ProtocolDiscussionPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset_id: UUID | None = None
    draft: ProtocolCausalDraftModel = Field(default_factory=ProtocolCausalDraftModel)
    phase: ProtocolDiscussionPhase = "DISCUSSING"
    assistant_message: str | None = None

    @field_validator("dataset_id", mode="before")
    @classmethod
    def _parse_dataset_id(cls, value: Any) -> UUID | None:
        return uuid_from_any(value)

    @field_validator(
        "assistant_message",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        raise TypeError("text fields must be str|null")


class ProtocolDiscussionState(NodeState):
    NAME: ClassVar[str] = "PROTOCOL_DISCUSSION"

    def __init__(self, payload: ProtocolDiscussionPayloadModel | None = None) -> None:
        self.payload = payload or ProtocolDiscussionPayloadModel()

    def name(self) -> str:
        return self.NAME

    def clear_state(self) -> None:
        self.payload = ProtocolDiscussionPayloadModel()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ProtocolDiscussionState:
        return cls(ProtocolDiscussionPayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> ProtocolDiscussionState:
        return cls()


__all__ = [
    "ProtocolCausalDraftModel",
    "ProtocolDiscussionPayloadModel",
    "ProtocolDiscussionPhase",
    "ProtocolDiscussionState",
    "StudyType",
]
