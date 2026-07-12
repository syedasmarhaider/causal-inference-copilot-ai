from __future__ import annotations

from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.models.models import ArtifactRef
from python.domain.workflows.node_state import NodeState
from python.implementation.workflows.utils.utils import uuid_from_any


class ShapExplanationPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_signature: str | None = None
    shap_values_dataset_id: UUID | None = None
    shap_values_summary: dict[str, Any] | None = None
    latest_query_result_raw_json_str: str | None = None
    latest_request_summary: str | None = None
    assistant_message: str | None = None
    message_artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    error_message: str | None = None

    @field_validator("shap_values_dataset_id", mode="before")
    @classmethod
    def _parse_uuid(cls, value: Any) -> UUID | None:
        return uuid_from_any(value)

    @field_validator(
        "source_signature",
        "latest_query_result_raw_json_str",
        "latest_request_summary",
        "assistant_message",
        "error_message",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        raise TypeError("text fields must be str|null")

    def reset_for_signature(self, *, source_signature: str) -> ShapExplanationPayloadModel:
        return self.model_copy(
            update={
                "source_signature": source_signature,
                "shap_values_dataset_id": None,
                "shap_values_summary": None,
                "latest_query_result_raw_json_str": None,
                "latest_request_summary": None,
                "assistant_message": None,
                "message_artifact_refs": [],
                "error_message": None,
            }
        )


class ShapExplanationState(NodeState):
    NAME: ClassVar[str] = "SHAP_EXPLANATION"

    def __init__(self, payload: ShapExplanationPayloadModel | None = None) -> None:
        self.payload = payload or ShapExplanationPayloadModel()

    def name(self) -> str:
        return self.NAME

    def clear_state(self) -> None:
        self.payload = ShapExplanationPayloadModel()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ShapExplanationState:
        return cls(ShapExplanationPayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> ShapExplanationState:
        return cls()


__all__ = [
    "ShapExplanationPayloadModel",
    "ShapExplanationState",
]
