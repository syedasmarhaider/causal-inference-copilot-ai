from __future__ import annotations

from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.models.models import ArtifactRef
from python.domain.workflows.node_state import NodeState
from python.implementation.workflows.utils.utils import uuid_from_any


class CausalValidatePayloadModel(BaseModel):
    """Persistent state owned by the outer-CV validation node.

    The upstream dataset, protocol, selected estimator, and trained-model identifier are
    deliberately not copied here.  They remain orchestrator dependencies and are folded
    into ``source_signature`` so a changed training context invalidates this cache.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_signature: str | None = None
    validation_dataset_id: UUID | None = None
    dr_test_summary_dataset_id: UUID | None = None
    validation_summary: dict[str, Any] | None = None
    latest_query_result_raw_json_str: str | None = None
    latest_request_summary: str | None = None
    assistant_message: str | None = None
    message_artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    error_message: str | None = None

    @field_validator(
        "validation_dataset_id",
        "dr_test_summary_dataset_id",
        mode="before",
    )
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

    def reset_for_signature(self, *, source_signature: str) -> CausalValidatePayloadModel:
        return self.model_copy(
            update={
                "source_signature": source_signature,
                "validation_dataset_id": None,
                "dr_test_summary_dataset_id": None,
                "validation_summary": None,
                "latest_query_result_raw_json_str": None,
                "latest_request_summary": None,
                "assistant_message": None,
                "message_artifact_refs": [],
                "error_message": None,
            }
        )


class CausalValidateState(NodeState):
    NAME: ClassVar[str] = "CAUSAL_VALIDATE"

    def __init__(self, payload: CausalValidatePayloadModel | None = None) -> None:
        self.payload = payload or CausalValidatePayloadModel()

    def name(self) -> str:
        return self.NAME

    def clear_state(self) -> None:
        self.payload = CausalValidatePayloadModel()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> CausalValidateState:
        return cls(CausalValidatePayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> CausalValidateState:
        return cls()


__all__ = ["CausalValidatePayloadModel", "CausalValidateState"]
