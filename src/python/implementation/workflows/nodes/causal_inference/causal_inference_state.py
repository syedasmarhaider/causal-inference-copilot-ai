from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.models.models import ArtifactRef
from python.domain.workflows.node_state import NodeState


class CausalInferencePayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_signature: str | None = None
    ate_result_raw_json_str: str | None = None
    latest_cate_result_raw_json_str: str | None = None
    latest_cate_request_summary: str | None = None
    assistant_message: str | None = None
    message_artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    error_message: str | None = None

    @field_validator(
        "source_signature",
        "ate_result_raw_json_str",
        "latest_cate_result_raw_json_str",
        "latest_cate_request_summary",
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

    def reset_for_signature(self, *, source_signature: str) -> CausalInferencePayloadModel:
        return self.model_copy(
            update={
                "source_signature": source_signature,
                "ate_result_raw_json_str": None,
                "latest_cate_result_raw_json_str": None,
                "latest_cate_request_summary": None,
                "assistant_message": None,
                "message_artifact_refs": [],
                "error_message": None,
            }
        )


class CausalInferenceState(NodeState):
    NAME: ClassVar[str] = "CAUSAL_INFERENCE"

    def __init__(self, payload: CausalInferencePayloadModel | None = None) -> None:
        self.payload = payload or CausalInferencePayloadModel()

    def name(self) -> str:
        return self.NAME

    def clear_state(self) -> None:
        self.payload = CausalInferencePayloadModel()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> CausalInferenceState:
        return cls(CausalInferencePayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> CausalInferenceState:
        return cls()


__all__ = [
    "CausalInferencePayloadModel",
    "CausalInferenceState",
]
