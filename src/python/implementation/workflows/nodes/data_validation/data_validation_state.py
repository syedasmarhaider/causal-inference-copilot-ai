from __future__ import annotations

from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.models.validation import ValidationIssueModel, ValidationStatus
from python.domain.workflows.node_state import NodeState
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.utils.utils import uuid_from_any

DataValidationPhase = Literal["INIT", "REVIEW_READY", "CONFIRMED", "FAILED"]


class DataValidationPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_dataset_id: UUID | None = None
    source_causal_spec: CausalSpec | None = None
    source_transformation_plan: TransformPlan | None = None
    validation_issues: list[ValidationIssueModel] = Field(default_factory=list)
    validation_status: ValidationStatus | None = None
    phase: DataValidationPhase = "INIT"
    assistant_message: str | None = None
    system_message: str | None = None
    error_message: str | None = None

    @field_validator("source_dataset_id", mode="before")
    @classmethod
    def _parse_dataset_id(cls, value: Any) -> UUID | None:
        return uuid_from_any(value)

    @field_validator(
        "assistant_message",
        "system_message",
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

    def bind_sources(
        self,
        *,
        dataset_id: UUID,
        causal_spec: CausalSpec,
        transformation_plan: TransformPlan,
    ) -> DataValidationPayloadModel:
        return self.model_copy(
            update={
                "source_dataset_id": dataset_id,
                "source_causal_spec": causal_spec,
                "source_transformation_plan": transformation_plan,
            }
        )

    def reset_for_revalidation(
        self,
        *,
        dataset_id: UUID,
        causal_spec: CausalSpec,
        transformation_plan: TransformPlan,
    ) -> DataValidationPayloadModel:
        return self.model_copy(
            update={
                "source_dataset_id": dataset_id,
                "source_causal_spec": causal_spec,
                "source_transformation_plan": transformation_plan,
                "validation_issues": [],
                "validation_status": None,
                "phase": "INIT",
                "assistant_message": None,
                "system_message": None,
                "error_message": None,
            }
        )


class DataValidationState(NodeState):
    NAME: ClassVar[str] = "DATA_VALIDATION"

    def __init__(self, payload: DataValidationPayloadModel | None = None) -> None:
        self.payload = payload or DataValidationPayloadModel()

    def name(self) -> str:
        return self.NAME

    def clear_state(self) -> None:
        self.payload = DataValidationPayloadModel()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DataValidationState:
        return cls(DataValidationPayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> DataValidationState:
        return cls()


__all__ = [
    "DataValidationPayloadModel",
    "DataValidationPhase",
    "DataValidationState",
]
