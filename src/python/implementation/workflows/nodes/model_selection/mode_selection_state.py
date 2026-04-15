from __future__ import annotations

from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.models.validation import ValidationIssueModel, ValidationStatus
from python.domain.workflows.node_state import NodeState
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.utils.utils import uuid_from_any

ModelSelectionPhase = Literal["INIT", "REVIEW_READY", "CONFIRMED", "FAILED"]


class ModelRecommendationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    estimator_fqcn: str
    display_label: str
    best_when: str
    why: str
    tradeoffs: str | None = None


class ConfirmedModelSelectionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    selected_model: str | None = None
    selected_model_display_label: str | None = None
    reasoning: str | None = None

    @field_validator(
        "selected_model",
        "selected_model_display_label",
        "reasoning",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        raise TypeError("confirmed model selection fields must be str|null")


class ModelSelectionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_dataset_id: UUID | None = None
    source_causal_spec: CausalSpec | None = None
    source_transformation_plan: TransformPlan | None = None
    source_validation_issues: list[ValidationIssueModel] = Field(default_factory=list)
    source_validation_status: ValidationStatus | None = None
    recommendations: list[ModelRecommendationModel] = Field(default_factory=list)
    confirmed_model_selection: ConfirmedModelSelectionPayload | None = None
    phase: ModelSelectionPhase = "INIT"
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
    def _normalize_payload_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        raise TypeError("payload text fields must be str|null")

    def bind_sources(
        self,
        *,
        dataset_id: UUID | None,
        causal_spec: CausalSpec,
        transformation_plan: TransformPlan,
        validation_issues: list[ValidationIssueModel],
        validation_status: ValidationStatus,
    ) -> ModelSelectionPayload:
        return self.model_copy(
            update={
                "source_dataset_id": dataset_id,
                "source_causal_spec": causal_spec,
                "source_transformation_plan": transformation_plan,
                "source_validation_issues": list(validation_issues),
                "source_validation_status": validation_status,
            }
        )

    def reset_for_reselection(
        self,
        *,
        dataset_id: UUID | None,
        causal_spec: CausalSpec,
        transformation_plan: TransformPlan,
        validation_issues: list[ValidationIssueModel],
        validation_status: ValidationStatus,
    ) -> ModelSelectionPayload:
        return self.model_copy(
            update={
                "source_dataset_id": dataset_id,
                "source_causal_spec": causal_spec,
                "source_transformation_plan": transformation_plan,
                "source_validation_issues": list(validation_issues),
                "source_validation_status": validation_status,
                "recommendations": [],
                "confirmed_model_selection": None,
                "phase": "INIT",
                "assistant_message": None,
                "system_message": None,
                "error_message": None,
            }
        )


class ModelSelectionState(NodeState):
    NAME: ClassVar[str] = "MODEL_SELECTION"

    def __init__(self, payload: ModelSelectionPayload | None = None) -> None:
        self.payload = payload or ModelSelectionPayload()

    def name(self) -> str:
        return self.NAME

    def clear_state(self) -> None:
        self.payload = ModelSelectionPayload()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ModelSelectionState:
        return cls(ModelSelectionPayload.model_validate(payload))

    @classmethod
    def init_empty(cls) -> ModelSelectionState:
        return cls()


__all__ = [
    "ConfirmedModelSelectionPayload",
    "ModelRecommendationModel",
    "ModelSelectionPayload",
    "ModelSelectionPhase",
    "ModelSelectionState",
]
