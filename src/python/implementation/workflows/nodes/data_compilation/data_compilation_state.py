from __future__ import annotations

from typing import Any, ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.models.validation import ValidationIssueModel, ValidationStatus
from python.domain.workflows.node_state import NodeState
from python.implementation.workflows.nodes.data_compilation.data_compilation_cleaning import (
    MissingnessDecisionList,
)
from python.implementation.workflows.nodes.data_compilation.data_compilation_transformation import (
    ColumnTransformationSuggestionList,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import (
    CausalSpecDraft,
)
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)
from python.implementation.workflows.utils.utils import uuid_from_any

DataCompilationPhase = Literal[
    "INIT",
    "ACTION_REQUIRED",
    "REVIEW_READY",
    "CONFIRMED",
    "FAILED",
]


class DataCompilationPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_dataset_id: UUID | None = None
    source_causal_spec_draft: CausalSpecDraft | None = None
    compiled_dataset_id: UUID | None = None
    compiled_dataset_summary: DatasetSummaryModel | None = None
    compiled_causal_spec: CausalSpec | None = None
    missingness_decisions: MissingnessDecisionList | None = None
    transformation_plan: TransformPlan | None = None
    transformation_suggestions: ColumnTransformationSuggestionList | None = None
    compilation_actions: list[str] = Field(default_factory=list)
    compilation_warnings: list[str] = Field(default_factory=list)
    validation_issues: list[ValidationIssueModel] = Field(default_factory=list)
    validation_status: ValidationStatus | None = None
    validation_retry_count: int = Field(default=0, ge=0)
    retry_feedback: str | None = None
    phase: DataCompilationPhase = "INIT"
    hard_failure: bool = False
    assistant_message: str | None = None
    system_message: str | None = None
    error_message: str | None = None

    @field_validator("source_dataset_id", "compiled_dataset_id", mode="before")
    @classmethod
    def _parse_dataset_id(cls, value: Any) -> UUID | None:
        return uuid_from_any(value)

    @field_validator(
        "retry_feedback",
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

    def bind_source(
        self,
        *,
        dataset_id: UUID,
        causal_spec_draft: CausalSpecDraft,
    ) -> DataCompilationPayloadModel:
        return self.model_copy(
            update={
                "source_dataset_id": dataset_id,
                "source_causal_spec_draft": causal_spec_draft,
            }
        )

    def reset_for_recompile(
        self,
        *,
        dataset_id: UUID,
        causal_spec_draft: CausalSpecDraft,
    ) -> DataCompilationPayloadModel:
        return self.model_copy(
            update={
                "source_dataset_id": dataset_id,
                "source_causal_spec_draft": causal_spec_draft,
                "compiled_dataset_id": None,
                "compiled_dataset_summary": None,
                "compiled_causal_spec": None,
                "missingness_decisions": None,
                "transformation_plan": None,
                "transformation_suggestions": None,
                "compilation_actions": [],
                "compilation_warnings": [],
                "validation_issues": [],
                "validation_status": None,
                "validation_retry_count": 0,
                "retry_feedback": None,
                "phase": "INIT",
                "hard_failure": False,
                "assistant_message": None,
                "system_message": None,
                "error_message": None,
            }
        )


class DataCompilationState(NodeState):
    NAME: ClassVar[str] = "DATA_COMPILATION"

    def __init__(self, payload: DataCompilationPayloadModel | None = None) -> None:
        self.payload = payload or DataCompilationPayloadModel()

    def name(self) -> str:
        return self.NAME

    def clear_state(self) -> None:
        self.payload = DataCompilationPayloadModel()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DataCompilationState:
        return cls(DataCompilationPayloadModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> DataCompilationState:
        return cls()


__all__ = [
    "DataCompilationPayloadModel",
    "DataCompilationPhase",
    "DataCompilationState",
]
