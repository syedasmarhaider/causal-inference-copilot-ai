from __future__ import annotations

from copy import deepcopy
from typing import Any, Final, Literal, overload
from collections.abc import Mapping
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import override

from python.domain.models.validation import ValidationIssueModel
from python.domain.workflows.ochestrator_state import ReadOnlyGlobalState, WritableGlobalState
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel

_VALID_GLOBAL_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "working_dataset_id",
        "working_dataset_summary",
        "working_dataset_frozen",
        "data_transformation_plan",
        "validation_issues",
        "validation_issues_accepted",
        "selected_model",
        "model_training_id",
    }
)

class GlobalStateModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    working_dataset_id: UUID | None = None
    working_dataset_summary: DatasetSummaryModel | None = None
    working_dataset_frozen: bool = False
    data_transformation_plan: TransformPlan | None = None
    validation_issues: list[ValidationIssueModel] = Field(default_factory=list)
    validation_issues_accepted: bool = False
    selected_model: str | None = None
    model_training_id: UUID | None = None

    @field_validator("selected_model")
    @classmethod
    def _normalize_selected_model(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        if not normalized:
            raise ValueError("selected_model must not be blank")

        return normalized

    @model_validator(mode="after")
    def _validate_cross_field_invariants(self) -> GlobalStateModel:
        if self.working_dataset_summary is not None and self.working_dataset_id is None:
            raise ValueError(
                "working_dataset_id must be set when working_dataset_summary is set"
            )

        if self.data_transformation_plan is not None and self.working_dataset_id is None:
            raise ValueError(
                "working_dataset_id must be set when data_transformation_plan is set"
            )

        if self.validation_issues_accepted and not self.validation_issues:
            raise ValueError(
                "validation_issues_accepted cannot be True when validation_issues is empty"
            )

        if self.model_training_id is not None and self.selected_model is None:
            raise ValueError(
                "selected_model must be set when model_training_id is set"
            )

        return self


class OrchestratorReadOnlyGlobalState(ReadOnlyGlobalState):
    def __init__(self, model: GlobalStateModel) -> None:
        self._model = model

    @overload
    def get(self, key: Literal["working_dataset_id"]) -> UUID | None: ...
    @overload
    def get(
        self, key: Literal["working_dataset_summary"]
    ) -> DatasetSummaryModel | None: ...
    @overload
    def get(self, key: Literal["working_dataset_frozen"]) -> bool: ...
    @overload
    def get(
        self, key: Literal["data_transformation_plan"]
    ) -> TransformPlan | None: ...
    @overload
    def get(
        self, key: Literal["validation_issues"]
    ) -> list[ValidationIssueModel]: ...
    @overload
    def get(self, key: Literal["validation_issues_accepted"]) -> bool: ...
    @overload
    def get(self, key: Literal["selected_model"]) -> str | None: ...
    @overload
    def get(self, key: Literal["model_training_id"]) -> UUID | None: ...

    @override
    def get(self, key: str) -> Any | None:
        if key not in _VALID_GLOBAL_STATE_KEYS:
            raise KeyError(f"unknown global state key: {key}")

        return getattr(self._model, key)

    def snapshot(self) -> GlobalStateModel:
        """
        Safe deep snapshot for callers that need the full state model.
        """
        return self._model.model_copy(deep=True)


class OrchestratorWritableGlobalState(
    OrchestratorReadOnlyGlobalState, WritableGlobalState
):
    def __init__(self, model: GlobalStateModel) -> None:
        super().__init__(model)

    @override
    def to_json_dict(self) -> dict[str, Any]:
        """
        JSON-safe output for persistence.
        """
        return self._model.model_dump(mode="json")

    @classmethod
    def from_json_dict(
        cls, payload: Mapping[str, Any]
    ) -> OrchestratorWritableGlobalState:
        normalized_payload = dict(payload)
        model = GlobalStateModel.model_validate(normalized_payload)
        return cls(model)

    @classmethod
    def init_empty(cls) -> OrchestratorWritableGlobalState:
        return cls(GlobalStateModel())


    def freeze_working_dataset(self) -> None:
        self._model.working_dataset_frozen = True

    def unfreeze_working_dataset(self) -> None:
        self._model.working_dataset_frozen = False

    def set_working_dataset_id(self, dataset_id: UUID) -> None:
        self._assert_dataset_mutable()

        if self._model.working_dataset_id == dataset_id:
            return

        self._model.working_dataset_id = dataset_id
        self._clear_dataset_dependent_state()

    def set_working_dataset_summary(
        self, summary: DatasetSummaryModel | None
    ) -> None:
        self._assert_dataset_mutable()

        if summary is not None and self._model.working_dataset_id is None:
            raise ValueError(
                "working_dataset_id must be set before working_dataset_summary"
            )

        self._model.working_dataset_summary = deepcopy(summary)

    def set_data_transformation_plan(
        self, plan: TransformPlan | None
    ) -> None:
        self._assert_dataset_mutable()

        if plan is not None and self._model.working_dataset_id is None:
            raise ValueError(
                "working_dataset_id must be set before data_transformation_plan"
            )

        self._model.data_transformation_plan = deepcopy(plan)

    def set_validation_issues(
        self, issues: list[ValidationIssueModel]
    ) -> None:
        self._model.validation_issues = issues
        self._model.validation_issues_accepted = False

    def clear_validation_issues(self) -> None:
        self._model.validation_issues = []
        self._model.validation_issues_accepted = False

    def accept_validation_issues(self) -> None:
        if not self._model.validation_issues:
            raise ValueError(
                "cannot accept validation issues when no validation issues exist"
            )

        self._model.validation_issues_accepted = True

    def set_selected_model(self, model_name: str | None) -> None:
        if model_name is None:
            self._model.selected_model = None
            self._model.model_training_id = None
            return

        if self._model.working_dataset_id is None:
            raise ValueError(
                "working_dataset_id must be set before selecting a model"
            )

        self._model.selected_model = model_name
        self._model.model_training_id = None

    def set_model_training_id(self, training_id: UUID | None) -> None:
        if training_id is not None and self._model.selected_model is None:
            raise ValueError(
                "selected_model must be set before model_training_id"
            )

        self._model.model_training_id = training_id

    def clear_training_state(self) -> None:
        self._model.selected_model = None
        self._model.model_training_id = None

    def _assert_dataset_mutable(self) -> None:
        if self._model.working_dataset_frozen:
            raise ValueError("working dataset is frozen")

    def _clear_dataset_dependent_state(self) -> None:
        self._model.working_dataset_summary = None
        self._model.data_transformation_plan = None
        self._model.validation_issues = []
        self._model.validation_issues_accepted = False
        self._model.selected_model = None
        self._model.model_training_id = None