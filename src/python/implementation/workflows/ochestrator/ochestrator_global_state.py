from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.validation import ValidationIssueModel
from python.domain.workflows.ochestrator_state import (
    ReadOnlyOchestratorState,
    WritableOchestratorState,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan import (
    TransformPlan,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)

log = logging.getLogger(__name__)


class GlobalStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # configuration fields
    last_active_node_name: str | None = None

    # stage 1 fields
    working_dataset_id: UUID | None = None
    working_dataset_summary: DatasetSummaryModel | None = None

    # stage 2 fields
    protocol_discussed: bool = False

    # stage 3 fields
    working_dataset_froozen: bool = False

    # stage 4 fields
    causal_spec: CausalSpec | None = None
    data_transformation_plan: TransformPlan | None = None
    validation_issues: list[ValidationIssueModel] = Field(default_factory=list)

    # stage 5 fields
    validation_issues_accepted: bool = False

    # stage 6 fields
    selected_model: str | None = None

    # stage 7 fields
    model_training_id: UUID | None = None


class OchestratorReadOnlyGlobalState(ReadOnlyOchestratorState):
    def __init__(self, model: GlobalStateModel) -> None:
        self._model = model

    def get(self, key: str) -> Any | None:
        if key not in GlobalStateModel.model_fields:
            raise KeyError(f"unknown global state key {key}")
        return deepcopy(getattr(self._model, key))


class OchestratorWritableGlobalState(
    OchestratorReadOnlyGlobalState, WritableOchestratorState
):
    _WORKFLOW_ORDER: Final[tuple[str, ...]] = (
        "working_dataset_id",
        "working_dataset_summary",
        "protocol_discussed",
        "working_dataset_froozen",
        "causal_spec",
        "data_transformation_plan",
        "validation_issues",
        "validation_issues_accepted",
        "selected_model",
        "model_training_id",
    )

    def __init__(self, model: GlobalStateModel) -> None:
        super().__init__(model)
        self._model = model

    def to_json_dict(self) -> dict[str, Any]:
        return self._model.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> OchestratorWritableGlobalState:
        model = GlobalStateModel.model_validate(payload)
        return cls(model)

    @classmethod
    def init_empty(cls) -> OchestratorWritableGlobalState:
        return cls(GlobalStateModel())

    # -----------------------------
    # public getters / setters
    # -----------------------------

    def get_last_active_node_name(self) -> str | None:
        return self._model.last_active_node_name

    def set_last_active_node_name(self, node_name: str) -> None:
        node_name = node_name.strip()
        if not node_name:
                raise ValueError("last_active_node_name cannot be blank")
        self._model.last_active_node_name = node_name

    def set_working_dataset_id(self, dataset_id: UUID) -> None:
        if self._model.working_dataset_id == dataset_id:
            return

        self._model.working_dataset_id = dataset_id
        self._clear_forward_from(
            "working_dataset_id",
            reason="working_dataset_id changed",
        )

    def clear_working_dataset(self) -> None:
        self._reset_from(
            "working_dataset_id",
            reason="working_dataset_id cleared",
        )

    def set_working_dataset_summary(
        self, summary: DatasetSummaryModel | None
    ) -> None:
        if summary is not None:
            self._require_working_dataset_id()

        if self._model.working_dataset_summary == summary:
            return

        self._model.working_dataset_summary = summary
        self._clear_forward_from(
            "working_dataset_summary",
            reason="working_dataset_summary changed",
        )

    def clear_working_dataset_summary(self) -> None:
        self.set_working_dataset_summary(None)

    def set_protocol_discussed(self, discussed: bool) -> None:
        if discussed:
            self._require_stage_1_complete()

        if self._model.protocol_discussed == discussed:
            return

        self._model.protocol_discussed = discussed
        self._clear_forward_from(
            "protocol_discussed",
            reason="protocol_discussed changed",
        )

    def set_working_dataset_froozen(self, frozen: bool) -> None:
        if frozen:
            self._require_stage_2_complete()

        if self._model.working_dataset_froozen == frozen:
            return

        self._model.working_dataset_froozen = frozen
        self._clear_forward_from(
            "working_dataset_froozen",
            reason="working_dataset_froozen changed",
        )

    def freeze_working_dataset(self) -> None:
        self.set_working_dataset_froozen(True)

    def unfreeze_working_dataset(self) -> None:
        self.set_working_dataset_froozen(False)

    def set_causal_spec(self, causal_spec: CausalSpec | None) -> None:
        if causal_spec is not None:
            self._require_stage_3_complete()

        if self._model.causal_spec == causal_spec:
            return

        self._model.causal_spec = causal_spec
        self._clear_forward_from(
            "causal_spec",
            reason="causal_spec changed",
        )

    def clear_causal_spec(self) -> None:
        self.set_causal_spec(None)

    def set_data_transformation_plan(self, plan: TransformPlan | None) -> None:
        if plan is not None:
            self._require_causal_spec_ready()

        if self._model.data_transformation_plan == plan:
            return

        self._model.data_transformation_plan = plan
        self._clear_forward_from(
            "data_transformation_plan",
            reason="data_transformation_plan changed",
        )

    def clear_data_transformation_plan(self) -> None:
        self.set_data_transformation_plan(None)

    def set_validation_issues(self, issues: list[ValidationIssueModel]) -> None:
        normalized_issues = list(issues)

        if normalized_issues:
            self._require_transformation_plan_ready()

        if self._model.validation_issues == normalized_issues:
            return

        self._model.validation_issues = normalized_issues
        self._clear_forward_from(
            "validation_issues",
            reason="validation_issues changed",
        )

    def clear_validation_issues(self) -> None:
        self.set_validation_issues([])

    def set_validation_issues_accepted(self, accepted: bool) -> None:
        if accepted:
            self._require_validation_issues_present()

        if self._model.validation_issues_accepted == accepted:
            return

        self._model.validation_issues_accepted = accepted
        self._clear_forward_from(
            "validation_issues_accepted",
            reason="validation_issues_accepted changed",
        )

    def accept_validation_issues(self) -> None:
        self.set_validation_issues_accepted(True)

    def reject_validation_issues(self) -> None:
        self.set_validation_issues_accepted(False)

    def set_selected_model(self, model_name: str | None) -> None:
        normalized_model_name: str | None = None
        if model_name is not None:
            normalized_model_name = model_name.strip()
            if not normalized_model_name:
                raise ValueError("selected_model cannot be blank")
            self._require_model_selection_ready()

        if self._model.selected_model == normalized_model_name:
            return

        self._model.selected_model = normalized_model_name
        self._clear_forward_from(
            "selected_model",
            reason="selected_model changed",
        )

    def clear_selected_model(self) -> None:
        self.set_selected_model(None)

    def set_model_training_id(self, training_id: UUID | None) -> None:
        if training_id is not None:
            self._require_model_training_ready()

        if self._model.model_training_id == training_id:
            return

        self._model.model_training_id = training_id

    def clear_model_training_id(self) -> None:
        self.set_model_training_id(None)

    # -----------------------------
    # stage guards
    # -----------------------------

    def _require_working_dataset_id(self) -> None:
        if self._model.working_dataset_id is None:
            raise ValueError("working_dataset_id must be set first")

    def _require_stage_1_complete(self) -> None:
        if self._model.working_dataset_id is None:
            raise ValueError("working_dataset_id must be set first")
        if self._model.working_dataset_summary is None:
            raise ValueError("working_dataset_summary must be set first")

    def _require_stage_2_complete(self) -> None:
        self._require_stage_1_complete()
        if not self._model.protocol_discussed:
            raise ValueError("protocol_discussed must be True first")

    def _require_stage_3_complete(self) -> None:
        self._require_stage_2_complete()
        if not self._model.working_dataset_froozen:
            raise ValueError("working_dataset_froozen must be True first")

    def _require_causal_spec_ready(self) -> None:
        self._require_stage_3_complete()
        if self._model.causal_spec is None:
            raise ValueError("causal_spec must be set first")

    def _require_transformation_plan_ready(self) -> None:
        self._require_causal_spec_ready()
        if self._model.data_transformation_plan is None:
            raise ValueError("data_transformation_plan must be set first")

    def _require_validation_issues_present(self) -> None:
        self._require_transformation_plan_ready()
        if not self._model.validation_issues:
            raise ValueError("validation_issues must be non-empty first")

    def _require_model_selection_ready(self) -> None:
        self._require_transformation_plan_ready()
        if self._model.validation_issues and not self._model.validation_issues_accepted:
            raise ValueError(
                "validation_issues must be accepted before selected_model is set"
            )

    def _require_model_training_ready(self) -> None:
        self._require_model_selection_ready()
        if self._model.selected_model is None:
            raise ValueError("selected_model must be set first")

    # -----------------------------
    # reset helpers
    # -----------------------------

    def _clear_forward_from(self, field_name: str, *, reason: str) -> None:
        field_index = self._WORKFLOW_ORDER.index(field_name)
        cleared_fields: list[str] = []

        for downstream_field in self._WORKFLOW_ORDER[field_index + 1 :]:
            if self._reset_field_if_needed(downstream_field):
                cleared_fields.append(downstream_field)

        if cleared_fields:
            log.warning(
                "%s; cleared forward fields: %s",
                reason,
                ", ".join(cleared_fields),
            )

    def _reset_from(self, field_name: str, *, reason: str) -> None:
        field_index = self._WORKFLOW_ORDER.index(field_name)
        cleared_fields: list[str] = []

        for current_field in self._WORKFLOW_ORDER[field_index:]:
            if self._reset_field_if_needed(current_field):
                cleared_fields.append(current_field)

        if cleared_fields:
            log.warning(
                "%s; cleared fields: %s",
                reason,
                ", ".join(cleared_fields),
            )

    def _reset_field_if_needed(self, field_name: str) -> bool:
        current_value = getattr(self._model, field_name)
        default_value = self._default_value_for(field_name)

        if current_value == default_value:
            return False

        setattr(self._model, field_name, default_value)
        return True

    @staticmethod
    def _default_value_for(field_name: str) -> Any:
        if field_name in {
            "protocol_discussed",
            "working_dataset_froozen",
            "validation_issues_accepted",
        }:
            return False
        if field_name == "validation_issues":
            return []
        return None