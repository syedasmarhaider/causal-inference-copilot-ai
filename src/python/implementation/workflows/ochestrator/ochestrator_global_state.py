from __future__ import annotations

from copy import deepcopy
from typing import Any, Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.validation import ValidationIssueModel
from python.domain.workflows.ochestrator_state import (
    ReadOnlyOchestratorState,
    WritableOchestratorState,
)
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.tools.causal.encoding.encoding_plan import (
    TransformPlan,
)
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import (
    DatasetSummaryModel,
)

log = get_logger(__name__)


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
    selected_model: str | None = None

    # stage 6 fields
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

    def get_last_active_node_name(self) -> str | None:
        return self._model.last_active_node_name

    def set_last_active_node_name(self, node_name: str) -> None:
        normalized_node_name = node_name.strip()
        if not normalized_node_name:
            raise ValueError("last_active_node_name cannot be blank")
        self._model.last_active_node_name = normalized_node_name

    # -----------------------------
    # stage 1
    # -----------------------------

    def set_working_dataset(
        self,
        dataset_id: UUID,
        summary: DatasetSummaryModel,
    ) -> None:
        previous_dataset_id = self._model.working_dataset_id
        previous_summary = self._model.working_dataset_summary
        previous_protocol_discussed = self._model.protocol_discussed

        if previous_dataset_id == dataset_id and previous_summary == summary:
            return

        self._model.working_dataset_id = dataset_id
        self._model.working_dataset_summary = summary

        if (
            previous_dataset_id == dataset_id
            and previous_summary != summary
            and previous_protocol_discussed
        ):
            # same dataset, refined summary after protocol discussion:
            # preserve protocol_discussed, invalidate only downstream stages
            self._clear_forward_from(
                "protocol_discussed",
                reason="working dataset summary changed after protocol discussion",
            )
            return

        # stage 1 changed => invalidate stages after stage 1
        self._clear_forward_from(
            "working_dataset_summary",
            reason="working dataset changed",
        )

    def clear_working_dataset(self) -> None:
        self._reset_from(
            "working_dataset_id",
            reason="working dataset cleared",
        )

    # -----------------------------
    # stage 2
    # -----------------------------

    def set_protocol_discussed(self) -> None:
        self._require_stage_1_complete()

        if self._model.protocol_discussed:
            return

        self._model.protocol_discussed = True
        self._clear_forward_from(
            "protocol_discussed",
            reason="protocol_discussed changed",
        )

    def clear_protocol_discussed(self) -> None:
        if not self._model.protocol_discussed:
            return

        self._model.protocol_discussed = False
        self._clear_forward_from(
            "protocol_discussed",
            reason="protocol_discussed cleared",
        )

    # -----------------------------
    # stage 3
    # -----------------------------

    def set_freeze_working_dataset(
        self, datasetid: UUID, datasetSummary: DatasetSummaryModel
    ) -> None:
        self._require_stage_2_complete()

        if self._model.working_dataset_froozen:
            return

        self._model.working_dataset_froozen = True
        self._model.working_dataset_id = datasetid
        self._model.working_dataset_summary = datasetSummary
        self._clear_forward_from(
            "working_dataset_froozen",
            reason="working_dataset_froozen changed",
        )

    def freeze_working_dataset(self) -> None:
        self._require_stage_2_complete()

        if self._model.working_dataset_froozen:
            return

        self._model.working_dataset_froozen = True
        self._clear_forward_from(
            "working_dataset_froozen",
            reason="working_dataset_froozen changed",
        )

    def unfreeze_working_dataset(self) -> None:
        self._require_stage_1_complete()

        if not self._model.working_dataset_froozen:
            return

        self._model.working_dataset_froozen = False
        self._clear_forward_from(
            "working_dataset_froozen",
            reason="working_dataset_froozen changed",
        )

    # -----------------------------
    # stage 4
    # -----------------------------

    def set_causal_configuration(
        self,
        causal_spec: CausalSpec,
        data_transformation_plan: TransformPlan,
        validation_issues: list[ValidationIssueModel],
    ) -> None:
        self._require_stage_3_complete()

        normalized_issues = list(validation_issues)

        if (
            self._model.causal_spec == causal_spec
            and self._model.data_transformation_plan == data_transformation_plan
            and self._model.validation_issues == normalized_issues
        ):
            return

        self._model.causal_spec = causal_spec
        self._model.data_transformation_plan = data_transformation_plan
        self._model.validation_issues = normalized_issues

        # clear only downstream stages, keep stage 4 atomically applied
        self._clear_forward_from(
            "validation_issues",
            reason="causal configuration changed",
        )

    def clear_causal_configuration(self) -> None:
        self._reset_from(
            "causal_spec",
            reason="causal configuration cleared",
        )

    # -----------------------------
    # stage 5
    # -----------------------------

    def set_selected_model(self, selected_model: str) -> None:
        self._require_model_selection_ready()

        self._model.selected_model = selected_model
        self._clear_forward_from(
            "selected_model",
            reason="selected_model changed",
        )

    def clear_selected_model(self) -> None:
        if self._model.selected_model is None:
            return

        self._model.selected_model = None
        self._clear_forward_from(
            "selected_model",
            reason="selected_model cleared",
        )

    # -----------------------------
    # stage 6
    # -----------------------------

    def set_model_training_id(self, training_id: UUID) -> None:
        self._require_model_training_ready()

        if self._model.model_training_id == training_id:
            return

        self._model.model_training_id = training_id

    def clear_model_training_id(self) -> None:
        if self._model.model_training_id is None:
            return

        self._model.model_training_id = None

    # -----------------------------
    # stage guards
    # -----------------------------

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

    def _require_stage_4_complete(self) -> None:
        self._require_stage_3_complete()
        if self._model.causal_spec is None:
            raise ValueError("causal_spec must be set first")
        if self._model.data_transformation_plan is None:
            raise ValueError("data_transformation_plan must be set first")

    def _require_model_selection_ready(self) -> None:
        self._require_stage_4_complete()

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
        }:
            return False
        if field_name == "validation_issues":
            return []
        return None