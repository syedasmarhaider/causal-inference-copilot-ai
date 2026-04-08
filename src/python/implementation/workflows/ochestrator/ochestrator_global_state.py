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
from python.implementation.workflows.nodes.causal_inference.causal_inference_node import CausalInferenceNode
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_node import CompileAndValidateNode
from python.implementation.workflows.nodes.dataset.dataset_node import DatasetNode
from python.implementation.workflows.nodes.model_selection.model_selection_node import ModelSelectionNode
from python.implementation.workflows.nodes.model_train.model_train_node import ModelTrainNode
from python.implementation.workflows.nodes.noop_done.noop_done_node import NoopDoneNode
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import ProtocolDiscussionNode
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
    
    # stage 1
    working_dataset_id: UUID | None = None
    working_dataset_summary: DatasetSummaryModel | None = None

    # stage 2
    protocol_discussion: str | None = None
    dataset_cleaning_pending: bool = False

    # stage 3
    working_dataset_frozen: bool = False

    # stage 4
    causal_spec: CausalSpec | None = None
    data_transformation_plan: TransformPlan | None = None
    validation_issues: list[ValidationIssueModel] = Field(default_factory=list)

    # stage 5
    selected_model: str | None = None

    # stage 6
    model_training_id: UUID | None = None


class OchestratorReadOnlyGlobalState(ReadOnlyOchestratorState):
    def __init__(self, model: GlobalStateModel) -> None:
        self._model = model

    def get(self, key: str) -> Any | None:
        if key not in GlobalStateModel.model_fields:
            raise KeyError(f"unknown global state key: {key}")
        return deepcopy(getattr(self._model, key))


class OchestratorWritableGlobalState(
    OchestratorReadOnlyGlobalState,
    WritableOchestratorState,
):
    _WORKFLOW_ORDER: Final[tuple[str, ...]] = (
        "working_dataset_id",
        "working_dataset_summary",
        "protocol_discussion",
        "dataset_cleaning_pending",
        "working_dataset_frozen",
        "causal_spec",
        "data_transformation_plan",
        "validation_issues",
        "selected_model",
        "model_training_id",
    )

    def __init__(self, model: GlobalStateModel) -> None:
        super().__init__(model)
        self._model = model

    @classmethod
    def init_empty(cls) -> OchestratorWritableGlobalState:
        return cls(GlobalStateModel())

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> OchestratorWritableGlobalState:
        model = GlobalStateModel.model_validate(payload)
        return cls(model)
    
    def name(self) -> str:
        return "OCHESTRATOR_STATE"

    def to_json_dict(self) -> dict[str, Any]:
        return self._model.model_dump(mode="json")

    # -------------------------------------------------------------------------
    # stage 1: working dataset
    # -------------------------------------------------------------------------

    def set_working_dataset(
        self,
        dataset_id: UUID,
        summary: DatasetSummaryModel,
    ) -> None:
        previous_dataset_id = self._model.working_dataset_id
        previous_summary = self._model.working_dataset_summary
        had_protocol_discussion = self._has_protocol_discussion()

        if previous_dataset_id == dataset_id and previous_summary == summary:
            return

        self._model.working_dataset_id = dataset_id
        self._model.working_dataset_summary = summary

        same_dataset_refined_summary = (
            previous_dataset_id == dataset_id
            and previous_summary != summary
        )

        if same_dataset_refined_summary and had_protocol_discussion:
            self._invalidate_downstream_of(
                "protocol_discussion",
                reason="working dataset summary changed for same dataset; preserving protocol discussion",
            )
            return

        self._invalidate_downstream_of(
            "working_dataset_summary",
            reason="working dataset changed",
        )

    def invalidate_working_dataset_and_downstream(self) -> None:
        self._reset_from_including(
            "working_dataset_id",
            reason="working dataset invalidated",
        )

    # -------------------------------------------------------------------------
    # stage 2: protocol discussion
    # -------------------------------------------------------------------------

    def set_protocol_discussion(self, protocol_discussion: str) -> None:
        self._require_stage_1_complete()

        normalized_protocol_discussion = self._normalize_non_blank_text(
            value=protocol_discussion,
            field_name="protocol_discussion",
        )

        if self._model.protocol_discussion == normalized_protocol_discussion:
            return

        self._model.protocol_discussion = normalized_protocol_discussion
        self._invalidate_downstream_of(
            "protocol_discussion",
            reason="protocol discussion changed",
        )

    def invalidate_protocol_discussion_and_downstream(self) -> None:
        self._reset_from_including(
            "protocol_discussion",
            reason="protocol discussion invalidated",
        )

    def mark_dataset_cleaning_pending(self) -> None:
        self._require_stage_2_complete()

        if self._model.dataset_cleaning_pending:
            return

        self._model.dataset_cleaning_pending = True
        self._invalidate_downstream_of(
            "dataset_cleaning_pending",
            reason="dataset cleaning pending",
        )

    # -------------------------------------------------------------------------
    # stage 3: dataset freeze
    # -------------------------------------------------------------------------

    def freeze_working_dataset(self) -> None:
        self._require_stage_2_complete()

        if self._model.working_dataset_frozen and not self._model.dataset_cleaning_pending:
            return

        self._model.dataset_cleaning_pending = False
        self._model.working_dataset_frozen = True
        self._invalidate_downstream_of(
            "working_dataset_frozen",
            reason="working dataset frozen",
        )

    def freeze_working_dataset_snapshot(
        self,
        dataset_id: UUID,
        dataset_summary: DatasetSummaryModel,
    ) -> None:
        self._require_stage_2_complete()

        dataset_changed = (
            self._model.working_dataset_id != dataset_id
            or self._model.working_dataset_summary != dataset_summary
        )
        frozen_changed = not self._model.working_dataset_frozen
        pending_changed = self._model.dataset_cleaning_pending

        if not dataset_changed and not frozen_changed and not pending_changed:
            return

        self._model.working_dataset_id = dataset_id
        self._model.working_dataset_summary = dataset_summary
        self._model.dataset_cleaning_pending = False
        self._model.working_dataset_frozen = True

        self._invalidate_downstream_of(
            "working_dataset_frozen",
            reason="working dataset snapshot frozen",
        )

    def unfreeze_working_dataset_and_downstream(self) -> None:
        self._require_stage_1_complete()

        if not self._model.working_dataset_frozen:
            return

        self._model.dataset_cleaning_pending = False
        self._model.working_dataset_frozen = False
        self._invalidate_downstream_of(
            "working_dataset_frozen",
            reason="working dataset unfrozen",
        )

    # -------------------------------------------------------------------------
    # stage 4: causal configuration
    # -------------------------------------------------------------------------

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

        self._invalidate_downstream_of(
            "validation_issues",
            reason="causal configuration changed",
        )

    def invalidate_causal_configuration_and_downstream(self) -> None:
        self._reset_from_including(
            "causal_spec",
            reason="causal configuration invalidated",
        )

    # -------------------------------------------------------------------------
    # stage 5: model selection
    # -------------------------------------------------------------------------

    def set_selected_model(self, selected_model: str) -> None:
        self._require_model_selection_ready()

        normalized_selected_model = self._normalize_non_blank_text(
            value=selected_model,
            field_name="selected_model",
        )

        if self._model.selected_model == normalized_selected_model:
            return

        self._model.selected_model = normalized_selected_model
        self._invalidate_downstream_of(
            "selected_model",
            reason="selected model changed",
        )

    def invalidate_selected_model_and_downstream(self) -> None:
        self._reset_from_including(
            "selected_model",
            reason="selected model invalidated",
        )

    # -------------------------------------------------------------------------
    # stage 6: training
    # -------------------------------------------------------------------------

    def set_model_training_id(self, training_id: UUID) -> None:
        self._require_model_training_ready()

        if self._model.model_training_id == training_id:
            return

        self._model.model_training_id = training_id

    def clear_model_training_id(self) -> None:
        if self._model.model_training_id is None:
            return
        self._model.model_training_id = None
    
    
    # -------------------------------------------------------------------------
    # active node tracking and rollback
    # -------------------------------------------------------------------------
    def needs_node_name(
        self,
    ) -> str:

        if self._model.working_dataset_id is None:
            return DatasetNode.NAME

        if self._model.working_dataset_summary is None:
            return DatasetNode.NAME

        if not self._has_protocol_discussion():
            return ProtocolDiscussionNode.NAME

        if self._model.dataset_cleaning_pending:
            return DatasetNode.NAME

        if not self._model.working_dataset_frozen:
            return DatasetNode.NAME

        if self._model.causal_spec is None:
            return CompileAndValidateNode.NAME

        if self._model.data_transformation_plan is None:
            return CompileAndValidateNode.NAME

        if self._model.selected_model is None:
            return ModelSelectionNode.NAME

        if self._model.model_training_id is None:
            return ModelTrainNode.NAME

        return CausalInferenceNode.NAME
    
    
    def rollback_orchestrator_global_state(
        self,
        recovery_state_name: str,
    ) -> None:
        if recovery_state_name == ProtocolDiscussionNode.NAME:
            self.invalidate_protocol_discussion_and_downstream()
    
        if recovery_state_name == DatasetNode.NAME:
            self.invalidate_protocol_discussion_and_downstream()

        if recovery_state_name == CompileAndValidateNode.NAME:
            self.invalidate_causal_configuration_and_downstream()

        if recovery_state_name == ModelSelectionNode.NAME:
            self.invalidate_selected_model_and_downstream()

        if recovery_state_name == ModelTrainNode.NAME:
            self.clear_model_training_id()        

    # -------------------------------------------------------------------------
    # guards
    # -------------------------------------------------------------------------

    def _require_stage_1_complete(self) -> None:
        if self._model.working_dataset_id is None:
            raise ValueError("working_dataset_id must be set first")
        if self._model.working_dataset_summary is None:
            raise ValueError("working_dataset_summary must be set first")

    def _require_stage_2_complete(self) -> None:
        self._require_stage_1_complete()
        if not self._has_protocol_discussion():
            raise ValueError("protocol_discussion must be set first")

    def _require_stage_3_complete(self) -> None:
        self._require_stage_2_complete()
        if not self._model.working_dataset_frozen:
            raise ValueError("working_dataset_frozen must be True first")

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

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------

    def _has_protocol_discussion(self) -> bool:
        return self._model.protocol_discussion is not None

    def _invalidate_downstream_of(self, field_name: str, *, reason: str) -> None:
        field_index = self._WORKFLOW_ORDER.index(field_name)
        cleared_fields: list[str] = []

        for downstream_field in self._WORKFLOW_ORDER[field_index + 1 :]:
            if self._reset_field_if_needed(downstream_field):
                cleared_fields.append(downstream_field)

        if cleared_fields:
            log.warning(
                "%s; cleared downstream fields: %s",
                reason,
                ", ".join(cleared_fields),
            )

    def _reset_from_including(self, field_name: str, *, reason: str) -> None:
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
        if field_name == "dataset_cleaning_pending":
            return False
        if field_name == "working_dataset_frozen":
            return False
        if field_name == "validation_issues":
            return []
        return None

    @staticmethod
    def _normalize_non_blank_text(*, value: str, field_name: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError(f"{field_name} cannot be blank")
        return normalized_value
