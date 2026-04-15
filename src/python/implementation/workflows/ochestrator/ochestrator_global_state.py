from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from python.domain.models.validation import ValidationIssueModel
from python.domain.workflows.ochestrator_state import OchestratorState
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.causal_inference.causal_inference_node import (
    CausalInferenceNode,
)
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import CausalInferenceState
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_node import (
    CompileAndValidateNode,
)
from python.implementation.workflows.nodes.compile_and_validate.compile_and_validate_state import CompileAndValidateState
from python.implementation.workflows.nodes.dataset.dataset_node import DatasetNode
from python.implementation.workflows.nodes.dataset.dataset_state import DatasetState
from python.implementation.workflows.nodes.model_selection.mode_selection_state import ModelSelectionState
from python.implementation.workflows.nodes.model_selection.model_selection_node import (
    ModelSelectionNode,
)
from python.implementation.workflows.nodes.model_train.model_train_node import (
    ModelTrainNode,
)
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainState
from python.implementation.workflows.nodes.noop_done.noop_done_state import NoopDoneState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import (
    ProtocolDiscussionNode,
)
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import ProtocolDiscussionState
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

    # stage 2.5
    data_cleaned: bool = False

    # stage 3
    working_dataset_frozen: bool = False

    # stage 4
    causal_spec: CausalSpec | None = None
    data_transformation_plan: TransformPlan | None = None
    validation_issues: list[ValidationIssueModel] = Field(default_factory=lambda: [])

    # stage 5
    selected_model: str | None = None

    # stage 6
    model_training_id: UUID | None = None






class OchestratorWritableGlobalState(OchestratorState):

    def __init__(self, model: GlobalStateModel) -> None:
        super().__init__(model)
        self._model = model

    @classmethod
    def init_empty(cls) -> OchestratorWritableGlobalState:
        return cls(GlobalStateModel())
    
    
    def get(self, key: str) -> Any | None:
        if key not in GlobalStateModel.model_fields:
            raise KeyError(f"unknown global state key: {key}")
        return deepcopy(getattr(self._model, key))
    
    
    def set(self, key: str, value: Any) -> None:
        

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> OchestratorWritableGlobalState:
        normalized_payload = dict(payload)
        legacy_data_cleaning_pending = normalized_payload.pop(
            "dataset_cleaning_pending",
            None,
        )
        if "data_cleaned" not in normalized_payload and legacy_data_cleaning_pending is not None:
            normalized_payload["data_cleaned"] = not bool(legacy_data_cleaning_pending)

        model = GlobalStateModel.model_validate(normalized_payload)
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
        preserve_protocol_discussion: bool,
    ) -> None:
        previous_dataset_id = self._model.working_dataset_id
        previous_summary = self._model.working_dataset_summary

        if previous_dataset_id == dataset_id and previous_summary == summary:
            return

        self._model.working_dataset_id = dataset_id
        self._model.working_dataset_summary = summary
        
        if preserve_protocol_discussion:
            self._invalidate_downstream_of(
                "protocol_discussion",
                reason="working dataset changed with protocol discussion preserved",
            )
        else:
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

    # -------------------------------------------------------------------------
    # stage 2.5: dataset cleaning
    # -------------------------------------------------------------------------

    def mark_data_cleaned(self) -> None:
        self._require_stage_2_complete()

        if self._model.data_cleaned:
            return

        self._model.data_cleaned = True
        self._invalidate_downstream_of(
            "data_cleaned",
            reason="dataset cleaned",
        )

    def mark_dataset_cleaning_pending(self) -> None:
        self._require_stage_2_complete()

        if not self._model.data_cleaned and not self._model.working_dataset_frozen:
            return

        self._reset_from_including(
            "data_cleaned",
            reason="dataset cleaning pending",
        )

    # -------------------------------------------------------------------------
    # stage 3: dataset freeze
    # -------------------------------------------------------------------------

    def freeze_working_dataset(self) -> None:
        self._require_stage_2_complete()

        if self._model.data_cleaned and self._model.working_dataset_frozen:
            return

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

        if (
            self._model.working_dataset_id == dataset_id
            and self._model.working_dataset_summary == dataset_summary
            and self._model.data_cleaned
            and self._model.working_dataset_frozen
        ):
            return

        self._model.working_dataset_id = dataset_id
        self._model.working_dataset_summary = dataset_summary
        self._model.data_cleaned = True
        self._model.working_dataset_frozen = True

        self._invalidate_downstream_of(
            "working_dataset_frozen",
            reason="working dataset snapshot frozen",
        )

    def unfreeze_working_dataset_and_downstream(self) -> None:
        self._require_stage_1_complete()

        if not self._model.data_cleaned and not self._model.working_dataset_frozen:
            return

        self._reset_from_including(
            "data_cleaned",
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

    def needs_node_name(self) -> str:
        if self._model.working_dataset_id is None:
            return DatasetNode.NAME

        if self._model.working_dataset_summary is None:
            return DatasetNode.NAME

        if not self._has_protocol_discussion():
            return ProtocolDiscussionNode.NAME

        if not self._model.data_cleaned:
            return DatasetNode.NAME

        if not self._model.working_dataset_frozen:
            return CompileAndValidateNode.NAME

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
        if not self._model.data_cleaned:
            raise ValueError("data_cleaned must be True first")

    def _require_stage_4_complete(self) -> None:
        self._require_stage_3_complete()
        if self._model.causal_spec is None:
            raise ValueError("causal_spec must be set first")
        if self._model.data_transformation_plan is None:
            raise ValueError("data_transformation_plan must be set first")
        if not self._model.working_dataset_frozen:
            raise ValueError("working_dataset must be frozen first")

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
        if field_name in {"data_cleaned", "working_dataset_frozen"}:
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
    
    
    def update_ochestration_working_state_if_node_done(
        self,
        *,
        state: State,
    ) -> None:
        if state.status() != "DONE" and state.name() != DatasetState.NAME:
            return

        match state:
            case DatasetState() as dataset_state:
                if not dataset_state.payload.dataset_iterations:
                    return 

                latest_iteration_dataset_id = (
                    dataset_state.payload.dataset_iterations[-1]
                )
                latest_iteration_dataset_summary = dataset_state.payload.latest_summary
                if latest_iteration_dataset_summary is None:
                    raise ValueError(
                        "Latest dataset summary must be set when dataset state is DONE"
                    )

                protocol_discussed = self._has_protocol_discussion()
                data_frozen = self.get("working_dataset_frozen") is True
                
                if data_frozen:
                    return

                if protocol_discussed:
                    self.set_working_dataset(
                        dataset_id=latest_iteration_dataset_id,
                        summary=latest_iteration_dataset_summary,
                        preserve_protocol_discussion=True,
                    )
                    self.mark_data_cleaned()
                else:
                    self.set_working_dataset(
                        dataset_id=latest_iteration_dataset_id,
                        summary=latest_iteration_dataset_summary,
                        preserve_protocol_discussion=False,
                    )
                    

            case ProtocolDiscussionState():
                self.set_protocol_discussion(
                    protocol_discussion=state.payload.discussion
                )

            case CompileAndValidateState() as compile_and_validate_state:
                inference_ready_spec = (
                    compile_and_validate_state.payload.inference_ready_causal_spec
                )
                if inference_ready_spec is None:
                    raise ValueError(
                        "Inference ready causal spec must be set when compile-and-validate is DONE"
                    )
                    
                self.set_causal_configuration(
                    causal_spec=inference_ready_spec.causal_spec,
                    data_transformation_plan=inference_ready_spec.transformation_plan,
                    validation_issues=compile_and_validate_state.payload.validation_issues,
                )
                self.freeze_working_dataset()

            case ModelSelectionState() as model_selection_state:
                confirmed = model_selection_state.payload.confirmed_model_selection
                if confirmed is None or confirmed.selected_model is None:
                    raise ValueError(
                        "Confirmed model selection must be set when model selection is DONE"
                    )

                self.set_selected_model(confirmed.selected_model)

            case ModelTrainState() as model_train_state:
                if model_train_state.payload.trained_model_id is None:
                    raise ValueError(
                        "Trained model ID must be set when model-train is DONE"
                    )

                self.set_model_training_id(
                    model_train_state.payload.trained_model_id
                )

            case CausalInferenceState():
                pass

            case NoopDoneState():
                pass

            case _:
                raise ValueError(
                    f"Unsupported DONE-state update for state {state.name()!r}"
                )
