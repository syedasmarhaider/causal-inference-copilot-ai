from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.models.validation import ValidationIssueModel
from python.domain.workflows.ochestrator_state import OchestratorState
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.causal_inference.causal_inference_node import CausalInferenceNode
from python.implementation.workflows.nodes.causal_inference.causal_inference_state import CausalInferenceState
from python.implementation.workflows.nodes.data_compilation.data_compilation_node import DataCompilationNode
from python.implementation.workflows.nodes.data_compilation.data_compilation_state import DataCompilationState
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_node import DataManupulationNode
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_state import DataManupulationState
from python.implementation.workflows.nodes.data_statistics.data_statistics_node import DataStatisticsNode
from python.implementation.workflows.nodes.general_queries.general_queries_node import GeneralQueriesNode
from python.implementation.workflows.nodes.model_selection.mode_selection_state import ModelSelectionState
from python.implementation.workflows.nodes.model_selection.model_selection_node import ModelSelectionNode
from python.implementation.workflows.nodes.model_train.model_train_node import ModelTrainNode
from python.implementation.workflows.nodes.model_train.model_train_state import ModelTrainState
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_node import ProtocolDiscussionNode
from python.implementation.workflows.nodes.protocol_discussion.protocol_discussion_state import ProtocolDiscussionState
from python.implementation.workflows.ochestrator.ochestrator_prompts import ROUTE_SYSTEM_PROMPT_CAUSAL
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.causal.specs.causal_spec_draft import CausalSpecDraft
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel

log = get_logger(__name__)


class GlobalStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # stage 1 — dataset
    working_dataset_ids: list[UUID] = Field(default_factory=list)
    latest_dataset_summary: DatasetSummaryModel | None = None

    # stage 2 — protocol discussion + draft
    protocol_discussion: str | None = None
    protocol_cleaning_instructions: str | None = None
    causal_spec_draft: CausalSpecDraft | None = None

    # stage 3 — compiled spec + transform plan + validation
    causal_spec: CausalSpec | None = None
    data_transformation_plan: TransformPlan | None = None
    working_dataset_frozen: bool = False
    validation_issues: list[ValidationIssueModel] = Field(default_factory=list)
    is_validated: bool = False

    # stage 4 — model selection
    selected_model: str | None = None
    selection_reasoning: str | None = None

    # stage 5 — training
    trained_model_id: UUID | None = None
    training_warnings: list[str] = Field(default_factory=list)

    @field_validator("working_dataset_ids", mode="before")
    @classmethod
    def _parse_ids(cls, v: Any) -> list[UUID]:
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [item if isinstance(item, UUID) else UUID(str(item)) for item in v]
        raise ValueError(f"working_dataset_ids must be a list, got {type(v).__name__}")

    @field_validator("trained_model_id", mode="before")
    @classmethod
    def _parse_trained_model_id(cls, v: Any) -> UUID | None:
        if v is None:
            return None
        return v if isinstance(v, UUID) else UUID(str(v))


class CausalOchestratorState(OchestratorState):
    INIT_DATA_ID = UUID("00000000-0000-0000-0000-000000000000")
    NAME: ClassVar[str] = "CAUSAL_OCHESTRATOR_STATE"
    

    _STAGE_FIELDS: ClassVar[dict[int, tuple[str, ...]]] = {
        1: ("working_dataset_ids", "latest_dataset_summary"),
        2: (
            "protocol_discussion",
            "protocol_cleaning_instructions",
            "causal_spec_draft",
        ),
        3: (
            "causal_spec",
            "data_transformation_plan",
            "working_dataset_frozen",
            "validation_issues",
            "is_validated",
        ),
        4: ("selected_model", "selection_reasoning"),
        5: ("trained_model_id", "training_warnings"),
    }
    _MAX_STAGE: ClassVar[int] = 5
    _BOOL_FIELDS: ClassVar[frozenset[str]] = frozenset({"working_dataset_frozen", "is_validated"})
    _LIST_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"working_dataset_ids", "validation_issues", "training_warnings"}
    )

    def __init__(self, model: GlobalStateModel) -> None:
        self._model = model

    def name(self) -> str:
        return self.NAME

    def to_json_dict(self) -> dict[str, Any]:
        return self._model.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> CausalOchestratorState:
        return cls(GlobalStateModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> CausalOchestratorState:
        global_model = GlobalStateModel()
        global_model.working_dataset_ids = [cls.INIT_DATA_ID]
        return cls(global_model)

    def get(self, key: str) -> Any:
        if key == "working_dataset_id":
            return self._model.working_dataset_ids[-1]
        if key not in GlobalStateModel.model_fields:
            raise KeyError(f"Unknown global state key: {key!r}")
        return deepcopy(getattr(self._model, key))
    
    def get_working_dataset_id_and_frozen_status(self) -> tuple[UUID | None, bool]:
        dataset_id = self._model.working_dataset_ids[-1] if self._model.working_dataset_ids else None
        return dataset_id, self._model.working_dataset_frozen

    def set(self, key: str, value: dict[str, Any]) -> None:  # noqa: C901
        match key:
            case _ if key == DataManupulationState.NAME:
                self._set_data_manipulation(value)
            case _ if key == ProtocolDiscussionState.NAME:
                self._set_protocol_discussion(value)
            case _ if key == DataCompilationState.NAME:
                self._set_data_compilation(value)
            case _ if key == ModelSelectionState.NAME:
                self._set_model_selection(value)
            case _ if key == ModelTrainState.NAME:
                self._set_model_train(value)
            case _:
                raise KeyError(f"Unknown node name for set: {key!r}")

    def _set_data_manipulation(self, value: dict[str, Any]) -> None:
        if "working_dataset_id" not in value or "latest_dataset_summary" not in value:
            raise KeyError(
                "DATA_MANUPULATION updates must include working_dataset_id and latest_dataset_summary"
            )

        dataset_id = self._parse_dataset_id(value["working_dataset_id"], field_name="working_dataset_id")
        latest_dataset_summary = self._parse_dataset_summary(value["latest_dataset_summary"])
        next_ids = list(self._model.working_dataset_ids)
        revert_requested = value.get("revert_request") is True or value.get("_revert_request") is True
        if revert_requested:
            if not next_ids:
                raise ValueError("No working dataset to revert from")
            if len(next_ids) == 1:
                raise ValueError("Cannot revert from initial dataset")
            if next_ids[-2] != dataset_id:
                raise ValueError(f"Dataset ID mismatch: expected {next_ids[-2]}, got {dataset_id}")
            next_ids = next_ids[:-1]

        changed = self._set_stage1_fields(
            working_dataset_ids=self._append_dataset_id_if_needed(next_ids, dataset_id),
            latest_dataset_summary=latest_dataset_summary,
        )
        if changed:
            self._clear_stages_from(2, reason="stage-1 dataset updated")

    def _set_protocol_discussion(self, value: dict[str, Any]) -> None:
        if "protocol_discussion" not in value:
            raise KeyError("PROTOCOL_DISCUSSION updates must include protocol_discussion")

        self._require_stage1()

        protocol_discussion = self._parse_optional_text(value["protocol_discussion"])
        if "protocol_cleaning_instructions" in value:
            protocol_cleaning_instructions = self._parse_optional_text(
                value["protocol_cleaning_instructions"]
            )
        else:
            protocol_cleaning_instructions = self._model.protocol_cleaning_instructions

        protocol_changed = (
            self._model.protocol_discussion != protocol_discussion
            or self._model.protocol_cleaning_instructions != protocol_cleaning_instructions
        )
        next_causal_spec_draft = (
            self._parse_causal_spec_draft(value["causal_spec_draft"])
            if "causal_spec_draft" in value
            else self._model.causal_spec_draft
        )

        if protocol_changed:
            self._model.protocol_discussion = protocol_discussion
            self._model.protocol_cleaning_instructions = protocol_cleaning_instructions
            self._model.causal_spec_draft = (
                self._parse_causal_spec_draft(value["causal_spec_draft"])
                if "causal_spec_draft" in value
                else None
            )
            self._clear_stages_from(3, reason="stage-2 protocol discussion updated")
            return

        if self._model.causal_spec_draft == next_causal_spec_draft:
            return

        self._model.causal_spec_draft = next_causal_spec_draft
        self._clear_stages_from(3, reason="stage-2 causal spec draft updated")

    def _set_data_compilation(self, value: dict[str, Any]) -> None:
        if (
            "working_dataset_id" not in value
            or "latest_dataset_summary" not in value
            or "causal_spec_draft" not in value
        ):
            raise KeyError(
                "DATA_COMPILATION updates must include working_dataset_id, "
                "latest_dataset_summary and causal_spec_draft"
            )

        self._require_stage2()

        dataset_id = self._parse_dataset_id(value["working_dataset_id"], field_name="working_dataset_id")
        latest_dataset_summary = self._parse_dataset_summary(value["latest_dataset_summary"])
        causal_spec_draft = self._parse_causal_spec_draft(value["causal_spec_draft"])
        self._require_matching_causal_spec_draft(causal_spec_draft)

        self._set_stage1_fields(
            working_dataset_ids=self._append_dataset_id_if_needed(self._model.working_dataset_ids, dataset_id),
            latest_dataset_summary=latest_dataset_summary,
        )
        self._model.causal_spec_draft = causal_spec_draft

        has_stage3_payload = any(field in value for field in self._STAGE_FIELDS[3])
        if not has_stage3_payload:
            self._clear_stages_from(3, reason="stage-3 dataset refreshed")
            return

        self._model.causal_spec = (
            self._parse_causal_spec(value["causal_spec"])
            if "causal_spec" in value
            else self._model.causal_spec
        )
        self._model.data_transformation_plan = (
            self._parse_transform_plan(value["data_transformation_plan"])
            if "data_transformation_plan" in value
            else self._model.data_transformation_plan
        )
        self._model.working_dataset_frozen = (
            self._parse_bool(value["working_dataset_frozen"], field_name="working_dataset_frozen")
            if "working_dataset_frozen" in value
            else self._model.working_dataset_frozen
        )
        self._model.validation_issues = (
            self._parse_validation_issues(value["validation_issues"])
            if "validation_issues" in value
            else []
        )
        self._model.is_validated = (
            self._parse_bool(value["is_validated"], field_name="is_validated")
            if "is_validated" in value
            else False
        )
        self._clear_stages_from(4, reason="stage-3 compilation updated")

    def _set_model_selection(self, value: dict[str, Any]) -> None:
        if "selected_model" not in value or "selection_reasoning" not in value:
            raise KeyError(
                "MODEL_SELECTION updates must include selected_model and selection_reasoning"
            )

        self._require_stage3()

        selected_model = self._parse_optional_text(value["selected_model"])
        selection_reasoning = self._parse_optional_text(value["selection_reasoning"])
        changed = (
            self._model.selected_model != selected_model
            or self._model.selection_reasoning != selection_reasoning
        )
        self._model.selected_model = selected_model
        self._model.selection_reasoning = selection_reasoning
        if changed:
            self._clear_stages_from(5, reason="stage-4 model selection updated")

    def _set_model_train(self, value: dict[str, Any]) -> None:
        if "trained_model_id" not in value or "training_warnings" not in value:
            raise KeyError("MODEL_TRAIN updates must include trained_model_id and training_warnings")

        self._require_stage4()

        self._model.trained_model_id = self._parse_optional_uuid(
            value["trained_model_id"], field_name="trained_model_id"
        )
        self._model.training_warnings = self._parse_string_list(
            value["training_warnings"], field_name="training_warnings"
        )

    def get_current_node_name(self) -> str:
        if not self._is_stage1_complete():
            return DataManupulationNode.NAME
        if not self._is_stage2_complete():
            return ProtocolDiscussionNode.NAME
        if not self._is_stage3_complete():
            return DataCompilationNode.NAME
        if not self._is_stage4_complete():
            return ModelSelectionNode.NAME
        if not self._is_stage5_complete():
            return ModelTrainNode.NAME
        return CausalInferenceNode.NAME

    def get_current_node_companion_names(self, node_name: str) -> list[str]:
        match node_name:
            case _ if node_name == DataManupulationNode.NAME:
                if not self._model.working_dataset_ids or self._model.latest_dataset_summary is None:
                    return []
                return [ProtocolDiscussionNode.NAME, DataStatisticsNode.NAME,GeneralQueriesNode.NAME]
            case _ if node_name == ProtocolDiscussionNode.NAME:
                return [DataManupulationNode.NAME, DataStatisticsNode.NAME,GeneralQueriesNode.NAME]
            case _ if node_name == DataCompilationNode.NAME:
                return [DataStatisticsNode.NAME,GeneralQueriesNode.NAME]
            case _ if node_name == ModelSelectionNode.NAME:
                return [DataStatisticsNode.NAME,GeneralQueriesNode.NAME]
            case _ if node_name == ModelTrainNode.NAME:
                return []
            case _ if node_name == CausalInferenceNode.NAME:
                return [DataStatisticsNode.NAME,GeneralQueriesNode.NAME]
            case _:
                raise ValueError(f"Unknown node name for companions: {node_name!r}")
    
    def get_completed_and_last_pending_nodes(self) -> list[str]:
        answer_arr: list[str] = []
        if self._is_stage1_complete():
            answer_arr.append(DataManupulationNode.NAME)
        if self._is_stage2_complete():
            answer_arr.append(ProtocolDiscussionNode.NAME)
        if self._is_stage3_complete():
            answer_arr.append(DataCompilationNode.NAME)
        if self._is_stage4_complete():
            answer_arr.append(ModelSelectionNode.NAME)
        if self._is_stage5_complete():
            answer_arr.append(ModelTrainNode.NAME)
        if len(answer_arr) < self._MAX_STAGE:
            answer_arr.append(self.get_current_node_name())
        return answer_arr    

    def rocover_failure(self, current_failed_node: str) -> None:
        match current_failed_node:
            case _ if current_failed_node == DataManupulationNode.NAME:
                self._clear_stages_from(1, reason=f"rollback from {current_failed_node}")
            case _ if current_failed_node == ProtocolDiscussionNode.NAME:
                self._clear_stages_from(2, reason=f"rollback from {current_failed_node}")
            case _ if current_failed_node == DataCompilationNode.NAME:
                self._clear_stages_from(2, reason=f"rollback from {current_failed_node}")
            case _ if current_failed_node == ModelSelectionNode.NAME:
                self._clear_stages_from(2, reason=f"rollback from {current_failed_node}")
            case _ if current_failed_node == ModelTrainNode.NAME:
                self._clear_stages_from(2, reason=f"rollback from {current_failed_node}")
            case _ if current_failed_node == CausalInferenceNode.NAME:
                pass
            case _:
                raise ValueError(f"Unknown node name for rollback: {current_failed_node!r}")

    def get_forward_states_after_node(self, node_name: str) -> list[str]:
        match node_name:
            case _ if node_name == DataManupulationNode.NAME:
                return [
                    DataCompilationNode.NAME,
                    ModelSelectionNode.NAME,
                    ModelTrainNode.NAME,
                    CausalInferenceNode.NAME,
                ]
            case _ if node_name == ProtocolDiscussionNode.NAME:
                return [
                    DataCompilationNode.NAME,
                    ModelSelectionNode.NAME,
                    ModelTrainNode.NAME,
                    CausalInferenceNode.NAME,
                ]
            case _ if node_name == DataCompilationNode.NAME:
                return [ModelSelectionNode.NAME, ModelTrainNode.NAME, CausalInferenceNode.NAME]
            case _ if node_name == ModelSelectionNode.NAME:
                return [ModelTrainNode.NAME, CausalInferenceNode.NAME]
            case _ if node_name == ModelTrainNode.NAME:
                return [CausalInferenceNode.NAME]
            case _ if node_name == CausalInferenceNode.NAME:
                return []
            case _:
                raise ValueError(f"Unknown node name for forward states: {node_name!r}")

    def roll_back_to_state(self, state_name: str) -> None:
        match state_name:
            case _ if state_name == DataManupulationState.NAME:
                self._clear_stages_from(1, reason=f"rollback to {state_name}")
            case _ if state_name == ProtocolDiscussionState.NAME:
                self._clear_stages_from(2, reason=f"rollback to {state_name}")
            case _ if state_name == DataCompilationState.NAME:
                self._clear_stages_from(3, reason=f"rollback to {state_name}")
            case _ if state_name == ModelSelectionState.NAME:
                self._clear_stages_from(4, reason=f"rollback to {state_name}")
            case _ if state_name == ModelTrainState.NAME:
                self._clear_stages_from(5, reason=f"rollback to {state_name}")
            case _ if state_name == CausalInferenceState.NAME:
                pass
            case _:
                raise ValueError(f"Unknown state name for rollback: {state_name!r}")
    
    def get_ochestration_prompt(self) -> str:
        return ROUTE_SYSTEM_PROMPT_CAUSAL     

    def _is_stage1_complete(self) -> bool:
        return bool(self._model.working_dataset_ids) and self._model.latest_dataset_summary is not None

    def _is_stage2_complete(self) -> bool:
        return self._is_stage1_complete() and (
            self._model.protocol_discussion is not None
            and self._model.causal_spec_draft is not None
        )

    def _is_stage3_complete(self) -> bool:
        return self._is_stage2_complete() and (
            self._model.causal_spec is not None
            and self._model.data_transformation_plan is not None
            and self._model.working_dataset_frozen is True
            and self._model.is_validated is True
        )

    def _is_stage4_complete(self) -> bool:
        return self._is_stage3_complete() and (
            self._model.selected_model is not None
            and self._model.selection_reasoning is not None
        )

    def _is_stage5_complete(self) -> bool:
        return self._is_stage4_complete() and self._model.trained_model_id is not None

    def _require_stage1(self) -> None:
        if not self._is_stage1_complete():
            raise ValueError("Stage 1 incomplete: dataset state not ready")

    def _require_stage2(self) -> None:
        if not self._is_stage2_complete():
            raise ValueError("Stage 2 incomplete: protocol discussion or causal_spec_draft not set")

    def _require_stage3(self) -> None:
        if not self._is_stage3_complete():
            raise ValueError("Stage 3 incomplete: compiled outputs are not ready")

    def _require_stage4(self) -> None:
        if not self._is_stage4_complete():
            raise ValueError("Stage 4 incomplete: model selection not set")

    def _set_stage1_fields(
        self,
        *,
        working_dataset_ids: list[UUID],
        latest_dataset_summary: DatasetSummaryModel | None,
    ) -> bool:
        changed = (
            self._model.working_dataset_ids != working_dataset_ids
            or self._model.latest_dataset_summary != latest_dataset_summary
        )
        self._model.working_dataset_ids = working_dataset_ids
        self._model.latest_dataset_summary = latest_dataset_summary
        return changed

    @staticmethod
    def _append_dataset_id_if_needed(existing_ids: list[UUID], dataset_id: UUID) -> list[UUID]:
        next_ids = list(existing_ids)
        if next_ids and next_ids[-1] == dataset_id:
            return next_ids
        next_ids.append(dataset_id)
        return next_ids

    @staticmethod
    def _parse_dataset_id(raw: Any, *, field_name: str) -> UUID:
        try:
            return raw if isinstance(raw, UUID) else UUID(str(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a UUID-compatible value") from exc

    @staticmethod
    def _parse_optional_uuid(raw: Any, *, field_name: str) -> UUID | None:
        if raw is None:
            return None
        try:
            return raw if isinstance(raw, UUID) else UUID(str(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a UUID-compatible value") from exc

    @staticmethod
    def _parse_dataset_summary(raw: Any) -> DatasetSummaryModel | None:
        if raw is None:
            return None
        return raw if isinstance(raw, DatasetSummaryModel) else DatasetSummaryModel.model_validate(raw)

    @staticmethod
    def _parse_optional_text(raw: Any) -> str | None:
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    @staticmethod
    def _parse_causal_spec_draft(raw: Any) -> CausalSpecDraft | None:
        if raw is None:
            return None
        return raw if isinstance(raw, CausalSpecDraft) else CausalSpecDraft.model_validate(raw)

    def _require_matching_causal_spec_draft(self, draft: CausalSpecDraft | None) -> None:
        current_draft = self._model.causal_spec_draft
        if current_draft is None:
            raise ValueError("Stage 2 incomplete: causal_spec_draft not set")
        if draft is None:
            raise ValueError("DATA_COMPILATION causal_spec_draft must not be None")
        if current_draft.model_dump(mode="json") != draft.model_dump(mode="json"):
            raise ValueError(
                "DATA_COMPILATION causal_spec_draft must match the current stored causal_spec_draft"
            )

    @staticmethod
    def _parse_causal_spec(raw: Any) -> CausalSpec | None:
        if raw is None:
            return None
        return raw if isinstance(raw, CausalSpec) else CausalSpec.model_validate(raw)

    @staticmethod
    def _parse_transform_plan(raw: Any) -> TransformPlan | None:
        if raw is None:
            return None
        return raw if isinstance(raw, TransformPlan) else TransformPlan.model_validate(raw)

    @staticmethod
    def _parse_validation_issues(raw: Any) -> list[ValidationIssueModel]:
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise TypeError("validation_issues must be a list")
        return [
            issue
            if isinstance(issue, ValidationIssueModel)
            else ValidationIssueModel.model_validate(issue)
            for issue in raw
        ]

    @staticmethod
    def _parse_string_list(raw: Any, *, field_name: str) -> list[str]:
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise TypeError(f"{field_name} must be a list")
        return [str(item) for item in raw]

    @staticmethod
    def _parse_bool(raw: Any, *, field_name: str) -> bool:
        if not isinstance(raw, bool):
            raise TypeError(f"{field_name} must be a bool")
        return raw

    def _clear_stages_from(self, stage: int, *, reason: str) -> None:
        cleared: list[str] = []
        for current_stage in range(stage, self._MAX_STAGE + 1):
            for field_name in self._STAGE_FIELDS[current_stage]:
                if self._reset_field(field_name):
                    cleared.append(field_name)
        if cleared:
            log.warning("%s; cleared: %s", reason, ", ".join(cleared))

    def _reset_field(self, field_name: str) -> bool:
        default = self._default(field_name)
        current = getattr(self._model, field_name)
        if current == default:
            return False
        setattr(self._model, field_name, default)
        return True

    @classmethod
    def _default(cls, field_name: str) -> Any:
        if field_name in cls._BOOL_FIELDS:
            return False
        if field_name in cls._LIST_FIELDS:
            return []
        return None
