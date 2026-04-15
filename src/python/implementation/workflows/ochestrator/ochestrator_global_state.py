from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.models.validation import ValidationIssueModel
from python.domain.workflows.ochestrator_state import OchestratorState
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.tools.causal.encoding.encoding_plan import TransformPlan
from python.implementation.workflows.tools.causal.specs.causal_spec import CausalSpec
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Node name constants — avoids circular imports with node modules
# ---------------------------------------------------------------------------

_NODE_DATA_STATISTICS: str = "DATA_STATISTICS"
_NODE_DATA_MANUPULATION: str = "DATA_MANUPULATION"
_NODE_PROTOCOL_DISCUSSION: str = "PROTOCOL_DISCUSSION"
_NODE_DATA_COMPILATION: str = "DATA_COMPILATION"
_NODE_DATA_VALIDATION: str = "DATA_VALIDATION"
_NODE_MODEL_SELECTION: str = "MODEL_SELECTION"
_NODE_MODEL_TRAIN: str = "MODEL_TRAIN"
_NODE_CAUSAL_INFERENCE: str = "CAUSAL_INFERENCE"
_NODE_NOOP_DONE: str = "NOOP_DONE"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class GlobalStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # stage 1 — dataset
    working_dataset_ids: list[UUID] = Field(default_factory=list)
    latest_dataset_summary: DatasetSummaryModel | None = None

    # stage 2 — protocol discussion
    protocol_discussion: str | None = None

    # stage 3 — data cleaning
    data_cleaned: bool = False

    # stage 4 — causal spec + dataset freeze
    causal_spec: CausalSpec | None = None
    data_transformation_plan: TransformPlan | None = None
    working_dataset_frozen: bool = False

    # stage 5 — validation + model selection
    validation_issues: list[ValidationIssueModel] = Field(default_factory=list)
    selected_model: str | None = None
    selection_reasoning: str | None = None

    # stage 6 — model training
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


# ---------------------------------------------------------------------------
# Orchestrator state
# ---------------------------------------------------------------------------


class OchestratorWritableGlobalState(OchestratorState):
    _WORKFLOW_ORDER: ClassVar[list[str]] = [
        "working_dataset_ids",
        "latest_dataset_summary",
        "protocol_discussion",
        "data_cleaned",
        "causal_spec",
        "data_transformation_plan",
        "working_dataset_frozen",
        "validation_issues",
        "selected_model",
        "selection_reasoning",
        "trained_model_id",
        "training_warnings",
    ]

    _BOOL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"data_cleaned", "working_dataset_frozen"}
    )
    _LIST_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"working_dataset_ids", "validation_issues", "training_warnings"}
    )

    def __init__(self, model: GlobalStateModel) -> None:
        self._model = model

    # -------------------------------------------------------------------------
    # OchestratorState ABC
    # -------------------------------------------------------------------------

    def name(self) -> str:
        return "OCHESTRATOR_STATE"

    def to_json_dict(self) -> dict[str, Any]:
        return self._model.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> OchestratorWritableGlobalState:
        normalized = dict(payload)
        # migrate legacy field name
        if "working_dataset_id" in normalized and "working_dataset_ids" not in normalized:
            old = normalized.pop("working_dataset_id")
            normalized["working_dataset_ids"] = [old] if old is not None else []
        # migrate legacy summary field name
        if "working_dataset_summary" in normalized and "latest_dataset_summary" not in normalized:
            normalized["latest_dataset_summary"] = normalized.pop("working_dataset_summary")
        # migrate legacy training_id field
        if "model_training_id" in normalized and "trained_model_id" not in normalized:
            normalized["trained_model_id"] = normalized.pop("model_training_id")
        # migrate legacy data_cleaning_pending flag
        legacy_pending = normalized.pop("dataset_cleaning_pending", None)
        if "data_cleaned" not in normalized and legacy_pending is not None:
            normalized["data_cleaned"] = not bool(legacy_pending)
        return cls(GlobalStateModel.model_validate(normalized))

    @classmethod
    def init_empty(cls) -> OchestratorWritableGlobalState:
        return cls(GlobalStateModel())

    # -------------------------------------------------------------------------
    # Public get / set (generic key-value interface)
    # -------------------------------------------------------------------------

    def get(self, key: str) -> Any:
        if key not in GlobalStateModel.model_fields:
            raise KeyError(f"Unknown global state key: {key!r}")
        return deepcopy(getattr(self._model, key))

    def set(self, key: str, value: dict[str, Any]) -> None:  # noqa: C901
        """key = node name; value = dict of field updates that node is allowed to write.
        Each node owns specific global state fields and dispatches to the
        appropriate private stage setter so cascading invalidation fires."""
        match key:
            case _ if key == _NODE_DATA_MANUPULATION:
                if "working_dataset_ids" in value or "latest_dataset_summary" in value:
                    protocol_discusson = self._model.protocol_discussion if self._model.protocol_discussion else None
                    self._set_stage1(
                        working_dataset_ids=self._parse_dataset_ids(
                            value.get("working_dataset_ids", self._model.working_dataset_ids)
                        ),
                        latest_dataset_summary=self._parse_dataset_summary(
                            value.get("latest_dataset_summary", self._model.latest_dataset_summary)
                        ),
                    )
                    if protocol_discusson is not None:
                            self._set_stage2(protocol_discussion=protocol_discusson)    
                if "data_cleaned" in value:
                    self._set_stage3(data_cleaned=bool(value["data_cleaned"]))

            case _ if key == _NODE_PROTOCOL_DISCUSSION:
                raw_text = value.get("protocol_discussion")
                self._set_stage2(
                    protocol_discussion=str(raw_text).strip() if raw_text is not None else None
                )

            case _ if key == _NODE_DATA_COMPILATION:
                raw_spec = value.get("causal_spec", self._model.causal_spec)
                raw_plan = value.get("data_transformation_plan", self._model.data_transformation_plan)
                self._set_stage4(
                    causal_spec=(
                        raw_spec if isinstance(raw_spec, CausalSpec) or raw_spec is None
                        else CausalSpec.model_validate(raw_spec)
                    ),
                    data_transformation_plan=(
                        raw_plan if isinstance(raw_plan, TransformPlan) or raw_plan is None
                        else TransformPlan.model_validate(raw_plan)
                    ),
                    working_dataset_frozen=True,
                )

            case _ if key == _NODE_DATA_VALIDATION:
                raw_issues = cast(list[Any], value.get("validation_issues", self._model.validation_issues))
                issues = [
                    v if isinstance(v, ValidationIssueModel) else ValidationIssueModel.model_validate(v)
                    for v in raw_issues
                ]
                self._set_stage5(
                    validation_issues=issues,
                )

            case _ if key == _NODE_MODEL_SELECTION:
                raw_model = value.get("selected_model", self._model.selected_model)
                raw_reasoning = value.get("selection_reasoning", self._model.selection_reasoning)
                self._set_stage5(
                    validation_issues=self._model.validation_issues,
                    selected_model=str(raw_model).strip() if raw_model else None,
                    selection_reasoning=str(raw_reasoning).strip() if raw_reasoning else None,
                )

            case _ if key == _NODE_MODEL_TRAIN:
                raw_tid = value.get("trained_model_id", self._model.trained_model_id)
                tid = raw_tid if isinstance(raw_tid, UUID) or raw_tid is None else UUID(str(raw_tid))
                raw_warnings = cast(list[Any], value.get("training_warnings", self._model.training_warnings))
                self._set_stage6(
                    trained_model_id=tid,
                    training_warnings=[str(w) for w in raw_warnings],
                )

            case _ if key in {_NODE_CAUSAL_INFERENCE, _NODE_NOOP_DONE}:
                pass  # terminal nodes — no global state writes

            case _:
                raise KeyError(f"Unknown node name for set: {key!r}")

    # -------------------------------------------------------------------------
    # Private stage setters — each invalidates all downstream fields
    # -------------------------------------------------------------------------

    def _set_stage1(
        self,
        *,
        working_dataset_ids: list[UUID],
        latest_dataset_summary: DatasetSummaryModel | None,
        preserve_protocol_discussion: bool = False,
    ) -> None:
        changed = (
            self._model.working_dataset_ids != working_dataset_ids
            or self._model.latest_dataset_summary != latest_dataset_summary
        )
        self._model.working_dataset_ids = working_dataset_ids
        self._model.latest_dataset_summary = latest_dataset_summary
        if changed:
              self._invalidate_downstream_of("latest_dataset_summary", reason="stage-1 dataset updated")

    def _set_stage2(self, *, protocol_discussion: str | None) -> None:
        self._require_stage1()
        if self._model.protocol_discussion == protocol_discussion:
            return
        self._model.protocol_discussion = protocol_discussion
        self._invalidate_downstream_of("protocol_discussion", reason="stage-2 protocol discussion updated")

    def _set_stage3(self, *, data_cleaned: bool) -> None:
        self._require_stage2()
        if self._model.data_cleaned == data_cleaned:
            return
        self._model.data_cleaned = data_cleaned
        self._invalidate_downstream_of("data_cleaned", reason="stage-3 data cleaning updated")

    def _set_stage4(
        self,
        *,
        causal_spec: CausalSpec | None,
        data_transformation_plan: TransformPlan | None,
        working_dataset_frozen: bool,
    ) -> None:
        self._require_stage3()
        changed = (
            self._model.causal_spec != causal_spec
            or self._model.data_transformation_plan != data_transformation_plan
            or self._model.working_dataset_frozen != working_dataset_frozen
        )
        self._model.causal_spec = causal_spec
        self._model.data_transformation_plan = data_transformation_plan
        self._model.working_dataset_frozen = working_dataset_frozen
        if changed:
            self._invalidate_downstream_of("working_dataset_frozen", reason="stage-4 causal config updated")

    def _set_stage5(
        self,
        *,
        validation_issues: list[ValidationIssueModel],
    ) -> None:
        self._require_stage4()
        changed = (
            self._model.validation_issues != validation_issues
            or self._model.selected_model != selected_model
            or self._model.selection_reasoning != selection_reasoning
        )
        self._model.validation_issues = validation_issues
        self._model.selected_model = selected_model
        self._model.selection_reasoning = selection_reasoning
        if changed:
            self._invalidate_downstream_of("selection_reasoning", reason="stage-5 model selection updated")

    def _set_stage6(
        self,
        *,
        trained_model_id: UUID | None,
        training_warnings: list[str],
    ) -> None:
        self._require_stage5()
        self._model.trained_model_id = trained_model_id
        self._model.training_warnings = training_warnings

    # -------------------------------------------------------------------------
    # needs_node_name — returns which node the workflow currently requires
    # -------------------------------------------------------------------------

    def needs_node_name(self) -> str:
        if not self._model.working_dataset_ids or self._model.latest_dataset_summary is None:
            return _NODE_DATA_STATISTICS

        if not self._model.protocol_discussion:
            return _NODE_PROTOCOL_DISCUSSION

        if not self._model.data_cleaned:
            return _NODE_DATA_MANUPULATION

        if (
            self._model.causal_spec is None
            or self._model.data_transformation_plan is None
            or not self._model.working_dataset_frozen
        ):
            return _NODE_DATA_COMPILATION

        if self._model.selected_model is None:
            return _NODE_MODEL_SELECTION

        if self._model.trained_model_id is None:
            return _NODE_MODEL_TRAIN

        return _NODE_CAUSAL_INFERENCE

    # -------------------------------------------------------------------------
    # companion — which other nodes can run alongside a given node
    # -------------------------------------------------------------------------

    def companion(self, node_name: str) -> list[str]:
        """Given a node name, return which other nodes are valid companions
        based on the current workflow stage."""
        has_dataset = bool(self._model.working_dataset_ids)
        has_protocol = bool(self._model.protocol_discussion)
        data_ready = has_dataset and has_protocol and self._model.data_cleaned
        spec_ready = (
            data_ready
            and self._model.causal_spec is not None
            and self._model.data_transformation_plan is not None
            and self._model.working_dataset_frozen
        )
        model_selected = spec_ready and self._model.selected_model is not None

        match node_name:
            case _ if node_name in {
                _NODE_DATA_STATISTICS,
                _NODE_DATA_MANUPULATION,
                _NODE_DATA_DASHBOARD,
            }:
                companions = [
                    n for n in [
                        _NODE_DATA_STATISTICS,
                        _NODE_DATA_MANUPULATION,
                        _NODE_DATA_DASHBOARD,
                    ]
                    if n != node_name
                ]
                if has_dataset:
                    companions.append(_NODE_PROTOCOL_DISCUSSION)
                return companions

            case _ if node_name == _NODE_PROTOCOL_DISCUSSION:
                return [_NODE_DATA_STATISTICS, _NODE_DATA_MANUPULATION, _NODE_DATA_DASHBOARD]

            case _ if node_name == _NODE_DATA_COMPILATION:
                return [_NODE_DATA_VALIDATION] if data_ready else []

            case _ if node_name == _NODE_DATA_VALIDATION:
                return [_NODE_DATA_COMPILATION, _NODE_MODEL_SELECTION] if spec_ready else [_NODE_DATA_COMPILATION]

            case _ if node_name == _NODE_MODEL_SELECTION:
                return [_NODE_DATA_VALIDATION] if spec_ready else []

            case _ if node_name == _NODE_MODEL_TRAIN:
                return []

            case _ if node_name == _NODE_CAUSAL_INFERENCE:
                return [_NODE_NOOP_DONE] if model_selected else []

            case _:
                return []

    # -------------------------------------------------------------------------
    # Rollback
    # -------------------------------------------------------------------------

    def rollback(self, recovery_node_name: str) -> None:
        match recovery_node_name:
            case _ if recovery_node_name in {_NODE_DATA_STATISTICS, _NODE_DATA_MANUPULATION, _NODE_DATA_DASHBOARD}:
                self._reset_from("working_dataset_ids", reason=f"rollback to {recovery_node_name}")
            case _ if recovery_node_name == _NODE_PROTOCOL_DISCUSSION:
                self._reset_from("protocol_discussion", reason="rollback to PROTOCOL_DISCUSSION")
            case _ if recovery_node_name == _NODE_DATA_COMPILATION:
                self._reset_from("causal_spec", reason="rollback to DATA_COMPILATION")
            case _ if recovery_node_name == _NODE_MODEL_SELECTION:
                self._reset_from("selected_model", reason="rollback to MODEL_SELECTION")
            case _ if recovery_node_name == _NODE_MODEL_TRAIN:
                self._model.trained_model_id = None
                self._model.training_warnings = []
            case _:
                pass

    # -------------------------------------------------------------------------
    # Guards
    # -------------------------------------------------------------------------

    def _require_stage1(self) -> None:
        if not self._model.working_dataset_ids:
            raise ValueError("Stage 1 incomplete: no working dataset")
        if self._model.latest_dataset_summary is None:
            raise ValueError("Stage 1 incomplete: no dataset summary")

    def _require_stage2(self) -> None:
        self._require_stage1()
        if not self._model.protocol_discussion:
            raise ValueError("Stage 2 incomplete: protocol discussion not set")

    def _require_stage3(self) -> None:
        self._require_stage2()
        if not self._model.data_cleaned:
            raise ValueError("Stage 3 incomplete: data not cleaned")

    def _require_stage4(self) -> None:
        self._require_stage3()
        if self._model.causal_spec is None:
            raise ValueError("Stage 4 incomplete: causal_spec not set")
        if self._model.data_transformation_plan is None:
            raise ValueError("Stage 4 incomplete: data_transformation_plan not set")
        if not self._model.working_dataset_frozen:
            raise ValueError("Stage 4 incomplete: dataset not frozen")

    def _require_stage5(self) -> None:
        self._require_stage4()
        if self._model.selected_model is None:
            raise ValueError("Stage 5 incomplete: selected_model not set")

    # -------------------------------------------------------------------------
    # Parse helpers (used by set dispatch)
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_dataset_ids(raw: Any) -> list[UUID]:
        if raw is None:
            return []
        items = cast(list[Any], raw)
        return [v if isinstance(v, UUID) else UUID(str(v)) for v in items]

    @staticmethod
    def _parse_dataset_summary(raw: Any) -> DatasetSummaryModel | None:
        if raw is None:
            return None
        return raw if isinstance(raw, DatasetSummaryModel) else DatasetSummaryModel.model_validate(raw)

    # -------------------------------------------------------------------------
    # Cascade invalidation helpers
    # -------------------------------------------------------------------------

    def _invalidate_downstream_of(self, field_name: str, *, reason: str) -> None:
        idx = self._WORKFLOW_ORDER.index(field_name)
        cleared: list[str] = []
        for field in self._WORKFLOW_ORDER[idx + 1:]:
            if self._reset_field(field):
                cleared.append(field)
        if cleared:
            log.warning("%s; cleared downstream: %s", reason, ", ".join(cleared))

    def _reset_from(self, field_name: str, *, reason: str) -> None:
        idx = self._WORKFLOW_ORDER.index(field_name)
        cleared: list[str] = []
        for field in self._WORKFLOW_ORDER[idx:]:
            if self._reset_field(field):
                cleared.append(field)
        if cleared:
            log.warning("%s; cleared: %s", reason, ", ".join(cleared))

    def _reset_field(self, field_name: str) -> bool:
        default = self._default(field_name)
        current = getattr(self._model, field_name)
        if current == default:
            return False
        setattr(self._model, field_name, default)
        return True

    @staticmethod
    def _default(field_name: str) -> Any:
        if field_name in {"data_cleaned", "working_dataset_frozen"}:
            return False
        if field_name in {"working_dataset_ids", "validation_issues", "training_warnings"}:
            return []
        return None


__all__ = ["GlobalStateModel", "OchestratorWritableGlobalState"]
