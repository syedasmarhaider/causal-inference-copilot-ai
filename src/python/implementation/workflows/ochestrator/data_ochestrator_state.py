from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from python.domain.workflows.ochestrator_state import OchestratorState
from python.implementation.service.logging.default_logging import get_logger
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_node import (
    DataManupulationNode,
)
from python.implementation.workflows.nodes.data_manupulation.data_manupulation_state import (
    DataManupulationState,
)
from python.implementation.workflows.nodes.data_statistics.data_statistics_node import (
    DataStatisticsNode,
)
from python.implementation.workflows.ochestrator.ochestrator_prompts import ROUTE_SYSTEM_PROMPT_DATA
from python.implementation.workflows.tools.common.model.data_summary import DatasetSummaryModel

log = get_logger(__name__)


class GlobalDataStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    update_counter: int = Field(default=0, ge=0)
    working_dataset_ids: list[UUID] = Field(default_factory=list)
    latest_dataset_summary: DatasetSummaryModel | None = None

    @field_validator("update_counter", mode="before")
    @classmethod
    def _parse_update_counter(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("update_counter must be a non-negative integer")
        if value < 0:
            raise ValueError("update_counter must be a non-negative integer")
        return value

    @field_validator("working_dataset_ids", mode="before")
    @classmethod
    def _parse_ids(cls, value: Any) -> list[UUID]:
        if value is None:
            return []

        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"working_dataset_ids must be a list or tuple, got {type(value).__name__}"
            )

        parsed_ids: list[UUID] = []
        for item in value:
            try:
                parsed_ids.append(item if isinstance(item, UUID) else UUID(str(item)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid UUID in working_dataset_ids: {item!r}") from exc

        return parsed_ids


class DataOchestratorState(OchestratorState):
    INIT_DATA_ID = UUID("00000000-0000-0000-0000-000000000000")
    NAME: ClassVar[str] = "DATA_OCHESTRATOR_STATE"

    def __init__(self, model: GlobalDataStateModel) -> None:
        self._model = model

    def name(self) -> str:
        return self.NAME

    def get_update_counter(self) -> int:
        return self._model.update_counter

    def set_update_counter(self, value: int) -> None:
        self._model.update_counter = value

    def to_json_dict(self) -> dict[str, Any]:
        return self._model.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DataOchestratorState:
        return cls(GlobalDataStateModel.model_validate(payload))

    @classmethod
    def init_empty(cls) -> DataOchestratorState:
        return cls(
            GlobalDataStateModel(
                working_dataset_ids=[cls.INIT_DATA_ID],
                latest_dataset_summary=None,
            )
        )

    def get(self, key: str) -> Any:
        if key == "working_dataset_id":
            return self.get_working_dataset_id()
        
        if key == "working_dataset_frozen":
              return False

        if key not in GlobalDataStateModel.model_fields:
            raise KeyError(f"Unknown global state key: {key!r}")

        return deepcopy(getattr(self._model, key))

    def get_working_dataset_id(self) -> UUID | None:
        if not self._model.working_dataset_ids:
            return None
        return self._model.working_dataset_ids[-1]

    def set(self, key: str, value: dict[str, Any]) -> None:
        if key != DataManupulationState.NAME:
            raise KeyError(
                f"Only {DataManupulationState.NAME!r} updates are supported, got {key!r}"
            )
        self._set_data_manipulation(value)

    def _set_data_manipulation(self, value: dict[str, Any]) -> None:
        if "working_dataset_id" not in value or "latest_dataset_summary" not in value:
            raise KeyError(
                "DATA_MANUPULATION updates must include working_dataset_id and latest_dataset_summary"
            )

        dataset_id = self._parse_dataset_id(
            value["working_dataset_id"],
            field_name="working_dataset_id",
        )
        latest_dataset_summary = self._parse_dataset_summary(value["latest_dataset_summary"])

        current_ids = list(self._model.working_dataset_ids)
        revert_requested = value.get("revert_request") is True or value.get("_revert_request") is True

        if revert_requested:
            if not current_ids:
                raise ValueError("No working dataset exists to revert from")

            if len(current_ids) == 1:
                raise ValueError("Cannot revert: no previous dataset version exists")

            expected_previous_id = current_ids[-2]
            if dataset_id != expected_previous_id:
                raise ValueError(
                    f"Dataset ID mismatch during revert: expected {expected_previous_id}, got {dataset_id}"
                )

            next_ids = current_ids[:-1]
        else:
            next_ids = self._append_dataset_id_if_needed(current_ids, dataset_id)

        changed = (
            self._model.working_dataset_ids != next_ids
            or self._model.latest_dataset_summary != latest_dataset_summary
        )

        self._model.working_dataset_ids = next_ids
        self._model.latest_dataset_summary = latest_dataset_summary

        if changed:
            log.info(
                "Dataset state updated: latest_dataset_id=%s, version_count=%d",
                self.get_working_dataset_id(),
                len(self._model.working_dataset_ids),
            )

    def get_current_node_name(self) -> str:
        return DataManupulationNode.NAME
    
    def get_ochestration_prompt(self) -> str:
        return ROUTE_SYSTEM_PROMPT_DATA

    def get_current_node_companion_names(self, node_name: str) -> list[str]:
        if node_name != DataManupulationNode.NAME:
            raise ValueError(f"Unknown node name for companions: {node_name!r}")

        if self._model.latest_dataset_summary is None:
            return []

        return [DataStatisticsNode.NAME]

    def get_completed_and_last_pending_nodes(self) -> list[str]:
        return [DataManupulationNode.NAME]
    
    
    def get_working_dataset_id_and_frozen_status(self) -> tuple[UUID | None, bool]:
        dataset_id = self._model.working_dataset_ids[-1] if self._model.working_dataset_ids else None
        return dataset_id, False

    def rocover_failure(self, current_failed_node: str) -> None:
        if current_failed_node != DataManupulationNode.NAME:
            raise ValueError(f"Unknown node name for rollback: {current_failed_node!r}")

        self._clear_stage(reason=f"rollback from {current_failed_node}")

    def get_forward_states_after_node(self, node_name: str) -> list[str]:
        if node_name != DataManupulationNode.NAME:
            raise ValueError(f"Unknown node name for forward states: {node_name!r}")
        return []

    def roll_back_to_state(self, state_name: str) -> None:
        if state_name != DataManupulationState.NAME:
            raise ValueError(f"Unknown state name for rollback: {state_name!r}")

        self._clear_stage(reason=f"rollback to {state_name}")

    def is_complete(self) -> bool:
        return (
            bool(self._model.working_dataset_ids)
            and self._model.latest_dataset_summary is not None
        )

    def _clear_stage(self, *, reason: str) -> None:
        changed = False

        if self._model.working_dataset_ids != [self.INIT_DATA_ID]:
            self._model.working_dataset_ids = [self.INIT_DATA_ID]
            changed = True

        if self._model.latest_dataset_summary is not None:
            self._model.latest_dataset_summary = None
            changed = True

        if changed:
            log.warning("%s; data stage cleared", reason)

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
    def _parse_dataset_summary(raw: Any) -> DatasetSummaryModel | None:
        if raw is None:
            return None
        return (
            raw
            if isinstance(raw, DatasetSummaryModel)
            else DatasetSummaryModel.model_validate(raw)
        )
