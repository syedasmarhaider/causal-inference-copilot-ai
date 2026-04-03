from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from python.domain.models.errors import NodeExecutionError
from python.domain.workflows.state import State, StateMessage, Status
from python.implementation.workflows.nodes.model_train.model_train_deps import ModelTrainDeps
from python.implementation.workflows.tools.encoding.encoding_plan import TransformPlan


class ModelTrainPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    trained_model_id: UUID | None = None
    
    column_transformation_plan: TransformPlan | None = None
    training_warnings: str | None = None
    order_effect_modifiers: list[str] | None = None
    order_covariates: list[str] | None = None
    prev_training_errors: str | None = None
    no_of_times_trained: int | None = None

    # UI / node-local
    user_message: str | None = None
    needs_user_input: bool | None = None
    
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ModelTrainState(State):
    NAME: ClassVar[str] = "MODEL_TRAIN"
    payload: ModelTrainPayload
    MaxNoOfInterationTrain = 3

    # ---- required by State ABC ----
    @property
    def name(self) -> str:
        return self.NAME

    @property
    def error(self) -> NodeExecutionError | None:
        if self.payload.error is not None:
            return NodeExecutionError(state_name=self.NAME, error=self.payload.error)
        return None

    @property
    def status(self) -> Status:
        if self.error is not None:
            return "ABORTED"
        if self.payload.trained_model_id is not None:
            return "DONE"
        return "PENDING"

    @property
    def message(self) -> StateMessage:
        if self.payload.user_message is None:
            raise ValueError(
                "ModelTrainState.message is required but missing. "
                "Don't access .message outside the node/UI context where user_message is guaranteed."
            )
        action = "NONE"
        if  self.payload.needs_user_input is not None and self.payload.needs_user_input:
            action = "NEEDS_INPUT"     
        return StateMessage(txt_message=self.payload.user_message, action=action)


    def pre_required_states_names(self) -> Sequence[str]:
        # Fill this based on your pipeline, e.g. ("MODEL_SELECTION", "ENCODING", "VALIDATE_INFERENCE_READY")
        return ModelTrainDeps.pre_required_states_names()

    def to_json_dict(self) -> dict[str, Any]:
        return self.payload.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> ModelTrainState:
        model = ModelTrainPayload.model_validate(payload)
        return cls(payload=model)

    @classmethod
    def init_empty(cls) -> ModelTrainState:
        return cls(payload=ModelTrainPayload())