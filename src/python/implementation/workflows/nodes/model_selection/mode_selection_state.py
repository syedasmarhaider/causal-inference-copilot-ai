from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from python.domain.workflows.state import State
from pydantic import BaseModel, ConfigDict


class ModelSelectionStatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    selected_model: Optional[str] = None
    user_message: Optional[str] = None

@dataclass(frozen=True)
class ModelSelectionState(State):
    NAME = "MODEL_SELECTION"
    payload: ModelSelectionStatePayload