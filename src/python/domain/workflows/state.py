from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from python.domain.models.errors import NodeExecutionError

Status = Literal["PENDING", "DONE", "ABORTED"]
ACTION = Literal["NONE", "NEEDS_INPUT","NEEDS_DATA"]


@dataclass(frozen=True, slots=True)
class StateMessage:
    txt_message: str
    action: ACTION 
    artifact_ids: Sequence[str] | None = None

class State(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def status(self) -> Status:
        raise NotImplementedError

    @property
    @abstractmethod
    def message(self) -> StateMessage:
        raise NotImplementedError

    @property
    @abstractmethod
    def error(self) -> NodeExecutionError | None:
        raise NotImplementedError

    @abstractmethod
    def pre_required_states_names(self) -> Sequence[str]:
        raise NotImplementedError
    
    @abstractmethod
    def to_json_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> State:
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def init_empty(cls) -> State:
        raise NotImplementedError
