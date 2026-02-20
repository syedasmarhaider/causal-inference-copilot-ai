from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Sequence

Status = Literal["PENDING", "DONE", "ABORTED"]
ACTION = Literal["NONE", "NEEDS_INPUT"]


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
    def message(self) -> Optional[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def error(self) -> Optional[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_action(self) -> ACTION:
        raise NotImplementedError

    @abstractmethod
    def required_states_keys(self) -> Sequence[str]:
        raise NotImplementedError
    
    @abstractmethod
    def to_json_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_json_dict(cls, payload: Dict[str, Any]) -> "State":
        raise NotImplementedError
