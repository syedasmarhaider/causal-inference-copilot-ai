from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class NodeState(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def clear_state(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def to_json_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> NodeState:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def init_empty(cls) -> NodeState:
        raise NotImplementedError
