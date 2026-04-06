from __future__ import annotations


from abc import ABC, abstractmethod
from typing import Any


class ReadOnlyOchestratorState(ABC):
    @abstractmethod
    def get(self, key: str) -> Any:
        raise NotImplementedError

class WritableOchestratorState(ReadOnlyOchestratorState):
    @abstractmethod
    def to_json_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> WritableOchestratorState:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def init_empty(cls) -> WritableOchestratorState:
        raise NotImplementedError

   
    
    
    
    
    
    
    
    
    
    
    