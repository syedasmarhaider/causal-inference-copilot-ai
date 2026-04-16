from __future__ import annotations


from abc import ABC, abstractmethod
from typing import Any


class OchestratorState(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
   
    @abstractmethod
    def get(self, key: str) -> Any:
     raise NotImplementedError
    
    @abstractmethod
    def set(self, key: str, value: dict[str, Any]) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def to_json_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> OchestratorState:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def init_empty(cls) -> OchestratorState:
        raise NotImplementedError

   
    
    
    
    
    
    
    
    
    
    
    