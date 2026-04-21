from __future__ import annotations


from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

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
    def get_current_node_name(self) -> str:
        raise NotImplementedError      

    @abstractmethod
    def get_current_node_companion_names(self, node_name: str) -> list[str]:
        raise NotImplementedError
       
    @abstractmethod
    def get_completed_and_last_pending_nodes(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def rocover_failure(self, current_failed_node: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_forward_states_after_node(self, node_name: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def roll_back_to_state(self, state_name: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def get_working_dataset_id_and_frozen_status(self) -> tuple[UUID | None, bool]:
        raise NotImplementedError
    
    @abstractmethod
    def get_ochestration_prompt(self) -> str:
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

   
    
    
    
    
    
    
    
    
    
    
    