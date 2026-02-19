from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

from python.domain.workflows.state import State


class StateDep(ABC):
    
    @classmethod
    @abstractmethod
    def from_loaded(cls, loaded: Mapping[str, State]) -> "StateDep":
        """
        Build a strongly-typed deps object from already-loaded states.
        Must validate types and raise if missing/invalid.
        """
        raise NotImplementedError
