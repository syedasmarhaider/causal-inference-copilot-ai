from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class StateDep(ABC):
    @classmethod
    @abstractmethod
    def from_loaded(cls, loaded: Mapping[str, Any]) -> "StateDep":
        """
        loaded: { state_name: payload_or_state_or_other }
        Each deps class must validate/normalize what it needs.
        """
        raise NotImplementedError