from __future__ import annotations


from abc import ABC, abstractmethod
from typing import Sequence

from python.domain.service.llm_service import ChatMessage
from python.domain.workflows.state import State


class Route(ABC):
    @abstractmethod
    def get_executable_state_name(self, 
                                  current_state: State,
                                  messages_history: Sequence[ChatMessage]
                                  ) -> str:
        raise NotImplementedError