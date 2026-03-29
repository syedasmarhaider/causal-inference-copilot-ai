from __future__ import annotations

from abc import ABC, abstractmethod


class Tool(ABC):
    @abstractmethod
    def get_tool_name(self) -> str: ...
    
    @abstractmethod
    def get_tool_info(self) -> str: ...
