from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

class ToolFactory(ABC):
    @abstractmethod
    def get_tool_names(self) -> list[str]:
        raise NotImplementedError
    
    @abstractmethod
    def get_tool_info(self, name: str) -> Optional[str]:
        raise NotImplementedError
    
    @abstractmethod
    def get_tools_info(self) -> dict[str, str]:
        raise NotImplementedError
    
    @abstractmethod
    def get_tool(self, name: str) -> Optional[object]:
        raise NotImplementedError 