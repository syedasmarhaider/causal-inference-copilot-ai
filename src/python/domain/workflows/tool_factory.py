from __future__ import annotations

from abc import ABC, abstractmethod

from python.domain.workflows.tool import Tool

class ToolFactory(ABC):
    @abstractmethod
    def get_tool_names(self) -> list[str]:
        raise NotImplementedError
    
    @abstractmethod
    def get_tool_info(self, name: str) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def get_tools_info(self) -> dict[str, str]:
        raise NotImplementedError
    
    @abstractmethod
    def has_tool(self, name: str) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def get_tool(self, name: str) -> Tool:
        raise NotImplementedError 