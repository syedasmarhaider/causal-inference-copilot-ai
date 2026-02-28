
from __future__ import annotations
from typing import Any
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import DatasetProfilingStateTool


class DefaultToolFactory(ToolFactory):
    _tools: dict[str, Any]
    _infos: dict[str, str]
    def __init__(self) -> None:
        self._tools = {
            "DATA_PROFILING_TOOL": DatasetProfilingStateTool(),
        }
        self._infos = {
            "DATA_PROFILING_TOOL": "Tool for profiling datasets",
        }

    def get_tool_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def get_tool_info(self, name: str) -> str:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}")
        return self._infos.get(name, "")

    def get_tools_info(self) -> dict[str, str]:
        return {k: self._infos.get(k, "") for k in self._tools.keys()}

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> Any:
        try:
            return self._tools[name]
        except KeyError as e:
            raise KeyError(f"Unknown tool: {name!r}") from e