
from __future__ import annotations
from typing import Any
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.workflows.tool import Tool
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.tools.causal.causal_model_factory import CausalInferenceFactory
from python.implementation.workflows.tools.data_processing.data_processing_tool import DataProcessingTool
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import DatasetProfilingStateTool
from python.implementation.workflows.tools.encoding.encoding_tool import EncodingTool


class DefaultToolFactory(ToolFactory):
    _tools: dict[str, Tool]
    def __init__(self, data_repo: DataRepo, models_repo: ModelsRepo) -> None:
        self._tools = {
            "DATA_PROFILING_TOOL": DatasetProfilingStateTool(),
            "ENCODING_TOOL": EncodingTool(),
            "DATA_PROCESSING_TOOL": DataProcessingTool(),
            "CAUSAL_INFERENCE_TOOL": CausalInferenceFactory.create_default(data_repo=data_repo, models_repo=models_repo),
        }

    def get_tool_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def get_tool_info(self, name: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name!r}")
        return tool.get_tool_info()

    def get_tools_info(self) -> dict[str, str]:
        return {k: self._tools[k].get_tool_info() for k in self._tools.keys()}

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> Any:
        try:
            return self._tools[name]
        except KeyError as e:
            raise KeyError(f"Unknown tool: {name!r}") from e