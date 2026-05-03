from __future__ import annotations

from typing import Any

from python.domain.repo.analytics_repo import AnalyticsRepo
from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.service.llm_service import LLMService
from python.domain.workflows.tool import Tool
from python.domain.workflows.tool_factory import ToolFactory
from python.implementation.workflows.tools.advanced_analytics.advanced_analytics_tool import (
    AdvancedAnalyticsTool,
)
from python.implementation.workflows.tools.causal.encoding.encoding_plan_tool import (
    EncodingPlanTool,
)
from python.implementation.workflows.tools.causal.inference.causal_model_factory_tool import (
    CausalModelFactoryTool,
)
from python.implementation.workflows.tools.causal.specs.causal_specs_tool import (
    CausalSpecsTool,
)
from python.implementation.workflows.tools.causal.validation.validation_backdoor_tool import (
    ValidationBackdoorTool,
)
from python.implementation.workflows.tools.data_manupulation_tool.data_manipulation_tool import (
    DataManipulationTool,
)
from python.implementation.workflows.tools.data_profiling.data_profiling_tool import (
    DatasetProfilingTool,
)
from python.implementation.workflows.tools.plot_tool.plot_tool import PlotTool
from python.implementation.workflows.tools.simple_data_transformation_tool.simple_data_transformation_tool import (
    SimpleDataTransformationTool,
)


class DefaultToolFactory(ToolFactory):
    _tools: dict[str, Tool]

    def __init__(
        self,
        data_repo: DataRepo,
        models_repo: ModelsRepo,
        analytics_repo: AnalyticsRepo,
        llm_service: LLMService,
    ) -> None:
        self._tools = {
            DatasetProfilingTool.NAME: DatasetProfilingTool(),
            DataManipulationTool.NAME: DataManipulationTool(
                llm=llm_service,
                analytics_repo=analytics_repo,
            ),
            PlotTool.NAME: PlotTool(llm=llm_service),
            AdvancedAnalyticsTool.NAME: AdvancedAnalyticsTool(llm=llm_service),
            SimpleDataTransformationTool.NAME: SimpleDataTransformationTool(),
            CausalSpecsTool.NAME: CausalSpecsTool(),
            EncodingPlanTool.NAME: EncodingPlanTool(),
            ValidationBackdoorTool.NAME: ValidationBackdoorTool(),
            CausalModelFactoryTool.NAME: CausalModelFactoryTool.create_default(
                data_repo=data_repo, models_repo=models_repo
            ),
        }

    def get_tool_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def get_tool_info(self, name: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name!r}")
        return tool.get_tool_info()

    def get_tools_info(self) -> dict[str, str]:
        return {k: self._tools[k].get_tool_info() for k in self._tools}

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_tool(self, name: str) -> Any:
        try:
            return self._tools[name]
        except KeyError as e:
            raise KeyError(f"Unknown tool: {name!r}") from e
