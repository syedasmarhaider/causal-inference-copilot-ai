from __future__ import annotations
from typing import Any, ClassVar, List

import pandas as pd

from python.domain.workflows.tool import Tool
from python.implementation.workflows.tools.data_profiling.plots.cate_distribution import plot_cate_distribution
from python.implementation.workflows.tools.data_profiling.plots.causal_missingness_by_group import generate_causal_missingness_by_group_graph
from python.implementation.workflows.tools.data_profiling.plots.comparability_overlap import generate_comparability_overlap_histogram_graph
from python.implementation.workflows.tools.data_profiling.plots.model import GraphImage
from python.implementation.workflows.tools.data_profiling.plots.propensity_vs_key_confounder import  generate_propensity_vs_top_confounders_graphs
from python.implementation.workflows.tools.data_profiling.plots.cate_forest import plot_cate_forest_mean_ci
from python.implementation.workflows.tools.data_profiling.plots.cate_sorted_curve import plot_cate_sorted_curve




class CausalDataProfilingTool(Tool):
    NAME: ClassVar[str] = "CAUSAL_DATA_PROFILING"
    
    def get_tool_name(self) -> str:
        return self.NAME
    
    def get_tool_info(self) -> str:
        return "Tool for generating causal data profiling graphs, such as propensity score overlap, covariate balance (Love) plots, and weight distribution plots. These graphs help diagnose potential issues with causal inference analyses, such as lack of common support or extreme weights."
    
    def generate_comparability_overlap_histogram(self, df: pd.DataFrame, protocol: Any) -> GraphImage:
        return generate_comparability_overlap_histogram_graph(df, protocol)
    
    def generate_causal_missingness_by_group_graph(self, df: pd.DataFrame, protocol: Any) -> GraphImage:
        return generate_causal_missingness_by_group_graph(df, protocol)
    
    def generate_propensity_vs_top_confounders_graphs(self, df: pd.DataFrame, protocol: Any) -> List[GraphImage]:
        return generate_propensity_vs_top_confounders_graphs(df, protocol)
    
    def plot_cate_distribution(self, cohorts: List[Any], protocol: Any) -> GraphImage:
        return plot_cate_distribution(cohorts, key="cate_distribution", title="CATE distribution")
    
    def plot_cate_forest_mean_ci(self, cohorts: List[Any], protocol: Any) -> GraphImage:
        return plot_cate_forest_mean_ci(cohorts, key="cate_forest_mean_ci", title="Mean CATE by cohort (bootstrap CI)")
    
    def plot_cate_sorted_curve(self, cohorts: List[Any], protocol: Any) -> GraphImage:
        return plot_cate_sorted_curve(cohorts, key="cate_sorted_curve", title="Sorted CATE curve (heterogeneity shape)")

