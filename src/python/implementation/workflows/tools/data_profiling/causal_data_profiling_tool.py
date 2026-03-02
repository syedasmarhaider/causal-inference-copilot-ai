from __future__ import annotations
from typing import Any, ClassVar, List

import pandas as pd






from python.domain.workflows.tool import Tool
from python.implementation.workflows.tools.data_profiling.plots.causal_missingness_by_group import generate_causal_missingness_by_group_graph
from python.implementation.workflows.tools.data_profiling.plots.comparability_overlap import generate_comparability_overlap_histogram_graph
from python.implementation.workflows.tools.data_profiling.plots.model import GraphImage
from python.implementation.workflows.tools.data_profiling.plots.propensity_vs_key_confounder import generate_propensity_vs_key_confounder_graph




class CausalDataProfilingTool(Tool):
    NAME: ClassVar[str] = "CAUSAL_DATA_PROFILING"
    
    def get_tool_name(self) -> str:
        return self.NAME
    
    def get_tool_info(self) -> str:
        return "Tool for generating causal data profiling graphs, such as propensity score overlap, covariate balance (Love) plots, and weight distribution plots. These graphs help diagnose potential issues with causal inference analyses, such as lack of common support or extreme weights."

    def generate_causal_graphs(
        self,
        df: pd.DataFrame,
        protocol: Any,
        *,
        compute_quantiles: bool = True,
        strict: bool = True,
    ) -> List[GraphImage]:
        """
        Generates a set of causal data profiling graphs based on the input DataFrame and protocol.
        - df: The dataset to profile, after exclusions and preprocessing.
        - protocol: The causal protocol containing treatment, covariate, and outcome specifications.
        - compute_quantiles: Whether to compute quantiles for numeric covariates (may be expensive).
        - strict: If True, raises errors for common issues (e.g., missing columns, invalid treatment values). If False, tries to proceed with best effort and may produce less accurate graphs.
        
        Returns a list of CausalGraphImage objects containing the generated graphs.
        """
        comparability_map = generate_comparability_overlap_histogram_graph(df, protocol) 
        causal_missingness = generate_causal_missingness_by_group_graph(df, protocol)
        propensity_vs_key_confounder = generate_propensity_vs_key_confounder_graph(df, protocol)
    
        return [comparability_map, causal_missingness, propensity_vs_key_confounder]


