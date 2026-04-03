from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from python.domain.repo.data_repo import DataRepo
from python.domain.repo.models_repo import ModelsRepo
from python.domain.workflows.tool import Tool
from python.implementation.workflows.tools.causal.inference.causal_model import CausalModel
from python.implementation.workflows.tools.causal.inference.econml.dml.causal_forest_dml import (
    CausalForestDMLCausalModel,
)
from python.implementation.workflows.tools.causal.inference.econml.dml.kernal_dml import KernelDMLCausalModel
from python.implementation.workflows.tools.causal.inference.econml.dml.linear_dml import LinearDMLCausalModel
from python.implementation.workflows.tools.causal.inference.econml.dml.sparse_linear_dml import (
    SparseLinearDMLCausalModel,
)
from python.implementation.workflows.tools.causal.inference.econml.dr.dr_learner import (
    ForestDRLearnerCausalModel,
    LinearDRLearnerCausalModel,
    SparseLinearDRLearnerCausalModel,
)
from python.implementation.workflows.tools.causal.inference.econml.models_meta import (
    SupportedModelsLiteralType,
)
from python.implementation.workflows.tools.encoding.encoding_util import EncodingUtil


@dataclass
class CausalModelFactoryTool(Tool):
    NAME: ClassVar[str] = "CAUSAL_MODEL_FACTORY"
    _by_fqcn: dict[SupportedModelsLiteralType, CausalModel]

    
    @classmethod
    def create_default(cls, *, data_repo: DataRepo, models_repo: ModelsRepo) -> CausalModelFactoryTool:
        encoding_util = EncodingUtil()
        return cls(
            _by_fqcn={
                "econml.dml.LinearDML": LinearDMLCausalModel(data_repo=data_repo, models_repo=models_repo, encoding_util=encoding_util),
                "econml.dml.SparseLinearDML": SparseLinearDMLCausalModel(data_repo=data_repo, models_repo=models_repo, encoding_util=encoding_util),
                "econml.dml.KernelDML": KernelDMLCausalModel(data_repo=data_repo, models_repo=models_repo, encoding_util=encoding_util),
                "econml.dml.CausalForestDML": CausalForestDMLCausalModel(data_repo=data_repo, models_repo=models_repo, encoding_util=encoding_util),
                "econml.dr.LinearDRLearner": LinearDRLearnerCausalModel(data_repo=data_repo, models_repo=models_repo, encoding_util=encoding_util),
                "econml.dr.SparseLinearDRLearner": SparseLinearDRLearnerCausalModel(data_repo=data_repo, models_repo=models_repo, encoding_util=encoding_util),
                "econml.dr.ForestDRLearner": ForestDRLearnerCausalModel(data_repo=data_repo, models_repo=models_repo, encoding_util=encoding_util),
            }
        )
     
    def get_tool_name(self) -> str:
        return self.NAME
    
    def get_tool_info(self) -> str:
        return "Tool for managing and resolving causal inference models."
        
    def supported_estimators(self) -> list[str]:
        return sorted(self._by_fqcn.keys())
    
    def has_estimator(self, estimator_fqcn: str) -> bool:
        return estimator_fqcn in self._by_fqcn

    def resolve(self, estimator_fqcn: str) -> CausalModel | None:
        if estimator_fqcn not in self._by_fqcn:
            return None
        return self._by_fqcn.get(estimator_fqcn)
    
    def get_all_esimators_info(self) -> dict[str, str]:
        return {fqcn: model.get_info() for fqcn, model in self._by_fqcn.items()}